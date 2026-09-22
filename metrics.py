"""Opt-in, content-free metrics for local server usage."""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Mapping


EVENT_NAMES = (
    "captures",
    "searches",
    "empty_searches",
    "capture_to_search",
    "retrievals",
    "search_to_retrieval",
    "evictions",
    "cleanups",
)

BYTE_COUNTER_NAMES = (
    "capture_input_bytes",
    "capture_retained_bytes",
    "capture_original_bytes",
    "tool_response_bytes",
    "search_response_bytes",
    "retrieval_response_bytes",
    "socket_request_bytes",
    "socket_response_bytes",
)

# Fixed buckets keep latency state bounded and make the distribution additive
# across snapshot-token windows. Percentiles are reported as the upper bound of
# the bucket containing the nearest-rank sample.
LATENCY_BUCKET_UPPER_BOUNDS_MS = (
    1,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
    30_000,
    60_000,
)
FAILURE_CATEGORIES = (
    "validation",
    "timeout",
    "socket",
    "embedding",
    "eviction",
    "other",
)

SEMANTIC_INDEX_SOURCES = ("prefetch", "on_demand")
SEMANTIC_INDEX_OUTCOMES = (
    "queued",
    "completed",
    "failed",
    "cancelled",
    "evicted",
    "cleared",
)
SEMANTIC_SEARCH_OUTCOMES = ("pending_hybrid_responses", "semantic_fallbacks")

MAX_SNAPSHOT_TOKENS = 128
MAX_SCOPE_STATES = 128

def metrics_enabled() -> bool:
    """Return whether local metrics were explicitly enabled."""
    return os.environ.get("EPHEMERAL_METRICS", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


class LocalMetrics:
    """Bounded in-process counters and timers with no content storage."""

    def __init__(self, enabled: bool | None = None):
        self.enabled = metrics_enabled() if enabled is None else enabled
        self._lock = threading.RLock()
        self._scope_context: ContextVar[str] = ContextVar(
            "metrics_scope", default="process"
        )
        self._scopes: OrderedDict[str, dict[str, Any]] = OrderedDict(
            {
                "process": self._new_runtime_state(
                    "process",
                    {
                        "kind": "process",
                        "mode": "aggregate",
                        "id": f"proc_{secrets.token_urlsafe(12)}",
                    },
                )
            }
        )
        self._active_measurement: ContextVar[dict[str, Any] | None] = ContextVar(
            "active_measurement", default=None
        )
        self._snapshot_tokens: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @classmethod
    def _new_runtime_state(
        cls,
        scope_key: str,
        attribution: Mapping[str, str],
    ) -> dict[str, Any]:
        return {
            "scope_key": scope_key,
            "started_at": time.time(),
            "attribution": dict(attribution),
            "tools": defaultdict(cls._new_tool),
            "events": defaultdict(int),
            "bytes": defaultdict(int),
            "semantic_index": cls._new_semantic_state(),
            "captured": set(),
            "searched": set(),
            "in_flight_measurements": {},
            "pins": 0,
        }

    def _scope_state(
        self,
        scope_key: str | None = None,
        *,
        attribution: Mapping[str, str] | None = None,
        pin: bool = False,
    ) -> dict[str, Any]:
        key = scope_key or self._scope_context.get()
        with self._lock:
            state = self._scopes.get(key)
            if state is not None:
                self._scopes.move_to_end(key)
                if pin:
                    state["pins"] += 1
            else:
                while len(self._scopes) >= MAX_SCOPE_STATES:
                    evictable_key = next(
                        (
                            candidate
                            for candidate, candidate_state in self._scopes.items()
                            if candidate != "process"
                            and not candidate_state["pins"]
                            and not candidate_state["in_flight_measurements"]
                        ),
                        None,
                    )
                    if evictable_key is None:
                        # Active calls keep their state alive until their
                        # measurements finalize. Attribute overflow activity
                        # to the aggregate rather than exceeding the hard
                        # cache bound while all client states are active.
                        state = self._scopes["process"]
                        if pin:
                            state["pins"] += 1
                        return state
                    self._scopes.pop(evictable_key, None)
                    self._discard_scope_tokens(evictable_key)
                state = self._new_runtime_state(
                    key,
                    attribution
                    or {
                        "kind": "mcp_session",
                        "mode": "private",
                        "id": f"scope_{secrets.token_urlsafe(12)}",
                    },
                )
                self._scopes[key] = state
                if pin:
                    state["pins"] += 1
            return state

    def _discard_scope_tokens(self, scope_key: str) -> None:
        """Drop task-window baselines when their scope leaves the cache."""
        for token, state in tuple(self._snapshot_tokens.items()):
            if state.get("scope_key") == scope_key:
                self._snapshot_tokens.pop(token, None)

    def _current_scope_states(
        self,
        measurement: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return the aggregate state and the active client state."""
        if measurement is not None:
            states = measurement["states"]
        else:
            scoped = self._scope_state()
            aggregate = self._scopes["process"]
            states = (aggregate,) if scoped is aggregate else (aggregate, scoped)
        return states

    @contextmanager
    def bind_scope(
        self,
        scope_key: str,
        *,
        attribution: Mapping[str, str] | None = None,
    ) -> Iterator[None]:
        """Bind subsequent metric writes and snapshots to one MCP session."""
        if not self.enabled or scope_key == "process":
            yield
            return
        self._scope_state(scope_key, attribution=attribution)
        token = self._scope_context.set(scope_key)
        try:
            yield
        finally:
            self._scope_context.reset(token)

    # These compatibility views keep the internal state inspection used by
    # existing embedders/tests scoped to the current context.
    @property
    def _started_at(self) -> float:
        return self._scope_state()["started_at"]

    @property
    def _tools(self) -> dict[str, dict[str, Any]]:
        return self._scope_state()["tools"]

    @property
    def _events(self) -> dict[str, int]:
        return self._scope_state()["events"]

    @property
    def _bytes(self) -> dict[str, int]:
        return self._scope_state()["bytes"]

    @property
    def _semantic_index(self) -> dict[str, Any]:
        return self._scope_state()["semantic_index"]

    @property
    def _captured(self) -> set[str]:
        return self._scope_state()["captured"]

    @property
    def _searched(self) -> set[str]:
        return self._scope_state()["searched"]

    @property
    def _in_flight_measurements(self) -> dict[int, dict[str, Any]]:
        return self._scope_state()["in_flight_measurements"]

    @staticmethod
    def _new_tool() -> dict[str, Any]:
        return {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_duration_ms": 0.0,
            "max_duration_ms": 0.0,
            "result_count": 0,
            "latency_buckets": [0] * (len(LATENCY_BUCKET_UPPER_BOUNDS_MS) + 1),
            "failure_categories": {category: 0 for category in FAILURE_CATEGORIES},
        }

    @staticmethod
    def _new_duration_state() -> dict[str, Any]:
        return {
            "total_ms": 0.0,
            "max_ms": 0.0,
            "buckets": [0] * (len(LATENCY_BUCKET_UPPER_BOUNDS_MS) + 1),
        }

    @classmethod
    def _new_semantic_state(cls) -> dict[str, Any]:
        return {
            source: {
                **{outcome: 0 for outcome in SEMANTIC_INDEX_OUTCOMES},
                "indexed_chunks": 0,
                "queue_wait_ms": cls._new_duration_state(),
                "indexing_duration_ms": cls._new_duration_state(),
            }
            for source in SEMANTIC_INDEX_SOURCES
        } | {
            "search": {outcome: 0 for outcome in SEMANTIC_SEARCH_OUTCOMES},
        }

    @classmethod
    def _copy_semantic_state(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        copied = {}
        for source in SEMANTIC_INDEX_SOURCES:
            source_state = state.get(source, {})
            copied[source] = {
                outcome: int(source_state.get(outcome, 0))
                for outcome in SEMANTIC_INDEX_OUTCOMES
            }
            copied[source]["indexed_chunks"] = int(source_state.get("indexed_chunks", 0))
            for duration_name in ("queue_wait_ms", "indexing_duration_ms"):
                duration = source_state.get(duration_name, {})
                copied[source][duration_name] = {
                    "total_ms": float(duration.get("total_ms", 0.0)),
                    "max_ms": float(duration.get("max_ms", 0.0)),
                    "buckets": list(duration.get("buckets", [])),
                }
        search = state.get("search", {})
        copied["search"] = {
            outcome: int(search.get(outcome, 0))
            for outcome in SEMANTIC_SEARCH_OUTCOMES
        }
        return copied

    @classmethod
    def _record_semantic_index_in(
        cls,
        target: dict[str, Any],
        source: str,
        outcome: str,
        *,
        queue_wait_ms: float | None = None,
        indexing_duration_ms: float | None = None,
        indexed_chunks: int = 0,
    ) -> None:
        if source not in SEMANTIC_INDEX_SOURCES:
            raise ValueError(f"unknown semantic index source: {source}")
        if outcome not in SEMANTIC_INDEX_OUTCOMES:
            raise ValueError(f"unknown semantic index outcome: {outcome}")
        source_state = target[source]
        source_state[outcome] += 1
        source_state["indexed_chunks"] += max(0, int(indexed_chunks))
        for duration_name, duration_ms in (
            ("queue_wait_ms", queue_wait_ms),
            ("indexing_duration_ms", indexing_duration_ms),
        ):
            if duration_ms is None:
                continue
            duration = max(0.0, float(duration_ms))
            duration_state = source_state[duration_name]
            duration_state["total_ms"] += duration
            duration_state["max_ms"] = max(duration_state["max_ms"], duration)
            duration_state["buckets"][cls._latency_bucket_index(duration)] += 1

    @classmethod
    def _subtract_semantic_state(
        cls,
        target: dict[str, Any],
        subtract: Mapping[str, Any],
    ) -> None:
        for source in SEMANTIC_INDEX_SOURCES:
            target_source = target[source]
            subtract_source = subtract.get(source, {})
            for outcome in SEMANTIC_INDEX_OUTCOMES:
                target_source[outcome] = max(
                    0,
                    target_source[outcome] - int(subtract_source.get(outcome, 0)),
                )
            target_source["indexed_chunks"] = max(
                0,
                target_source["indexed_chunks"]
                - int(subtract_source.get("indexed_chunks", 0)),
            )
            for duration_name in ("queue_wait_ms", "indexing_duration_ms"):
                target_duration = target_source[duration_name]
                subtract_duration = subtract_source.get(duration_name, {})
                target_duration["total_ms"] = max(
                    0.0,
                    target_duration["total_ms"]
                    - float(subtract_duration.get("total_ms", 0.0)),
                )
                target_duration["buckets"] = [
                    max(0, current - int(previous))
                    for current, previous in zip(
                        target_duration["buckets"],
                        list(subtract_duration.get("buckets", []))
                        + [0] * len(target_duration["buckets"]),
                    )
                ]
        target_search = target["search"]
        subtract_search = subtract.get("search", {})
        for outcome in SEMANTIC_SEARCH_OUTCOMES:
            target_search[outcome] = max(
                0,
                target_search[outcome] - int(subtract_search.get(outcome, 0)),
            )

    @staticmethod
    def _latency_bucket_index(duration_ms: float) -> int:
        # Timer subtraction can produce values such as 10.000000000000009
        # for an exact boundary in tests or on a stable clock. Normalize the
        # value before assigning it so boundary semantics are deterministic.
        duration_ms = round(max(0.0, float(duration_ms)), 6)
        for index, upper_bound in enumerate(LATENCY_BUCKET_UPPER_BOUNDS_MS):
            if duration_ms <= upper_bound:
                return index
        return len(LATENCY_BUCKET_UPPER_BOUNDS_MS)

    @staticmethod
    def _latency_distribution(buckets: Iterable[int]) -> dict[str, Any]:
        counts = [max(0, int(count)) for count in buckets]
        expected_count = len(LATENCY_BUCKET_UPPER_BOUNDS_MS) + 1
        if len(counts) != expected_count:
            counts = (counts + [0] * expected_count)[:expected_count]
        count = sum(counts)

        def percentile(percentile_value: float) -> float | None:
            if not count:
                return None
            rank = max(1, int((count * percentile_value) + 0.999999999))
            cumulative = 0
            bucket_index = len(counts) - 1
            for index, bucket_count in enumerate(counts):
                cumulative += bucket_count
                if cumulative >= rank:
                    bucket_index = index
                    break
            if bucket_index < len(LATENCY_BUCKET_UPPER_BOUNDS_MS):
                return float(LATENCY_BUCKET_UPPER_BOUNDS_MS[bucket_index])
            return None

        return {
            "count": count,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "histogram": [
                {
                    "upper_bound_ms": float(upper_bound),
                    "count": counts[index],
                }
                for index, upper_bound in enumerate(LATENCY_BUCKET_UPPER_BOUNDS_MS)
            ] + [{"upper_bound_ms": None, "count": counts[-1]}],
            "overflow_count": counts[-1],
        }

    @contextmanager
    def measure(self, tool: str) -> Iterator[dict[str, Any]]:
        """Measure one tool call; the yielded state may contain result_count.

        Count the call at entry so snapshots generated by a diagnostic tool
        include that in-flight tool in interface coverage.
        """
        scoped_state = self._scope_state(pin=self.enabled)
        try:
            aggregate_state = self._scopes["process"]
            target_states = (
                (aggregate_state,)
                if scoped_state is aggregate_state
                else (aggregate_state, scoped_state)
            )
        except Exception:
            if self.enabled:
                with self._lock:
                    scoped_state["pins"] = max(0, scoped_state["pins"] - 1)
            raise
        state = {
            "success": True,
            "result_count": None,
            "failure_category": None,
            "record": True,
            "tool_existed": tool in scoped_state["tools"],
        }
        started = time.perf_counter()
        measurement = {
            "tool": tool,
            "events_by_scope": {
                target["scope_key"]: defaultdict(int)
                for target in target_states
            },
            "bytes": defaultdict(int),
            "semantic_index": self._new_semantic_state(),
            "result_count": 0,
            "states": target_states,
            "scope_state": scoped_state,
        }
        measurement_token = None
        if self.enabled:
            measurement_token = self._active_measurement.set(measurement)
            with self._lock:
                for target in target_states:
                    target["tools"][tool]["calls"] += 1
                    target["in_flight_measurements"][id(measurement)] = measurement
                scoped_state["pins"] = max(0, scoped_state["pins"] - 1)
        try:
            yield state
        except Exception:
            state["success"] = False
            raise
        finally:
            if self.enabled:
                duration_ms = (time.perf_counter() - started) * 1000
                with self._lock:
                    for target in target_states:
                        stats = target["tools"][tool]
                        if state["record"]:
                            if state["success"]:
                                stats["successes"] += 1
                            else:
                                stats["failures"] += 1
                                category = state.get("failure_category")
                                if category not in FAILURE_CATEGORIES:
                                    category = "other"
                                stats["failure_categories"][category] += 1
                            stats["total_duration_ms"] += duration_ms
                            stats["max_duration_ms"] = max(stats["max_duration_ms"], duration_ms)
                            stats["latency_buckets"][self._latency_bucket_index(duration_ms)] += 1
                            if state["result_count"] is not None:
                                stats["result_count"] += int(state["result_count"])
                        else:
                            stats["calls"] = max(0, stats["calls"] - 1)
                            if (
                                not state["tool_existed"]
                                and stats["calls"] == 0
                                and target["tools"].get(tool) is stats
                            ):
                                target["tools"].pop(tool, None)
                        target["in_flight_measurements"].pop(id(measurement), None)
                self._active_measurement.reset(measurement_token)

    def record_event(self, event: str) -> None:
        if self.enabled:
            with self._lock:
                measurement = self._active_measurement.get()
                targets = self._current_scope_states(measurement)
                for target in targets:
                    target["events"][event] += 1
                if measurement is not None:
                    for target in targets:
                        measurement["events_by_scope"][target["scope_key"]][event] += 1

    def record_bytes(self, counter: str, amount: int) -> None:
        """Record a non-content byte count for the current server session."""
        if self.enabled:
            if counter not in BYTE_COUNTER_NAMES:
                raise ValueError(f"unknown byte counter: {counter}")
            with self._lock:
                amount = max(0, int(amount))
                measurement = self._active_measurement.get()
                for target in self._current_scope_states(measurement):
                    target["bytes"][counter] += amount
                if measurement is not None:
                    measurement["bytes"][counter] += amount

    def record_result_count(self, tool: str, count: int) -> None:
        if self.enabled:
            with self._lock:
                count = max(0, int(count))
                measurement = self._active_measurement.get()
                for target in self._current_scope_states(measurement):
                    target["tools"][tool]["result_count"] += count
                if measurement is not None and measurement["tool"] == tool:
                    measurement["result_count"] += count

    def measurement_handle(self) -> dict[str, Any] | None:
        """Return the current tool measurement for asynchronous attribution."""
        if not self.enabled:
            return None
        return self._active_measurement.get()

    def record_semantic_index(
        self,
        source: str,
        outcome: str,
        *,
        queue_wait_ms: float | None = None,
        indexing_duration_ms: float | None = None,
        indexed_chunks: int = 0,
        measurement: dict[str, Any] | None = None,
    ) -> None:
        """Record content-free semantic indexing lifecycle work."""
        if self.enabled:
            with self._lock:
                if measurement is None:
                    measurement = self._active_measurement.get()
                for target in self._current_scope_states(measurement):
                    self._record_semantic_index_in(
                        target["semantic_index"],
                        source,
                        outcome,
                        queue_wait_ms=queue_wait_ms,
                        indexing_duration_ms=indexing_duration_ms,
                        indexed_chunks=indexed_chunks,
                    )
                if measurement is not None:
                    self._record_semantic_index_in(
                        measurement["semantic_index"],
                        source,
                        outcome,
                        queue_wait_ms=queue_wait_ms,
                        indexing_duration_ms=indexing_duration_ms,
                        indexed_chunks=indexed_chunks,
                    )

    def record_semantic_search(self, outcome: str) -> None:
        """Record a content-free semantic search outcome."""
        if self.enabled:
            if outcome not in SEMANTIC_SEARCH_OUTCOMES:
                raise ValueError(f"unknown semantic search outcome: {outcome}")
            with self._lock:
                measurement = self._active_measurement.get()
                for target in self._current_scope_states(measurement):
                    target["semantic_index"]["search"][outcome] += 1
                if measurement is not None:
                    measurement["semantic_index"]["search"][outcome] += 1

    def record_capture(self, capture_id: str) -> None:
        if self.enabled:
            with self._lock:
                measurement = self._active_measurement.get()
                targets = self._current_scope_states(measurement)
                for target in targets:
                    target["captured"].add(capture_id)
                    target["events"]["captures"] += 1
                if measurement is not None:
                    for target in targets:
                        measurement["events_by_scope"][target["scope_key"]]["captures"] += 1

    def record_search(self, capture_id: str, match_count: int) -> None:
        if self.enabled:
            with self._lock:
                measurement = self._active_measurement.get()
                targets = self._current_scope_states(measurement)
                for target in targets:
                    if capture_id in target["captured"]:
                        target["events"]["capture_to_search"] += 1
                        target["searched"].add(capture_id)
                        if measurement is not None:
                            measurement["events_by_scope"][target["scope_key"]][
                                "capture_to_search"
                            ] += 1
                    target["events"]["searches"] += 1
                    target["events"]["empty_searches"] += int(match_count == 0)
                if measurement is not None:
                    for target in targets:
                        target_events = measurement["events_by_scope"][target["scope_key"]]
                        target_events["searches"] += 1
                        target_events["empty_searches"] += int(match_count == 0)

    def record_retrieval(self, capture_id: str) -> None:
        if self.enabled:
            with self._lock:
                measurement = self._active_measurement.get()
                targets = self._current_scope_states(measurement)
                for target in targets:
                    if capture_id in target["searched"]:
                        target["events"]["search_to_retrieval"] += 1
                        if measurement is not None:
                            measurement["events_by_scope"][target["scope_key"]][
                                "search_to_retrieval"
                            ] += 1
                    target["events"]["retrievals"] += 1
                if measurement is not None:
                    for target in targets:
                        measurement["events_by_scope"][target["scope_key"]]["retrievals"] += 1

    def forget_capture(self, capture_id: str) -> None:
        if self.enabled:
            with self._lock:
                for target in self._scopes.values():
                    target["captured"].discard(capture_id)
                    target["searched"].discard(capture_id)

    @staticmethod
    def _format_timestamp(timestamp: float) -> str:
        return datetime.fromtimestamp(
            timestamp, tz=timezone.utc
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _current_state(
        self,
        snapshot_at: float,
        *,
        runtime_state: dict[str, Any] | None = None,
        exclude_in_flight: bool = False,
    ) -> dict[str, Any]:
        """Copy additive counters and tool state for a snapshot boundary."""
        runtime_state = runtime_state or self._scope_state()
        tools = {}
        for name, stats in runtime_state["tools"].items():
            copied_stats = dict(stats)
            copied_stats["latency_buckets"] = list(stats["latency_buckets"])
            copied_stats["failure_categories"] = dict(stats["failure_categories"])
            tools[name] = copied_stats
        events = {
            name: runtime_state["events"][name]
            for name in EVENT_NAMES
        }
        bytes_snapshot = {
            name: runtime_state["bytes"][name]
            for name in BYTE_COUNTER_NAMES
        }
        semantic_index = self._copy_semantic_state(runtime_state["semantic_index"])
        if exclude_in_flight:
            for measurement in runtime_state["in_flight_measurements"].values():
                tool_stats = tools.get(measurement["tool"])
                if tool_stats is not None:
                    tool_stats["calls"] = max(0, tool_stats["calls"] - 1)
                    tool_stats["result_count"] = max(
                        0,
                        tool_stats["result_count"] - measurement["result_count"],
                    )
                for name, amount in measurement["events_by_scope"].get(
                    runtime_state["scope_key"], {}
                ).items():
                    if name in events:
                        events[name] = max(0, events[name] - amount)
                for name, amount in measurement["bytes"].items():
                    if name in bytes_snapshot:
                        bytes_snapshot[name] = max(0, bytes_snapshot[name] - amount)
                self._subtract_semantic_state(
                    semantic_index,
                    measurement["semantic_index"],
                )
        return {
            "snapshot_at": snapshot_at,
            "tools": tools,
            "events": events,
            "bytes": bytes_snapshot,
            "semantic_index": semantic_index,
        }

    def _store_snapshot_token(self, state: dict[str, Any]) -> str:
        token = secrets.token_urlsafe(18)
        self._snapshot_tokens[token] = state
        self._snapshot_tokens.move_to_end(token)
        while len(self._snapshot_tokens) > MAX_SNAPSHOT_TOKENS:
            self._snapshot_tokens.popitem(last=False)
        return token

    @staticmethod
    def _public_scope_name(runtime_state: Mapping[str, Any]) -> str:
        return "process" if runtime_state.get("scope_key") == "process" else "mcp_session"

    @staticmethod
    def _delta_value(current: Any, baseline: Any) -> Any:
        return max(0, current - baseline)

    def _delta_tools(
        self,
        current_tools: Mapping[str, Mapping[str, Any]],
        baseline_tools: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Return tools with activity since the requested baseline.

        ``max_duration_ms`` is a non-additive process gauge, so delta windows
        retain the current process maximum while additive fields are reduced.
        """
        delta_tools: dict[str, dict[str, Any]] = {}
        additive_fields = (
            "calls",
            "successes",
            "failures",
            "total_duration_ms",
            "result_count",
        )
        for name in sorted(set(current_tools) | set(baseline_tools)):
            current = current_tools.get(name, self._new_tool())
            baseline = baseline_tools.get(name, self._new_tool())
            stats = {
                field: self._delta_value(current[field], baseline[field])
                for field in additive_fields
            }
            stats["max_duration_ms"] = current["max_duration_ms"]
            stats["latency_buckets"] = [
                self._delta_value(current_bucket, baseline_bucket)
                for current_bucket, baseline_bucket in zip(
                    current["latency_buckets"], baseline["latency_buckets"]
                )
            ]
            stats["failure_categories"] = {
                category: self._delta_value(
                    current["failure_categories"].get(category, 0),
                    baseline["failure_categories"].get(category, 0),
                )
                for category in FAILURE_CATEGORIES
            }
            if (
                any(stats[field] for field in additive_fields)
                or any(stats["latency_buckets"])
                or any(stats["failure_categories"].values())
            ):
                delta_tools[name] = stats
        return delta_tools

    @classmethod
    def _public_tool_stats(cls, stats: Mapping[str, Any]) -> dict[str, Any]:
        """Project internal additive distribution state into the wire schema."""
        return {
            "calls": stats["calls"],
            "successes": stats["successes"],
            "failures": stats["failures"],
            "total_duration_ms": round(stats["total_duration_ms"], 3),
            "max_duration_ms": round(stats["max_duration_ms"], 3),
            "result_count": stats["result_count"],
            "latency_ms": cls._latency_distribution(stats["latency_buckets"]),
            "failure_categories": dict(stats["failure_categories"]),
        }

    @classmethod
    def _delta_semantic_state(
        cls,
        current: Mapping[str, Any],
        baseline: Mapping[str, Any],
    ) -> dict[str, Any]:
        delta = cls._new_semantic_state()
        for source in SEMANTIC_INDEX_SOURCES:
            current_source = current.get(source, {})
            baseline_source = baseline.get(source, {})
            delta_source = delta[source]
            for outcome in SEMANTIC_INDEX_OUTCOMES:
                delta_source[outcome] = cls._delta_value(
                    int(current_source.get(outcome, 0)),
                    int(baseline_source.get(outcome, 0)),
                )
            delta_source["indexed_chunks"] = cls._delta_value(
                int(current_source.get("indexed_chunks", 0)),
                int(baseline_source.get("indexed_chunks", 0)),
            )
            for duration_name in ("queue_wait_ms", "indexing_duration_ms"):
                current_duration = current_source.get(duration_name, {})
                baseline_duration = baseline_source.get(duration_name, {})
                delta_duration = delta_source[duration_name]
                delta_duration["total_ms"] = max(
                    0.0,
                    float(current_duration.get("total_ms", 0.0))
                    - float(baseline_duration.get("total_ms", 0.0)),
                )
                delta_duration["max_ms"] = float(current_duration.get("max_ms", 0.0))
                delta_duration["buckets"] = [
                    cls._delta_value(current_bucket, baseline_bucket)
                    for current_bucket, baseline_bucket in zip(
                        list(current_duration.get("buckets", []))
                        + [0] * (len(LATENCY_BUCKET_UPPER_BOUNDS_MS) + 1),
                        list(baseline_duration.get("buckets", []))
                        + [0] * (len(LATENCY_BUCKET_UPPER_BOUNDS_MS) + 1),
                    )
                ][: len(LATENCY_BUCKET_UPPER_BOUNDS_MS) + 1]
        current_search = current.get("search", {})
        baseline_search = baseline.get("search", {})
        for outcome in SEMANTIC_SEARCH_OUTCOMES:
            delta["search"][outcome] = cls._delta_value(
                int(current_search.get(outcome, 0)),
                int(baseline_search.get(outcome, 0)),
            )
        return delta

    @classmethod
    def _public_duration_stats(cls, duration: Mapping[str, Any]) -> dict[str, Any]:
        distribution = cls._latency_distribution(duration.get("buckets", []))
        return {
            "count": distribution["count"],
            "total_ms": round(float(duration.get("total_ms", 0.0)), 3),
            "max_ms": round(float(duration.get("max_ms", 0.0)), 3),
            "p50": distribution["p50"],
            "p95": distribution["p95"],
            "p99": distribution["p99"],
            "overflow_count": distribution["overflow_count"],
        }

    @staticmethod
    def _throughput_metric(indexed_chunks: int, indexing_duration_ms: float) -> dict[str, Any]:
        metric = {
            "status": "ok" if indexed_chunks and indexing_duration_ms else "unavailable",
            "indexed_chunks": indexed_chunks,
            "indexing_duration_ms": round(indexing_duration_ms, 3),
            "chunks_per_second": round(
                indexed_chunks / (indexing_duration_ms / 1000), 3
            ) if indexed_chunks and indexing_duration_ms else None,
        }
        if not indexed_chunks:
            metric["reason"] = "zero_indexed_chunks"
        elif not indexing_duration_ms:
            metric["reason"] = "zero_indexing_duration"
        return metric

    @classmethod
    def _public_semantic_index(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        result = {}
        for source in SEMANTIC_INDEX_SOURCES:
            source_state = state.get(source, {})
            result[source] = {
                outcome: int(source_state.get(outcome, 0))
                for outcome in SEMANTIC_INDEX_OUTCOMES
            }
            result[source]["indexed_chunks"] = int(source_state.get("indexed_chunks", 0))
            result[source]["queue_wait_ms"] = cls._public_duration_stats(
                source_state.get("queue_wait_ms", {})
            )
            result[source]["indexing_duration_ms"] = cls._public_duration_stats(
                source_state.get("indexing_duration_ms", {})
            )
            result[source]["throughput"] = cls._throughput_metric(
                int(source_state.get("indexed_chunks", 0)),
                float(
                    source_state.get("indexing_duration_ms", {}).get(
                        "total_ms", 0.0
                    )
                ),
            )
        result["search"] = {
            outcome: int(state.get("search", {}).get(outcome, 0))
            for outcome in SEMANTIC_SEARCH_OUTCOMES
        }
        return result

    @staticmethod
    def _percentage_metric(
        numerator: int,
        denominator: int,
        denominator_name: str,
    ) -> dict[str, Any]:
        metric = {
            "status": "ok" if denominator else "unavailable",
            "numerator": numerator,
            "denominator": denominator,
            "denominator_name": denominator_name,
            "percentage": round(numerator / denominator * 100, 1)
            if denominator
            else None,
        }
        if not denominator:
            metric["reason"] = "zero_denominator"
        return metric

    @staticmethod
    def _ratio_metric(
        numerator: int,
        denominator: int,
        denominator_name: str,
    ) -> dict[str, Any]:
        metric = {
            "status": "ok" if denominator else "unavailable",
            "numerator": numerator,
            "denominator": denominator,
            "denominator_name": denominator_name,
            "ratio": round(numerator / denominator, 4) if denominator else None,
        }
        if not denominator:
            metric["reason"] = "zero_denominator"
        return metric

    @staticmethod
    def _reduction_metric(
        response_bytes: int,
        baseline_bytes: int,
        operation_count: int,
    ) -> dict[str, Any]:
        metric = {
            "status": "ok" if baseline_bytes and operation_count else "unavailable",
            "numerator": baseline_bytes - response_bytes,
            "denominator": baseline_bytes,
            "denominator_name": "capture_input_bytes",
            "operation_count": operation_count,
            "percentage": round(
                (baseline_bytes - response_bytes) / baseline_bytes * 100, 1
            ) if baseline_bytes and operation_count else None,
        }
        if not baseline_bytes:
            metric["reason"] = "zero_denominator"
        elif not operation_count:
            metric["reason"] = "zero_operation_count"
        return metric

    @classmethod
    def _workflow_effectiveness(
        cls,
        tools: Mapping[str, Mapping[str, Any]],
        events: Mapping[str, int],
        bytes_snapshot: Mapping[str, int],
    ) -> dict[str, dict[str, Any]]:
        """Derive operational funnel and response-size signals."""
        successful_calls = sum(stats["successes"] for stats in tools.values())
        failed_calls = sum(stats["failures"] for stats in tools.values())
        captured_bytes = bytes_snapshot["capture_input_bytes"]
        return {
            "capture_to_search_rate": cls._percentage_metric(
                events["capture_to_search"],
                events["searches"],
                "searches",
            ),
            "search_to_retrieval_rate": cls._percentage_metric(
                events["search_to_retrieval"],
                events["retrievals"],
                "retrievals",
            ),
            "empty_search_rate": cls._percentage_metric(
                events["empty_searches"],
                events["searches"],
                "searches",
            ),
            "successful_call_rate": cls._percentage_metric(
                successful_calls,
                successful_calls + failed_calls,
                "completed_calls",
            ),
            "response_bytes_per_captured_byte": cls._ratio_metric(
                bytes_snapshot["tool_response_bytes"],
                captured_bytes,
                "capture_input_bytes",
            ),
            "search_response_reduction": cls._reduction_metric(
                bytes_snapshot["search_response_bytes"],
                captured_bytes,
                events["searches"],
            ),
            "retrieval_response_reduction": cls._reduction_metric(
                bytes_snapshot["retrieval_response_bytes"],
                captured_bytes,
                events["retrievals"],
            ),
        }

    def snapshot(
        self,
        available_tools: Iterable[str] = (),
        tool_categories: Mapping[str, str] | None = None,
        since_snapshot: str | None = None,
        include_snapshot_token: bool = False,
        scope_key: str | None = None,
    ) -> dict[str, Any]:
        """Return scoped metrics suitable for a content-free diagnostic.

        ``available_tools`` should be the live interface inventory when the
        metrics are used by an MCP server. Standalone metric users can omit it
        when interface coverage is not applicable. When supplied,
        ``tool_categories`` maps each available tool to its primary capability
        category for descriptive per-category coverage. ``since_snapshot``
        requests a scope-local delta from a token previously returned by a
        token-enabled snapshot. Tokens cannot be used across MCP sessions.
        Delta baselines are independent and do not reset cumulative metrics.
        Calls still in flight at the boundary are excluded from delta state
        and attributed to the following window. ``scope_key="process"``
        explicitly selects the aggregate process view.
        """
        if not self.enabled:
            return {"enabled": False}
        with self._lock:
            snapshot_at = time.time()
            runtime_state = self._scope_state(scope_key)
            current_state = self._current_state(
                snapshot_at,
                runtime_state=runtime_state,
                exclude_in_flight=since_snapshot is not None,
            )
            baseline = (
                self._snapshot_tokens.get(since_snapshot)
                if since_snapshot is not None
                else None
            )
            if baseline is not None and baseline.get("scope_key") != runtime_state["scope_key"]:
                baseline = None
            return_token = include_snapshot_token or since_snapshot is not None
            snapshot_token = (
                self._store_snapshot_token(
                    {
                        **self._current_state(
                            snapshot_at,
                            runtime_state=runtime_state,
                            exclude_in_flight=True,
                        ),
                        "scope_key": runtime_state["scope_key"],
                    }
                )
                if return_token
                else None
            )
            if since_snapshot is not None and baseline is None:
                response = {
                    "enabled": True,
                    "scope": self._public_scope_name(runtime_state),
                    "attribution": dict(runtime_state["attribution"]),
                    "started_at": self._format_timestamp(runtime_state["started_at"]),
                    "snapshot_at": self._format_timestamp(snapshot_at),
                    "window": {
                        "status": "unavailable",
                        "kind": "delta",
                        "started_at": None,
                        "ended_at": self._format_timestamp(snapshot_at),
                        "reason": "snapshot_token_unavailable",
                    },
                }
                if snapshot_token is not None:
                    response["snapshot_token"] = snapshot_token
                return response

            state = (
                current_state
                if baseline is None
                else {
                    "tools": self._delta_tools(
                        current_state["tools"], baseline["tools"]
                    ),
                    "events": {
                        name: self._delta_value(
                            current_state["events"].get(name, 0),
                            baseline["events"].get(name, 0),
                        )
                        for name in EVENT_NAMES
                    },
                    "bytes": {
                        name: self._delta_value(
                            current_state["bytes"].get(name, 0),
                            baseline["bytes"].get(name, 0),
                        )
                        for name in BYTE_COUNTER_NAMES
                    },
                    "semantic_index": self._delta_semantic_state(
                        current_state["semantic_index"],
                        baseline.get("semantic_index", self._new_semantic_state()),
                    ),
                }
            )
            available_tool_names = tuple(dict.fromkeys(available_tools))
            used_tool_names = set(state["tools"])
            used_tools = [name for name in available_tool_names if name in used_tool_names]
            unused_tools = [name for name in available_tool_names if name not in used_tool_names]
            available_tool_count = len(available_tool_names)
            category_coverage: dict[str, dict[str, Any]] = {}
            if tool_categories is not None:
                tools_by_category: dict[str, list[str]] = defaultdict(list)
                for name in available_tool_names:
                    category = tool_categories.get(name, "uncategorized")
                    tools_by_category[category].append(name)
                for category in sorted(tools_by_category):
                    category_tools = tools_by_category[category]
                    category_used = [name for name in category_tools if name in used_tool_names]
                    category_unused = [name for name in category_tools if name not in used_tool_names]
                    category_available = len(category_tools)
                    category_coverage[category] = {
                        "used": len(category_used),
                        "available": category_available,
                        "percentage": round(
                            len(category_used) / category_available * 100, 1
                        ) if category_available else 0.0,
                        "unused_tools": category_unused,
                    }
            response = {
                "enabled": True,
                "scope": self._public_scope_name(runtime_state),
                "attribution": dict(runtime_state["attribution"]),
                "started_at": self._format_timestamp(runtime_state["started_at"]),
                "snapshot_at": self._format_timestamp(snapshot_at),
                "window": {
                    "status": "ok",
                    "kind": "delta" if baseline is not None else "process",
                    "started_at": self._format_timestamp(
                        baseline["snapshot_at"]
                        if baseline is not None
                        else runtime_state["started_at"]
                    ),
                    "ended_at": self._format_timestamp(snapshot_at),
                },
                "interface_coverage": {
                    "used": len(used_tools),
                    "available": available_tool_count,
                    "percentage": round(len(used_tools) / available_tool_count * 100, 1)
                    if available_tool_count
                    else 0.0,
                    "unused_tools": unused_tools,
                    "by_category": category_coverage,
                },
                "tools": {
                    name: self._public_tool_stats(stats)
                    for name, stats in state["tools"].items()
                },
                "events": state["events"],
                "bytes": state["bytes"],
                "semantic_index": self._public_semantic_index(state["semantic_index"]),
                "workflow_effectiveness": self._workflow_effectiveness(
                    state["tools"],
                    state["events"],
                    state["bytes"],
                ),
            }
            if snapshot_token is not None:
                response["snapshot_token"] = snapshot_token
            return response
