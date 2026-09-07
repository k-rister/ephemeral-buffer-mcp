"""Opt-in, content-free metrics for local server usage."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator


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


def metrics_enabled() -> bool:
    """Return whether local metrics were explicitly enabled."""
    return os.environ.get("EPHEMERAL_METRICS", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


class LocalMetrics:
    """Bounded in-process counters and timers with no content storage."""

    def __init__(self, enabled: bool | None = None):
        self.enabled = metrics_enabled() if enabled is None else enabled
        self._lock = threading.Lock()
        self._tools: dict[str, dict[str, Any]] = defaultdict(self._new_tool)
        self._events: dict[str, int] = defaultdict(int)
        self._captured: set[str] = set()
        self._searched: set[str] = set()

    @staticmethod
    def _new_tool() -> dict[str, Any]:
        return {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_duration_ms": 0.0,
            "max_duration_ms": 0.0,
            "result_count": 0,
        }

    @contextmanager
    def measure(self, tool: str) -> Iterator[dict[str, Any]]:
        """Measure one tool call; the yielded state may contain result_count."""
        state = {"success": True, "result_count": None}
        started = time.perf_counter()
        try:
            yield state
        except Exception:
            state["success"] = False
            raise
        finally:
            if self.enabled:
                duration_ms = (time.perf_counter() - started) * 1000
                with self._lock:
                    stats = self._tools[tool]
                    stats["calls"] += 1
                    stats["successes" if state["success"] else "failures"] += 1
                    stats["total_duration_ms"] += duration_ms
                    stats["max_duration_ms"] = max(stats["max_duration_ms"], duration_ms)
                    if state["result_count"] is not None:
                        stats["result_count"] += int(state["result_count"])

    def record_event(self, event: str) -> None:
        if self.enabled:
            with self._lock:
                self._events[event] += 1

    def record_result_count(self, tool: str, count: int) -> None:
        if self.enabled:
            with self._lock:
                self._tools[tool]["result_count"] += max(0, int(count))

    def record_capture(self, capture_id: str) -> None:
        if self.enabled:
            with self._lock:
                self._captured.add(capture_id)
                self._events["captures"] += 1

    def record_search(self, capture_id: str, match_count: int) -> None:
        if self.enabled:
            with self._lock:
                if capture_id in self._captured:
                    self._events["capture_to_search"] += 1
                    self._searched.add(capture_id)
                self._events["searches"] += 1
                self._events["empty_searches"] += int(match_count == 0)

    def record_retrieval(self, capture_id: str) -> None:
        if self.enabled:
            with self._lock:
                if capture_id in self._searched:
                    self._events["search_to_retrieval"] += 1
                self._events["retrievals"] += 1

    def forget_capture(self, capture_id: str) -> None:
        if self.enabled:
            with self._lock:
                self._captured.discard(capture_id)
                self._searched.discard(capture_id)

    def snapshot(self) -> dict[str, Any]:
        """Return aggregate metrics suitable for a content-free diagnostic."""
        if not self.enabled:
            return {"enabled": False}
        with self._lock:
            return {
                "enabled": True,
                "tools": {
                    name: {
                        **stats,
                        "total_duration_ms": round(stats["total_duration_ms"], 3),
                        "max_duration_ms": round(stats["max_duration_ms"], 3),
                    }
                    for name, stats in self._tools.items()
                },
                "events": {name: self._events[name] for name in EVENT_NAMES},
            }
