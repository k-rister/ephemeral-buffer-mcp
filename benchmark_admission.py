#!/usr/bin/env python3
"""Measure MCP and Unix-socket admission under a controlled saturation burst."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shlex
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import workload_results as wr


SCHEMA_VERSION = 1
DEFAULT_HOLD_SECONDS = 1.0
DEFAULT_OVERFLOW_REQUESTS = 4
DEFAULT_REPETITIONS = 3
DEFAULT_MAX_REQUESTS = 256
SOCKET_RESPONSE_MAX_BYTES = 1024 * 1024


def _work_counts(snapshot: dict[str, Any], work_type: str) -> dict[str, int]:
    """Return one work type's current admission counters."""
    return {
        "active": int(snapshot["admission_active_by_type"].get(work_type, 0)),
        "queued": int(snapshot["admission_queued_by_type"].get(work_type, 0)),
        "rejected": int(snapshot["admission_rejected_by_type"].get(work_type, 0)),
    }


async def _wait_for_admission(
    admission,
    *,
    work_type: str,
    active: int,
    queued: int,
    rejected_at_least: int,
    rejected_before: int,
    timeout_seconds: float,
) -> dict[str, int]:
    """Wait until a burst reaches its expected active, queued, and rejected shape."""
    deadline = time.perf_counter() + timeout_seconds
    last = {"active": 0, "queued": 0, "rejected": 0}
    while time.perf_counter() < deadline:
        last = _work_counts(admission.admission_snapshot(), work_type)
        rejected_delta = last["rejected"] - rejected_before
        if (
            last["active"] == active
            and last["queued"] == queued
            and rejected_delta >= rejected_at_least
        ):
            return {**last, "rejected_delta": rejected_delta}
        await asyncio.sleep(0.002)
    raise RuntimeError(
        f"{work_type} admission did not reach the expected saturation shape "
        f"(active={active}, queued={queued}, rejected_delta>={rejected_at_least}); "
        f"observed active={last['active']}, queued={last['queued']}, "
        f"rejected_delta={last['rejected'] - rejected_before}"
    )


async def _wait_for_idle(admission, work_type: str, timeout_seconds: float) -> dict[str, int]:
    """Wait for one lane to release all active and queued requests."""
    deadline = time.perf_counter() + timeout_seconds
    last = {"active": 0, "queued": 0, "rejected": 0}
    while time.perf_counter() < deadline:
        last = _work_counts(admission.admission_snapshot(), work_type)
        if last["active"] == 0 and last["queued"] == 0:
            return last
        await asyncio.sleep(0.002)
    raise RuntimeError(
        f"{work_type} admission did not return to idle "
        f"(active={last['active']}, queued={last['queued']})"
    )


def _mcp_payload(result: Any) -> dict[str, Any]:
    """Extract structured content from a public FastMCP app call result."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured

    content = getattr(result, "content", result)
    if isinstance(content, tuple):
        content = content[0] if content else None
    if isinstance(content, list):
        content = content[0] if content else None
    text = getattr(content, "text", content if isinstance(content, str) else None)
    if isinstance(text, str):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {"status": "error", "error": {"code": "unstructured_response"}}
        if isinstance(payload, dict):
            return payload
    return {"status": "error", "error": {"code": "unstructured_response"}}


def _outcome(payload: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    """Keep response classification and latency while discarding response content."""
    error = payload.get("error")
    error_code = error.get("code") if isinstance(error, dict) else None
    return {
        "status": payload.get("status", "error"),
        "error_code": error_code,
        "elapsed_seconds": elapsed_seconds,
    }


async def _mcp_request(app, command: str, label: str, timeout_seconds: float) -> dict[str, Any]:
    """Run one public FastMCP tool call and return its privacy-safe outcome."""
    started = time.perf_counter()
    try:
        result = await app.call_tool(
            "execute_and_capture",
            {
                "command": command,
                "cwd": os.getcwd(),
                "label": label,
                "content_type": "text",
                "timeout_seconds": timeout_seconds,
            },
        )
        return _outcome(_mcp_payload(result), time.perf_counter() - started)
    except Exception as exc:  # The benchmark records error types, never output content.
        return {
            "status": "exception",
            "error_code": type(exc).__name__,
            "elapsed_seconds": time.perf_counter() - started,
        }


async def _run_mcp_repetition(
    app,
    admission,
    *,
    active_limit: int,
    queue_limit: int,
    overflow: int,
    hold_seconds: float,
    repetition: int,
) -> dict[str, Any]:
    """Fill the MCP lane, observe the bounded queue, then collect every result."""
    admitted = active_limit + queue_limit
    request_count = admitted + overflow
    before = _work_counts(admission.admission_snapshot(), "execute_and_capture")
    command = shlex.join([
        sys.executable,
        "-c",
        f"import time; time.sleep({hold_seconds!r}); print('admission benchmark')",
    ])
    command_timeout = max(30.0, hold_seconds * (math.ceil(admitted / active_limit) + 3))
    started = time.perf_counter()
    tasks = [
        asyncio.create_task(
            _mcp_request(
                app,
                command,
                f"admission-mcp-{repetition}-{index}",
                command_timeout,
            )
        )
        for index in range(request_count)
    ]
    try:
        occupancy = await _wait_for_admission(
            admission,
            work_type="execute_and_capture",
            active=active_limit,
            queued=queue_limit,
            rejected_at_least=overflow,
            rejected_before=before["rejected"],
            timeout_seconds=max(5.0, hold_seconds * 2),
        )
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    outcomes = await asyncio.gather(*tasks)
    wall_seconds = time.perf_counter() - started
    after = await _wait_for_idle(
        admission,
        "execute_and_capture",
        timeout_seconds=max(5.0, command_timeout),
    )
    return _summarize_repetition(
        outcomes,
        wall_seconds,
        occupancy,
        rejected_delta=after["rejected"] - before["rejected"],
        idle=after["active"] == 0 and after["queued"] == 0,
        expected_admitted=admitted,
        expected_rejected=overflow,
    )


async def _read_socket_response(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read one bounded framed response from the Unix socket protocol."""
    from socket_protocol import FRAME_HEADER_SIZE, decode_header

    header = await reader.readexactly(FRAME_HEADER_SIZE)
    body_size = decode_header(header)
    if body_size > SOCKET_RESPONSE_MAX_BYTES:
        raise ValueError(f"socket response exceeded {SOCKET_RESPONSE_MAX_BYTES} bytes")
    body = await reader.readexactly(body_size)
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("socket response was not a JSON object")
    return payload


async def _socket_request_outcome(
    reader: asyncio.StreamReader,
    started: float,
) -> dict[str, Any]:
    """Read and classify one socket response without retaining its content."""
    try:
        payload = await _read_socket_response(reader)
        return _outcome(payload, time.perf_counter() - started)
    except Exception as exc:
        return {
            "status": "exception",
            "error_code": type(exc).__name__,
            "elapsed_seconds": time.perf_counter() - started,
        }


async def _run_socket_repetition(
    admission,
    socket_path: str,
    *,
    active_limit: int,
    queue_limit: int,
    overflow: int,
    repetition: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Hold socket slots with incomplete frames, reject overflow, then complete them."""
    from socket_protocol import FRAME_HEADER_SIZE, encode_frame

    admitted = active_limit + queue_limit
    request_count = admitted + overflow
    before = _work_counts(admission.admission_snapshot(), "socket")
    clients: list[tuple[asyncio.StreamReader, asyncio.StreamWriter, float, bytes]] = []
    outcomes: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index in range(request_count):
            request_started = time.perf_counter()
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(socket_path),
                timeout=timeout_seconds,
            )
            payload = json.dumps({
                "label": f"admission-socket-{repetition}-{index}",
                "text": "socket admission benchmark payload",
                "content_type": "text",
            }).encode("utf-8")
            frame = encode_frame(payload)
            writer.write(frame[:FRAME_HEADER_SIZE])
            await writer.drain()
            clients.append((reader, writer, request_started, frame[FRAME_HEADER_SIZE:]))

            expected_admitted_so_far = min(index + 1, admitted)
            expected_active = min(expected_admitted_so_far, active_limit)
            expected_queued = max(0, expected_admitted_so_far - active_limit)
            expected_rejected = max(0, index + 1 - admitted)
            await _wait_for_admission(
                admission,
                work_type="socket",
                active=expected_active,
                queued=expected_queued,
                rejected_at_least=expected_rejected,
                rejected_before=before["rejected"],
                timeout_seconds=timeout_seconds,
            )

            if index >= admitted:
                outcomes.append(
                    await asyncio.wait_for(
                        _socket_request_outcome(reader, request_started),
                        timeout=timeout_seconds,
                    )
                )

        saturation = _work_counts(admission.admission_snapshot(), "socket")
        if saturation["active"] != active_limit or saturation["queued"] != queue_limit:
            raise RuntimeError(
                "socket requests did not hold the configured active slots and queue"
            )

        for _reader, writer, _request_started, payload in clients[:admitted]:
            writer.write(payload)
            await writer.drain()

        admitted_outcomes = await asyncio.gather(*[
            asyncio.wait_for(
                _socket_request_outcome(reader, request_started),
                timeout=timeout_seconds,
            )
            for reader, _writer, request_started, _payload in clients[:admitted]
        ])
        outcomes.extend(admitted_outcomes)
        wall_seconds = time.perf_counter() - started
        after = await _wait_for_idle(admission, "socket", timeout_seconds)
        return _summarize_repetition(
            outcomes,
            wall_seconds,
            {**saturation, "rejected_delta": after["rejected"] - before["rejected"]},
            rejected_delta=after["rejected"] - before["rejected"],
            idle=after["active"] == 0 and after["queued"] == 0,
            expected_admitted=admitted,
            expected_rejected=overflow,
        )
    finally:
        for _reader, writer, _request_started, _payload in clients:
            writer.close()
        await asyncio.gather(*[
            writer.wait_closed()
            for _reader, writer, _request_started, _payload in clients
        ], return_exceptions=True)


def _summarize_repetition(
    outcomes: list[dict[str, Any]],
    wall_seconds: float,
    occupancy: dict[str, int],
    *,
    rejected_delta: int,
    idle: bool,
    expected_admitted: int,
    expected_rejected: int,
) -> dict[str, Any]:
    """Summarize one burst and state whether it exercised the configured gate."""
    successes = [item for item in outcomes if item["status"] == "ok"]
    rejections = [item for item in outcomes if item["error_code"] == "server_busy"]
    errors = [
        item["error_code"] or item["status"]
        for item in outcomes
        if item["status"] != "ok" and item["error_code"] != "server_busy"
    ]
    request_count = len(outcomes)
    failures: list[str] = []
    if len(successes) != expected_admitted:
        failures.append(
            f"expected {expected_admitted} successful requests, observed {len(successes)}"
        )
    if len(rejections) != expected_rejected or rejected_delta != expected_rejected:
        failures.append(
            f"expected {expected_rejected} busy rejections, observed "
            f"{len(rejections)} responses and {rejected_delta} admission rejects"
        )
    if errors:
        failures.append("unexpected request errors: " + ", ".join(sorted(set(errors))))
    if not idle:
        failures.append("admission counters did not return to idle after the burst")
    if wall_seconds <= 0 or not math.isfinite(wall_seconds):
        failures.append("burst duration was not a finite positive value")

    return {
        "requests": request_count,
        "successful_requests": len(successes),
        "busy_rejections": len(rejections),
        "other_errors": len(errors),
        "success_rate": len(successes) / request_count if request_count else 0.0,
        "busy_rejection_rate": len(rejections) / request_count if request_count else 0.0,
        "wall_seconds": wall_seconds,
        "accepted_throughput_per_second": (
            len(successes) / wall_seconds if wall_seconds > 0 else 0.0
        ),
        "request_latency_seconds": wr.summarize(
            (item["elapsed_seconds"] for item in outcomes),
            "seconds",
            note="Includes queue wait and response time; socket clients first hold incomplete frames.",
        ),
        "successful_latency_seconds": wr.summarize(
            (item["elapsed_seconds"] for item in successes),
            "seconds",
        ),
        "rejected_latency_seconds": wr.summarize(
            (item["elapsed_seconds"] for item in rejections),
            "seconds",
        ),
        "observed_active_at_saturation": occupancy["active"],
        "observed_queued_at_saturation": occupancy["queued"],
        "admission_rejects": rejected_delta,
        "passed": not failures,
        "errors": failures,
    }


def _scenario_summary(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repetition-level admission and latency measurements."""
    request_counts = sum(item["requests"] for item in repetitions)
    successful = sum(item["successful_requests"] for item in repetitions)
    rejected = sum(item["busy_rejections"] for item in repetitions)
    throughput = [item["accepted_throughput_per_second"] for item in repetitions]
    return {
        "repetition_count": len(repetitions),
        "requests": request_counts,
        "successful_requests": successful,
        "busy_rejections": rejected,
        "success_rate": successful / request_counts if request_counts else 0.0,
        "busy_rejection_rate": rejected / request_counts if request_counts else 0.0,
        "accepted_throughput_per_second_median": statistics.median(throughput),
        "request_latency_seconds_by_repetition": [
            item["request_latency_seconds"] for item in repetitions
        ],
        "successful_latency_seconds_by_repetition": [
            item["successful_latency_seconds"] for item in repetitions
        ],
        "rejected_latency_seconds_by_repetition": [
            item["rejected_latency_seconds"] for item in repetitions
        ],
        "max_active_at_saturation": max(
            item["observed_active_at_saturation"] for item in repetitions
        ),
        "max_queued_at_saturation": max(
            item["observed_queued_at_saturation"] for item in repetitions
        ),
        "passed": all(item["passed"] for item in repetitions),
        "errors": [error for item in repetitions for error in item["errors"]],
    }


def _workload_result(record: dict[str, Any]) -> dict[str, Any]:
    """Return the tool-agnostic workload result for MCP and socket bursts."""
    runs = []
    for transport in ("mcp", "socket"):
        limits = record["configuration"][transport]
        for index, repetition in enumerate(record["scenarios"][transport]["repetitions"], start=1):
            runs.append(wr.run(
                f"{transport}-repetition-{index}",
                labels={
                    "transport": transport,
                    "repetition": index,
                    "active_limit": limits["active"],
                    "queue_limit": limits["queued"],
                },
                status="success" if repetition["passed"] else "failure",
                measurements={
                    "success_rate": wr.measurement(
                        "ratio",
                        value=repetition["success_rate"],
                        samples=repetition["requests"],
                    ),
                    "busy_rejection_rate": wr.measurement(
                        "ratio",
                        value=repetition["busy_rejection_rate"],
                        samples=repetition["requests"],
                    ),
                    "throughput_per_second": wr.measurement(
                        "per_second",
                        value=repetition["accepted_throughput_per_second"],
                        samples=repetition["successful_requests"],
                    ),
                    "request_latency_seconds": repetition["request_latency_seconds"],
                    "successful_latency_seconds": repetition["successful_latency_seconds"],
                    "rejected_latency_seconds": repetition["rejected_latency_seconds"],
                    "active_slots_at_saturation": wr.measurement(
                        "count",
                        value=repetition["observed_active_at_saturation"],
                        samples=1,
                    ),
                    "queued_slots_at_saturation": wr.measurement(
                        "count",
                        value=repetition["observed_queued_at_saturation"],
                        samples=1,
                    ),
                },
                phases=[wr.phase("saturation_burst", value=repetition["wall_seconds"], samples=repetition["requests"])],
                errors=repetition["errors"],
            ))

    errors = [
        f"{transport}: {error}"
        for transport in ("mcp", "socket")
        for error in record["scenarios"][transport]["errors"]
    ]
    return wr.build_result(
        workload="foreground-admission",
        kind="benchmark",
        producer="benchmark_admission.py",
        producer_schema_version=SCHEMA_VERSION,
        parameters=record["configuration"],
        runs=runs,
        details=record,
        errors=errors,
        description=(
            "Measures bounded MCP and Unix-socket request admission with a controlled saturation burst. "
            "MCP workers run a short local sleep command; socket requests hold incomplete frames until "
            "the active slots and queue are occupied."
        ),
        privacy="Counts, timing, capacity settings, and response status codes only; no output content is retained.",
    )


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run repeated MCP and socket saturation bursts using configured limits."""
    # Import only after main disables semantic prefetch for this isolated run.
    import admission
    import server

    snapshot = admission.admission_snapshot()
    capacities = {
        "mcp": {
            "active": int(snapshot["admission_max_active_tool_work"]),
            "queued": int(snapshot["admission_max_queued_tool_work"]),
        },
        "socket": {
            "active": int(snapshot["admission_max_active_socket_clients"]),
            "queued": int(snapshot["admission_max_queued_socket_clients"]),
        },
    }
    for transport, limits in capacities.items():
        burst_size = limits["active"] + limits["queued"] + args.overflow
        if burst_size > args.max_requests:
            raise ValueError(
                f"{transport} burst needs {burst_size} requests, over --max-requests "
                f"limit {args.max_requests}"
            )

    context = server.create_service_context(
        engine_options={"semantic_prefetch": False, "embedding_warmup": False}
    )
    app = server.create_mcp_server(context)
    scenarios: dict[str, list[dict[str, Any]]] = {"mcp": [], "socket": []}
    timeout_seconds = max(5.0, args.hold_seconds * 2 + 2)

    try:
        with tempfile.TemporaryDirectory(prefix="ephemeral-admission-") as temp_dir:
            socket_path = str(Path(temp_dir) / "admission.sock")
            listener = await asyncio.start_unix_server(
                server.handle_socket_client,
                path=socket_path,
                backlog=max(100, args.max_requests),
            )
            try:
                for repetition in range(1, args.repetitions + 1):
                    mcp_result = await _run_mcp_repetition(
                        app,
                        admission,
                        active_limit=capacities["mcp"]["active"],
                        queue_limit=capacities["mcp"]["queued"],
                        overflow=args.overflow,
                        hold_seconds=args.hold_seconds,
                        repetition=repetition,
                    )
                    scenarios["mcp"].append(mcp_result)
                    socket_result = await _run_socket_repetition(
                        admission,
                        socket_path,
                        active_limit=capacities["socket"]["active"],
                        queue_limit=capacities["socket"]["queued"],
                        overflow=args.overflow,
                        repetition=repetition,
                        timeout_seconds=timeout_seconds,
                    )
                    scenarios["socket"].append(socket_result)
            finally:
                listener.close()
                await listener.wait_closed()
    finally:
        context.close()

    scenario_records = {
        transport: {
            **_scenario_summary(repetitions),
            "repetitions": repetitions,
        }
        for transport, repetitions in scenarios.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "mcp": capacities["mcp"],
            "socket": capacities["socket"],
            "overflow_per_repetition": args.overflow,
            "repetitions": args.repetitions,
            "mcp_hold_seconds": args.hold_seconds,
            "semantic_prefetch": False,
        },
        "scenarios": scenario_records,
        "passed": all(item["passed"] for item in scenario_records.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold-seconds", type=float, default=DEFAULT_HOLD_SECONDS)
    parser.add_argument("--overflow", type=int, default=DEFAULT_OVERFLOW_REQUESTS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument("--output", type=Path, help="Write detailed benchmark JSON")
    wr.add_result_argument(parser)
    args = parser.parse_args()
    if not math.isfinite(args.hold_seconds) or not 0.05 <= args.hold_seconds <= 30:
        parser.error("--hold-seconds must be finite and between 0.05 and 30")
    if args.overflow < 1 or args.repetitions < 1 or args.max_requests < 1:
        parser.error("--overflow, --repetitions, and --max-requests must be positive")

    # Admission latency is the subject of this benchmark; background embedding
    # work is disabled so local model startup cannot dominate the queue results.
    os.environ["EPHEMERAL_SEMANTIC_PREFETCH"] = "0"
    try:
        record = asyncio.run(run_benchmark(args))
    except (OSError, RuntimeError, ValueError, asyncio.TimeoutError) as exc:
        parser.error(str(exc))

    for transport in ("mcp", "socket"):
        summary = record["scenarios"][transport]
        print(
            f"transport={transport} repetitions={summary['repetition_count']} "
            f"success_rate={summary['success_rate']:.3f} "
            f"busy_rejection_rate={summary['busy_rejection_rate']:.3f} "
            f"accepted_throughput_median={summary['accepted_throughput_per_second_median']:.2f}/s "
            f"active={summary['max_active_at_saturation']} "
            f"queued={summary['max_queued_at_saturation']} "
            f"passed={summary['passed']}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if args.result:
        wr.write_result(
            _workload_result(record),
            args.result,
            experiment=wr.experiment_from_args(args),
        )
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
