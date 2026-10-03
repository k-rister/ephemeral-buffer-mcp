"""Bounded admission for MCP tool and socket work."""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict

from config import startup_settings

_SETTINGS = startup_settings()
MAX_ACTIVE_TOOL_WORK = _SETTINGS.max_active_tool_work.value
MAX_QUEUED_TOOL_WORK = _SETTINGS.max_queued_tool_work.value
MAX_ACTIVE_SOCKET_CLIENTS = _SETTINGS.max_active_socket_clients.value
MAX_QUEUED_SOCKET_CLIENTS = _SETTINGS.max_queued_socket_clients.value
MAX_ACTIVE_DIAGNOSTIC_WORK = 1


class AdmissionBusy(RuntimeError):
    """Raised when both active capacity and the bounded wait queue are full."""


class AdmissionMetrics:
    """Thread-safe, content-free counters for admitted work."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Dict[str, int] = {}
        self._queued: Dict[str, int] = {}
        self._rejected: Dict[str, int] = {}

    def change(self, state: str, work_type: str, amount: int) -> None:
        values = {
            "active": self._active,
            "queued": self._queued,
            "rejected": self._rejected,
        }[state]
        with self._lock:
            updated = values.get(work_type, 0) + amount
            if updated < 0:
                raise RuntimeError(f"admission {state} counter underflow for {work_type}")
            if updated:
                values[work_type] = updated
            else:
                values.pop(work_type, None)

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            active_by_type = dict(sorted(self._active.items()))
            queued_by_type = dict(sorted(self._queued.items()))
            rejected_by_type = dict(sorted(self._rejected.items()))
        return {
            "admission_active": sum(active_by_type.values()),
            "admission_queued": sum(queued_by_type.values()),
            "admission_rejected": sum(rejected_by_type.values()),
            "admission_active_by_type": active_by_type,
            "admission_queued_by_type": queued_by_type,
            "admission_rejected_by_type": rejected_by_type,
            "admission_max_active_tool_work": MAX_ACTIVE_TOOL_WORK,
            "admission_max_queued_tool_work": MAX_QUEUED_TOOL_WORK,
            "admission_max_active_socket_clients": MAX_ACTIVE_SOCKET_CLIENTS,
            "admission_max_queued_socket_clients": MAX_QUEUED_SOCKET_CLIENTS,
            "admission_max_active_diagnostic_work": MAX_ACTIVE_DIAGNOSTIC_WORK,
        }


ADMISSION_METRICS = AdmissionMetrics()


@dataclass
class _Waiter:
    work_type: str
    future: asyncio.Future[None]
    granted: bool = False


class AdmissionTicket:
    """An idempotently releasable active-work reservation."""

    def __init__(self, gate: "BoundedAdmissionGate", work_type: str) -> None:
        self._gate = gate
        self._work_type = work_type
        self._released = False
        self._release_deferred = False

    def defer_release_until(self, task: asyncio.Future[object]) -> None:
        """Keep the reservation until detached worker work actually finishes."""
        if self._released or self._release_deferred:
            return
        self._release_deferred = True
        task.add_done_callback(self._finish_deferred_release)

    def _finish_deferred_release(self, _: asyncio.Future[object]) -> None:
        self._release_deferred = False
        self.release()

    def release(self) -> None:
        if self._released or self._release_deferred:
            return
        self._released = True
        self._gate._release(self._work_type)


class BoundedAdmissionGate:
    """FIFO active slots with a bounded async wait queue, scoped to one loop."""

    def __init__(self, max_active: int, max_queued: int) -> None:
        if max_active < 1 or max_queued < 0:
            raise ValueError("admission limits must be positive active and non-negative queued values")
        self.max_active = max_active
        self.max_queued = max_queued
        self._loop_ref = weakref.ref(asyncio.get_running_loop())
        self._active = 0
        self._waiters: Deque[_Waiter] = deque()

    async def acquire(self, work_type: str) -> AdmissionTicket:
        loop = asyncio.get_running_loop()
        if loop is not self._loop_ref():
            raise RuntimeError("admission gate used from a different event loop")
        if self._active < self.max_active and not self._waiters:
            self._active += 1
            ADMISSION_METRICS.change("active", work_type, 1)
            return AdmissionTicket(self, work_type)

        if len(self._waiters) >= self.max_queued:
            ADMISSION_METRICS.change("rejected", work_type, 1)
            raise AdmissionBusy(work_type)

        waiter = _Waiter(work_type, loop.create_future())
        self._waiters.append(waiter)
        ADMISSION_METRICS.change("queued", work_type, 1)
        try:
            await waiter.future
        except asyncio.CancelledError:
            if waiter.granted:
                self._release(work_type)
            else:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
                else:
                    ADMISSION_METRICS.change("queued", work_type, -1)
            raise
        return AdmissionTicket(self, work_type)

    def _release(self, work_type: str) -> None:
        if self._active < 1:
            raise RuntimeError("admission active counter underflow")
        self._active -= 1
        ADMISSION_METRICS.change("active", work_type, -1)
        while self._waiters:
            waiter = self._waiters.popleft()
            ADMISSION_METRICS.change("queued", waiter.work_type, -1)
            if waiter.future.cancelled():
                continue
            self._active += 1
            ADMISSION_METRICS.change("active", waiter.work_type, 1)
            waiter.granted = True
            waiter.future.set_result(None)
            return


_GATES_LOCK = threading.Lock()
_GATES: Dict[str, weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, BoundedAdmissionGate]] = {
    "mcp": weakref.WeakKeyDictionary(),
    "socket": weakref.WeakKeyDictionary(),
    "diagnostics": weakref.WeakKeyDictionary(),
}


def admission_gate(kind: str) -> BoundedAdmissionGate:
    """Return the gate for this event loop and work lane."""
    loop = asyncio.get_running_loop()
    if kind == "mcp":
        limits = (MAX_ACTIVE_TOOL_WORK, MAX_QUEUED_TOOL_WORK)
    elif kind == "socket":
        limits = (MAX_ACTIVE_SOCKET_CLIENTS, MAX_QUEUED_SOCKET_CLIENTS)
    elif kind == "diagnostics":
        limits = (MAX_ACTIVE_DIAGNOSTIC_WORK, 0)
    else:
        raise ValueError(f"unknown admission lane: {kind}")
    with _GATES_LOCK:
        gates = _GATES[kind]
        gate = gates.get(loop)
        if gate is None:
            gate = BoundedAdmissionGate(*limits)
            gates[loop] = gate
        return gate


def admission_snapshot() -> Dict[str, object]:
    """Return active, queued, rejected, and configured admission counters."""
    return ADMISSION_METRICS.snapshot()
