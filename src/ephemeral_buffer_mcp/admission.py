"""Bounded process-wide admission for MCP tool and socket work."""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict

from .config import startup_settings

_SETTINGS = startup_settings()
MAX_ACTIVE_TOOL_WORK = _SETTINGS.max_active_tool_work.value
MAX_QUEUED_TOOL_WORK = _SETTINGS.max_queued_tool_work.value
MAX_ACTIVE_SOCKET_CLIENTS = _SETTINGS.max_active_socket_clients.value
MAX_QUEUED_SOCKET_CLIENTS = _SETTINGS.max_queued_socket_clients.value
MAX_ACTIVE_DIAGNOSTIC_WORK = 1


class AdmissionBusy(RuntimeError):
    """Raised when the per-loop wait queue is full."""


class AdmissionMetrics:
    """Thread-safe, content-free counters for admitted work."""

    def __init__(self) -> None:
        self._lock = threading.Condition()
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
            self._lock.notify_all()

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

    def wait_for_idle(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for all admitted work to release its ticket."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._lock.wait(remaining)
            return True


ADMISSION_METRICS = AdmissionMetrics()


@dataclass(eq=False)
class _Waiter:
    gate: "BoundedAdmissionGate"
    work_type: str
    future: asyncio.Future[None]
    submitted: bool = False
    state: str = "local"


class _GrantCallback:
    """Deliver a grant while making discarded loop callbacks observable."""

    __slots__ = ("waiter", "__weakref__")

    def __init__(self, waiter: _Waiter) -> None:
        self.waiter = waiter

    def __call__(self) -> None:
        self.waiter.gate._deliver_grant(self.waiter)


class ProcessAdmissionArbiter:
    """Share active permits across loops and fairly rotate queued loops."""

    def __init__(self, limits: Dict[str, int]) -> None:
        if not limits or any(limit < 1 for limit in limits.values()):
            raise ValueError("process admission limits must be positive")
        self._limits = dict(limits)
        self._active = {lane: 0 for lane in limits}
        self._queues: Dict[str, Dict[BoundedAdmissionGate, Deque[_Waiter]]] = {
            lane: {} for lane in limits
        }
        self._rotation: Dict[str, Deque[BoundedAdmissionGate]] = {
            lane: deque() for lane in limits
        }
        self._lock = threading.Lock()

    def try_acquire(self, gate: "BoundedAdmissionGate") -> bool:
        """Reserve an immediately available process permit without bypassing waiters."""
        lane = gate.lane
        with self._lock:
            self._check_lane(lane)
            if self._active[lane] >= self._limits[lane] or self._rotation[lane]:
                return False
            self._active[lane] += 1
            return True

    def enqueue(self, waiter: _Waiter) -> None:
        """Queue one locally reserved request and grant available permits fairly."""
        lane = waiter.gate.lane
        with self._lock:
            self._check_lane(lane)
            if waiter.state != "local":
                raise RuntimeError("admission waiter submitted more than once")
            queues = self._queues[lane]
            queue = queues.get(waiter.gate)
            if queue is None:
                queue = deque()
                queues[waiter.gate] = queue
                self._rotation[lane].append(waiter.gate)
            queue.append(waiter)
            waiter.state = "queued"
            self._grant_available(lane)

    def cancel(self, waiter: _Waiter) -> bool:
        """Remove a queued waiter or release a grant not yet delivered to its loop."""
        lane = waiter.gate.lane
        with self._lock:
            if waiter.state == "queued":
                queue = self._queues[lane][waiter.gate]
                queue.remove(waiter)
                if not queue:
                    del self._queues[lane][waiter.gate]
                    self._rotation[lane].remove(waiter.gate)
                waiter.state = "cancelled"
                return True
            if waiter.state == "granted":
                if self._active[lane] < 1:
                    raise RuntimeError("process admission active counter underflow")
                waiter.state = "cancelled"
                self._active[lane] -= 1
                self._grant_available(lane)
                return True
            return False

    def mark_delivered(self, waiter: _Waiter) -> bool:
        """Claim a scheduled grant on its owning event loop."""
        with self._lock:
            if waiter.state != "granted":
                return False
            waiter.state = "delivered"
            return True

    def release(self, gate: "BoundedAdmissionGate") -> None:
        """Release a process permit and grant the next waiting loop, if any."""
        lane = gate.lane
        with self._lock:
            self._check_lane(lane)
            if self._active[lane] < 1:
                raise RuntimeError("process admission active counter underflow")
            self._active[lane] -= 1
            self._grant_available(lane)

    def _reclaim_undelivered_grant(self, lane: str, waiter: _Waiter) -> None:
        """Release a grant whose scheduled callback was discarded by loop close."""
        with self._lock:
            if waiter.state != "granted":
                return
            if self._active[lane] < 1:
                raise RuntimeError("process admission active counter underflow")
            waiter.state = "cancelled"
            self._active[lane] -= 1
            waiter.gate._abandon_waiter_after_loop_close(waiter)
            self._grant_available(lane)

    def active_count(self, lane: str) -> int:
        """Return the current process-wide reservations for tests and diagnostics."""
        with self._lock:
            self._check_lane(lane)
            return self._active[lane]

    def _check_lane(self, lane: str) -> None:
        if lane not in self._limits:
            raise ValueError(f"unknown process admission lane: {lane}")

    def _grant_available(self, lane: str) -> None:
        rotation = self._rotation[lane]
        queues = self._queues[lane]
        while self._active[lane] < self._limits[lane] and rotation:
            gate = rotation.popleft()
            queue = queues[gate]
            waiter = queue.popleft()
            if queue:
                rotation.append(gate)
            else:
                del queues[gate]
            waiter.state = "granted"
            self._active[lane] += 1
            loop = gate._loop_ref()
            try:
                if loop is None:
                    raise RuntimeError("admission event loop is no longer available")
                # A successful schedule can still be discarded by loop.close().
                # The finalizer reclaims only grants the loop never delivered.
                callback = _GrantCallback(waiter)
                finalizer = weakref.finalize(
                    callback,
                    self._reclaim_undelivered_grant,
                    lane,
                    waiter,
                )
                finalizer.atexit = False
                loop.call_soon_threadsafe(callback)
            except RuntimeError:
                waiter.state = "cancelled"
                self._active[lane] -= 1
                gate._abandon_waiter_after_loop_close(waiter)


PROCESS_ADMISSION_ARBITER = ProcessAdmissionArbiter(
    {
        "mcp": MAX_ACTIVE_TOOL_WORK,
        "socket": MAX_ACTIVE_SOCKET_CLIENTS,
        "diagnostics": MAX_ACTIVE_DIAGNOSTIC_WORK,
    }
)


class AdmissionTicket:
    """An idempotently releasable reservation of local and process capacity."""

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
    """A FIFO, bounded queue with a local cap and shared process permits."""

    def __init__(
        self,
        max_active: int,
        max_queued: int,
        *,
        lane: str = "mcp",
        arbiter: ProcessAdmissionArbiter | None = None,
    ) -> None:
        if max_active < 1 or max_queued < 0:
            raise ValueError("admission limits must be positive active and non-negative queued values")
        self.max_active = max_active
        self.max_queued = max_queued
        self.lane = lane
        self._arbiter = arbiter or PROCESS_ADMISSION_ARBITER
        self._loop_ref = weakref.ref(asyncio.get_running_loop())
        self._active = 0
        # Reservations include active tickets and requests already in the
        # process arbiter, so a loop cannot overbook its own active limit.
        self._reserved = 0
        self._waiters: Deque[_Waiter] = deque()

    async def acquire(self, work_type: str) -> AdmissionTicket:
        loop = asyncio.get_running_loop()
        if loop is not self._loop_ref():
            raise RuntimeError("admission gate used from a different event loop")
        if self._reserved < self.max_active and not self._waiters:
            if self._arbiter.try_acquire(self):
                self._reserved += 1
                self._active += 1
                ADMISSION_METRICS.change("active", work_type, 1)
                return AdmissionTicket(self, work_type)

        if len(self._waiters) >= self.max_queued:
            ADMISSION_METRICS.change("rejected", work_type, 1)
            raise AdmissionBusy(work_type)

        waiter = _Waiter(self, work_type, loop.create_future())
        self._waiters.append(waiter)
        ADMISSION_METRICS.change("queued", work_type, 1)
        self._promote_waiters()
        try:
            await waiter.future
        except asyncio.CancelledError:
            self._cancel_waiter(waiter)
            raise
        return AdmissionTicket(self, work_type)

    def _promote_waiters(self) -> None:
        for waiter in self._waiters:
            if self._reserved >= self.max_active:
                return
            if waiter.submitted:
                continue
            if waiter.future.cancelled():
                return
            self._reserved += 1
            waiter.submitted = True
            self._arbiter.enqueue(waiter)

    def _deliver_grant(self, waiter: _Waiter) -> None:
        if waiter.future.cancelled():
            self._cancel_waiter(waiter)
            return
        if not self._arbiter.mark_delivered(waiter):
            return
        self._waiters.remove(waiter)
        ADMISSION_METRICS.change("queued", waiter.work_type, -1)
        self._active += 1
        ADMISSION_METRICS.change("active", waiter.work_type, 1)
        waiter.future.set_result(None)

    def _cancel_waiter(self, waiter: _Waiter) -> None:
        if waiter.state == "delivered":
            self._release(waiter.work_type)
            return
        if waiter.state == "cancelled":
            return

        if waiter.submitted:
            self._arbiter.cancel(waiter)
            self._reserved -= 1
        else:
            waiter.state = "cancelled"

        self._waiters.remove(waiter)
        ADMISSION_METRICS.change("queued", waiter.work_type, -1)
        self._promote_waiters()

    def _abandon_waiter_after_loop_close(self, waiter: _Waiter) -> None:
        """Drop an undeliverable grant after its owning loop has closed."""
        self._waiters.remove(waiter)
        self._reserved -= 1
        ADMISSION_METRICS.change("queued", waiter.work_type, -1)

    def _release(self, work_type: str) -> None:
        if self._active < 1:
            raise RuntimeError("admission active counter underflow")
        self._active -= 1
        self._reserved -= 1
        ADMISSION_METRICS.change("active", work_type, -1)
        self._arbiter.release(self)
        self._promote_waiters()


_GATES_LOCK = threading.Lock()
_GATES: Dict[str, weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, BoundedAdmissionGate]] = {
    "mcp": weakref.WeakKeyDictionary(),
    "socket": weakref.WeakKeyDictionary(),
    "diagnostics": weakref.WeakKeyDictionary(),
}


def admission_gate(kind: str) -> BoundedAdmissionGate:
    """Return the local gate for this event loop and process-wide work lane."""
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
            gate = BoundedAdmissionGate(*limits, lane=kind)
            gates[loop] = gate
        return gate


def admission_snapshot() -> Dict[str, object]:
    """Return process-wide active counts and per-loop queue counters."""
    return ADMISSION_METRICS.snapshot()
