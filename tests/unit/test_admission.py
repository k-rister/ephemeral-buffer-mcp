"""Regression tests for bounded asynchronous work admission."""

import asyncio
import gc
import queue
import threading
import time
import unittest
from concurrent.futures import CancelledError

import ephemeral_buffer_mcp.admission as admission


class _LoopThread:
    """Small event-loop host for exercising process-wide admission."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.started = threading.Event()

        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.started.set()
            self.loop.run_forever()

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        if not self.started.wait(2):
            raise RuntimeError("event loop thread did not start")

    def submit(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def close(self):
        if not self.loop.is_running():
            self.thread.join(timeout=2)
            return

        async def cancel_pending_tasks():
            current = asyncio.current_task()
            pending = [
                task
                for task in asyncio.all_tasks()
                if task is not current and not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            self.submit(cancel_pending_tasks()).result(timeout=2)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=2)
            self.loop.close()


class TestProcessAdmissionArbiter(unittest.TestCase):
    def setUp(self):
        self.original_metrics = admission.ADMISSION_METRICS
        admission.ADMISSION_METRICS = admission.AdmissionMetrics()

    def tearDown(self):
        admission.ADMISSION_METRICS = self.original_metrics

    def test_process_permits_round_robin_between_loops_and_preserve_fifo(self):
        arbiter = admission.ProcessAdmissionArbiter({"mcp": 2})
        loop_a = _LoopThread()
        loop_b = _LoopThread()
        grants = queue.Queue()
        tickets = {}

        async def create_gate(label):
            gate = admission.BoundedAdmissionGate(
                max_active=3,
                max_queued=2,
                lane="mcp",
                arbiter=arbiter,
            )
            ticket = await gate.acquire(f"{label}_holder")
            return gate, ticket

        async def wait_and_report(gate, name):
            ticket = await gate.acquire(name)
            grants.put(name)
            return ticket

        async def release(ticket):
            ticket.release()

        def await_queued(count):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if admission.ADMISSION_METRICS.snapshot()["admission_queued"] == count:
                    return
                time.sleep(0.002)
            self.fail(f"expected {count} queued waiters")

        def collect(name):
            self.assertEqual(grants.get(timeout=2), name)
            ticket = futures[name].result(timeout=2)
            tickets[name] = ticket
            self.assertLessEqual(arbiter.active_count("mcp"), 2)
            return ticket

        try:
            gate_a, holder_a = loop_a.submit(create_gate("a")).result(timeout=2)
            gate_b, holder_b = loop_b.submit(create_gate("b")).result(timeout=2)
            self.assertEqual(arbiter.active_count("mcp"), 2)

            futures = {}
            for name in ("a_first", "a_second"):
                futures[name] = loop_a.submit(wait_and_report(gate_a, name))
            await_queued(2)
            for name in ("b_first", "b_second"):
                futures[name] = loop_b.submit(wait_and_report(gate_b, name))
            await_queued(4)

            loop_a.submit(release(holder_a)).result(timeout=2)
            first_a = collect("a_first")
            loop_b.submit(release(holder_b)).result(timeout=2)
            first_b = collect("b_first")
            self.assertEqual(
                admission.ADMISSION_METRICS.snapshot()["admission_queued"], 2
            )

            loop_a.submit(release(first_a)).result(timeout=2)
            second_a = collect("a_second")
            loop_b.submit(release(first_b)).result(timeout=2)
            second_b = collect("b_second")
            loop_a.submit(release(second_a)).result(timeout=2)
            loop_b.submit(release(second_b)).result(timeout=2)

            self.assertEqual(arbiter.active_count("mcp"), 0)
            self.assertEqual(
                admission.ADMISSION_METRICS.snapshot()["admission_active"], 0
            )
            self.assertEqual(
                admission.ADMISSION_METRICS.snapshot()["admission_queued"], 0
            )
        finally:
            loop_a.close()
            loop_b.close()

    def test_mcp_and_socket_caps_are_shared_across_loops_with_reserved_diagnostics(self):
        arbiter = admission.ProcessAdmissionArbiter(
            {"mcp": 2, "socket": 2, "diagnostics": 1}
        )
        loop_a = _LoopThread()
        loop_b = _LoopThread()
        tickets = []
        blocked = []

        async def create_gates(label):
            mcp_gate = admission.BoundedAdmissionGate(
                2, 1, lane="mcp", arbiter=arbiter
            )
            socket_gate = admission.BoundedAdmissionGate(
                2, 1, lane="socket", arbiter=arbiter
            )
            return (
                mcp_gate,
                socket_gate,
                await mcp_gate.acquire(f"mcp_{label}"),
                await socket_gate.acquire(f"socket_{label}"),
            )

        async def acquire_extra(gate, work_type):
            return await gate.acquire(work_type)

        async def reject_extra(gate, work_type):
            with self.assertRaises(admission.AdmissionBusy):
                await gate.acquire(work_type)

        async def acquire_diagnostics():
            gate = admission.BoundedAdmissionGate(
                1, 0, lane="diagnostics", arbiter=arbiter
            )
            return await gate.acquire("diagnostic_probe")

        async def release(ticket):
            ticket.release()

        def await_queued(count):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if admission.ADMISSION_METRICS.snapshot()["admission_queued"] == count:
                    return
                time.sleep(0.002)
            self.fail(f"expected {count} queued waiters")

        try:
            mcp_a, socket_a, mcp_ticket_a, socket_ticket_a = (
                loop_a.submit(create_gates("a")).result(timeout=2)
            )
            mcp_b, socket_b, mcp_ticket_b, socket_ticket_b = (
                loop_b.submit(create_gates("b")).result(timeout=2)
            )
            tickets.extend(
                (
                    (loop_a, mcp_ticket_a),
                    (loop_a, socket_ticket_a),
                    (loop_b, mcp_ticket_b),
                    (loop_b, socket_ticket_b),
                )
            )
            self.assertEqual(arbiter.active_count("mcp"), 2)
            self.assertEqual(arbiter.active_count("socket"), 2)

            blocked.extend(
                (
                    loop_a.submit(acquire_extra(mcp_a, "mcp_extra")),
                    loop_a.submit(acquire_extra(socket_a, "socket_extra")),
                )
            )
            await_queued(2)
            loop_a.submit(reject_extra(mcp_a, "mcp_overflow")).result(timeout=2)
            loop_a.submit(reject_extra(socket_a, "socket_overflow")).result(timeout=2)

            diagnostic_ticket = loop_b.submit(acquire_diagnostics()).result(timeout=2)
            tickets.append((loop_b, diagnostic_ticket))
            snapshot = admission.ADMISSION_METRICS.snapshot()
            self.assertEqual(snapshot["admission_active"], 5)
            self.assertEqual(snapshot["admission_queued"], 2)
            self.assertEqual(snapshot["admission_active_by_type"]["mcp_a"], 1)
            self.assertEqual(snapshot["admission_active_by_type"]["mcp_b"], 1)
            self.assertEqual(snapshot["admission_active_by_type"]["socket_a"], 1)
            self.assertEqual(snapshot["admission_active_by_type"]["socket_b"], 1)
            self.assertEqual(snapshot["admission_active_by_type"]["diagnostic_probe"], 1)

            for future in blocked:
                future.cancel()
            for future in blocked:
                with self.assertRaises(CancelledError):
                    future.result(timeout=2)
            await_queued(0)

            for loop, ticket in tickets:
                loop.submit(release(ticket)).result(timeout=2)
            self.assertEqual(arbiter.active_count("mcp"), 0)
            self.assertEqual(arbiter.active_count("socket"), 0)
            self.assertEqual(arbiter.active_count("diagnostics"), 0)
            self.assertEqual(
                admission.ADMISSION_METRICS.snapshot()["admission_active"], 0
            )
        finally:
            for future in blocked:
                future.cancel()
            loop_a.close()
            loop_b.close()

    def test_closing_loop_reclaims_a_scheduled_but_undelivered_grant(self):
        arbiter = admission.ProcessAdmissionArbiter({"mcp": 1})
        closed_loop = asyncio.new_event_loop()

        async def acquire_holder():
            gate = admission.BoundedAdmissionGate(
                2, 1, lane="mcp", arbiter=arbiter
            )
            return gate, await gate.acquire("holder")

        gate, holder = closed_loop.run_until_complete(acquire_holder())
        waiter = admission._Waiter(
            gate, "discarded_grant", closed_loop.create_future()
        )
        gate._waiters.append(waiter)
        gate._reserved += 1
        waiter.submitted = True
        admission.ADMISSION_METRICS.change("queued", waiter.work_type, 1)
        arbiter.enqueue(waiter)

        live_loop = asyncio.new_event_loop()
        try:
            # The target loop is stopped but open, so scheduling succeeds and
            # leaves the grant callback pending until close discards it.
            holder.release()
            self.assertEqual(arbiter.active_count("mcp"), 1)
            self.assertEqual(gate._reserved, 1)
            self.assertEqual(
                admission.ADMISSION_METRICS.snapshot()["admission_queued"], 1
            )

            closed_loop.close()
            gc.collect()

            snapshot = admission.ADMISSION_METRICS.snapshot()
            self.assertEqual(arbiter.active_count("mcp"), 0)
            self.assertEqual(gate._reserved, 0)
            self.assertEqual(snapshot["admission_active"], 0)
            self.assertEqual(snapshot["admission_queued"], 0)

            async def acquire_after_shutdown():
                next_gate = admission.BoundedAdmissionGate(
                    1, 0, lane="mcp", arbiter=arbiter
                )
                return await next_gate.acquire("after_shutdown")

            next_ticket = live_loop.run_until_complete(acquire_after_shutdown())
            self.assertEqual(arbiter.active_count("mcp"), 1)
            next_ticket.release()
            self.assertEqual(arbiter.active_count("mcp"), 0)
            self.assertEqual(
                admission.ADMISSION_METRICS.snapshot()["admission_active"], 0
            )
        finally:
            if not closed_loop.is_closed():
                closed_loop.close()
            live_loop.close()


class TestAdmissionMetrics(unittest.TestCase):
    def setUp(self):
        self.original_metrics = admission.ADMISSION_METRICS
        admission.ADMISSION_METRICS = admission.AdmissionMetrics()

    def tearDown(self):
        admission.ADMISSION_METRICS = self.original_metrics

    def test_counters_are_aggregated_and_zero_entries_are_removed(self):
        metrics = admission.ADMISSION_METRICS
        metrics.change("active", "search", 2)
        metrics.change("queued", "socket", 1)
        metrics.change("rejected", "search", 3)
        metrics.change("active", "search", -2)
        metrics.change("queued", "socket", -1)

        snapshot = metrics.snapshot()

        self.assertEqual(snapshot["admission_active"], 0)
        self.assertEqual(snapshot["admission_queued"], 0)
        self.assertEqual(snapshot["admission_rejected"], 3)
        self.assertEqual(snapshot["admission_active_by_type"], {})
        self.assertEqual(snapshot["admission_queued_by_type"], {})
        self.assertEqual(snapshot["admission_rejected_by_type"], {"search": 3})
        self.assertIn("admission_max_active_tool_work", snapshot)
        self.assertIn("admission_max_active_diagnostic_work", snapshot)

    def test_counter_underflow_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "counter underflow"):
            admission.ADMISSION_METRICS.change("active", "search", -1)

    def test_wait_for_idle_observes_timeout_and_ticket_release(self):
        metrics = admission.AdmissionMetrics()
        metrics.change("active", "worker", 1)
        self.assertFalse(metrics.wait_for_idle(0))

        def release_later():
            time.sleep(0.01)
            metrics.change("active", "worker", -1)

        releaser = threading.Thread(target=release_later)
        releaser.start()
        self.assertTrue(metrics.wait_for_idle(1))
        releaser.join(timeout=1)


class TestBoundedAdmissionGate(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_metrics = admission.ADMISSION_METRICS
        admission.ADMISSION_METRICS = admission.AdmissionMetrics()

    def tearDown(self):
        admission.ADMISSION_METRICS = self.original_metrics

    async def test_fifo_admission_and_full_queue_rejection(self):
        gate = admission.BoundedAdmissionGate(max_active=1, max_queued=1)
        active = await gate.acquire("capture_text")
        waiting = asyncio.create_task(gate.acquire("search_capture"))
        await asyncio.sleep(0)

        self.assertEqual(admission.admission_snapshot()["admission_queued"], 1)
        with self.assertRaises(admission.AdmissionBusy):
            await gate.acquire("get_capture_slice")

        active.release()
        waiting_ticket = await waiting
        self.assertEqual(admission.admission_snapshot()["admission_queued"], 0)
        self.assertEqual(admission.admission_snapshot()["admission_active"], 1)
        waiting_ticket.release()
        self.assertEqual(admission.admission_snapshot()["admission_active"], 0)
        self.assertEqual(admission.admission_snapshot()["admission_rejected"], 1)

    async def test_cancelled_waiter_is_removed_from_the_queue(self):
        gate = admission.BoundedAdmissionGate(max_active=1, max_queued=1)
        active = await gate.acquire("active")
        waiting = asyncio.create_task(gate.acquire("waiting"))
        await asyncio.sleep(0)

        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertEqual(admission.admission_snapshot()["admission_queued"], 0)
        active.release()
        self.assertEqual(admission.admission_snapshot()["admission_active"], 0)

    async def test_cancelled_granted_waiter_releases_its_reserved_slot(self):
        gate = admission.BoundedAdmissionGate(max_active=1, max_queued=1)
        active = await gate.acquire("active")
        waiting = asyncio.create_task(gate.acquire("waiting"))
        await asyncio.sleep(0)

        active.release()
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting

        snapshot = admission.admission_snapshot()
        self.assertEqual(snapshot["admission_active"], 0)
        self.assertEqual(snapshot["admission_queued"], 0)

    async def test_release_skips_a_cancelled_waiter_still_in_the_queue(self):
        gate = admission.BoundedAdmissionGate(max_active=1, max_queued=1)
        active = await gate.acquire("active")
        waiting = asyncio.create_task(gate.acquire("waiting"))
        await asyncio.sleep(0)

        waiting.cancel()
        active.release()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertEqual(admission.admission_snapshot()["admission_active"], 0)
        self.assertEqual(admission.admission_snapshot()["admission_queued"], 0)

    async def test_ticket_release_is_idempotent_and_can_follow_a_future(self):
        gate = admission.BoundedAdmissionGate(max_active=1, max_queued=0)
        ticket = await gate.acquire("worker")
        completion = asyncio.get_running_loop().create_future()
        ticket.defer_release_until(completion)
        ticket.defer_release_until(completion)
        ticket.release()
        self.assertEqual(admission.admission_snapshot()["admission_active"], 1)

        completion.set_result(None)
        await asyncio.sleep(0)
        ticket.release()
        ticket.defer_release_until(completion)
        self.assertEqual(admission.admission_snapshot()["admission_active"], 0)

    async def test_ticket_release_rejects_gate_underflow(self):
        gate = admission.BoundedAdmissionGate(max_active=1, max_queued=0)
        with self.assertRaisesRegex(RuntimeError, "admission active counter underflow"):
            gate._release("unreserved")

    async def test_process_arbiter_validates_limits_and_rejects_invalid_states(self):
        with self.assertRaisesRegex(ValueError, "limits must be positive"):
            admission.ProcessAdmissionArbiter({})
        with self.assertRaisesRegex(ValueError, "limits must be positive"):
            admission.ProcessAdmissionArbiter({"mcp": 0})

        arbiter = admission.ProcessAdmissionArbiter({"mcp": 1})
        gate = admission.BoundedAdmissionGate(
            1, 1, lane="mcp", arbiter=arbiter
        )
        with self.assertRaisesRegex(ValueError, "unknown process admission lane"):
            arbiter.active_count("unknown")
        gate.lane = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown process admission lane"):
            arbiter.try_acquire(gate)
        gate.lane = "mcp"

        waiter = admission._Waiter(
            gate, "probe", asyncio.get_running_loop().create_future()
        )
        self.assertFalse(arbiter.mark_delivered(waiter))
        self.assertFalse(arbiter.cancel(waiter))
        with self.assertRaisesRegex(RuntimeError, "process admission active counter underflow"):
            arbiter.release(gate)
        waiter.state = "granted"
        with self.assertRaisesRegex(RuntimeError, "process admission active counter underflow"):
            arbiter.cancel(waiter)
        with self.assertRaisesRegex(RuntimeError, "process admission active counter underflow"):
            arbiter._reclaim_undelivered_grant("mcp", waiter)

        waiter.state = "local"
        holder = await gate.acquire("holder")
        arbiter.enqueue(waiter)
        with self.assertRaisesRegex(RuntimeError, "submitted more than once"):
            arbiter.enqueue(waiter)
        self.assertTrue(arbiter.cancel(waiter))
        holder.release()

        cancelled_waiter = admission._Waiter(
            gate,
            "cancelled",
            asyncio.get_running_loop().create_future(),
            state="cancelled",
        )
        gate._deliver_grant(cancelled_waiter)
        gate._cancel_waiter(cancelled_waiter)

        self.assertTrue(arbiter.try_acquire(gate))
        gate._active = 1
        gate._reserved = 1
        admission.ADMISSION_METRICS.change("active", "delivered", 1)
        delivered_waiter = admission._Waiter(
            gate,
            "delivered",
            asyncio.get_running_loop().create_future(),
            submitted=True,
            state="delivered",
        )
        gate._cancel_waiter(delivered_waiter)
        self.assertEqual(arbiter.active_count("mcp"), 0)

    async def test_grant_to_closed_loop_releases_process_and_local_reservations(self):
        arbiter = admission.ProcessAdmissionArbiter({"mcp": 1})
        gate = admission.BoundedAdmissionGate(
            2, 1, lane="mcp", arbiter=arbiter
        )
        holder = await gate.acquire("holder")
        waiter = admission._Waiter(
            gate, "closed_loop", asyncio.get_running_loop().create_future()
        )
        gate._waiters.append(waiter)
        gate._reserved += 1
        waiter.submitted = True
        admission.ADMISSION_METRICS.change("queued", waiter.work_type, 1)
        arbiter.enqueue(waiter)
        gate._loop_ref = lambda: None

        holder.release()

        snapshot = admission.ADMISSION_METRICS.snapshot()
        self.assertEqual(arbiter.active_count("mcp"), 0)
        self.assertEqual(gate._reserved, 0)
        self.assertEqual(snapshot["admission_active"], 0)
        self.assertEqual(snapshot["admission_queued"], 0)

    async def test_invalid_limits_and_cross_loop_use_are_rejected(self):
        with self.assertRaises(ValueError):
            admission.BoundedAdmissionGate(max_active=0, max_queued=0)
        with self.assertRaises(ValueError):
            admission.BoundedAdmissionGate(max_active=1, max_queued=-1)

        gate = admission.BoundedAdmissionGate(max_active=1, max_queued=0)
        with self.assertRaisesRegex(ValueError, "unknown admission lane"):
            admission.admission_gate("unknown")
        with self.assertRaisesRegex(RuntimeError, "different event loop"):
            await asyncio.to_thread(asyncio.run, gate.acquire("wrong-loop"))

    async def test_gate_registry_is_per_loop_and_work_lane(self):
        mcp_gate = admission.admission_gate("mcp")
        diagnostics_gate = admission.admission_gate("diagnostics")
        socket_gate = admission.admission_gate("socket")

        self.assertIs(admission.admission_gate("mcp"), mcp_gate)
        self.assertIsNot(mcp_gate, diagnostics_gate)
        self.assertIsNot(mcp_gate, socket_gate)
        self.assertEqual((diagnostics_gate.max_active, diagnostics_gate.max_queued), (1, 0))

        first_loop_gate = mcp_gate
        new_loop_gate = await asyncio.to_thread(
            lambda: asyncio.run(_get_mcp_gate())
        )
        self.assertIsNot(first_loop_gate, new_loop_gate)


async def _get_mcp_gate():
    return admission.admission_gate("mcp")


if __name__ == "__main__":
    unittest.main()
