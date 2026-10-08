"""Regression tests for bounded asynchronous work admission."""

import asyncio
import threading
import time
import unittest

import ephemeral_buffer_mcp.admission as admission


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
