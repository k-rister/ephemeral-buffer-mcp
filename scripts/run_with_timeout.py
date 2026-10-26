#!/usr/bin/env python3
"""Run one command in a process group with a bounded termination grace."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


TERMINATION_GRACE_SECONDS = 5.0
GROUP_POLL_INTERVAL_SECONDS = 0.05


class _RunnerInterrupted(Exception):
    """Raised when the runner receives a cancellation signal."""

    def __init__(self, signal_number: int) -> None:
        super().__init__(signal_number)
        self.signal_number = signal_number


def _process_group_exists(process_group_id: int) -> bool:
    """Return whether a process group still has members."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(process_group_id: int, requested_signal: int) -> bool:
    """Send a signal to a process group, returning false if it has exited."""
    try:
        os.killpg(process_group_id, requested_signal)
    except ProcessLookupError:
        return False
    return True


def _shell_return_code(return_code: int) -> int:
    """Represent a signal exit using the conventional shell status."""
    return return_code if return_code >= 0 else 128 + -return_code


def _terminate_process_group(process: subprocess.Popen[bytes], reason: str) -> None:
    """Terminate a command group, allow graceful cleanup, then reap its leader."""
    process_group_id = process.pid
    cancellation_signals = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, cancellation_signals)
    previous_handlers: dict[int, object] = {}
    try:
        for requested_signal in cancellation_signals:
            previous_handlers[requested_signal] = signal.signal(requested_signal, signal.SIG_IGN)
    finally:
        # Pending cancellation signals are ignored once the mask is restored.
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    try:
        print(
            f"{reason}; sending SIGTERM to process group {process_group_id} "
            f"({TERMINATION_GRACE_SECONDS:g}s grace)",
            file=sys.stderr,
            flush=True,
        )
        _signal_process_group(process_group_id, signal.SIGTERM)

        grace_deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
        while time.monotonic() < grace_deadline:
            process.poll()
            if not _process_group_exists(process_group_id):
                break
            time.sleep(min(GROUP_POLL_INTERVAL_SECONDS, max(0.0, grace_deadline - time.monotonic())))

        if _process_group_exists(process_group_id):
            print(
                f"process group {process_group_id} remained after the grace period; sending SIGKILL",
                file=sys.stderr,
                flush=True,
            )
            _signal_process_group(process_group_id, signal.SIGKILL)

        process.wait()
    finally:
        for requested_signal, previous_handler in previous_handlers.items():
            signal.signal(requested_signal, previous_handler)


def run(command: list[str], timeout_seconds: float) -> int:
    """Run a command, returning 124 after terminating it on timeout."""
    process: subprocess.Popen[bytes] | None = None
    previous_handlers: dict[int, object] = {}
    launching = True
    interrupted_signal: int | None = None

    def handle_cancellation(signal_number: int, _frame: object) -> None:
        """Defer cancellation during Popen until its process handle is available."""
        nonlocal interrupted_signal
        if interrupted_signal is None:
            interrupted_signal = signal_number
        if not launching:
            raise _RunnerInterrupted(signal_number)

    try:
        try:
            # Install handlers before Popen so a signal during launch is
            # remembered until we have the child process handle.
            for requested_signal in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[requested_signal] = signal.signal(
                    requested_signal,
                    handle_cancellation,
                )

            launch_error: OSError | None = None
            try:
                process = subprocess.Popen(command, start_new_session=True)
            except OSError as exc:
                launch_error = exc
            finally:
                launching = False

            if interrupted_signal is not None:
                signal_name = signal.Signals(interrupted_signal).name
                if process is not None:
                    _terminate_process_group(process, f"runner received {signal_name}")
                else:
                    print(
                        f"runner received {signal_name} before the command started",
                        file=sys.stderr,
                        flush=True,
                    )
                return 128 + interrupted_signal

            if launch_error is not None:
                print(f"could not start command: {launch_error}", file=sys.stderr, flush=True)
                return 127
            if process is None:
                raise RuntimeError("command launch returned no process")

            return _shell_return_code(process.wait(timeout=timeout_seconds))
        except subprocess.TimeoutExpired:
            if process is None:
                raise
            _terminate_process_group(
                process,
                f"command timed out after {timeout_seconds:g} seconds",
            )
            return 124
        except _RunnerInterrupted as interrupted:
            signal_name = signal.Signals(interrupted.signal_number).name
            if process is not None:
                _terminate_process_group(process, f"runner received {signal_name}")
            else:
                print(
                    f"runner received {signal_name} before the command started",
                    file=sys.stderr,
                    flush=True,
                )
            return 128 + interrupted.signal_number
        except KeyboardInterrupt:
            if process is not None:
                _terminate_process_group(process, "runner received SIGINT")
            else:
                print("runner received SIGINT before the command started", file=sys.stderr, flush=True)
            return 128 + signal.SIGINT
    finally:
        for requested_signal, previous_handler in previous_handlers.items():
            signal.signal(requested_signal, previous_handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --timeout")
    return run(command, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
