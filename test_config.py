"""Tests for environment-backed configuration."""

import io
import os
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from config import DEFAULT_SOCKET_PATH, positive_int_env, socket_path


class TestPositiveIntEnv(unittest.TestCase):
    def test_missing_value_uses_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(positive_int_env("TEST_LIMIT", 25), 25)

    def test_valid_value_is_used(self):
        with patch.dict(os.environ, {"TEST_LIMIT": "128"}, clear=True):
            self.assertEqual(positive_int_env("TEST_LIMIT", 25), 128)

    def test_invalid_value_warns_and_uses_default(self):
        stderr = io.StringIO()
        with patch.dict(os.environ, {"TEST_LIMIT": "0"}, clear=True), redirect_stderr(stderr):
            self.assertEqual(positive_int_env("TEST_LIMIT", 25), 25)
        self.assertIn("Ignoring invalid TEST_LIMIT", stderr.getvalue())

    def test_socket_path_can_be_overridden(self):
        with patch.dict(os.environ, {"EPHEMERAL_SOCKET_PATH": "/tmp/test-ephemeral.sock"}):
            self.assertEqual(socket_path(), "/tmp/test-ephemeral.sock")

    def test_session_id_derives_stable_socket_path(self):
        with patch.dict(os.environ, {"EPHEMERAL_SESSION_ID": "agent-session-1"}, clear=True):
            first_path = socket_path()
            second_path = socket_path()

        self.assertEqual(first_path, second_path)
        self.assertIn("ephemeral_buffer-", first_path)
        self.assertTrue(first_path.endswith(".sock"))
        self.assertNotEqual(first_path, DEFAULT_SOCKET_PATH)

    def test_explicit_socket_path_takes_precedence(self):
        with patch.dict(
            os.environ,
            {
                "EPHEMERAL_SOCKET_PATH": "/tmp/explicit.sock",
                "EPHEMERAL_SESSION_ID": "agent-session-1",
            },
            clear=True,
        ):
            self.assertEqual(socket_path(), "/tmp/explicit.sock")
        self.assertTrue(DEFAULT_SOCKET_PATH)


if __name__ == "__main__":
    unittest.main()
