"""Tests for environment-backed configuration."""

import io
import os
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from config import positive_int_env


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


if __name__ == "__main__":
    unittest.main()
