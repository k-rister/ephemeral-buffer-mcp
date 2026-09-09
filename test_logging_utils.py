"""Tests for privacy-safe structured logging configuration."""

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import logging_utils


class TestLoggingUtils(unittest.TestCase):
    def tearDown(self):
        logger = logging.getLogger(f"{logging_utils.LOGGER_NAME}.test_file_logging")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    def test_file_logging_writes_structured_events(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.jsonl"
            with patch.dict(
                logging_utils.os.environ,
                {
                    "EPHEMERAL_LOG_FILE": str(log_path),
                    "EPHEMERAL_LOG_LEVEL": "INFO",
                },
                clear=True,
            ):
                logger = logging_utils.get_logger("test_file_logging")
                logging_utils.log_event(logger, logging.INFO, "test_event", count=2)
                for handler in logger.handlers:
                    handler.flush()

            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(record["event"], "test_event")
            self.assertEqual(record["count"], 2)

    def test_unwritable_file_does_not_prevent_logger_configuration(self):
        with patch.dict(
            logging_utils.os.environ,
            {"EPHEMERAL_LOG_FILE": "/path/that/does/not/exist/events.jsonl"},
            clear=True,
        ), patch.object(logging_utils.logging, "FileHandler", side_effect=OSError):
            logger = logging_utils.get_logger("test_file_logging")

        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)


if __name__ == "__main__":
    unittest.main()
