"""Tests for privacy-safe structured logging configuration."""

import errno
import json
import logging
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
            original_umask = os.umask(0o022)
            try:
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
            finally:
                os.umask(original_umask)

            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(record["event"], "test_event")
            self.assertEqual(record["count"], 2)
            self.assertEqual(stat.S_IMODE(log_path.stat().st_mode), 0o600)

    def test_symlink_log_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "target.jsonl"
            log_path = Path(directory) / "events.jsonl"
            target_path.write_text("do not append\n", encoding="utf-8")
            log_path.symlink_to(target_path)

            with patch.dict(
                logging_utils.os.environ,
                {
                    "EPHEMERAL_LOG_FILE": str(log_path),
                    "EPHEMERAL_LOG_LEVEL": "INFO",
                },
                clear=True,
            ):
                logger = logging_utils.get_logger("test_file_logging")
                logging_utils.log_event(logger, logging.INFO, "must_not_reach_target")

            self.assertEqual(target_path.read_text(encoding="utf-8"), "do not append\n")
            self.assertFalse(
                any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
            )

    def test_unavailable_secure_file_creation_falls_back_to_stderr(self):
        original_hasattr = hasattr

        def secure_open_unavailable(value, name):
            if value is logging_utils.os and name == "O_NOFOLLOW":
                return False
            return original_hasattr(value, name)

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            logging_utils.os.environ,
            {"EPHEMERAL_LOG_FILE": str(Path(directory) / "events.jsonl")},
            clear=True,
        ), patch("builtins.hasattr", side_effect=secure_open_unavailable):
            logger = logging_utils.get_logger("test_file_logging")

        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)

    def test_invalid_log_file_metadata_is_rejected_and_descriptor_closed(self):
        invalid_metadata = (
            (stat.S_IFDIR | 0o700, os.geteuid(), 1, errno.EINVAL),
            (stat.S_IFREG | 0o600, os.geteuid() + 1, 1, errno.EPERM),
            (stat.S_IFREG | 0o600, os.geteuid(), 2, errno.EMLINK),
        )
        for mode, owner, link_count, expected_errno in invalid_metadata:
            with self.subTest(mode=mode, owner=owner, link_count=link_count):
                with tempfile.TemporaryDirectory() as directory:
                    log_path = Path(directory) / "events.jsonl"
                    log_path.touch()
                    fake_stat = SimpleNamespace(
                        st_mode=mode,
                        st_uid=owner,
                        st_nlink=link_count,
                    )
                    with patch.object(
                        logging_utils.os, "fstat", return_value=fake_stat
                    ), patch.object(
                        logging_utils.os, "close", wraps=os.close
                    ) as close_descriptor:
                        with self.assertRaises(OSError) as raised:
                            logging_utils._PrivateFileHandler(
                                str(log_path), encoding="utf-8"
                            )

                    self.assertEqual(raised.exception.errno, expected_errno)
                    close_descriptor.assert_called_once()

    def test_unwritable_file_does_not_prevent_logger_configuration(self):
        with patch.dict(
            logging_utils.os.environ,
            {"EPHEMERAL_LOG_FILE": "/path/that/does/not/exist/events.jsonl"},
            clear=True,
        ), patch.object(logging_utils, "_PrivateFileHandler", side_effect=OSError):
            logger = logging_utils.get_logger("test_file_logging")

        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)


if __name__ == "__main__":
    unittest.main()
