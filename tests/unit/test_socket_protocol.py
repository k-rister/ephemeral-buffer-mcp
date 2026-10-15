"""Unit tests for versioned socket-frame encoding and header validation."""

import json
import math
import unittest
from unittest.mock import patch

import ephemeral_buffer_mcp.socket_protocol as socket_protocol
from ephemeral_buffer_mcp.socket_protocol import (
    FRAME_HEADER_SIZE,
    SocketPayloadError,
    decode_header,
    decode_socket_request,
    decode_socket_response,
    encode_frame,
    encode_socket_request,
    encode_socket_response,
)


class TestSocketProtocol(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        payload = b"protocol payload"
        frame = encode_frame(payload)

        self.assertEqual(decode_header(frame[:FRAME_HEADER_SIZE]), len(payload))
        self.assertEqual(frame[FRAME_HEADER_SIZE:], payload)

    def test_decode_header_rejects_truncated_magic_and_unsupported_version(self):
        frame = encode_frame(b"payload")

        with self.assertRaisesRegex(ValueError, "truncated frame header"):
            decode_header(frame[: FRAME_HEADER_SIZE - 1])
        with self.assertRaisesRegex(ValueError, "invalid frame magic"):
            decode_header(b"XXXX" + frame[4:FRAME_HEADER_SIZE])
        with self.assertRaisesRegex(ValueError, "unsupported frame version"):
            decode_header(frame[:4] + b"\x02" + frame[5:FRAME_HEADER_SIZE])

    def test_encode_frame_enforces_maximum_length_boundary(self):
        with patch.object(socket_protocol, "MAX_FRAME_LENGTH", 3):
            frame = encode_frame(b"123")
            self.assertEqual(decode_header(frame[:FRAME_HEADER_SIZE]), 3)
            self.assertEqual(frame[FRAME_HEADER_SIZE:], b"123")
            with self.assertRaisesRegex(ValueError, "too large"):
                encode_frame(b"1234")

    def test_socket_request_round_trip_applies_defaults_and_ignores_unknown_fields(self):
        payload = {
            "text": "captured output",
            "content_type": "log",
            "structured_metrics": {"tests": 2},
            "future_field": "ignored",
        }

        encoded = encode_socket_request(payload)
        self.assertNotIn(b"future_field", encoded)
        self.assertEqual(
            decode_socket_request(encoded),
            {
                "label": "CLI pipe",
                "text": "captured output",
                "content_type": "log",
                "truncated": False,
                "original_byte_size": None,
                "command_exit_code": None,
                "timed_out": False,
                "duration_ms": None,
                "structured_metrics": {"tests": 2},
            },
        )

    def test_socket_request_rejects_missing_required_and_invalid_fields(self):
        invalid_payloads = [
            [],
            {},
            {"text": 1},
            {"text": "x", "label": None},
            {"text": "x", "content_type": False},
            {"text": "x", "content_type": "html"},
            {"text": "x", "truncated": 1},
            {"text": "x", "original_byte_size": -1},
            {"text": "x", "original_byte_size": True},
            {"text": "x", "command_exit_code": False},
            {"text": "x", "timed_out": "false"},
            {"text": "x", "duration_ms": "1"},
            {"text": "x", "duration_ms": math.nan},
            {"text": "x", "duration_ms": 10**400},
            {"text": "x", "structured_metrics": []},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(SocketPayloadError):
                    decode_socket_request(json.dumps(payload).encode("utf-8"))

        with self.assertRaisesRegex(SocketPayloadError, "missing required field 'text'"):
            encode_socket_request({})

    def test_malformed_request_payloads_keep_plain_text_fallback(self):
        for payload, expected_text in ((b"not-json", "not-json"), (b"\xff", "\ufffd")):
            with self.subTest(payload=payload):
                request = decode_socket_request(payload)
                self.assertEqual(request["text"], expected_text)
                self.assertEqual(request["label"], "CLI pipe")
                self.assertEqual(request["content_type"], "auto")

    def test_socket_response_round_trip_and_ignores_unknown_fields(self):
        success = {
            "status": "ok",
            "capture_id": "cap_test",
            "label": "build",
            "line_count": 2,
            "byte_size": 15,
            "summary": {"status": "success"},
            "future_field": "ignored",
        }
        encoded_success = encode_socket_response(success)
        self.assertNotIn(b"future_field", encoded_success)
        self.assertEqual(
            decode_socket_response(encoded_success),
            {key: value for key, value in success.items() if key != "future_field"},
        )

        error = {"status": "error", "code": "server_busy", "message": "retry"}
        self.assertEqual(decode_socket_response(encode_socket_response(error)), error)

        legacy_error = {"status": "error", "message": "legacy failure"}
        self.assertEqual(
            decode_socket_response(json.dumps(legacy_error).encode("utf-8")),
            {
                "status": "error",
                "code": "socket_request_failed",
                "message": "legacy failure",
            },
        )

    def test_socket_response_rejects_invalid_shapes_and_fields(self):
        invalid_responses = [
            [],
            {},
            {"status": "unknown"},
            {"status": "error", "message": "failed"},
            {"status": "error", "code": "", "message": "failed"},
            {"status": "error", "code": "failed"},
            {"status": "ok", "label": "build", "line_count": 1, "byte_size": 1},
            {"status": "ok", "capture_id": "cap", "line_count": 1, "byte_size": 1},
            {"status": "ok", "capture_id": "cap", "label": "build", "line_count": True, "byte_size": 1},
            {"status": "ok", "capture_id": "cap", "label": "build", "line_count": 1, "byte_size": -1},
            {"status": "ok", "capture_id": "cap", "label": "build", "line_count": 1, "byte_size": 1, "summary": []},
        ]
        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaises(SocketPayloadError):
                    encode_socket_response(response)

        for payload in (b"not-json", b"\xff"):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(SocketPayloadError, "valid UTF-8 JSON"):
                    decode_socket_response(payload)


if __name__ == "__main__":
    unittest.main()
