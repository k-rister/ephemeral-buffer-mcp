"""Unit tests for versioned socket-frame encoding and header validation."""

import unittest
from unittest.mock import patch

import socket_protocol
from socket_protocol import FRAME_HEADER_SIZE, decode_header, encode_frame


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


if __name__ == "__main__":
    unittest.main()
