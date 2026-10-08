"""Versioned length-prefixed framing for the private CLI socket protocol."""

import struct


FRAME_MAGIC = b"EBUF"
FRAME_VERSION = 1
FRAME_HEADER_SIZE = len(FRAME_MAGIC) + 1 + 4
MAX_FRAME_LENGTH = 0xFFFFFFFF


def encode_frame(payload: bytes) -> bytes:
    """Return one versioned frame containing ``payload``."""
    if not isinstance(payload, bytes):
        raise TypeError("frame payload must be bytes")
    if len(payload) > MAX_FRAME_LENGTH:
        raise ValueError("frame payload is too large")
    return FRAME_MAGIC + bytes((FRAME_VERSION,)) + struct.pack("!I", len(payload)) + payload


def decode_header(header: bytes) -> int:
    """Validate a frame header and return its declared payload length."""
    if len(header) != FRAME_HEADER_SIZE:
        raise ValueError("truncated frame header")
    if header[: len(FRAME_MAGIC)] != FRAME_MAGIC:
        raise ValueError("invalid frame magic")
    version = header[len(FRAME_MAGIC)]
    if version != FRAME_VERSION:
        raise ValueError(f"unsupported frame version: {version}")
    return struct.unpack("!I", header[-4:])[0]
