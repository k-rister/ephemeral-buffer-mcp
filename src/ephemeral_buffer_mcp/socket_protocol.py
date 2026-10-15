"""Framing and JSON payload contracts for the private CLI socket protocol."""

import json
import math
import struct
from typing import Any, Literal, TypedDict, cast


FRAME_MAGIC = b"EBUF"
# This version identifies both the frame format and its JSON payload schemas.
# Incompatible payload changes must increment it.
FRAME_VERSION = 1
FRAME_HEADER_SIZE = len(FRAME_MAGIC) + 1 + 4
MAX_FRAME_LENGTH = 0xFFFFFFFF


class SocketPayloadError(ValueError):
    """A framed JSON payload does not match the socket protocol contract."""


class SocketRequestError(SocketPayloadError):
    """A JSON request does not match the socket request contract."""


class SocketRequestPayload(TypedDict, total=False):
    """Wire fields accepted for a CLI capture request."""

    label: str
    text: str
    content_type: str
    truncated: bool
    original_byte_size: int | None
    command_exit_code: int | None
    timed_out: bool
    duration_ms: float | None
    structured_metrics: dict[str, Any] | None


class ParsedSocketRequest(TypedDict):
    """A request with protocol defaults applied for server ingestion."""

    label: str
    text: str
    content_type: str
    truncated: bool
    original_byte_size: int | None
    command_exit_code: int | None
    timed_out: bool
    duration_ms: float | None
    structured_metrics: dict[str, Any] | None


class SocketSuccessResponse(TypedDict):
    status: Literal["ok"]
    capture_id: str
    label: str
    line_count: int
    byte_size: int


class SocketSuccessResponseWithSummary(SocketSuccessResponse):
    summary: dict[str, Any]


class SocketErrorResponse(TypedDict):
    status: Literal["error"]
    code: str
    message: str


SocketResponse = SocketSuccessResponse | SocketSuccessResponseWithSummary | SocketErrorResponse


_REQUEST_FIELDS = frozenset(
    {
        "label",
        "text",
        "content_type",
        "truncated",
        "original_byte_size",
        "command_exit_code",
        "timed_out",
        "duration_ms",
        "structured_metrics",
    }
)
_CONTENT_TYPES = frozenset({"auto", "diff", "log", "text"})
_REQUEST_DEFAULTS: ParsedSocketRequest = {
    "label": "CLI pipe",
    "text": "",
    "content_type": "auto",
    "truncated": False,
    "original_byte_size": None,
    "command_exit_code": None,
    "timed_out": False,
    "duration_ms": None,
    "structured_metrics": None,
}


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


def encode_socket_request(payload: SocketRequestPayload) -> bytes:
    """Validate and JSON-encode a CLI request, dropping unknown fields."""
    normalized = _normalize_socket_request(payload, apply_defaults=False)
    return json.dumps(normalized).encode("utf-8")


def decode_socket_request(
    payload: bytes,
    *,
    allow_legacy_missing_text: bool = False,
) -> ParsedSocketRequest:
    """Decode a request, optionally accepting v1 peers that omitted ``text``."""
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _plain_text_request(payload.decode("utf-8", errors="replace"))
    return cast(
        ParsedSocketRequest,
        _normalize_socket_request(
            decoded,
            apply_defaults=True,
            allow_missing_text=allow_legacy_missing_text,
        ),
    )


def encode_socket_response(response: SocketResponse) -> bytes:
    """Validate and JSON-encode a success or error response."""
    normalized = _normalize_socket_response(response)
    return json.dumps(normalized).encode("utf-8")


def decode_socket_response(payload: bytes) -> SocketResponse:
    """Decode and validate a response from the socket server."""
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SocketPayloadError("socket response must be valid UTF-8 JSON") from exc
    # Older v1 servers sent generic errors without a code. Normalize those
    # responses so callers can still report the server's original message.
    return _normalize_socket_response(decoded, allow_legacy_error_code=True)


def _plain_text_request(text: str) -> ParsedSocketRequest:
    request = dict(_REQUEST_DEFAULTS)
    request["text"] = text
    return cast(ParsedSocketRequest, request)


def _normalize_socket_request(
    payload: Any,
    *,
    apply_defaults: bool,
    allow_missing_text: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SocketRequestError("socket payload must be a JSON object")
    if "text" not in payload and not allow_missing_text:
        raise SocketRequestError("socket payload is missing required field 'text'")

    normalized: dict[str, Any] = dict(_REQUEST_DEFAULTS) if apply_defaults else {}
    for field, value in payload.items():
        if not _valid_request_field(field, value):
            raise SocketRequestError(f"socket payload field '{field}' has an invalid value")
        if field in _REQUEST_FIELDS:
            normalized[field] = value
    return cast(ParsedSocketRequest, normalized)


def _valid_request_field(field: str, value: Any) -> bool:
    if field in {"label", "text"}:
        return isinstance(value, str)
    if field == "content_type":
        return isinstance(value, str) and value in _CONTENT_TYPES
    if field in {"truncated", "timed_out"}:
        return isinstance(value, bool)
    if field == "original_byte_size":
        return value is None or _is_integer(value, nonnegative=True)
    if field == "command_exit_code":
        return value is None or _is_integer(value)
    if field == "duration_ms":
        if value is None:
            return True
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    if field == "structured_metrics":
        return value is None or isinstance(value, dict)
    return field not in _REQUEST_FIELDS


def _is_integer(value: Any, *, nonnegative: bool = False) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (not nonnegative or value >= 0)
    )


def _normalize_socket_response(
    response: Any,
    *,
    allow_legacy_error_code: bool = False,
) -> SocketResponse:
    if not isinstance(response, dict):
        raise SocketPayloadError("socket response must be a JSON object")

    status = response.get("status")
    if status == "error":
        code = response.get("code")
        message = response.get("message")
        if "code" not in response and allow_legacy_error_code:
            code = "socket_request_failed"
        if not isinstance(code, str) or not code:
            raise SocketPayloadError("socket error response requires a non-empty code")
        if not isinstance(message, str):
            raise SocketPayloadError("socket error response requires a message")
        return cast(SocketErrorResponse, {"status": "error", "code": code, "message": message})

    if status == "ok":
        capture_id = response.get("capture_id")
        label = response.get("label")
        line_count = response.get("line_count")
        byte_size = response.get("byte_size")
        if not isinstance(capture_id, str):
            raise SocketPayloadError("socket success response requires a capture_id string")
        if not isinstance(label, str):
            raise SocketPayloadError("socket success response requires a label string")
        if not _is_integer(line_count, nonnegative=True):
            raise SocketPayloadError("socket success response requires a non-negative line_count")
        if not _is_integer(byte_size, nonnegative=True):
            raise SocketPayloadError("socket success response requires a non-negative byte_size")

        normalized: dict[str, Any] = {
            "status": "ok",
            "capture_id": capture_id,
            "label": label,
            "line_count": line_count,
            "byte_size": byte_size,
        }
        if "summary" in response:
            summary = response["summary"]
            if not isinstance(summary, dict):
                raise SocketPayloadError("socket success response summary must be a JSON object")
            normalized["summary"] = summary
        return cast(SocketResponse, normalized)

    raise SocketPayloadError("socket response status must be 'ok' or 'error'")
