"""Typed MCP payload contracts for execution, search, and retrieval tools."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, TypeAdapter


class PayloadModel(BaseModel):
    """Base for additive payload contracts with strict declared field types."""

    model_config = ConfigDict(extra="allow", strict=True)


class ExecutionError(PayloadModel):
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class ExecutionFailure(PayloadModel):
    status: Literal["error"]
    error: ExecutionError


class ExecutionPhase(PayloadModel):
    name: str
    status: Literal["pending", "started", "completed", "failed", "interrupted", "timed_out"]
    side_effects: Optional[Literal["none", "unsafe"]] = None
    unsafe_side_effects: Optional[bool] = None
    attempts: Optional[int] = None
    error: Optional[str] = None
    blocked_reason_code: Optional[str] = None
    blocked_reason: Optional[str] = None
    cwd: Optional[str] = None
    timeout_seconds: Optional[float] = None
    max_output_bytes: Optional[int] = None
    idempotency_key: Optional[str] = None
    structured_metrics: Optional[dict[str, Any]] = None
    events: Optional[list[dict[str, Any]]] = None
    attempt_results: Optional[list[dict[str, Any]]] = None
    result: Optional[dict[str, Any]] = None
    output: Optional[str] = None
    output_bytes: Optional[int] = None


class ExecutionResume(PayloadModel):
    available: bool
    retry_required: bool
    unsafe_confirmation_required: bool
    attempt_limit_reached: bool
    next_phase: Optional[str]
    policy: Literal["safe", "allow-unsafe"]
    blocked_reason: Optional[str]


class ExecutionPublic(PayloadModel):
    status: Literal["ok"]
    schema_version: int
    execution_id: str
    label: str
    execution_status: Literal["pending", "running", "completed", "partial", "interrupted"]
    execution_in_progress: bool
    partial: bool
    summary: str
    created_at: str
    updated_at: str
    resume: ExecutionResume
    completed_phase_count: int
    phase_count: int
    phases: list[ExecutionPhase]
    cancellation_requested: Optional[bool] = None
    background_error: Optional[str] = None


class ExecutionPublicTruncated(ExecutionPublic):
    """Compact successful execution response retained under the byte limit."""

    response_truncated: Literal[True]


class ExecutionRecordDiagnostic(PayloadModel):
    status: Literal["invalid_record", "unsupported_record"]
    execution_id: str
    schema_version: Optional[Union[str, int, float, bool]] = None
    read_only: Literal[True]
    reason_code: str
    message: str
    record_bytes: int
    record_preview: Optional[str] = None
    record_preview_sanitized: bool
    record_preview_redacted: bool
    record_preview_truncated: bool
    error: ExecutionError


class ExecutionListFileDiagnostic(PayloadModel):
    """Malformed listing entry identified by its durable record filename."""

    status: Literal["invalid_record"]
    execution_id: Optional[str]
    record_file_id: str
    schema_version: Optional[Union[str, int, float, bool]] = None
    read_only: Literal[True]
    reason_code: str
    message: str
    record_bytes: int
    record_preview: Optional[str] = None
    record_preview_sanitized: bool
    record_preview_redacted: bool
    record_preview_truncated: bool
    error: ExecutionError


class ExecutionListFileDiagnosticSummary(PayloadModel):
    """Compact malformed-record entry retaining its filename identity."""

    status: Literal["invalid_record"]
    execution_id: Optional[str]
    record_file_id: str
    schema_version: Optional[Union[str, int, float, bool]] = None
    read_only: Optional[Literal[True]] = None
    reason_code: Optional[str] = None
    message: Optional[str] = None
    record_bytes: Optional[int] = None
    error: Optional[ExecutionError] = None


class ExecutionListSummary(PayloadModel):
    """Compact item retained when a list response exceeds its byte budget."""

    status: Literal["ok", "invalid_record", "unsupported_record"]
    execution_id: str
    label: Optional[str] = None
    execution_status: Optional[
        Literal["pending", "running", "completed", "partial", "interrupted"]
    ] = None
    partial: Optional[bool] = None
    execution_in_progress: Optional[bool] = None
    cancellation_requested: Optional[bool] = None
    background_error: Optional[str] = None
    updated_at: Optional[str] = None
    completed_phase_count: Optional[int] = None
    phase_count: Optional[int] = None
    reason_code: Optional[str] = None
    schema_version: Optional[Union[str, int, float, bool]] = None
    read_only: Optional[bool] = None
    error: Optional[ExecutionError] = None


class ExecutionCancelAcknowledgement(PayloadModel):
    status: Literal["ok"]
    execution_id: str
    execution_in_progress: bool
    cancellation_requested: bool
    message: str


class ExecutionOutputPhase(PayloadModel):
    name: str
    status: Literal["pending", "started", "completed", "failed", "interrupted", "timed_out"]
    output: str
    output_byte_size: int
    offset: int
    next_offset: int
    truncated: bool
    result: dict[str, Any]


class ExecutionOutput(PayloadModel):
    status: Literal["ok"]
    execution_id: str
    execution_status: Literal["pending", "running", "completed", "partial", "interrupted"]
    partial: bool
    max_bytes: int
    phases: list[ExecutionOutputPhase]


class ExecutionOutputPhaseSummary(PayloadModel):
    """Phase identity retained by the compact oversized-output fallback."""

    name: str
    status: Literal["pending", "started", "completed", "failed", "interrupted", "timed_out"]
    attempts: Optional[int] = None
    error: Optional[str] = None


class ExecutionOutputCompact(PayloadModel):
    """Successful summary returned when a full output response cannot fit."""

    status: Literal["ok"]
    execution_id: str
    execution_status: Literal["pending", "running", "completed", "partial", "interrupted"]
    partial: bool
    response_truncated: Literal[True]
    phases: list[ExecutionOutputPhaseSummary]


class ExecutionList(PayloadModel):
    status: Literal["ok"]
    limit: int
    offset: int
    executions: list[
        Union[
            ExecutionPublicTruncated,
            ExecutionPublic,
            ExecutionRecordDiagnostic,
            ExecutionListFileDiagnostic,
            ExecutionListFileDiagnosticSummary,
            ExecutionListSummary,
        ]
    ]
    response_truncated: Optional[bool] = None
    returned_count: Optional[int] = None
    omitted_count: Optional[int] = None


class ExecutionFilesystemCapacity(PayloadModel):
    total_bytes: Optional[int]
    free_bytes: Optional[int]


class ExecutionCapacity(PayloadModel):
    status: Literal["ok"]
    state_directory: str
    state_directory_exists: bool
    quota_bytes: int
    checkpoint_reserve_bytes: int
    record_storage_limit_bytes: int
    record_count: int
    summary_count: int
    paired_record_count: int
    unpaired_record_count: int
    unpaired_summary_count: int
    record_bytes: int
    summary_bytes: int
    committed_bytes: int
    temporary_file_bytes: int
    accounted_bytes: int
    active_reserved_bytes: int
    stale_reserved_bytes: int
    available_bytes: int
    capacity_confident: bool
    active_record_count: int
    fence_pending_record_count: int
    invalid_record_count: int
    invalid_reservation_count: int
    uncertain_lock_count: int
    other_file_count: int
    other_file_bytes: int
    symlink_count: int
    filesystem: ExecutionFilesystemCapacity
    background_execution_limit: int
    background_execution_active: int
    background_execution_queued: int


class ExecutionRetirementItem(PayloadModel):
    execution_id: str
    eligible: bool
    record_bytes: int
    summary_bytes: int
    projected_reclaimed_bytes: int
    label: Optional[str] = None
    execution_status: Optional[str] = None
    updated_at: Optional[str] = None
    reason: Optional[str] = None


class ExecutionRemovalError(PayloadModel):
    execution_id: Optional[str]
    error: str


class ExecutionRetirementPreview(PayloadModel):
    status: Literal["ok"]
    dry_run: Literal[True]
    items: list[ExecutionRetirementItem]
    projected_reclaimed_bytes: int
    archive_required_for_retirement: Optional[bool] = None
    eligible_count: Optional[int] = None
    blocked_count: Optional[int] = None


class ExecutionRetirementRefused(PayloadModel):
    status: Literal["refused"]
    dry_run: Literal[False]
    items: list[ExecutionRetirementItem]
    projected_reclaimed_bytes: int
    reason: Optional[str] = None


class ExecutionRetirementCompleted(PayloadModel):
    status: Literal["ok", "partial"]
    dry_run: Literal[False]
    projected_reclaimed_bytes: int
    archive_path: str
    archive_bytes: int
    retired_count: int
    retired: list[ExecutionRetirementItem]
    removal_errors: list[ExecutionRemovalError]
    archive_is_durable: bool


class ExecutionRetirementCompactItem(PayloadModel):
    """Retirement summary that keeps the durable execution identity."""

    execution_id: str
    eligible: bool
    record_bytes: int
    summary_bytes: int
    projected_reclaimed_bytes: int


class ExecutionRetirementPreviewCompact(PayloadModel):
    """Bounded retirement preview retaining each returned execution ID."""

    status: Literal["ok"]
    dry_run: Literal[True]
    items: list[ExecutionRetirementCompactItem]
    projected_reclaimed_bytes: int
    response_truncated: Literal[True]
    eligible_count: Optional[int] = None
    blocked_count: Optional[int] = None


class ExecutionRetirementRefusedCompact(PayloadModel):
    """Bounded refusal retaining the selected execution identities."""

    status: Literal["refused"]
    dry_run: Literal[False]
    items: list[ExecutionRetirementCompactItem]
    projected_reclaimed_bytes: int
    response_truncated: Literal[True]
    reason: Optional[str] = None


class ExecutionRetirementCompletedCompact(PayloadModel):
    """Bounded completed retirement retaining retired execution identities."""

    status: Literal["ok", "partial"]
    dry_run: Literal[False]
    projected_reclaimed_bytes: int
    response_truncated: Literal[True]
    retired: list[ExecutionRetirementCompactItem]
    removal_errors: list[ExecutionRemovalError]
    retired_count: int
    archive_bytes: int
    archive_is_durable: bool


class SearchMatch(PayloadModel):
    chunk_id: int
    chunk_index: Union[int, str]
    score: float
    matched_range: str
    context_range: str
    context_start_line: int
    context_end_line: int
    snippet: str
    snippet_truncated: bool
    context: str
    context_truncated: bool


class SearchPayload(PayloadModel):
    capture_id: str
    label: str
    label_truncated: bool
    total_lines: int
    mode: Literal["hybrid", "bm25", "semantic"]
    query: str
    semantic_coverage: Literal["complete", "unavailable", "pending", "not-requested"]
    matches: list[SearchMatch]
    match_count: int
    available_match_count: Optional[int] = None
    semantic_fallback: Optional[str] = None
    semantic_index_state: Optional[str] = None
    semantic_wait_seconds: Optional[float] = None
    message: Optional[str] = None


class CaptureSliceSegment(PayloadModel):
    line: int
    byte_offset: int
    byte_length: int
    line_complete: bool
    separator_after: bool


class CaptureSlicePayload(PayloadModel):
    capture_id: str
    label: str
    requested_start_line: int
    requested_end_line: int
    start_line: int
    end_line: int
    total_lines: int
    content: str
    content_bytes: int
    segments: list[CaptureSliceSegment]


class CaptureSummaryDiff(PayloadModel):
    stats: str
    file_map: str
    file_map_truncated: bool
    omitted_file_count: int


class CaptureSummaryPreviews(PayloadModel):
    head: str
    tail: str


class CaptureSummaryPayload(PayloadModel):
    schema_version: int
    capture_id: str
    label: str
    label_truncated: bool
    source: str
    status: Literal["captured", "success", "failed", "timed_out"]
    partial: bool
    content_type: str
    timestamp: Optional[str]
    duration_ms: Optional[float]
    exit_code: Optional[int]
    timed_out: bool
    total_lines: int
    byte_size: int
    original_byte_size: Optional[int]
    estimated_tokens: Optional[int]
    original_estimated_tokens: Optional[int]
    truncated: bool
    signals: dict[str, Any]
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    structured_metrics: dict[str, Any]
    evicted_captures: list[Union[str, dict[str, Any]]]
    evicted_captures_omitted: int
    retrieval: dict[str, str]
    diff: Optional[CaptureSummaryDiff] = None
    previews: Optional[CaptureSummaryPreviews] = None


class CaptureListEntry(PayloadModel):
    capture_id: str
    label: str
    label_truncated: bool
    total_lines: int
    byte_size: int
    timestamp: str


class CaptureListPayload(PayloadModel):
    captures: list[CaptureListEntry]
    capture_count: int
    page_count: int


PAYLOAD_CONTRACTS: dict[str, TypeAdapter] = {
    "start_execution": TypeAdapter(Union[
        ExecutionPublicTruncated,
        ExecutionPublic,
        ExecutionFailure,
    ]),
    "resume_execution": TypeAdapter(Union[
        ExecutionPublicTruncated,
        ExecutionPublic,
        ExecutionFailure,
    ]),
    "cancel_execution": TypeAdapter(Union[
        ExecutionPublicTruncated,
        ExecutionPublic,
        ExecutionCancelAcknowledgement,
        ExecutionRecordDiagnostic,
        ExecutionFailure,
    ]),
    "get_execution": TypeAdapter(Union[
        ExecutionPublicTruncated,
        ExecutionPublic,
        ExecutionRecordDiagnostic,
        ExecutionFailure,
    ]),
    "get_execution_output": TypeAdapter(
        Union[ExecutionOutput, ExecutionOutputCompact, ExecutionFailure]
    ),
    "list_executions": TypeAdapter(Union[ExecutionList, ExecutionFailure]),
    "get_execution_capacity": TypeAdapter(Union[ExecutionCapacity, ExecutionFailure]),
    "retire_executions": TypeAdapter(Union[
        ExecutionRetirementPreviewCompact,
        ExecutionRetirementRefusedCompact,
        ExecutionRetirementCompletedCompact,
        ExecutionRetirementPreview,
        ExecutionRetirementRefused,
        ExecutionRetirementCompleted,
        ExecutionFailure,
    ]),
    "search_capture": TypeAdapter(SearchPayload),
    "get_capture_slice": TypeAdapter(CaptureSlicePayload),
    "get_capture_summary": TypeAdapter(CaptureSummaryPayload),
    "list_captures": TypeAdapter(CaptureListPayload),
}

PAYLOAD_SCHEMA_VERSIONS = {name: 1 for name in PAYLOAD_CONTRACTS}
