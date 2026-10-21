"""Execution domain handlers."""

import inspect

from ..server import *  # noqa: F403
from .registry import register_tool


class ExecutionPhaseInput(BaseModel):
    """Schema contract for one phase exposed through the MCP tool."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_PHASE_NAME_BYTES})
    command: str = Field(min_length=1, json_schema_extra={"maxUtf8Bytes": 16 * 1024})
    cwd: Optional[str] = Field(
        default=None, min_length=1, json_schema_extra={"maxUtf8Bytes": 4096}
    )
    timeout_seconds: Optional[float] = Field(default=None, gt=0)
    max_output_bytes: Optional[int] = Field(default=None, ge=512)
    structured_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra={"maxJsonBytes": MAX_STRUCTURED_METRICS_BYTES},
    )
    side_effects: Literal["none", "unsafe"] = "none"
    unsafe_side_effects: Optional[bool] = None
    idempotency_key: Optional[str] = Field(
        default=None, min_length=1,
        json_schema_extra={"maxUtf8Bytes": MAX_PHASE_NAME_BYTES},
    )

    @field_validator("name", "command", "cwd", "idempotency_key")
    @classmethod
    def validate_utf8_byte_limit(cls, value, info):
        """Apply the same UTF-8 byte bounds as durable execution records."""
        limits = {
            "name": MAX_PHASE_NAME_BYTES,
            "command": 16 * 1024,
            "cwd": 4096,
            "idempotency_key": MAX_PHASE_NAME_BYTES,
        }
        if value is not None and len(value.encode("utf-8")) > limits[info.field_name]:
            raise ValueError(
                f"{info.field_name} exceeds the {limits[info.field_name]:,}-byte limit"
            )
        return value

    @field_validator("structured_metrics")
    @classmethod
    def validate_structured_metrics_size(cls, value):
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("structured_metrics must contain JSON-compatible values") from exc
        if len(encoded) > MAX_STRUCTURED_METRICS_BYTES:
            raise ValueError(
                f"structured_metrics exceeds the {MAX_STRUCTURED_METRICS_BYTES:,}-byte limit"
            )
        return value

    @field_validator("max_output_bytes")
    @classmethod
    def validate_output_limit(cls, value):
        if value is not None and value > _active_engine().max_buffer_bytes:
            raise ValueError(
                f"max_output_bytes cannot exceed the configured {_active_engine().max_buffer_bytes:,}-byte limit"
            )
        return value

    @model_validator(mode="after")
    def normalize_side_effect_alias(self):
        """Treat the boolean alias as an override only when the enum was omitted."""
        if "unsafe_side_effects" not in self.model_fields_set:
            return self
        alias_side_effects = "unsafe" if self.unsafe_side_effects else "none"
        if "side_effects" in self.model_fields_set and self.side_effects != alias_side_effects:
            raise ValueError("side_effects and unsafe_side_effects must agree")
        self.side_effects = alias_side_effects
        return self


def _resolve_preflight_cwd(cwd: Optional[str]) -> dict[str, Any]:
    """Resolve a requested working directory without executing user commands."""
    requested = cwd if cwd is not None else os.getcwd()
    path = Path(requested).expanduser()
    resolved = path.resolve(strict=False)
    exists = os.path.lexists(path)
    is_symlink = path.is_symlink()
    target_exists = resolved.exists() if is_symlink else exists
    if is_symlink and not target_exists:
        status = "dangling-symlink"
    elif not exists:
        status = "missing"
    elif not resolved.is_dir():
        status = "not-a-directory"
    else:
        status = "ok"
    return {
        "input": requested,
        "resolved": str(resolved),
        "source": "explicit" if cwd is not None else "process-cwd",
        "status": status,
        "exists": exists,
        "is_directory": resolved.is_dir(),
        "is_symlink": is_symlink,
        "symlink_target": str(resolved) if is_symlink else None,
        "symlink_target_exists": target_exists if is_symlink else None,
    }


def _resolve_preflight_executable(tokens: list[str], resolved_cwd: str) -> dict[str, Any]:
    """Resolve only the first parsed token; shell expansion remains out of scope."""
    if not tokens:
        return {"status": "unavailable", "reason": "command contains no parseable tokens"}
    token = tokens[0]
    if os.path.sep in token or (os.altsep and os.altsep in token):
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = Path(resolved_cwd) / candidate
        candidate = candidate.resolve(strict=False)
        if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK):
            return {"status": "resolved", "requested": token, "resolved": str(candidate)}
        return {"status": "unavailable", "requested": token, "reason": "path is not an executable file"}

    search_path = os.environ.get("PATH", os.defpath)
    search_entries = [
        entry if os.path.isabs(entry) else os.path.abspath(os.path.join(resolved_cwd, entry))
        for entry in search_path.split(os.pathsep)
    ]
    resolved = shutil.which(token, path=os.pathsep.join(search_entries))
    if resolved:
        return {"status": "resolved", "requested": token, "resolved": str(Path(resolved).resolve(strict=False))}
    return {"status": "unavailable", "requested": token, "reason": "executable was not found on PATH"}


@register_tool("preflight_command", "diagnostics")
@_instrument_tool("preflight_command")
def preflight_command(
    command: str,
    cwd: Optional[str] = None,
    __invocation_mode: str = "direct",
) -> str:
    """Return content-free path and executable diagnostics without running ``command``.

    This resolves the working directory, symlink target, detectable Git
    repository root, and first executable token. Relative and empty PATH entries
    are resolved against the requested working directory. Shell expansion, aliases,
    pipelines, redirections, environment changes, and arbitrary shell logic
    cannot be verified here. The requested command is never executed.
    """
    try:
        working_directory = _resolve_preflight_cwd(cwd)
        try:
            tokens = shlex.split(command)
            parse_status = "ok" if tokens else "empty"
        except ValueError as exc:
            tokens = []
            parse_status = f"unavailable: {exc}"

        executable = _resolve_preflight_executable(tokens, working_directory["resolved"])
        repository: dict[str, Any] = {
            "status": "unavailable",
            "root": None,
            "reason": "working directory is not an existing directory",
        }
        if working_directory["status"] == "ok":
            try:
                result = subprocess.run(
                    ["git", "-C", working_directory["resolved"], "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=2,
                )
                if result.returncode == 0 and result.stdout.strip():
                    repository = {"status": "detected", "root": str(Path(result.stdout.strip()).resolve())}
                else:
                    repository = {"status": "not-detected", "root": None, "reason": "not inside a Git work tree"}
            except (OSError, subprocess.TimeoutExpired) as exc:
                repository = {"status": "unavailable", "root": None, "reason": type(exc).__name__}

        return json.dumps(
            {
                "status": "ok",
                "command": {
                    "parse_status": parse_status,
                    "token_count": len(tokens),
                    "executable": executable,
                },
                "working_directory": working_directory,
                "repository": repository,
                "limitations": [
                    "The requested command was not executed.",
                    "Shell expansion, aliases, pipelines, redirections, and environment changes are not resolved.",
                    "Repository detection reports only a local Git work-tree root; it does not verify user intent or remotes.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    except Exception as exc:
        if __invocation_mode == "mcp":
            return _execution_error_json("preflight_failed", str(exc))
        return json.dumps({"status": "error", "reason": type(exc).__name__})


_preflight_command_signature = inspect.signature(preflight_command)
preflight_command.__signature__ = _preflight_command_signature.replace(
    parameters=tuple(
        parameter
        for parameter in _preflight_command_signature.parameters.values()
        if parameter.name != "__invocation_mode"
    )
)


@register_tool("start_execution", "execution", background_execution=True)
@_instrument_tool("start_execution")
def start_execution(
    phases: Annotated[List[ExecutionPhaseInput], Field(min_length=1, max_length=MAX_EXECUTION_PHASES)],
    execution_id: Optional[Annotated[str, Field(
        min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_EXECUTION_ID_BYTES}
    )]] = None,
    label: Annotated[str, Field(json_schema_extra={"maxUtf8Bytes": 1024})] = "",
    resume_policy: Literal["safe", "allow-unsafe"] = "safe",
    cwd: Optional[Annotated[str, Field(json_schema_extra={"maxUtf8Bytes": 4096})]] = None,
    timeout_seconds: Annotated[Optional[float], Field(gt=0)] = None,
    max_output_bytes: Annotated[
        Optional[int], Field(ge=512)
    ] = None,
    __background: bool = False,
) -> str:
    """Start a sequential, durably checkpointed workflow.

    Use for commands likely to approach or exceed the MCP client's tool-call
    window, including one command as one phase, regardless of output size. When
    invoked through MCP, phases run in the background and the call returns an ID
    promptly; use get_execution and get_execution_output to follow work. Direct
    Python calls run phases synchronously and return after the workflow completes
    or encounters a stopping condition.
    Caller disconnection detaches; cancel_execution requests termination.

    Requires Linux process-recovery support and admits eight active or queued
    executions. Check get_execution_capacity before scheduling. Configure
    EPHEMERAL_SESSION_ID, EPHEMERAL_SOCKET_PATH, or
    EPHEMERAL_EXECUTION_STATE_DIR to preserve state across restarts.

    Each phase requires name and command. Completed phases are never rerun.
    Failed or timed-out phases require retry_failed=True. Retrying an unsafe
    phase also requires confirm_unsafe=True unless the execution used
    resume_policy='allow-unsafe'. State uses the session identity or a
    process-private directory by default.
    """
    phase_payloads = [
        phase.model_dump(exclude_none=True) if isinstance(phase, ExecutionPhaseInput) else phase
        for phase in phases
    ]
    manager = _active_execution_manager()
    start_method = (
        manager.start_background
        if __background
        else manager.start
    )
    return _execution_json(
        lambda: _execution_public_payload(
            start_method(
                phase_payloads,
                execution_id=execution_id,
                label=label,
                resume_policy=resume_policy,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
                output_handler=_background_execution_output_handler(),
            )
        ),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


def _hide_internal_background_parameter(function):
    """Keep the internal dispatch flag out of MCP and public introspection."""
    signature = inspect.signature(function)
    function.__signature__ = signature.replace(
        parameters=tuple(
            parameter
            for parameter in signature.parameters.values()
            if parameter.name != "__background"
        )
    )
    return function


start_execution = _hide_internal_background_parameter(start_execution)


@register_tool("resume_execution", "execution", background_execution=True)
@_instrument_tool("resume_execution")
def resume_execution(
    execution_id: Annotated[str, Field(
        min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_EXECUTION_ID_BYTES}
    )],
    retry_failed: bool = False,
    confirm_unsafe: bool = False,
    __background: bool = False,
) -> str:
    """Resume at the first incomplete phase.

    When invoked through MCP, remaining phases run in the background and the
    call returns promptly. Direct Python calls run resumed phases synchronously
    and return after the run completes or encounters a stopping condition.

    Use get_execution for progress; caller disconnection detaches. Completed
    phases are skipped, and safe interrupted phases resume normally. Failed or
    timed-out phases require retry_failed=True. Unsafe retries also require
    confirm_unsafe=True unless the execution used the allow-unsafe policy.
    """
    manager = _active_execution_manager()
    resume_method = (
        manager.resume_background
        if __background
        else manager.resume
    )
    return _execution_json(
        lambda: _execution_public_payload(
            resume_method(
                execution_id,
                retry_failed=retry_failed,
                confirm_unsafe=confirm_unsafe,
                output_handler=_background_execution_output_handler(),
            )
        ),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


resume_execution = _hide_internal_background_parameter(resume_execution)


@register_tool("cancel_execution", "execution")
@_instrument_tool("cancel_execution")
def cancel_execution(
    execution_id: Annotated[str, Field(
        min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_EXECUTION_ID_BYTES}
    )],
) -> str:
    """Request cancellation of an active durable execution by its ID."""
    return _execution_json(
        lambda: _active_execution_manager().request_cancel(execution_id),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


@register_tool("get_execution", "execution")
@_instrument_tool("get_execution")
def get_execution(
    execution_id: Annotated[str, Field(
        min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_EXECUTION_ID_BYTES}
    )],
    include_output: bool = False,
) -> str:
    """Return durable phase metadata and event history for one execution."""
    return _execution_json(
        lambda: _execution_public_payload(
            _execution_get_payload(execution_id, include_output)
        ),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


@register_tool("get_execution_output", "execution")
@_instrument_tool("get_execution_output")
def get_execution_output(
    execution_id: Annotated[str, Field(
        min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_EXECUTION_ID_BYTES}
    )],
    phase_name: Optional[Annotated[str, Field(
        min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_PHASE_NAME_BYTES}
    )]] = None,
    offset: Annotated[
        int, Field(ge=0, description="UTF-8 byte offset when paging one phase")
    ] = 0,
    max_bytes: Annotated[int, Field(ge=512, le=MAX_EXECUTION_OUTPUT_CHUNK_BYTES)] = MAX_EXECUTION_OUTPUT_CHUNK_BYTES,
) -> str:
    """Retrieve bounded output; page one phase using UTF-8 byte offsets."""
    return _execution_json(
        lambda: _execution_public_payload(
            _active_execution_manager().output(
                execution_id,
                phase_name,
                offset=offset,
                max_bytes=max_bytes,
            )
        ),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


@register_tool("list_executions", "execution")
@_instrument_tool("list_executions")
def list_executions(
    limit: Annotated[
        int, Field(ge=1, le=MAX_EXECUTION_LIST_LIMIT)
    ] = DEFAULT_EXECUTION_LIST_LIMIT,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> str:
    """List durable executions with compact partial/completion summaries."""
    return _execution_json(
        lambda: {
            "status": "ok",
            "limit": limit,
            "offset": offset,
            "executions": _active_execution_manager().list_public(limit=limit, offset=offset),
        },
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


@register_tool("get_execution_capacity", "execution")
@_instrument_tool("get_execution_capacity")
def get_execution_capacity() -> str:
    """Report durable storage diagnostics and active/queued background execution counts."""
    return _execution_json(
        lambda: _active_execution_manager().capacity(),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


@register_tool("retire_executions", "execution")
@_instrument_tool("retire_executions")
def retire_executions(
    execution_ids: Annotated[
        List[str], Field(min_length=1, max_length=MAX_EXECUTION_RETIRE_BATCH)
    ],
    archive_path: Optional[Annotated[str, Field(
        json_schema_extra={"maxUtf8Bytes": 4096}
    )]] = None,
    dry_run: bool = True,
) -> str:
    """Preview selected durable records or archive and retire them.

    A dry run is read-only and reports eligibility and projected space reclaimed.
    Actual retirement requires an archive path outside the state directory; the
    archived bundle is durably written before the source record pair is removed.
    Active and fence-pending executions are never retired.
    """
    return _execution_json(
        lambda: _active_execution_manager().retire(
            execution_ids,
            archive_path=archive_path,
            dry_run=dry_run,
        ),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )
