"""Exercise the installed distribution through its public MCP stdio API."""

import asyncio
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import uuid

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULES_FROM_DISTRIBUTION = (
    "ephemeral_buffer_mcp",
    "ephemeral_buffer_mcp.server",
    "ephemeral_buffer_mcp.engine",
    "ephemeral_buffer_mcp.metrics",
)


def _report_step(message: str) -> None:
    print(f"[installed-mcp-smoke] {message}", file=sys.stderr, flush=True)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _assert_installed_modules() -> None:
    """Fail if Python would resolve application modules from the checkout."""
    prefix = Path(sys.prefix).resolve()
    distribution = importlib.metadata.distribution("ephemeral-buffer-mcp")

    for module_name in MODULES_FROM_DISTRIBUTION:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise AssertionError(f"Could not locate installed module {module_name!r}")

        module_path = Path(spec.origin).resolve()
        module_parts = module_name.split(".")
        expected_relative_path = (
            Path(*module_parts) / "__init__.py"
            if len(module_parts) == 1
            else Path(*module_parts[:-1]) / f"{module_parts[-1]}.py"
        )
        expected_path = Path(distribution.locate_file(expected_relative_path)).resolve()
        if module_path != expected_path:
            raise AssertionError(
                f"{module_name} resolved to {module_path}, expected the installed "
                f"distribution file {expected_path}"
            )
        if not _is_within(module_path, prefix):
            raise AssertionError(
                f"{module_name} resolved outside the isolated environment: {module_path}"
            )
        if _is_within(module_path, REPOSITORY_ROOT):
            raise AssertionError(
                f"{module_name} resolved from the source checkout: {module_path}"
            )


def _structured_result(result, tool_name: str) -> dict:
    if result.isError:
        raise AssertionError(f"MCP tool {tool_name!r} returned an error: {result.content}")

    structured = result.structuredContent
    if not isinstance(structured, dict):
        raise AssertionError(
            f"MCP tool {tool_name!r} did not return structured content: {result.content}"
        )
    if structured.get("status") != "ok":
        raise AssertionError(f"MCP tool {tool_name!r} failed: {structured}")
    return structured


async def _exercise_mcp_roundtrip() -> None:
    working_directory = Path.cwd().resolve()
    if _is_within(working_directory, REPOSITORY_ROOT):
        raise AssertionError(
            f"Run this smoke test outside the source checkout; current directory is "
            f"{working_directory}"
        )

    with tempfile.TemporaryDirectory(
        prefix="ephemeral-mcp-smoke-", dir=working_directory
    ) as runtime_directory_name:
        runtime_directory = Path(runtime_directory_name)
        server_environment = os.environ.copy()
        server_environment.pop("PYTHONPATH", None)
        server_environment.update(
            {
                "EPHEMERAL_TEST_EMBEDDINGS": "1",
                "EPHEMERAL_DISABLE_SOCKET_SERVER": "1",
                "EPHEMERAL_REQUIRE_ISOLATION": "1",
                "EPHEMERAL_SESSION_ID": f"installed-smoke-{uuid.uuid4().hex}",
                "EPHEMERAL_SOCKET_PATH": str(runtime_directory / "disabled.sock"),
                "EPHEMERAL_EXECUTION_STATE_DIR": str(runtime_directory / "executions"),
                "EPHEMERAL_METRICS_FILE": str(runtime_directory / "metrics.json"),
                "EPHEMERAL_LOG_FILE": str(runtime_directory / "logs.jsonl"),
            }
        )

        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ephemeral_buffer_mcp.server"],
            env=server_environment,
            cwd=str(runtime_directory),
        )

        _report_step("starting installed server over stdio")
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                _report_step("initializing MCP session")
                await session.initialize()
                _report_step("listing MCP tools")

                listed_tools = await session.list_tools()
                tool_names = {tool.name for tool in listed_tools.tools}
                expected_tools = {
                    "capture_text",
                    "search_capture",
                    "get_capture_slice",
                    "list_captures",
                }
                missing_tools = expected_tools - tool_names
                if missing_tools:
                    raise AssertionError(
                        f"MCP server did not list expected tools: {sorted(missing_tools)}"
                    )
                _report_step("capturing text")

                label = f"installed-mcp-smoke-{uuid.uuid4().hex}"
                needle = f"roundtrip-needle-{uuid.uuid4().hex}"
                content = f"Installed package {needle}\nSecond line for retrieval."

                capture = _structured_result(
                    await session.call_tool(
                        "capture_text",
                        {"content": content, "label": label, "content_type": "text"},
                    ),
                    "capture_text",
                )
                capture_id = capture["data"]["capture_id"]

                _report_step("listing captures")
                captures = _structured_result(
                    await session.call_tool("list_captures", {}),
                    "list_captures",
                )
                listed_ids = {
                    item["capture_id"] for item in captures["data"]["captures"]
                }
                if capture_id not in listed_ids:
                    raise AssertionError(
                        f"New capture {capture_id!r} was not returned by list_captures"
                    )

                _report_step("searching capture with deterministic embeddings")
                search = _structured_result(
                    await session.call_tool(
                        "search_capture",
                        {"query": needle, "mode": "hybrid", "capture_id": capture_id},
                    ),
                    "search_capture",
                )
                matches = search["data"].get("matches", [])
                if not matches or not any(
                    needle in match.get("snippet", "") for match in matches
                ):
                    raise AssertionError(
                        f"Hybrid search did not find the captured needle: {search}"
                    )

                _report_step("retrieving captured lines")
                retrieved = _structured_result(
                    await session.call_tool(
                        "get_capture_slice",
                        {
                            "start_line": 1,
                            "end_line": 2,
                            "capture_id": capture_id,
                            "max_bytes": 8192,
                        },
                    ),
                    "get_capture_slice",
                )
                if retrieved["data"].get("content") != content:
                    raise AssertionError(
                        f"Retrieved capture did not match input: {retrieved}"
                    )
                _report_step("MCP roundtrip complete")


def main() -> None:
    """Verify the installed distribution through an MCP initialization.

    The smoke test lists tools, captures text, searches it, and retrieves lines.
    """
    _assert_installed_modules()
    asyncio.run(asyncio.wait_for(_exercise_mcp_roundtrip(), timeout=60))
    print("Installed package MCP initialize/list/capture/search/retrieve smoke passed.")


if __name__ == "__main__":
    main()
