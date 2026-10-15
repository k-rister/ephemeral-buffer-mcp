# Architecture and flow

```mermaid
flowchart TD
    subgraph Ingestion["1. Ingestion & Diagnostics"]
        A["CLI Pipe: command 2>&1 | ephbuf"] --> D["Unix Socket (platform temp dir)"]
        B["Agent Tool: execute_and_capture(cmd)"] --> E["Ephemeral Ring Buffer Engine"]
        C["Agent Tool: capture_text / capture_file"] --> E
        D --> E
        P["Agent Tool: preflight_command(cmd, cwd)"] --> Q["Path, executable & repository diagnostics"]
    end

    subgraph Execution["2. Durable Phase Execution"]
        X["Agent Tools: start_execution / resume_execution"] --> Y["Phase Execution Manager"]
        Y --> Z["Persistent Execution State & Bounded Phase Output"]
        Y --> E
        Z --> AA["get_execution / get_execution_output / list_executions / capacity / retirement"]
    end

    subgraph Indexing["3. Classification & Search Indexing"]
        E --> F["SQLite FTS5 (BM25 Lexical) or Python lexical fallback"]
        E --> G["Required FastEmbed dependency; model loads lazily or in background"]
        E --> K["Diff & Signal Parser (File Maps & Conflict Detection)"]
        E --> M["consolidate_captures: bounded source-aware JSON"]
        M --> E
    end

    subgraph Querying["4. Agent Query & Retrieval"]
        F & G --> H["Reciprocal Rank Fusion (RRF)"]
        H --> I["search_capture(query, mode='hybrid')"]
        K --> L["get_capture_summary: diff stats & bounded file map"]
        L --> N["get_capture_slice: exact lines"]
        I --> J["Precise Context Chunk + Line Numbers"]
    end
```

## Admission and storage bounds

Admission control limits active and queued MCP and socket work before requests
run. The capture engine maintains a bounded LRU buffer; its default limits are
256 captures and 50 MiB of captured UTF-8 content, with a separate lexical
index-chunk budget. When a new capture would exceed a configured limit, least-
recently-used captures are evicted to make room when possible; an individual
capture that cannot fit within a configured budget is rejected.

Each evicted capture records every limit still exceeded when that LRU removal
is planned (`count`, `bytes`, and/or `chunks`). Runtime index-budget reductions
and explicit clears are recorded as `budget` and `clear`. Fixed reason counters
are available in buffer stats and opt-in usage metrics. Capture admission
responses include up to 25 evicted capture IDs and their reasons, plus an
omitted count; runtime budget adjustment results use the same bound. A missing
capture is reported as an empty buffer only when no captures remain.

Durable execution records use a separate state directory and quota. The
aggregate state quota defaults to 4 GiB, with a configurable checkpoint
reserve for atomic writes. Capture eviction does not remove durable execution
records.

FastEmbed is an import-time dependency of the engine. Loading the dense model
is lazy when semantic search first needs it; deployments may also warm the
model in the background after startup.

---
