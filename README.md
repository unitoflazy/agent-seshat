# Agent-Seshat (`sesh`) v2.0 — Unified MCP Gateway, Deep Rule Sync & Rule Gatekeeper

**Agent-Seshat** (`agent-seshat` / CLI alias `sesh`) is a zero-cost, 100% local **Unified MCP Gateway, Interactive Rule Gatekeeper & Hybrid Skill Librarian** named after **Seshat**, the Egyptian goddess of writing, architecture, and the *"Mistress of the House of Books"*.

It solves **context-window bloat**, **MCP server sprawl**, and **rule fragmentation** across **Claude Code (first-class citizen)**, **OpenAI Codex**, **Cursor**, **GitHub Copilot**, **Windsurf**, and **Cline**.

---

## Core Capabilities in v2.0

1. **Deep Discovery & Bidirectional Sync (`sesh deep-sync`)**:
   - Fast `os.scandir` traversal across `CLAUDE.md`, `.claude/commands/*.md`, `~/.claude/skills/**`, `AGENTS.md`, `CODEX.md`, `~/.codex/instructions.md`, `.cursorrules`, `.cursor/rules/*.mdc` (`alwaysApply`, `globs`), `.github/copilot-instructions.md`, `.github/instructions/*.instructions.md` (`applyTo`), `.windsurfrules`, `.clinerules`, and `.roorules`.
   - Automatically scrubs hardcoded secrets (`[REDACTED_SECRET]`), deduplicates via `content_hash`, preserves `origin_path` for re-syncing, and auto-pins Cursor `.mdc` rules with `alwaysApply: true`.
2. **Mandatory Pre-Task Rule Gatekeeper & "Always-On" Pinned Rules (`gatekeeper.py` & `sesh rules`)**:
   - Exposes `propose_session_rules` (Step 0 before code execution) which auto-loads **Pinned (`Always-Use`)** rules for `global` and `project:<repo-name>` scopes (plus `active_files` glob matches) and presents top candidate rules via Hybrid Search for engineer confirmation.
   - Exposes `activate_rules` to load selected rules and persist in-chat `"always"` preferences (`pinned = 1`) in SQLite.
   - Injects a compact 6-line **Seshat Gatekeeper Protocol** bootstrap stub (`--enforce-gatekeeper`) into root agent instruction files.
3. **Centralized MCP Aggregator & Lazy Meta-Broker (`mcp_aggregator.py` & `sesh mcp-hub`)**:
   - Harvests and deduplicates external MCP servers from Claude, Cursor, Copilot/VS Code, and Codex into `~/.agent-seshat/mcp_registry.json` (ignoring self-references).
   - Indexes downstream tool schemas into SQLite + FTS5 + ONNX (`mcp_tools_meta` & `mcp_tools_fts`) and exposes two meta-broker tools (`search_mcp_tools` and `call_mcp_tool`) so agents connect exclusively to `agent-seshat`.
4. **Zero-Miss Hybrid Search Engine (`search.py` + `indexer.py`)**:
   - Combines quantized CPU ONNX embeddings (`BAAI/bge-small-en-v1.5` via `fastembed` — no PyTorch, no external API keys), weighted SQLite FTS5 BM25 (`10x name, 8x tags, 5x description, 1x body`), exact identifier/tag boosting, project-scope multipliers (`1.35x / 1.0x / 0.40x`), and Reciprocal Rank Fusion ($k = 60$).

---

## CLI Reference (`sesh` / `agent-seshat`)

```bash
# Deep-scan & sync all agent rules/skills and inject the Gatekeeper bootstrap block
sesh deep-sync ~/Workspace --enforce-gatekeeper

# Inspect, pin, unpin, or interactively toggle Always-Choose rules
sesh rules list
sesh rules list --agent cursor --pinned-only
sesh rules pin go-chi-router --scope project:payments-api
sesh rules unpin go-chi-router
sesh rules select

# Centralize & proxy downstream MCP servers across Claude, Cursor, Copilot, and Codex
sesh mcp-hub harvest
sesh mcp-hub sync
sesh mcp-hub list
sesh mcp-hub consolidate

# Terminal hybrid search, status, and daemon modes
sesh search "errgroup.WithContext"
sesh connect --target all --mode auto --enforce-gatekeeper
sesh serve --transport http --host 127.0.0.1 --port 8765
sesh status
```

---

## Testing

```bash
uv run pytest -v
```
