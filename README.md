# Agent-Seshat (`sesh`)

**One local library for your AI coding agents' rules, skills, and MCP servers.**

Agent-Seshat scans the instruction files and MCP configs that Claude Code, Codex, Cursor, Copilot, Windsurf, and Cline each keep in their own format. It indexes them in one local SQLite database, and lets every agent pull in only what the current task needs.

Everything runs on your machine. No API keys, no hosted service.

## Why

- **Rule sprawl.** The same conventions live in `CLAUDE.md`, `.cursorrules`, `AGENTS.md`, Copilot instructions, and more. They drift apart.
- **Context bloat.** Loading every rule and every MCP tool schema up front wastes the context window.
- **MCP sprawl.** Each agent has its own list of MCP servers to configure and keep in sync.

## What it does

| Feature | Command | In one line |
|---|---|---|
| Rule sync | `sesh deep-sync` | Finds rule/skill files across all supported agents and indexes them in one place |
| Rule gatekeeper | `sesh rules` | Before a task starts, the agent loads your pinned rules and confirms the best-matching ones |
| MCP hub | `sesh mcp-hub` | Collects your MCP servers from every agent; agents see just two tools: `search_mcp_tools` and `call_mcp_tool` |
| Hybrid search | `sesh search` | Semantic + keyword search over rules, skills, and MCP tool schemas |

Claude Code is the primary target. The others are supported through the same index.

## Quickstart

> Requires Python and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Index your rules and skills
sesh deep-sync ~/Workspace

# 2. Wire up your agents
sesh connect --target all

# 3. Check it worked
sesh status
sesh search "errgroup.WithContext"
```

The first run downloads the embedding model (`BAAI/bge-small-en-v1.5`, quantized, CPU-only).

## How it works

### 1. Deep sync

`sesh deep-sync <dir>` scans for:

| Agent | Files |
|---|---|
| Claude Code | `CLAUDE.md`, `.claude/commands/*.md`, `~/.claude/skills/**` |
| Codex | `AGENTS.md`, `CODEX.md`, `~/.codex/instructions.md` |
| Cursor | `.cursorrules`, `.cursor/rules/*.mdc` (`alwaysApply`, `globs`) |
| Copilot | `.github/copilot-instructions.md`, `.github/instructions/*.instructions.md` (`applyTo`) |
| Windsurf / Cline / Roo | `.windsurfrules`, `.clinerules`, `.roorules` |

While indexing, it:
- redacts hardcoded secrets (`[REDACTED_SECRET]`)
- deduplicates by content hash
- records each rule's `origin_path` so it can sync back
- auto-pins Cursor rules marked `alwaysApply: true`

### 2. Rule gatekeeper

Agents call `propose_session_rules` before touching code. It returns:
1. your **pinned** rules for the `global` scope and the current `project:<repo-name>` scope, plus any whose globs match the open files
2. the top matching candidates from hybrid search, for you to confirm

The agent then calls `activate_rules` to load your selection. Saying "always" in chat pins a rule for next time.

```bash
sesh rules list --agent cursor --pinned-only
sesh rules pin go-chi-router --scope project:payments-api
sesh rules unpin go-chi-router
sesh rules select            # interactive picker
```

`sesh deep-sync --enforce-gatekeeper` (or `sesh connect --enforce-gatekeeper`) adds a short 6-line bootstrap block to your root agent files so they call the gatekeeper first.

### 3. MCP hub

```bash
sesh mcp-hub harvest       # read MCP servers from Claude, Cursor, Copilot/VS Code, Codex
sesh mcp-hub sync          # index their tool schemas
sesh mcp-hub list          # show the merged registry
sesh mcp-hub consolidate   # point agents at agent-seshat only
```

The registry lives at `~/.agent-seshat/mcp_registry.json`. Agents connect to Agent-Seshat alone, search for the tool they need, and call it through the broker. This avoids loading hundreds of tool schemas into context.

### 4. Search

Results combine four signals:
- vector similarity (ONNX embeddings via `fastembed`)
- keyword ranking (SQLite FTS5 BM25, with name and tag matches weighted highest)
- exact identifier and tag boosts
- a project-scope multiplier: current project 1.35×, global 1.0×, other projects 0.40×

These are merged with Reciprocal Rank Fusion (k = 60).

## What it changes on your machine

| Action | Touches |
|---|---|
| `deep-sync` | Reads rule files. Writes only to its own database. |
| `deep-sync --enforce-gatekeeper`, `connect` | **Edits** root agent instruction files and agent MCP configs |
| `mcp-hub consolidate` | **Edits** agent MCP configs |
| `mcp-hub harvest` | Reads MCP configs, which may contain credentials |

<!-- TODO: document where the database lives, whether edits are backed up, and how to undo `connect` / `consolidate`. -->

## Running as a server

```bash
sesh serve --transport http --host 127.0.0.1 --port 8765
```

## Development

```bash
uv run pytest -v
```

---

*Named for Seshat, the Egyptian goddess of writing and record-keeping, "Mistress of the House of Books."*