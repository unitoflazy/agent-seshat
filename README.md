# Agent-Seshat (`sesh`)

**One local library for your AI coding agents' rules, skills, environment variables, and MCP servers.**

Agent-Seshat scans the instruction files and MCP configs that Claude Code, Codex, Cursor, Copilot, Windsurf, and Cline each keep in their own format. It indexes them in one local SQLite + ONNX database, and lets every agent pull in only what the current task needs via MCP or native CLI (`sesh`).

Everything runs locally in Docker with **zero host Python dependencies** required. No API keys, no hosted service.

## Why

- **Rule sprawl.** The same conventions live in `CLAUDE.md`, `.cursorrules`, `AGENTS.md`, Copilot instructions, and more. They drift apart.
- **Context bloat.** Loading every rule and every MCP tool schema up front wastes the context window.
- **MCP sprawl.** Each agent has its own list of MCP servers to configure and keep in sync.
- **Env sprawl.** Sharing scoped environment variables (`global`, `project:<repo>`, `mcp:<server>`) across agents and downstream MCP servers is tedious.

## What it does

| Feature | Command | In one line |
|---|---|---|
| Docker daemon | `sesh daemon` | Starts/stops/inspects the warm background Docker container (`127.0.0.1:8765`) |
| Rule sync | `sesh deep-sync` | Finds rule/skill files across all supported agents and indexes them in one place |
| Rule gatekeeper | `sesh rules` | Before a task starts, the agent loads your pinned rules and confirms the best-matching ones |
| MCP hub | `sesh mcp-hub` | Collects your MCP servers from every agent; agents see just two tools: `search_mcp_tools` and `call_mcp_tool` |
| Env store | `sesh env` | Stores and retrieves scoped environment variables (`global`, `project:<repo>`, `mcp:<server>`) |
| Hybrid search | `sesh search` | Sub-25ms semantic (ONNX `bge-small-en-v1.5`) + keyword (SQLite FTS5 BM25) search |

Claude Code is the primary target. Cursor, Codex, Copilot, Windsurf, and Cline are supported through the same index.

## Installation (Docker + Native `sesh` CLI)

> Requires **Docker** (`docker compose`). No Python or `uv` installation needed on the host!

```bash
# 1. Start the warm Docker container & install `sesh` onto your PATH
./install.sh
# (or: make install)

# 2. Verify installation & warm daemon status
sesh daemon status
sesh status

# 3. Index your rules/skills and wire up your agents
sesh deep-sync ~/Workspace
sesh connect --all

# 4. Run a hybrid search from anywhere in your terminal
sesh search "errgroup.WithContext"
```

### How the Docker CLI Wrapper Works
- `./install.sh` starts the `agent-seshat` container with pre-baked ONNX weights (`BAAI/bge-small-en-v1.5`) and links [`bin/sesh`](bin/sesh) into `~/.local/bin/sesh` (and `/opt/homebrew/bin/sesh` if writable).
- Running `sesh <command>` on your host executes against the warm container while preserving your current working directory and `$HOME` paths, and routes `sesh search` directly to the warm HTTP daemon (`http://127.0.0.1:8765/search`) for ~18ms latency.
- To uninstall at any time:
  ```bash
  ./uninstall.sh          # Stops container & removes CLI links (keeps ~/.agent-seshat)
  ./uninstall.sh --purge  # Also deletes ~/.agent-seshat
  ```

### Container Lifecycle (`sesh daemon` / `Makefile`)

```bash
sesh daemon status    # or: make status
sesh daemon up        # or: make up
sesh daemon restart   # or: make restart
sesh daemon logs -f   # or: make logs
sesh daemon down      # or: make down
```

## Features

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

### 4. Scoped Environment Variable Store (`sesh env`)

Store and retrieve environment variables scoped to `global`, `project:<repo>`, or `mcp:<server>` (automatically injected into downstream MCP servers and masked when sensitive):

```bash
sesh env set DATABASE_URL "postgres://localhost:5432/app" --scope project:agent-seshat --secret
sesh env get DATABASE_URL --scope project:agent-seshat --reveal
sesh env list
sesh env import .env --scope project:agent-seshat
sesh env export --scope project:agent-seshat
```

### 5. Hybrid Search
Results combine four signals:
- vector similarity (ONNX embeddings via `fastembed`)
- keyword ranking (SQLite FTS5 BM25, with name and tag matches weighted highest)
- exact identifier and tag boosts
- a project-scope multiplier: current project 1.35×, global 1.0×, other projects 0.40×

These are merged with Reciprocal Rank Fusion ($k = 60$).

## Data & Backups

- All data is stored in `~/.agent-seshat/` (`skills/`, `cache/index.db`, `mcp_registry.json`, `env_store.json`).
- Whenever `sesh connect` or `sesh mcp-hub consolidate` edits an agent config file, it creates a timestamped `.bak.<timestamp>` backup alongside the original file first.

## Development (Local Python)

```bash
make test
# or: uv run pytest -v
```

---

*Named for Seshat, the Egyptian goddess of writing and record-keeping, "Mistress of the House of Books."*
