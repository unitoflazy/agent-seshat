"""Auto-connector: safely backs up and merges `agent-seshat` into coding agent MCP configs.

First-class support for:
- Claude Code (`~/.claude.json`) & Claude Desktop (`claude_desktop_config.json`)
- OpenAI Codex (`~/.codex/config.toml`)
- Cursor (`~/.cursor/mcp.json`)
- GitHub Copilot (`~/.copilot/mcp-config.json`)
- Windsurf / Codeium (`~/.codeium/windsurf/mcp_config.json`)
- Cline (`~/.cline/mcp_settings.json`)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit

from .config import DEFAULT_HOST, DEFAULT_PORT, SERVER_NAME, Settings, get_settings
from .gatekeeper import GATEKEEPER_END, GATEKEEPER_PROTOCOL_BLOCK, GATEKEEPER_START
from .parser import strip_gatekeeper_block

VALID_TARGETS = ("all", "claude", "claude-desktop", "codex", "cursor", "copilot", "windsurf", "cline")
VALID_MODES = ("auto", "stdio", "http")


@dataclass(frozen=True)
class TargetSpec:
    target: str
    label: str
    path: Path
    fmt: str  # "json" | "toml"
    root_key: str  # e.g. "mcpServers" or "mcp_servers"
    optional: bool = False  # If True, only create when explicitly targeted or parent dir exists


@dataclass
class ConnectResult:
    target: str
    label: str
    path: Path
    status: str  # "created" | "updated" | "unchanged" | "skipped" | "error"
    backup_path: Path | None = None
    mode: str = "auto"
    note: str = ""


def _claude_desktop_path(home: Path) -> Path:
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (home / "AppData" / "Roaming")
        return base / "Claude" / "claude_desktop_config.json"
    return home / ".config" / "Claude" / "claude_desktop_config.json"


def get_target_specs(user_home: Path | None = None) -> list[TargetSpec]:
    home = (user_home or Path.home()).expanduser().resolve()
    return [
        TargetSpec(
            target="claude",
            label="Claude Code",
            path=home / ".claude.json",
            fmt="json",
            root_key="mcpServers",
        ),
        TargetSpec(
            target="claude-desktop",
            label="Claude Desktop",
            path=_claude_desktop_path(home),
            fmt="json",
            root_key="mcpServers",
            optional=True,
        ),
        TargetSpec(
            target="codex",
            label="OpenAI Codex",
            path=home / ".codex" / "config.toml",
            fmt="toml",
            root_key="mcp_servers",
        ),
        TargetSpec(
            target="cursor",
            label="Cursor",
            path=home / ".cursor" / "mcp.json",
            fmt="json",
            root_key="mcpServers",
        ),
        TargetSpec(
            target="copilot",
            label="GitHub Copilot",
            path=home / ".copilot" / "mcp-config.json",
            fmt="json",
            root_key="mcpServers",
        ),
        TargetSpec(
            target="windsurf",
            label="Windsurf (Codeium)",
            path=home / ".codeium" / "windsurf" / "mcp_config.json",
            fmt="json",
            root_key="mcpServers",
        ),
        TargetSpec(
            target="cline",
            label="Cline",
            path=home / ".cline" / "mcp_settings.json",
            fmt="json",
            root_key="mcpServers",
        ),
    ]


def build_server_entry(
    mode: str = "auto",
    *,
    fmt: str = "json",
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    command: str = "sesh",
) -> dict[str, Any]:
    """Construct the MCP server config payload for `agent-seshat`."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {VALID_MODES}")

    if mode == "http":
        url = f"http://{host}:{port}/mcp"
        if fmt == "toml":
            return {"url": url}
        return {"type": "http", "url": url}

    args = ["mcp"] if mode == "auto" else ["serve", "--transport", "stdio"]
    return {
        "command": command,
        "args": args,
    }


def _backup_file(path: Path) -> Path:
    backup = path.with_name(f"{path.name}.seshat.bak")
    shutil.copy2(path, backup)
    return backup


def _patch_json_config(
    spec: TargetSpec,
    entry: dict[str, Any],
    dry_run: bool = False,
) -> ConnectResult:
    path = spec.path
    exists = path.exists()
    data: dict[str, Any] = {}

    if exists:
        raw = path.read_text(encoding="utf-8").strip()
        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    return ConnectResult(
                        target=spec.target,
                        label=spec.label,
                        path=path,
                        status="error",
                        note="Top-level JSON value is not an object",
                    )
                data = parsed
            except json.JSONDecodeError as exc:
                return ConnectResult(
                    target=spec.target,
                    label=spec.label,
                    path=path,
                    status="error",
                    note=f"Invalid JSON: {exc}",
                )

    servers = data.get(spec.root_key)
    if not isinstance(servers, dict):
        servers = {}
        data[spec.root_key] = servers

    prev = servers.get(SERVER_NAME)
    if prev == entry:
        return ConnectResult(
            target=spec.target,
            label=spec.label,
            path=path,
            status="unchanged",
        )

    servers[SERVER_NAME] = entry
    status = "updated" if exists else "created"
    backup: Path | None = None

    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        if exists:
            backup = _backup_file(path)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    return ConnectResult(
        target=spec.target,
        label=spec.label,
        path=path,
        status=status,
        backup_path=backup,
    )


def _patch_toml_config(
    spec: TargetSpec,
    entry: dict[str, Any],
    dry_run: bool = False,
) -> ConnectResult:
    path = spec.path
    exists = path.exists()
    if exists:
        raw = path.read_text(encoding="utf-8")
        try:
            doc = tomlkit.parse(raw) if raw.strip() else tomlkit.document()
        except Exception as exc:
            return ConnectResult(
                target=spec.target,
                label=spec.label,
                path=path,
                status="error",
                note=f"Invalid TOML: {exc}",
            )
    else:
        doc = tomlkit.document()

    if spec.root_key not in doc or not isinstance(doc[spec.root_key], dict):
        doc[spec.root_key] = tomlkit.table(is_super_table=True)

    servers = doc[spec.root_key]
    existing = servers.get(SERVER_NAME)
    if isinstance(existing, dict) and dict(existing) == entry:
        return ConnectResult(
            target=spec.target,
            label=spec.label,
            path=path,
            status="unchanged",
        )

    tbl = tomlkit.table()
    for k, v in entry.items():
        tbl[k] = v
    servers[SERVER_NAME] = tbl

    status = "updated" if exists else "created"
    backup: Path | None = None
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        if exists:
            backup = _backup_file(path)
        path.write_text(tomlkit.dumps(doc), encoding="utf-8")

    return ConnectResult(
        target=spec.target,
        label=spec.label,
        path=path,
        status=status,
        backup_path=backup,
    )


def _prepend_gatekeeper_to_markdown(path: Path, create_if_missing: bool = False) -> bool:
    if not path.is_file() and not create_if_missing:
        return False
    existing = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    if GATEKEEPER_START in existing and GATEKEEPER_END in existing:
        cleaned = strip_gatekeeper_block(existing).lstrip()
    else:
        cleaned = existing.lstrip()
    new_text = f"{GATEKEEPER_PROTOCOL_BLOCK}\n\n{cleaned}".rstrip() + "\n"
    if existing == new_text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")
    return True


def _write_cursor_gatekeeper_mdc(workspace: Path) -> Path:
    mdc_path = workspace / ".cursor" / "rules" / "00-seshat-gatekeeper.mdc"
    mdc_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "---\n"
        "description: Mandatory Agent-Seshat Pre-Flight Rule Gatekeeper Protocol\n"
        'globs: ["*"]\n'
        "alwaysApply: true\n"
        "---\n\n"
        f"{GATEKEEPER_PROTOCOL_BLOCK}\n"
    )
    mdc_path.write_text(content, encoding="utf-8")
    return mdc_path


def enforce_gatekeeper_protocol(
    *,
    workspace: Path | None = None,
    user_home: Path | None = None,
    create_missing: bool = True,
) -> list[Path]:
    """Inject the 6-line Seshat Gatekeeper Protocol block at the top of workspace/global
    `CLAUDE.md`, `AGENTS.md`, `.cursor/rules/00-seshat-gatekeeper.mdc` (with `alwaysApply: true`),
    and `.github/copilot-instructions.md`."""
    modified: list[Path] = []
    if workspace is not None:
        ws = workspace.expanduser().resolve()
        for rel in ("CLAUDE.md", "AGENTS.md", ".github/copilot-instructions.md"):
            target = ws / rel
            if _prepend_gatekeeper_to_markdown(target, create_if_missing=create_missing):
                modified.append(target)
        mdc_file = _write_cursor_gatekeeper_mdc(ws)
        modified.append(mdc_file)

    if user_home is not None:
        home = user_home.expanduser().resolve()
        for rel in (".claude/CLAUDE.md", ".codex/AGENTS.md", ".copilot/copilot-instructions.md"):
            target = home / rel
            if target.is_file() or (create_missing and target.parent.is_dir()):
                if _prepend_gatekeeper_to_markdown(target, create_if_missing=create_missing):
                    modified.append(target)
    return modified


def connect_agents(
    target: str = "all",
    mode: str = "auto",
    *,
    user_home: Path | None = None,
    workspace: Path | None = None,
    settings: Settings | None = None,
    command: str = "sesh",
    dry_run: bool = False,
    enforce_gatekeeper: bool = False,
) -> list[ConnectResult]:
    """Wire coding agents to `agent-seshat` while preserving existing MCP server entries."""
    target = target.strip().lower()
    mode = mode.strip().lower()
    if target not in VALID_TARGETS:
        raise ValueError(f"Unsupported target {target!r}; choose from {', '.join(VALID_TARGETS)}")
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode {mode!r}; choose from {', '.join(VALID_MODES)}")

    settings = settings or get_settings()
    specs = get_target_specs(user_home)

    selected: list[TargetSpec] = []
    for spec in specs:
        if target == "all":
            if spec.optional and not spec.path.exists() and not spec.path.parent.exists():
                continue
            selected.append(spec)
        elif target == "claude" and spec.target in ("claude", "claude-desktop"):
            if spec.optional and not spec.path.exists() and not spec.path.parent.exists():
                continue
            selected.append(spec)
        elif spec.target == target:
            selected.append(spec)

    results: list[ConnectResult] = []
    for spec in selected:
        entry = build_server_entry(
            mode,
            fmt=spec.fmt,
            host=settings.host,
            port=settings.port,
            command=command,
        )
        if spec.fmt == "toml":
            res = _patch_toml_config(spec, entry, dry_run=dry_run)
        else:
            res = _patch_json_config(spec, entry, dry_run=dry_run)
        res.mode = mode
        results.append(res)

    if enforce_gatekeeper and not dry_run:
        enforce_gatekeeper_protocol(workspace=workspace, user_home=user_home, create_missing=True)

    return results


def inspect_connections(user_home: Path | None = None) -> list[dict[str, Any]]:
    """Return connection status across all supported agent config files (used by `sesh status`)."""
    statuses: list[dict[str, Any]] = []
    for spec in get_target_specs(user_home):
        info: dict[str, Any] = {
            "target": spec.target,
            "label": spec.label,
            "path": spec.path,
            "exists": spec.path.exists(),
            "connected": False,
            "entry": None,
        }
        if spec.path.exists():
            try:
                raw = spec.path.read_text(encoding="utf-8")
                if spec.fmt == "toml":
                    doc = tomlkit.parse(raw) if raw.strip() else {}
                    servers = doc.get(spec.root_key, {})
                else:
                    doc = json.loads(raw) if raw.strip() else {}
                    servers = doc.get(spec.root_key, {}) if isinstance(doc, dict) else {}
                if isinstance(servers, dict) and SERVER_NAME in servers:
                    info["connected"] = True
                    info["entry"] = dict(servers[SERVER_NAME]) if isinstance(servers[SERVER_NAME], dict) else servers[SERVER_NAME]
            except Exception:
                info["error"] = True
        statuses.append(info)
    return statuses
