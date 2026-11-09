"""Centralized MCP Server Aggregator, Tool Indexer & Lazy Meta-Broker (`sesh mcp-hub`).

Harvests MCP server definitions across Claude Code, Claude Desktop, Cursor, GitHub Copilot /
VS Code, OpenAI Codex, Windsurf, and Cline into `~/.agent-seshat/mcp_registry.json`, indexes
downstream tool schemas into SQLite + FTS5 + ONNX, and exposes `search_mcp_tools` +
`call_mcp_tool`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomlkit

from .config import DEFAULT_HOST, DEFAULT_PORT, SERVER_NAME, Settings, get_settings, user_home_dir
from .connector import _claude_desktop_path, _is_macos_home, build_server_entry
from .indexer import SkillIndex

log = logging.getLogger(__name__)

SELF_SERVER_NAMES = frozenset({"agent-seshat", "agent_seshat", "sesh", "seshat"})


@dataclass
class MCPServerEntry:
    name: str
    transport: str  # "stdio" | "http"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    source_agents: list[str] = field(default_factory=list)

    def signature(self) -> tuple:
        if self.transport == "http":
            return ("http", self.url.rstrip("/"))
        return ("stdio", self.command, tuple(self.args))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerEntry":
        return cls(
            name=str(data.get("name") or ""),
            transport=str(data.get("transport") or ("http" if data.get("url") else "stdio")),
            command=str(data.get("command") or ""),
            args=[str(a) for a in (data.get("args") or [])],
            env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
            url=str(data.get("url") or ""),
            headers={str(k): str(v) for k, v in (data.get("headers") or {}).items()},
            enabled=bool(data.get("enabled", True)),
            source_agents=list(dict.fromkeys(str(a) for a in (data.get("source_agents") or []))),
        )


@dataclass
class MCPHarvestReport:
    servers: list[MCPServerEntry] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped_self: int = 0
    scanned_files: list[Path] = field(default_factory=list)


@dataclass
class MCPSyncResult:
    server_name: str
    tools_count: int = 0
    elapsed_ms: float = 0.0
    error: str = ""


def _normalize_server_name(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "-", s)
    return re.sub(r"-{2,}", "-", s).strip("-") or "mcp-server"


def is_self_server(name: str, raw_cfg: dict[str, Any]) -> bool:
    """Return True if the server definition refers to `agent-seshat` / `sesh` itself."""
    norm = _normalize_server_name(name)
    if norm in SELF_SERVER_NAMES:
        return True
    cmd = Path(str(raw_cfg.get("command") or "")).name.lower()
    if cmd in SELF_SERVER_NAMES:
        return True
    args_joined = " ".join(str(a).lower() for a in (raw_cfg.get("args") or []))
    if "agent-seshat" in args_joined or "agent_seshat" in args_joined:
        return True
    url = str(raw_cfg.get("url") or "").lower()
    if f":{DEFAULT_PORT}/mcp" in url:
        return True
    return False


def _vscode_settings_path(home: Path) -> Path:
    if _is_macos_home(home):
        return home / "Library" / "Application Support" / "Code" / "User" / "settings.json"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (home / "AppData" / "Roaming")
        return base / "Code" / "User" / "settings.json"
    return home / ".config" / "Code" / "User" / "settings.json"


def _candidate_mcp_config_sources(
    workspace: Path | None = None,
    user_home: Path | None = None,
    extra_paths: list[Path] | None = None,
) -> list[tuple[str, Path, str]]:
    """Return `(source_agent, file_path, fmt)` tuples for MCP config discovery."""
    home = (user_home or user_home_dir()).expanduser().resolve()
    ws = (workspace or Path.cwd()).expanduser().resolve()

    sources: list[tuple[str, Path, str]] = [
        ("claude", home / ".claude.json", "json"),
        ("claude", ws / ".mcp.json", "json"),
        ("claude", _claude_desktop_path(home), "json"),
        ("cursor", home / ".cursor" / "mcp.json", "json"),
        ("cursor", ws / ".cursor" / "mcp.json", "json"),
        ("copilot", home / ".copilot" / "mcp-config.json", "json"),
        ("copilot", ws / ".vscode" / "mcp.json", "json"),
        ("copilot", _vscode_settings_path(home), "json"),
        ("codex", home / ".codex" / "config.toml", "toml"),
        ("windsurf", home / ".codeium" / "windsurf" / "mcp_config.json", "json"),
        ("windsurf", home / ".cline" / "mcp_settings.json", "json"),
    ]
    for p in extra_paths or []:
        rp = Path(p).expanduser().resolve()
        fmt = "toml" if rp.suffix.lower() == ".toml" else "json"
        lower = str(rp).lower()
        if "claude" in lower or rp.name == ".mcp.json":
            agent = "claude"
        elif "cursor" in lower:
            agent = "cursor"
        elif "codex" in lower:
            agent = "codex"
        elif "copilot" in lower or "vscode" in lower or rp.name == "settings.json":
            agent = "copilot"
        else:
            agent = "windsurf"
        sources.append((agent, rp, fmt))

    dedup: list[tuple[str, Path, str]] = []
    seen: set[Path] = set()
    for agent, path, fmt in sources:
        if path not in seen:
            seen.add(path)
            dedup.append((agent, path, fmt))
    return dedup


def _extract_servers_from_dict(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract server definitions from various JSON/TOML MCP config layouts."""
    out: dict[str, dict[str, Any]] = {}
    for key in ("mcpServers", "mcp_servers", "servers"):
        block = data.get(key)
        if isinstance(block, dict):
            for sname, scfg in block.items():
                if isinstance(scfg, dict):
                    out[str(sname)] = dict(scfg)
    # VS Code settings.json: `{"mcp": {"servers": {...}}}` or `{"mcp.servers": {...}}`
    mcp_block = data.get("mcp")
    if isinstance(mcp_block, dict) and isinstance(mcp_block.get("servers"), dict):
        for sname, scfg in mcp_block["servers"].items():
            if isinstance(scfg, dict):
                out[str(sname)] = dict(scfg)
    if isinstance(data.get("mcp.servers"), dict):
        for sname, scfg in data["mcp.servers"].items():
            if isinstance(scfg, dict):
                out[str(sname)] = dict(scfg)
    return out


def _parse_server_entry(name: str, raw: dict[str, Any], source_agent: str) -> MCPServerEntry | None:
    url = str(raw.get("url") or raw.get("serverUrl") or "").strip()
    command = str(raw.get("command") or "").strip()
    if not url and not command:
        return None
    transport = "http" if url else "stdio"
    args = [str(a) for a in (raw.get("args") or [])]
    env = {str(k): str(v) for k, v in (raw.get("env") or {}).items()} if isinstance(raw.get("env"), dict) else {}
    headers = (
        {str(k): str(v) for k, v in (raw.get("headers") or {}).items()}
        if isinstance(raw.get("headers"), dict)
        else {}
    )
    enabled = not bool(raw.get("disabled", False)) and bool(raw.get("enabled", True))
    return MCPServerEntry(
        name=_normalize_server_name(name),
        transport=transport,
        command=command,
        args=args,
        env=env,
        url=url,
        headers=headers,
        enabled=enabled,
        source_agents=[source_agent],
    )


def load_mcp_registry(settings: Settings | None = None) -> dict[str, MCPServerEntry]:
    settings = settings or get_settings()
    reg_path = settings.mcp_registry_path
    if not reg_path.is_file():
        return {}
    try:
        raw = json.loads(reg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    servers_raw = raw.get("servers", raw) if isinstance(raw, dict) else {}
    if isinstance(servers_raw, list):
        return {
            e.name: e
            for item in servers_raw
            if isinstance(item, dict) and (e := MCPServerEntry.from_dict(item)).name
        }
    if isinstance(servers_raw, dict):
        out: dict[str, MCPServerEntry] = {}
        for k, v in servers_raw.items():
            if isinstance(v, dict):
                d = dict(v)
                d.setdefault("name", k)
                entry = MCPServerEntry.from_dict(d)
                out[entry.name] = entry
        return out
    return {}


def save_mcp_registry(
    entries: dict[str, MCPServerEntry] | list[MCPServerEntry],
    settings: Settings | None = None,
) -> Path:
    settings = (settings or get_settings()).ensure_dirs()
    reg_path = settings.mcp_registry_path
    by_name = (
        {e.name: e for e in entries}
        if isinstance(entries, list)
        else dict(entries)
    )
    payload = {
        "version": "2.0",
        "servers": {name: by_name[name].to_dict() for name in sorted(by_name)},
    }
    reg_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return reg_path


def harvest_mcp_configs(
    *,
    workspace: Path | None = None,
    user_home: Path | None = None,
    extra_paths: list[Path] | None = None,
    settings: Settings | None = None,
) -> MCPHarvestReport:
    """Scan agent MCP config files, deduplicate servers, ignore `agent-seshat`, and save `mcp_registry.json`."""
    settings = (settings or get_settings()).ensure_dirs()
    existing = load_mcp_registry(settings)
    report = MCPHarvestReport()

    by_sig: dict[tuple, MCPServerEntry] = {e.signature(): e for e in existing.values()}
    by_name: dict[str, MCPServerEntry] = dict(existing)

    for source_agent, cfg_path, fmt in _candidate_mcp_config_sources(workspace, user_home, extra_paths):
        if not cfg_path.is_file():
            continue
        try:
            raw_text = cfg_path.read_text(encoding="utf-8")
            if not raw_text.strip():
                continue
            parsed = dict(tomlkit.parse(raw_text)) if fmt == "toml" else json.loads(raw_text)
        except Exception as exc:
            log.warning("Skipping unreadable MCP config %s: %s", cfg_path, exc)
            continue

        if not isinstance(parsed, dict):
            continue
        report.scanned_files.append(cfg_path)
        raw_servers = _extract_servers_from_dict(parsed)

        for raw_name, raw_cfg in raw_servers.items():
            if is_self_server(raw_name, raw_cfg):
                report.skipped_self += 1
                continue
            entry = _parse_server_entry(raw_name, raw_cfg, source_agent)
            if entry is None:
                continue

            sig = entry.signature()
            if sig in by_sig:
                existing_entry = by_sig[sig]
                if source_agent not in existing_entry.source_agents:
                    existing_entry.source_agents.append(source_agent)
                    if existing_entry.name not in report.added and existing_entry.name not in report.updated:
                        report.updated.append(existing_entry.name)
                if entry.env:
                    existing_entry.env = {**entry.env, **existing_entry.env}
                continue

            if entry.name in by_name:
                prev = by_name[entry.name]
                for ag in prev.source_agents:
                    if ag not in entry.source_agents:
                        entry.source_agents.append(ag)
                by_name[entry.name] = entry
                by_sig[sig] = entry
                if entry.name not in report.updated:
                    report.updated.append(entry.name)
            else:
                by_name[entry.name] = entry
                by_sig[sig] = entry
                report.added.append(entry.name)

    save_mcp_registry(by_name, settings)
    report.servers = [by_name[k] for k in sorted(by_name)]
    return report


# ---------------------------------------------------------------------------
# Downstream MCP client operations (sync & call)
# ---------------------------------------------------------------------------


async def _fetch_server_tools_async(
    entry: MCPServerEntry,
    timeout: float = 15.0,
    extra_env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    async def _collect(read_stream: Any, write_stream: Any) -> list[dict[str, Any]]:
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            res = await session.list_tools()
            tools_out: list[dict[str, Any]] = []
            for t in res.tools:
                schema = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
                tools_out.append(
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "inputSchema": schema,
                    }
                )
            return tools_out

    if entry.transport == "http":
        async with asyncio.timeout(timeout):
            async with streamablehttp_client(entry.url, headers=entry.headers or None) as (r_stream, w_stream, _):
                return await _collect(r_stream, w_stream)

    merged_env = {**os.environ, **(extra_env or {}), **(entry.env or {})}
    params = StdioServerParameters(command=entry.command, args=entry.args, env=merged_env)
    with open(os.devnull, "w") as err_sink:
        async with asyncio.timeout(timeout):
            async with stdio_client(params, errlog=err_sink) as (r_stream, w_stream):
                return await _collect(r_stream, w_stream)


async def _call_server_tool_async(
    entry: MCPServerEntry,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    timeout: float = 30.0,
    extra_env: dict[str, str] | None = None,
) -> str:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    async def _invoke(read_stream: Any, write_stream: Any) -> str:
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments or {})
            parts: list[str] = []
            for block in result.content:
                text = getattr(block, "text", None)
                if text is not None:
                    parts.append(str(text))
                elif hasattr(block, "model_dump"):
                    parts.append(json.dumps(block.model_dump()))
                else:
                    parts.append(str(block))
            joined = "\n".join(parts).strip()
            if getattr(result, "isError", False):
                return f"[MCP Tool Error from {entry.name}::{tool_name}]\n{joined}"
            return joined

    if entry.transport == "http":
        async with asyncio.timeout(timeout):
            async with streamablehttp_client(entry.url, headers=entry.headers or None) as (r_stream, w_stream, _):
                return await _invoke(r_stream, w_stream)

    merged_env = {**os.environ, **(extra_env or {}), **(entry.env or {})}
    params = StdioServerParameters(command=entry.command, args=entry.args, env=merged_env)
    with open(os.devnull, "w") as err_sink:
        async with asyncio.timeout(timeout):
            async with stdio_client(params, errlog=err_sink) as (r_stream, w_stream):
                return await _invoke(r_stream, w_stream)


def _run_coro_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def sync_mcp_tools(
    index: SkillIndex | None = None,
    *,
    settings: Settings | None = None,
    server_names: list[str] | None = None,
    timeout: float = 15.0,
) -> list[MCPSyncResult]:
    """Connect to enabled downstream MCP servers, list their tools, and index them in SQLite + ONNX."""
    settings = settings or (index.settings if index else get_settings())
    idx = index or SkillIndex(settings)
    registry = load_mcp_registry(settings)

    targets = [
        entry
        for name, entry in sorted(registry.items())
        if entry.enabled and (not server_names or name in server_names)
    ]

    from .env_store import EnvStore

    env_store = EnvStore(idx, settings=settings)
    results: list[MCPSyncResult] = []
    for entry in targets:
        t0 = time.perf_counter()
        try:
            injected_env = env_store.resolve_for_mcp_server(entry.name)
            tools = _run_coro_sync(_fetch_server_tools_async(entry, timeout=timeout, extra_env=injected_env))
            count = idx.upsert_mcp_tools(entry.name, tools)
            results.append(
                MCPSyncResult(
                    server_name=entry.name,
                    tools_count=count,
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )
            )
        except Exception as exc:
            results.append(
                MCPSyncResult(
                    server_name=entry.name,
                    tools_count=0,
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                    error=str(exc),
                )
            )
    return results


def search_mcp_tools(
    index: SkillIndex,
    query: str,
    limit: int = 5,
) -> str:
    """Search cached upstream MCP tool schemas using Hybrid Search (ONNX + BM25)."""
    hits = index.search_mcp_tools(query, limit=limit)
    if not hits:
        return (
            f"No matching downstream MCP tools found for '{query}'. "
            "(Run `sesh mcp-hub harvest && sesh mcp-hub sync` to index upstream MCP servers.)"
        )

    lines: list[str] = [f"## Agent-Seshat MCP Hub: {len(hits)} matching tool(s) for '{query}'", ""]
    for i, h in enumerate(hits, 1):
        schema_str = json.dumps(h["input_schema"], indent=2)
        lines.append(
            f"{i}. **`{h['server_name']}` -> `{h['tool_name']}`** (score: {h['score']:.4f})"
        )
        if h["description"]:
            lines.append(f"   - Description: {h['description']}")
        lines.append("   - Input Schema:")
        lines.append("```json")
        lines.append(schema_str)
        lines.append("```")
    lines.append("")
    lines.append(
        'Invoke any tool above with `call_mcp_tool(server_name="<server>", tool_name="<tool>", arguments={...})`.'
    )
    return "\n".join(lines)


def call_mcp_tool(
    index: SkillIndex,
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    settings: Settings | None = None,
    timeout: float = 30.0,
) -> str:
    """Invoke `tool_name` on downstream MCP server `server_name`."""
    settings = settings or index.settings
    registry = load_mcp_registry(settings)
    norm_name = _normalize_server_name(server_name)
    entry = registry.get(norm_name) or registry.get(server_name)
    if entry is None:
        known = ", ".join(sorted(registry)) or "none"
        raise LookupError(f"Unknown MCP server '{server_name}'. Registered servers: {known}")
    if not entry.enabled:
        raise RuntimeError(f"Downstream MCP server '{entry.name}' is currently disabled.")
    from .env_store import EnvStore

    injected_env = EnvStore(index, settings=settings).resolve_for_mcp_server(entry.name)
    return _run_coro_sync(
        _call_server_tool_async(entry, tool_name, arguments or {}, timeout=timeout, extra_env=injected_env)
    )


def consolidate_mcp_configs(
    *,
    workspace: Path | None = None,
    user_home: Path | None = None,
    settings: Settings | None = None,
    mode: str = "auto",
    command: str = "sesh",
) -> list[tuple[Path, Path]]:
    """Harvest scattered MCP servers into `mcp_registry.json`, back up each config to
    `*.pre-seshat.bak`, and replace the MCP server lists with ONLY `agent-seshat`."""
    settings = (settings or get_settings()).ensure_dirs()
    harvest_mcp_configs(workspace=workspace, user_home=user_home, settings=settings)

    consolidated: list[tuple[Path, Path]] = []
    for source_agent, cfg_path, fmt in _candidate_mcp_config_sources(workspace, user_home):
        if not cfg_path.is_file():
            continue
        if cfg_path.name == "settings.json":
            continue
        backup_path = cfg_path.with_name(f"{cfg_path.name}.pre-seshat.bak")
        shutil.copy2(cfg_path, backup_path)

        entry = build_server_entry(
            mode,
            fmt=fmt,
            host=settings.host,
            port=settings.port,
            command=command,
        )
        if fmt == "toml":
            raw = cfg_path.read_text(encoding="utf-8")
            doc = tomlkit.parse(raw) if raw.strip() else tomlkit.document()
            sup = tomlkit.table(is_super_table=True)
            tbl = tomlkit.table()
            for k, v in entry.items():
                tbl[k] = v
            sup[SERVER_NAME] = tbl
            doc["mcp_servers"] = sup
            cfg_path.write_text(tomlkit.dumps(doc), encoding="utf-8")
        else:
            raw = cfg_path.read_text(encoding="utf-8")
            try:
                data = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                data = {}
            if not isinstance(data, dict):
                data = {}
            root_key = "servers" if "servers" in data and "mcpServers" not in data else "mcpServers"
            data[root_key] = {SERVER_NAME: entry}
            cfg_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

        consolidated.append((cfg_path, backup_path))
    return consolidated
