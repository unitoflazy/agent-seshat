"""FastMCP server: 3 progressive-disclosure tools, a dynamic catalog, and seshat:// resources."""

from __future__ import annotations

import difflib
import logging
import os
import threading
from pathlib import Path

from typing import Any

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.resources import FunctionResource
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import __version__
from .config import SERVER_NAME, Settings, get_settings, user_home_dir
from .env_store import EnvStore
from .gatekeeper import activate_rules as gk_activate_rules
from .gatekeeper import propose_session_rules as gk_propose_session_rules
from .indexer import SkillIndex, SkillRecord, discover_skill_files
from .mcp_aggregator import call_mcp_tool as hub_call_mcp_tool
from .mcp_aggregator import search_mcp_tools as hub_search_mcp_tools
from .parser import find_git_root, sanitize_secrets
from .search import HybridSearcher, SearchResult

log = logging.getLogger(__name__)

CATALOG_PREFIX = (
    "MANDATORY FIRST STEP: You are connected to Agent-Seshat (the House of Books). Before planning "
    "architecture, writing code, or running commands, search this library for project conventions and "
    "specialized skills. Available domains/skills include: "
)
CATALOG_MAX_SLUGS = 40
MAX_RESOURCE_BYTES = 512 * 1024

SERVER_INSTRUCTIONS = (
    "Agent-Seshat v2.0 is the Unified MCP Gateway, Rule Gatekeeper, and House of Books. Mandatory workflow: "
    "0) Call propose_session_rules(user_prompt_summary, workspace_path, active_files) FIRST on every new task, "
    "present any unpinned candidate rules to the engineer for confirmation, then call activate_rules(...). "
    "1) Use search_skills / load_skill / load_skill_resource for on-demand skills and templates. "
    "2) Use search_mcp_tools and call_mcp_tool to discover and invoke downstream MCP servers."
)


class SkillSecurityError(ValueError):
    """Raised when a resource request escapes its skill directory."""


class SkillNotFoundError(LookupError):
    pass


def detect_project(cwd: Path | None = None) -> str:
    """Project name for scope-aware ranking: ``$AGENT_SESHAT_PROJECT`` or the git repo name of ``cwd``."""
    env = os.environ.get("AGENT_SESHAT_PROJECT", "").strip()
    if env:
        return env
    try:
        root = find_git_root((cwd or Path.cwd()).resolve())
    except OSError:
        return ""
    if root is None or root == user_home_dir():
        return ""
    return root.name


# ---------------------------------------------------------------------------
# Service layer (transport-agnostic, unit-testable)
# ---------------------------------------------------------------------------


class SeshatService:
    def __init__(self, index: SkillIndex | None = None, default_project: str = ""):
        self.index = index or SkillIndex()
        self.searcher = HybridSearcher(self.index)
        self.env_store = EnvStore(self.index)
        self.default_project = default_project
        self._catalog_cache: tuple[tuple, str] | None = None

    # -- catalog -----------------------------------------------------------------

    def catalog(self) -> str:
        """Compact skill catalog built from disk (no embedding required)."""
        files = discover_skill_files(self.index.settings.skills_dir)
        slugs = sorted(files)
        key = (tuple(slugs), self.index.generation)
        if self._catalog_cache and self._catalog_cache[0] == key:
            return self._catalog_cache[1]
        if not slugs:
            text = "(library is empty - run `sesh harvest` or add skills to ~/.agent-seshat/skills)"
        elif len(slugs) <= CATALOG_MAX_SLUGS:
            text = ", ".join(slugs)
        else:
            counts: dict[str, int] = {}
            for rec in self.index.list_skills():
                for tag in rec.tags[:6]:
                    counts[tag] = counts.get(tag, 0) + 1
            top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:CATALOG_MAX_SLUGS]
            text = (
                f"{len(slugs)} skills across domains: " + ", ".join(f"{t}({n})" for t, n in top)
                if top
                else ", ".join(slugs[:CATALOG_MAX_SLUGS]) + f", ... (+{len(slugs) - CATALOG_MAX_SLUGS} more)"
            )
        self._catalog_cache = (key, text)
        return text

    def search_description(self) -> str:
        return (
            f"{CATALOG_PREFIX}[{self.catalog()}]. Returns ranked skill summaries (slug, scope, best-matching "
            "section, tags, companion files) - never full bodies. Then call load_skill(slug)."
        )

    # -- tools ---------------------------------------------------------------------

    def search(self, query: str, project: str = "", limit: int = 5) -> list[SearchResult]:
        limit = max(1, min(int(limit or 5), 20))
        return self.searcher.search(query, project=(project or self.default_project).strip(), top_k=limit)

    def search_markdown(self, query: str, project: str = "", limit: int = 5) -> str:
        if not (query or "").strip():
            return "Provide a non-empty query, e.g. search_skills('postgres index migration')."
        project = (project or self.default_project).strip()
        results = self.search(query, project, limit)
        if not results:
            return f"No matching skills found for '{query}'."
        header = f"## Seshat: {len(results)} skill(s) for '{query}'" + (f" (project: {project})" if project else "")
        lines = [header, ""]
        for i, r in enumerate(results, 1):
            scope = r.scope_label if r.scope_multiplier < 1.0 else (
                f"project:{project} (current repo)" if r.scope_multiplier > 1.0 else r.scope
            )
            lines.append(f"{i}. **{r.slug}** - {r.description}")
            lines.append(f"   - scope: {scope} | best section: \"{r.matched_heading}\" | score: {r.score:.4f}")
            if r.tags:
                lines.append(f"   - tags: {', '.join(r.tags[:10])}")
            if r.resources:
                lines.append(f"   - resources: {', '.join(r.resources[:10])}")
        lines += ["", "Next: call load_skill(\"<slug>\") for each relevant skill before acting."]
        return "\n".join(lines)

    def _require(self, slug: str) -> SkillRecord:
        slug = (slug or "").strip().strip("/")
        rec = self.index.get_skill(slug)
        if rec is None:
            self.index.ensure_fresh()
            rec = self.index.get_skill(slug)
        if rec is None:
            known = [s.slug for s in self.index.list_skills()]
            close = difflib.get_close_matches(slug, known, n=3, cutoff=0.4)
            hint = f" Did you mean: {', '.join(close)}?" if close else " Use search_skills to find valid slugs."
            raise SkillNotFoundError(f"Unknown skill '{slug}'.{hint}")
        return rec

    def load_skill(self, slug: str) -> str:
        rec = self._require(slug)
        parts = [
            f"# {rec.name}",
            f"<!-- seshat skill: {rec.slug} | scope: {rec.scope} | tags: {', '.join(rec.tags)} -->",
            f"> {rec.description}",
            "",
            rec.body.strip(),
            "",
            "---",
        ]
        if rec.resources:
            parts.append(
                f"Companion files for '{rec.slug}' (fetch with load_skill_resource(\"{rec.slug}\", \"<filename>\")):"
            )
            parts += [f"- {name}" for name in rec.resources]
        else:
            parts.append(f"No companion files for '{rec.slug}'.")
        return "\n".join(parts)

    def resolve_resource_path(self, slug: str, filename: str) -> Path:
        if not filename or "\x00" in filename:
            raise SkillSecurityError("Invalid filename.")
        rel = Path(filename)
        if rel.is_absolute() or rel.drive:
            raise SkillSecurityError(f"Absolute paths are not allowed: {filename!r}")
        rec = self._require(slug)
        skill_dir = rec.skill_dir
        if skill_dir is None:
            raise SkillSecurityError(f"Skill '{rec.slug}' is a flat file and has no companion resources.")
        base = skill_dir.resolve()
        target = (base / rel).resolve()
        if not target.is_relative_to(base) or target == base:
            raise SkillSecurityError(f"Path traversal blocked: {filename!r} escapes skill '{rec.slug}'.")
        if not target.is_file():
            available = ", ".join(rec.resources) or "none"
            raise FileNotFoundError(f"'{filename}' not found in skill '{rec.slug}'. Available: {available}")
        return target

    def load_skill_resource(self, slug: str, filename: str) -> str:
        target = self.resolve_resource_path(slug, filename)
        size = target.stat().st_size
        if size > MAX_RESOURCE_BYTES:
            raise ValueError(f"'{filename}' is {size} bytes; limit is {MAX_RESOURCE_BYTES}.")
        data = target.read_bytes()
        if b"\x00" in data[:4096]:
            return f"'{filename}' is a binary file ({size} bytes) and cannot be returned as text."
        text, _ = sanitize_secrets(data.decode("utf-8", errors="replace"))
        return text

    def propose_session_rules(
        self,
        user_prompt_summary: str,
        workspace_path: str = "",
        active_files: list[str] | None = None,
    ) -> str:
        eff_workspace = workspace_path or self.default_project
        return gk_propose_session_rules(
            self.index,
            user_prompt_summary=user_prompt_summary,
            workspace_path=eff_workspace,
            active_files=active_files or [],
        )

    def activate_rules(
        self,
        load_slugs: list[str],
        pin_always_slugs: list[str] | None = None,
        pin_scope: str = "global",
    ) -> str:
        return gk_activate_rules(
            self.index,
            load_slugs=load_slugs,
            pin_always_slugs=pin_always_slugs or [],
            pin_scope=pin_scope,
        )

    def search_mcp_tools(self, query: str, limit: int = 5) -> str:
        return hub_search_mcp_tools(self.index, query=query, limit=limit)

    def call_mcp_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        return hub_call_mcp_tool(
            self.index,
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments or {},
        )

    # -- environment variables --------------------------------------------------

    def list_env_vars(
        self,
        project: str = "",
        scope: str = "",
        include_global: bool = True,
    ) -> str:
        eff_project = (project or self.default_project).strip() if not scope else ""
        records = self.env_store.list(
            scope=scope or None,
            project=eff_project,
            include_global=include_global,
        )
        if not records:
            target = scope or (f"project:{eff_project}" if eff_project else "all scopes")
            return f"No environment variables stored for {target}."

        lines = [
            f"## Agent-Seshat Environment Variables ({len(records)} entry/entries)",
            "",
        ]
        for r in records:
            badge = " [SECRET - masked]" if r.is_secret else ""
            desc = f" — {r.description}" if r.description else ""
            lines.append(
                f"- **`{r.key}`** = `{r.display_value(reveal_secret=False)}` "
                f"(scope: `{r.scope}`){badge}{desc}"
            )
        lines.append("")
        lines.append("Use `get_env_var(key, project=..., reveal_secret=True)` to fetch a specific value.")
        return "\n".join(lines)

    def get_env_var(
        self,
        key: str,
        project: str = "",
        scope: str = "",
        reveal_secret: bool = True,
    ) -> str:
        eff_project = (project or self.default_project).strip()
        rec = self.env_store.get(
            key,
            scope=scope or None,
            project=eff_project,
            fallback_global=True,
        )
        if rec is None:
            return f"Environment variable '{key}' not found in Agent-Seshat."
        val = rec.display_value(reveal_secret=reveal_secret)
        secret_note = "yes" if rec.is_secret else "no"
        desc_line = f"\n- **Description:** {rec.description}" if rec.description else ""
        return (
            f"### `{rec.key}`\n"
            f"- **Value:** `{val}`\n"
            f"- **Scope:** `{rec.scope}`\n"
            f"- **Secret:** `{secret_note}`{desc_line}"
        )

    def set_env_var(
        self,
        key: str,
        value: str,
        scope: str = "global",
        description: str = "",
        is_secret: bool | None = None,
    ) -> str:
        rec = self.env_store.set(
            key,
            value,
            scope=scope,
            description=description,
            is_secret=is_secret,
        )
        shown = rec.display_value(reveal_secret=False)
        return (
            f"Stored environment variable `{rec.key}` = `{shown}` "
            f"(scope: `{rec.scope}`, secret: `{rec.is_secret}`)."
        )

    def unset_env_var(self, key: str, scope: str = "global") -> str:
        deleted = self.env_store.unset(key, scope=scope)
        if not deleted:
            return f"Environment variable `{key}` not found in scope `{scope}`."
        return f"Removed environment variable `{key}` from scope `{scope}`."

    def health(self) -> dict:
        skills, chunks = self.index.counts()
        embedder = self.index._embedder
        return {
            "status": "ok",
            "server": SERVER_NAME,
            "version": __version__,
            "skills": skills,
            "chunks": chunks,
            "model_loaded": bool(getattr(embedder, "loaded", False)),
            "home": str(self.index.settings.home),
        }

    def warm_up(self) -> None:
        """Index + load the ONNX model so the first real query is fast."""
        try:
            self.index.ensure_fresh()
            self.index.embedder.embed_query("warm up")
            self.index.vectors()
        except Exception:  # never crash the server because of warm-up
            log.exception("Warm-up failed")


# ---------------------------------------------------------------------------
# FastMCP wiring
# ---------------------------------------------------------------------------

_DEFAULT_INDEX: dict[Path, SkillIndex] = {}
_DEFAULT_LOCK = threading.Lock()


def get_default_index(settings: Settings | None = None) -> SkillIndex:
    settings = settings or get_settings()
    with _DEFAULT_LOCK:
        idx = _DEFAULT_INDEX.get(settings.home)
        if idx is None:
            idx = _DEFAULT_INDEX[settings.home] = SkillIndex(settings)
        return idx


def build_server(
    service: SeshatService | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    http_mode: bool = False,
):
    service = service or SeshatService(get_default_index())
    mcp = FastMCP(
        SERVER_NAME,
        instructions=SERVER_INSTRUCTIONS,
        host=host,
        port=port,
        # Stateless JSON responses make the daemon trivially proxyable by `sesh mcp`.
        stateless_http=http_mode,
        json_response=http_mode,
        log_level="WARNING",
    )
    read_only = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    state = {"description": service.search_description()}

    async def search_skills(query: str, project: str = "", limit: int = 5, ctx: Context = None) -> str:  # type: ignore[assignment]
        """(description replaced dynamically with the live catalog)"""
        result = await anyio.to_thread.run_sync(service.search_markdown, query, project, limit)
        new_desc = service.search_description()
        if new_desc != state["description"]:
            state["description"] = new_desc
            tool = mcp._tool_manager.get_tool("search_skills")
            if tool is not None:
                tool.description = new_desc
            if ctx is not None:
                try:
                    await ctx.session.send_tool_list_changed()
                except Exception:
                    pass
        return result

    async def load_skill(slug: str) -> str:
        """Load the full Markdown body of a skill by slug (from search_skills), plus its companion file list."""
        return await anyio.to_thread.run_sync(service.load_skill, slug)

    async def load_skill_resource(slug: str, filename: str) -> str:
        """Read a companion file (template, schema, example) that lives inside a skill's directory."""
        return await anyio.to_thread.run_sync(service.load_skill_resource, slug, filename)

    async def propose_session_rules(
        user_prompt_summary: str,
        workspace_path: str = "",
        active_files: list[str] = [],
    ) -> str:
        """
        MANDATORY STEP 0 BEFORE ANY PLANNING OR CODE EXECUTION:
        Call this tool immediately upon receiving a new user request.
        It returns:
        1. Pinned ('Always-Use') rules that MUST be followed automatically.
        2. Candidate rules/skills matched via Hybrid Search (ONNX + BM25) for the task.
        You MUST present the candidate rules to the engineer and ask which ones to activate
        before writing code (unless all relevant rules are already Pinned).
        """
        return await anyio.to_thread.run_sync(
            service.propose_session_rules,
            user_prompt_summary,
            workspace_path,
            active_files,
        )

    async def activate_rules(
        load_slugs: list[str],
        pin_always_slugs: list[str] = [],
        pin_scope: str = "global",
    ) -> str:
        """
        Load the full Markdown instructions for the rules selected by the engineer.
        If the engineer asked to 'always choose' specific rules, pass them in `pin_always_slugs`
        to persist that preference in the Agent-Seshat database.
        """
        return await anyio.to_thread.run_sync(
            service.activate_rules,
            load_slugs,
            pin_always_slugs,
            pin_scope,
        )

    async def search_mcp_tools(query: str, limit: int = 5) -> str:
        """Search centralized downstream MCP servers via Hybrid Search (ONNX + BM25) and return matching tool schemas."""
        return await anyio.to_thread.run_sync(service.search_mcp_tools, query, limit)

    async def call_mcp_tool(server_name: str, tool_name: str, arguments: dict[str, Any] = {}) -> str:
        """Lazily spawn/connect to a centralized downstream MCP server (`server_name`) and invoke `tool_name` with `arguments`."""
        return await anyio.to_thread.run_sync(service.call_mcp_tool, server_name, tool_name, arguments)

    async def list_env_vars(project: str = "", scope: str = "", include_global: bool = True) -> str:
        """List stored environment variables in Agent-Seshat (global and/or project scope). Secret values are masked in listings."""
        return await anyio.to_thread.run_sync(service.list_env_vars, project, scope, include_global)

    async def get_env_var(
        key: str,
        project: str = "",
        scope: str = "",
        reveal_secret: bool = True,
    ) -> str:
        """Retrieve a specific environment variable by `key` (checking `project:<project>` first, then `global`)."""
        return await anyio.to_thread.run_sync(service.get_env_var, key, project, scope, reveal_secret)

    async def set_env_var(
        key: str,
        value: str,
        scope: str = "global",
        description: str = "",
        is_secret: bool | None = None,
    ) -> str:
        """Store or update an environment variable in Agent-Seshat (`scope`: `global`, `project:<repo>`, or `mcp:<server>`)."""
        return await anyio.to_thread.run_sync(service.set_env_var, key, value, scope, description, is_secret)

    async def unset_env_var(key: str, scope: str = "global") -> str:
        """Delete an environment variable from Agent-Seshat for the given `scope`."""
        return await anyio.to_thread.run_sync(service.unset_env_var, key, scope)

    mcp.add_tool(
        propose_session_rules,
        name="propose_session_rules",
        description=propose_session_rules.__doc__.strip() if propose_session_rules.__doc__ else None,
        annotations=read_only,
    )
    mcp.add_tool(
        activate_rules,
        name="activate_rules",
        description=activate_rules.__doc__.strip() if activate_rules.__doc__ else None,
    )
    mcp.add_tool(search_skills, name="search_skills", description=state["description"], annotations=read_only)
    mcp.add_tool(load_skill, name="load_skill", description=load_skill.__doc__, annotations=read_only)
    mcp.add_tool(
        load_skill_resource, name="load_skill_resource", description=load_skill_resource.__doc__, annotations=read_only
    )
    mcp.add_tool(
        search_mcp_tools,
        name="search_mcp_tools",
        description=search_mcp_tools.__doc__,
        annotations=read_only,
    )
    mcp.add_tool(
        call_mcp_tool,
        name="call_mcp_tool",
        description=call_mcp_tool.__doc__,
    )
    mcp.add_tool(
        list_env_vars,
        name="list_env_vars",
        description=list_env_vars.__doc__,
        annotations=read_only,
    )
    mcp.add_tool(
        get_env_var,
        name="get_env_var",
        description=get_env_var.__doc__,
        annotations=read_only,
    )
    mcp.add_tool(
        set_env_var,
        name="set_env_var",
        description=set_env_var.__doc__,
    )
    mcp.add_tool(
        unset_env_var,
        name="unset_env_var",
        description=unset_env_var.__doc__,
    )

    # Resources: one concrete resource per skill (for @-mention pickers) + a template for new ones.
    def _skill_resource(slug: str, description: str) -> FunctionResource:
        return FunctionResource(
            uri=f"seshat://skill/{slug}",
            name=slug,
            description=description,
            mime_type="text/markdown",
            fn=lambda s=slug: service.load_skill(s),
        )

    try:
        for rec in service.index.list_skills():
            mcp.add_resource(_skill_resource(rec.slug, rec.description))
    except Exception:
        log.exception("Could not register skill resources")

    @mcp.resource("seshat://skill/{slug}", name="seshat-skill", mime_type="text/markdown")
    def skill_by_slug(slug: str) -> str:
        """Any Agent-Seshat skill by slug."""
        return service.load_skill(slug)

    @mcp.resource("seshat://catalog", name="seshat-catalog", mime_type="text/markdown")
    def catalog() -> str:
        """Catalog of every skill in the library."""
        service.index.ensure_fresh()
        rows = [f"- `{r.slug}` ({r.scope}) - {r.description}" for r in service.index.list_skills()]
        return "# Agent-Seshat catalog\n\n" + ("\n".join(rows) or "(empty)")

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> JSONResponse:
        return JSONResponse(await anyio.to_thread.run_sync(service.health))

    @mcp.custom_route("/search", methods=["GET"])
    async def http_search(request: Request) -> JSONResponse:
        q = request.query_params.get("q", "")
        proj = request.query_params.get("project", "")
        top_k = int(request.query_params.get("top_k", "5") or 5)

        def _do_search() -> dict[str, Any]:
            hits = service.searcher.search(q, project=proj, top_k=top_k)
            diag = service.searcher.last_diagnostics
            return {
                "results": [h.to_dict() for h in hits],
                "elapsed_ms": diag.elapsed_ms if diag else 0.0,
            }

        return JSONResponse(await anyio.to_thread.run_sync(_do_search))

    mcp._seshat_service = service  # type: ignore[attr-defined]
    return mcp


def _start_watcher(service: SeshatService) -> None:
    def _run() -> None:
        try:
            from watchfiles import watch
        except ImportError:
            return
        skills_dir = service.index.settings.skills_dir
        for _changes in watch(skills_dir, debounce=800, recursive=True):
            try:
                service.index.ensure_fresh()
            except Exception:
                log.exception("Background reindex failed")

    threading.Thread(target=_run, name="seshat-watch", daemon=True).start()


def run_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765, watch: bool = False) -> None:
    """Run the MCP server. ``transport``: stdio | http (streamable-http) | sse."""
    transport = {"http": "streamable-http", "streamable-http": "streamable-http", "sse": "sse"}.get(transport, "stdio")
    settings = get_settings()
    default_project = detect_project() if transport == "stdio" else ""
    service = SeshatService(get_default_index(settings), default_project=default_project)
    mcp = build_server(service, host=host, port=port, http_mode=transport != "stdio")
    threading.Thread(target=service.warm_up, name="seshat-warmup", daemon=True).start()
    if watch:
        _start_watcher(service)
    mcp.run(transport=transport)  # type: ignore[arg-type]
