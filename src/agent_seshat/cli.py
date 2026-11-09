"""Typer CLI (`agent-seshat` / `sesh`): serve, mcp, index, search, harvest, connect, status."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import DEFAULT_HOST, DEFAULT_PORT, get_settings
from .connector import VALID_MODES, VALID_TARGETS, connect_agents, inspect_connections
from .deep_scanner import deep_sync as run_deep_sync
from .env_store import EnvStore
from .harvester import harvest as run_harvest
from .indexer import SkillIndex
from .mcp_aggregator import (
    consolidate_mcp_configs,
    harvest_mcp_configs,
    load_mcp_registry,
    sync_mcp_tools,
)
from .proxy import daemon_health, run_mcp_entrypoint
from .search import HybridSearcher
from .server import detect_project, run_server

app = typer.Typer(
    name="agent-seshat",
    help=(
        "Agent-Seshat (sesh) v2.0 - Mistress of the House of Books.\n\n"
        "Unified MCP Gateway, Deep Rule Sync, Interactive Rule Gatekeeper & Hybrid ONNX + FTS5 "
        "Search Engine for Claude Code, Codex, Cursor, and GitHub Copilot."
    ),
    no_args_is_help=True,
    add_completion=False,
)

rules_app = typer.Typer(
    name="rules",
    help="Inspect, filter, and pin 'Always-Choose' rules across Claude, Cursor, Copilot, and Codex.",
    no_args_is_help=True,
)
mcp_hub_app = typer.Typer(
    name="mcp-hub",
    help="Centralized MCP Server Aggregator & Lazy Tool Broker (`mcp_registry.json`).",
    no_args_is_help=True,
)
env_app = typer.Typer(
    name="env",
    help="Store, retrieve, import, and export scoped environment variables (`global`, `project:<repo>`, `mcp:<server>`).",
    no_args_is_help=True,
)
app.add_typer(rules_app, name="rules")
app.add_typer(mcp_hub_app, name="mcp-hub")
app.add_typer(env_app, name="env")

console = Console()


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


@app.callback()
def _main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-v", help="Show version and exit.", is_eager=True),
    ] = None,
) -> None:
    if version:
        console.print(f"agent-seshat {__version__}")
        raise typer.Exit()


@app.command("serve")
def serve_cmd(
    transport: Annotated[
        str,
        typer.Option("--transport", "-t", help="MCP transport: stdio | http | sse"),
    ] = "stdio",
    host: Annotated[
        str,
        typer.Option("--host", help="Bind host for HTTP/SSE transport."),
    ] = DEFAULT_HOST,
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Bind port for HTTP/SSE transport."),
    ] = DEFAULT_PORT,
    watch: Annotated[
        bool,
        typer.Option("--watch/--no-watch", help="Watch ~/.agent-seshat/skills for live changes."),
    ] = True,
) -> None:
    """Start the Agent-Seshat MCP server daemon."""
    norm_transport = transport.strip().lower()
    if norm_transport not in ("stdio", "http", "streamable-http", "sse"):
        console.print(f"[red]Invalid transport '{transport}'. Expected stdio, http, or sse.[/red]")
        raise typer.Exit(code=2)

    if norm_transport != "stdio":
        settings = get_settings()
        console.print(
            Panel.fit(
                f"[bold cyan]Agent-Seshat MCP Daemon[/bold cyan] v{__version__}\n"
                f"Transport : [green]{norm_transport}[/green] (http://{host}:{port}/mcp)\n"
                f"Health    : http://{host}:{port}/health\n"
                f"Library   : {settings.skills_dir}",
                title="House of Books",
                border_style="cyan",
            )
        )
    run_server(transport=norm_transport, host=host, port=port, watch=watch)


@app.command("mcp")
def mcp_cmd(
    local: Annotated[
        bool,
        typer.Option("--local", help="Force in-process stdio server without probing HTTP daemon."),
    ] = False,
) -> None:
    """Zero-downtime stdio MCP entrypoint (proxies to warm HTTP daemon or falls back in-process)."""
    run_mcp_entrypoint(force_local=local)


@app.command("index")
def index_cmd(
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Rebuild all chunk embeddings from scratch."),
    ] = False,
) -> None:
    """Build or incrementally update the SQLite FTS5 + ONNX vector index."""
    settings = get_settings()
    idx = SkillIndex(settings)
    with console.status("[bold cyan]Indexing skills with FastEmbed ONNX (BAAI/bge-small-en-v1.5)..."):
        stats = idx.index(force=force)

    table = Table(title="Agent-Seshat Index Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Skills Directory", str(settings.skills_dir))
    table.add_row("Database Path", str(settings.db_path))
    table.add_row("Total Skills", str(stats.total_skills))
    table.add_row("Total Chunks", str(stats.total_chunks))
    table.add_row("Added / Updated / Removed", f"+{len(stats.added)} / ~{len(stats.updated)} / -{len(stats.removed)}")
    table.add_row("Unchanged (Hash Hit)", str(stats.unchanged))
    table.add_row("Chunks Embedded (ONNX)", str(stats.chunks_embedded))
    table.add_row("Indexing Latency", f"{stats.elapsed_ms:.1f} ms")
    table.add_row("SQLite DB Size", _format_bytes(idx.db_size_bytes()))
    console.print(table)

    if stats.errors:
        for err in stats.errors:
            console.print(f"[yellow]Warning:[/yellow] {err}")


@app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument(help="Natural-language or technical search query.")],
    project: Annotated[
        str,
        typer.Option("--project", "-p", help="Current project/repo name for scope boosting."),
    ] = "",
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Maximum number of skills to return."),
    ] = 5,
) -> None:
    """Run a hybrid search (ONNX Cosine + FTS5 BM25 + Exact Boost + RRF) and display diagnostics."""
    settings = get_settings()
    idx = SkillIndex(settings)
    searcher = HybridSearcher(idx)
    effective_project = project or detect_project()
    results = searcher.search(query, project=effective_project, top_k=top_k)
    diag = searcher.last_diagnostics

    if not results:
        console.print(
            Panel(
                f"No matching skills passed the confidence threshold for [bold]{query!r}[/bold].",
                title="Zero-Miss Hybrid Search",
                border_style="yellow",
            )
        )
        return

    subtitle = f"project={effective_project or 'global'}"
    if diag:
        subtitle += f" | latency={diag.elapsed_ms:.1f}ms"
    table = Table(
        title=f"Hybrid Search Results for {query!r} ({subtitle})",
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Slug", style="bold cyan")
    table.add_column("Scope", style="magenta")
    table.add_column("Matched Heading", style="green")
    table.add_column("RRF Score", justify="right", style="bold yellow")
    table.add_column("Cosine Sim", justify="right")
    table.add_column("BM25", justify="right")
    table.add_column("Tags", style="dim")

    for rank, r in enumerate(results, 1):
        cos_str = f"{r.cosine:.3f}" if r.cosine is not None else "-"
        bm25_str = f"{r.bm25:.2f}" if r.bm25 is not None else "-"
        scope_display = r.scope_label if r.scope_multiplier < 1.0 else r.scope
        table.add_row(
            str(rank),
            r.slug,
            scope_display,
            r.matched_heading,
            f"{r.score:.4f}",
            cos_str,
            bm25_str,
            ", ".join(r.tags[:6]),
        )
    console.print(table)


@app.command("harvest")
def harvest_cmd(
    paths: Annotated[
        Optional[list[Path]],
        typer.Argument(help="Directories or rule files to scan (defaults to cwd + home agent dirs)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview discovered skills and redactions without writing files."),
    ] = False,
    write_stubs: Annotated[
        bool,
        typer.Option(
            "--write-stubs",
            help="Back up original CLAUDE.md/.cursorrules/AGENTS.md to .bak and replace with an 8-line Seshat stub.",
        ),
    ] = False,
) -> None:
    """Harvest scattered Claude/Codex/Cursor/Copilot/Windsurf/Cline rules into ~/.agent-seshat/skills."""
    settings = get_settings()
    report = run_harvest(paths=paths, dry_run=dry_run, write_stubs=write_stubs, settings=settings)

    if not report.items:
        console.print("[yellow]No agent rule or skill files discovered in the target paths.[/yellow]")
        return

    table = Table(
        title="Agent-Seshat Harvester Report" + (" [DRY RUN]" if dry_run else ""),
        border_style="cyan",
    )
    table.add_column("Slug", style="bold cyan")
    table.add_column("Kind", style="dim")
    table.add_column("Scope", style="magenta")
    table.add_column("Status")
    table.add_column("Secrets Redacted", style="bold red")
    table.add_column("Source Path", overflow="fold")

    for item in report.items:
        secret_text = ", ".join(item.secrets) if item.secrets else "-"
        status_style = {
            "new": "[green]new[/green]",
            "updated": "[cyan]updated[/cyan]",
            "unchanged": "[dim]unchanged[/dim]",
            "duplicate": f"[yellow]duplicate ({item.note})[/yellow]",
            "would-create": "[green]would-create[/green]",
            "would-update": "[cyan]would-update[/cyan]",
        }.get(item.status, item.status)
        table.add_row(item.slug, item.kind, item.scope, status_style, secret_text, str(item.source))
    console.print(table)

    if report.secrets_total > 0:
        console.print(
            Panel(
                f"[bold red]Security Warning:[/bold red] Scrubbed {report.secrets_total} hardcoded credential(s) "
                "with [bold][REDACTED_SECRET][/bold] before importing into the skill library.",
                border_style="red",
            )
        )

    if report.stubs:
        stub_table = Table(title="Seshat Index Stubs Written", border_style="green")
        stub_table.add_column("Original File")
        stub_table.add_column("Backup File (.bak)")
        for src, bak in report.stubs:
            stub_table.add_row(str(src), str(bak))
        console.print(stub_table)

    if not dry_run:
        idx = SkillIndex(settings)
        with console.status("[bold cyan]Updating hybrid search index..."):
            stats = idx.index()
        console.print(
            f"[green]Indexed {stats.total_skills} skill(s) ({stats.total_chunks} chunks) in {stats.elapsed_ms:.1f} ms.[/green]"
        )


@app.command("connect")
def connect_cmd(
    target: Annotated[
        str,
        typer.Option(
            "--target",
            "-t",
            help=f"Target client ({'|'.join(VALID_TARGETS)}).",
        ),
    ] = "all",
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            "-m",
            help="Connection mode: auto (`sesh mcp` proxy+fallback) | stdio | http.",
        ),
    ] = "auto",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview config changes without writing."),
    ] = False,
    enforce_gatekeeper: Annotated[
        bool,
        typer.Option(
            "--enforce-gatekeeper",
            help="Inject the 6-line Mandatory Pre-Flight Gatekeeper stub into agent instruction files.",
        ),
    ] = False,
) -> None:
    """Safely back up (*.seshat.bak) and wire coding agents to Agent-Seshat."""
    try:
        results = connect_agents(
            target=target,
            mode=mode,
            dry_run=dry_run,
            workspace=Path.cwd(),
            enforce_gatekeeper=enforce_gatekeeper,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2)

    table = Table(title="Agent-Seshat MCP Connector", border_style="cyan")
    table.add_column("Agent", style="bold cyan")
    table.add_column("Mode", style="magenta")
    table.add_column("Status")
    table.add_column("Config Path", overflow="fold")
    table.add_column("Backup", overflow="fold", style="dim")

    for r in results:
        table.add_row(
            r.label,
            r.mode,
            r.status,
            str(r.path),
            str(r.backup_path) if r.backup_path else "-",
        )
    console.print(table)


@app.command("status")
def status_cmd() -> None:
    """Show skill counts, SQLite DB size, warm daemon status, and connected coding agents."""
    settings = get_settings()
    idx = SkillIndex(settings)
    idx.ensure_fresh()
    skills_count, chunks_count = idx.counts()
    db_bytes = idx.db_size_bytes()
    health = daemon_health(settings.daemon_url)

    daemon_text = (
        Text(f"ONLINE at {settings.daemon_url} (warm ONNX ready)", style="bold green")
        if health
        else Text(f"OFFLINE at {settings.daemon_url} (stdio fallback active)", style="yellow")
    )

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Seshat Home", str(settings.home))
    summary.add_row("Skills Directory", str(settings.skills_dir))
    summary.add_row("Total Skills", str(skills_count))
    summary.add_row("Total Chunks", str(chunks_count))
    summary.add_row("SQLite Index DB", f"{settings.db_path} ({_format_bytes(db_bytes)})")
    summary.add_row("HTTP Daemon", daemon_text)
    console.print(Panel(summary, title=f"Agent-Seshat v{__version__} Status", border_style="cyan"))

    conn_table = Table(title="Connected Coding Agents", border_style="cyan")
    conn_table.add_column("Agent", style="bold")
    conn_table.add_column("Target")
    conn_table.add_column("Connected")
    conn_table.add_column("Config File", overflow="fold")
    for info in inspect_connections():
        connected = "[green]yes[/green]" if info["connected"] else ("[dim]no[/dim]" if info["exists"] else "[dim]not installed[/dim]")
        conn_table.add_row(info["label"], info["target"], connected, str(info["path"]))
    console.print(conn_table)


# ---------------------------------------------------------------------------
# v2.0: Deep Sync (`sesh deep-sync`)
# ---------------------------------------------------------------------------


@app.command("deep-sync")
def deep_sync_cmd(
    paths: Annotated[
        Optional[list[Path]],
        typer.Argument(help="Directories or files to deep-scan (defaults to cwd + home agent configs)."),
    ] = None,
    enforce_gatekeeper: Annotated[
        bool,
        typer.Option(
            "--enforce-gatekeeper",
            help="Inject the Mandatory Pre-Flight Gatekeeper protocol block into agent instruction files.",
        ),
    ] = False,
    always_pin: Annotated[
        bool,
        typer.Option("--always-pin", help="Pin all discovered rules as Always-Choose on import."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview deep-sync results without writing files."),
    ] = False,
) -> None:
    """Deep-scan and sync rules & skills from Claude, Cursor, Copilot, Codex, and Windsurf into SQLite + ONNX."""
    settings = get_settings()
    with console.status("[bold cyan]Running multi-agent deep scan & ONNX/FTS5 sync..."):
        report = run_deep_sync(
            paths=paths,
            settings=settings,
            enforce_gatekeeper=enforce_gatekeeper,
            always_pin=always_pin,
            dry_run=dry_run,
        )

    if not report.items:
        console.print("[yellow]No agent rule or skill files discovered in the target paths.[/yellow]")
        return

    table = Table(
        title=f"Agent-Seshat Deep Sync Summary ({report.elapsed_ms:.0f} ms)" + (" [DRY RUN]" if dry_run else ""),
        border_style="cyan",
    )
    table.add_column("Slug", style="bold cyan")
    table.add_column("Agent", style="bold")
    table.add_column("Kind", style="dim")
    table.add_column("Scope", style="magenta")
    table.add_column("Pinned (Always)", justify="center")
    table.add_column("Globs", style="dim")
    table.add_column("Status")
    table.add_column("Origin Path", overflow="fold")

    for item in report.items:
        pin_str = f"[bold green]YES ({item.pinned_scope})[/bold green]" if item.pinned else "[dim]no[/dim]"
        table.add_row(
            item.slug,
            item.source_agent,
            item.rule_kind,
            item.scope,
            pin_str,
            item.file_globs or "-",
            item.status,
            str(item.origin_path),
        )
    console.print(table)

    if report.gatekeeper_files:
        gk_table = Table(title="Gatekeeper Bootstrap Protocol Injected", border_style="green")
        gk_table.add_column("Instruction File")
        for gf in report.gatekeeper_files:
            gk_table.add_row(str(gf))
        console.print(gk_table)


# ---------------------------------------------------------------------------
# v2.0: Rules Management (`sesh rules ...`)
# ---------------------------------------------------------------------------


@rules_app.command("list")
def rules_list_cmd(
    agent: Annotated[
        Optional[str],
        typer.Option("--agent", "-a", help="Filter by source agent (claude|cursor|copilot|codex|windsurf|custom)."),
    ] = None,
    scope: Annotated[
        Optional[str],
        typer.Option("--scope", "-s", help="Filter by scope (global or project:<name>)."),
    ] = None,
    pinned_only: Annotated[
        bool,
        typer.Option("--pinned-only", "-p", help="Show only rules pinned as Always-Choose."),
    ] = False,
) -> None:
    """List discovered rules & skills with Pinned status, Scope, Source Agent, Globs, and Origin Path."""
    idx = SkillIndex(get_settings())
    idx.ensure_fresh()
    records = idx.list_skills(agent=agent, scope=scope, pinned_only=pinned_only)

    if not records:
        console.print("[yellow]No matching rules or skills found.[/yellow]")
        return

    table = Table(title=f"Agent-Seshat Rules & Skills ({len(records)})", border_style="cyan", show_lines=True)
    table.add_column("Slug", style="bold cyan")
    table.add_column("Pinned (Always)", justify="center")
    table.add_column("Scope", style="magenta")
    table.add_column("Source Agent", style="bold")
    table.add_column("Globs", style="dim")
    table.add_column("Description")
    table.add_column("Origin Path", overflow="fold", style="dim")

    for r in records:
        pin_badge = f"[bold green]ALWAYS ({r.pinned_scope})[/bold green]" if r.pinned else "[dim]on-demand[/dim]"
        table.add_row(
            r.slug,
            pin_badge,
            r.scope,
            r.source_agent,
            r.file_globs or "-",
            r.description,
            r.origin_path or str(r.file_path),
        )
    console.print(table)


@rules_app.command("pin")
def rules_pin_cmd(
    slug: Annotated[str, typer.Argument(help="Skill/rule slug to mark as Always-Choose.")],
    scope: Annotated[
        str,
        typer.Option("--scope", "-s", help="Pin scope: 'global' or 'project:<repo-name>'."),
    ] = "global",
) -> None:
    """Mark a rule/skill as Always-Choose (`pinned = 1`) so agents automatically load and follow it."""
    idx = SkillIndex(get_settings())
    try:
        rec = idx.pin_skill(slug, scope=scope)
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"[bold green]Pinned[/bold green] `{rec.slug}` as Always-Choose (`pinned = 1`, scope = `{rec.pinned_scope}`)."
    )


@rules_app.command("unpin")
def rules_unpin_cmd(
    slug: Annotated[str, typer.Argument(help="Skill/rule slug to revert to on-demand discovery.")],
) -> None:
    """Revert a rule back to on-demand discovery (`pinned = 0`)."""
    idx = SkillIndex(get_settings())
    try:
        rec = idx.unpin_skill(slug)
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[yellow]Unpinned[/yellow] `{rec.slug}` (`pinned = 0`, now on-demand).")


@rules_app.command("select")
def rules_select_cmd(
    scope: Annotated[
        str,
        typer.Option("--scope", "-s", help="Pin scope to apply when toggling rules on ('global' or 'project:<name>')."),
    ] = "global",
) -> None:
    """Interactive terminal checklist grouped by Scope/Agent to toggle Always-Choose vs On-Demand rules."""
    idx = SkillIndex(get_settings())
    idx.ensure_fresh()
    records = idx.list_skills()
    if not records:
        console.print("[yellow]No rules in library yet. Run `sesh deep-sync` first.[/yellow]")
        return

    table = Table(title="Interactive Rule Selector (Grouped by Scope & Agent)", border_style="cyan")
    table.add_column("#", justify="right", style="bold")
    table.add_column("State", justify="center")
    table.add_column("Slug", style="bold cyan")
    table.add_column("Scope", style="magenta")
    table.add_column("Agent")
    table.add_column("Description")

    ordered = sorted(records, key=lambda r: (r.scope, r.source_agent, r.slug))
    for i, r in enumerate(ordered, 1):
        state = f"[bold green][x] ALWAYS ({r.pinned_scope})[/bold green]" if r.pinned else "[dim][ ] on-demand[/dim]"
        table.add_row(str(i), state, r.slug, r.scope, r.source_agent, r.description)
    console.print(table)

    raw_choice = typer.prompt(
        "Enter rule numbers or slugs to toggle Always-Choose (comma-separated, 'all', 'none', or 'q' to quit)",
        default="q",
    ).strip()
    if not raw_choice or raw_choice.lower() == "q":
        console.print("[dim]No changes made.[/dim]")
        return

    if raw_choice.lower() == "none":
        for r in ordered:
            if r.pinned:
                idx.unpin_skill(r.slug)
        console.print("[yellow]Cleared all pinned rules (all rules set to on-demand).[/yellow]")
        return

    if raw_choice.lower() == "all":
        for r in ordered:
            target_scope = r.scope if scope == "auto" else scope
            idx.pin_skill(r.slug, scope=target_scope)
        console.print(f"[green]Pinned all {len(ordered)} rule(s) as Always-Choose.[/green]")
        return

    tokens = [t.strip() for t in raw_choice.split(",") if t.strip()]
    by_slug = {r.slug: r for r in ordered}
    for tok in tokens:
        rec = None
        if tok.isdigit():
            pos = int(tok)
            if 1 <= pos <= len(ordered):
                rec = ordered[pos - 1]
        else:
            rec = by_slug.get(tok)
        if rec is None:
            console.print(f"[yellow]Skipping unknown entry '{tok}'[/yellow]")
            continue
        if rec.pinned:
            idx.unpin_skill(rec.slug)
            console.print(f"  [yellow][-] {rec.slug}[/yellow] -> on-demand")
        else:
            updated = idx.pin_skill(rec.slug, scope=scope)
            console.print(f"  [green][+] {updated.slug}[/green] -> ALWAYS ({updated.pinned_scope})")


# ---------------------------------------------------------------------------
# v2.0: MCP Hub (`sesh mcp-hub ...`)
# ---------------------------------------------------------------------------


@mcp_hub_app.command("harvest")
def mcp_hub_harvest_cmd(
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace directory to scan for local MCP configs."),
    ] = None,
) -> None:
    """Discover and import MCP servers from Claude, Cursor, Copilot, and Codex into ~/.agent-seshat/mcp_registry.json."""
    settings = get_settings()
    report = harvest_mcp_configs(workspace=workspace or Path.cwd(), settings=settings)

    if not report.servers:
        console.print("[yellow]No external MCP servers discovered across agent configs.[/yellow]")
        return

    table = Table(title=f"Centralized MCP Registry ({settings.mcp_registry_path})", border_style="cyan")
    table.add_column("Server Name", style="bold cyan")
    table.add_column("Transport", style="magenta")
    table.add_column("Command / URL", overflow="fold")
    table.add_column("Source Agents", style="green")
    table.add_column("Enabled", justify="center")

    for srv in report.servers:
        endpoint = srv.url if srv.transport == "http" else f"{srv.command} {' '.join(srv.args)}".strip()
        table.add_row(
            srv.name,
            srv.transport,
            endpoint,
            ", ".join(srv.source_agents),
            "yes" if srv.enabled else "no",
        )
    console.print(table)
    console.print(
        f"[green]Harvested {len(report.servers)} server(s) "
        f"(+{len(report.added)} added, ~{len(report.updated)} updated, {report.skipped_self} self-reference(s) skipped).[/green]"
    )


@mcp_hub_app.command("list")
def mcp_hub_list_cmd() -> None:
    """Show all centralized MCP servers, their origin agents, status, and cached tool counts."""
    settings = get_settings()
    idx = SkillIndex(settings)
    registry = load_mcp_registry(settings)
    cached_tools = idx.list_mcp_tools()
    counts_by_server: dict[str, int] = {}
    for t in cached_tools:
        counts_by_server[t["server_name"]] = counts_by_server.get(t["server_name"], 0) + 1

    if not registry:
        console.print("[yellow]MCP registry is empty. Run `sesh mcp-hub harvest` first.[/yellow]")
        return

    table = Table(title="Agent-Seshat MCP Hub Servers", border_style="cyan")
    table.add_column("Server Name", style="bold cyan")
    table.add_column("Transport", style="magenta")
    table.add_column("Origin Agents", style="green")
    table.add_column("Enabled", justify="center")
    table.add_column("Cached Tools", justify="right", style="bold yellow")
    table.add_column("Endpoint / Command", overflow="fold")

    for name, srv in sorted(registry.items()):
        endpoint = srv.url if srv.transport == "http" else f"{srv.command} {' '.join(srv.args)}".strip()
        table.add_row(
            name,
            srv.transport,
            ", ".join(srv.source_agents),
            "[green]enabled[/green]" if srv.enabled else "[dim]disabled[/dim]",
            str(counts_by_server.get(name, 0)),
            endpoint,
        )
    console.print(table)


@mcp_hub_app.command("sync")
def mcp_hub_sync_cmd() -> None:
    """Connect to enabled downstream MCP servers and index their tool schemas in SQLite + ONNX."""
    settings = get_settings()
    idx = SkillIndex(settings)
    with console.status("[bold cyan]Syncing tool schemas from downstream MCP servers..."):
        results = sync_mcp_tools(idx, settings=settings)

    if not results:
        console.print("[yellow]No enabled MCP servers in registry to sync.[/yellow]")
        return

    table = Table(title="Downstream MCP Tool Sync", border_style="cyan")
    table.add_column("Server Name", style="bold cyan")
    table.add_column("Tools Indexed", justify="right", style="bold green")
    table.add_column("Latency", justify="right")
    table.add_column("Status")

    for r in results:
        status_str = f"[red]ERROR: {r.error}[/red]" if r.error else "[green]OK[/green]"
        table.add_row(r.server_name, str(r.tools_count), f"{r.elapsed_ms:.1f} ms", status_str)
    console.print(table)


@mcp_hub_app.command("consolidate")
def mcp_hub_consolidate_cmd(
    mode: Annotated[
        str,
        typer.Option("--mode", "-m", help="Gateway connection mode: auto | stdio | http."),
    ] = "auto",
) -> None:
    """Back up agent configs (*.pre-seshat.bak) and route Claude, Cursor, Copilot, and Codex exclusively through agent-seshat."""
    settings = get_settings()
    consolidated = consolidate_mcp_configs(workspace=Path.cwd(), settings=settings, mode=mode)
    if not consolidated:
        console.print("[yellow]No existing agent MCP config files found to consolidate.[/yellow]")
        return

    table = Table(title="Consolidated Agent MCP Configs -> agent-seshat Gateway", border_style="green")
    table.add_column("Agent Config File", overflow="fold", style="bold cyan")
    table.add_column("Pre-Seshat Backup", overflow="fold", style="dim")
    for cfg, bak in consolidated:
        table.add_row(str(cfg), str(bak))
    console.print(table)


# ---------------------------------------------------------------------------
# v2.0: Environment Variables (`sesh env ...`)
# ---------------------------------------------------------------------------


@env_app.command("set")
def env_set_cmd(
    key_or_assignment: Annotated[
        str,
        typer.Argument(help="Variable name (`API_URL`) or `KEY=VALUE` assignment."),
    ],
    value: Annotated[
        Optional[str],
        typer.Argument(help="Variable value (when not using `KEY=VALUE` syntax)."),
    ] = None,
    scope: Annotated[
        str,
        typer.Option("--scope", "-s", help="Scope: 'global', 'project:<repo>', or 'mcp:<server>'."),
    ] = "global",
    description: Annotated[
        str,
        typer.Option("--description", "-d", help="Optional human/agent description of this variable."),
    ] = "",
    secret: Annotated[
        Optional[bool],
        typer.Option("--secret/--no-secret", help="Mark variable as secret (masked in listings; auto-detected if omitted)."),
    ] = None,
) -> None:
    """Store or update an environment variable in Agent-Seshat."""
    if value is None:
        if "=" not in key_or_assignment:
            console.print("[red]Provide either `sesh env set KEY VALUE` or `sesh env set KEY=VALUE`.[/red]")
            raise typer.Exit(code=2)
        key_str, val_str = key_or_assignment.split("=", 1)
    else:
        key_str, val_str = key_or_assignment, value

    store = EnvStore(settings=get_settings())
    try:
        rec = store.set(key_str, val_str, scope=scope, description=description, is_secret=secret)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2)

    badge = "[bold red]SECRET (masked)[/bold red]" if rec.is_secret else "[green]standard[/green]"
    console.print(
        f"[bold green]Saved[/bold green] `{rec.key}` = `{rec.display_value(reveal_secret=False)}` "
        f"(scope: [magenta]{rec.scope}[/magenta], type: {badge})"
    )


@env_app.command("get")
def env_get_cmd(
    key: Annotated[str, typer.Argument(help="Environment variable name to retrieve.")],
    scope: Annotated[
        Optional[str],
        typer.Option("--scope", "-s", help="Explicit scope to query ('global', 'project:<repo>', 'mcp:<server>')."),
    ] = None,
    project: Annotated[
        str,
        typer.Option("--project", "-p", help="Project name to resolve (`project:<name>` -> `global`)."),
    ] = "",
    raw: Annotated[
        bool,
        typer.Option("--raw", "-r", help="Print only the raw value without formatting."),
    ] = False,
) -> None:
    """Retrieve an environment variable by key (resolves project scope first, then global)."""
    store = EnvStore(settings=get_settings())
    eff_project = project or detect_project()
    rec = store.get(key, scope=scope, project=eff_project, fallback_global=True)
    if rec is None:
        if not raw:
            console.print(f"[red]Environment variable '{key}' not found.[/red]")
        raise typer.Exit(code=1)

    if raw:
        typer.echo(rec.value)
        return

    table = Table(title=f"Environment Variable: {rec.key}", border_style="cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Key", rec.key)
    table.add_row("Value", rec.value)
    table.add_row("Scope", rec.scope)
    table.add_row("Secret", "yes" if rec.is_secret else "no")
    if rec.description:
        table.add_row("Description", rec.description)
    console.print(table)


@env_app.command("list")
def env_list_cmd(
    scope: Annotated[
        Optional[str],
        typer.Option("--scope", "-s", help="Filter by scope ('global', 'project:<repo>', 'mcp:<server>')."),
    ] = None,
    project: Annotated[
        str,
        typer.Option("--project", "-p", help="Filter for a specific project (includes global)."),
    ] = "",
    show_secrets: Annotated[
        bool,
        typer.Option("--show-secrets", help="Reveal secret values in plaintext."),
    ] = False,
) -> None:
    """List stored environment variables across scopes (secrets are masked by default)."""
    store = EnvStore(settings=get_settings())
    records = store.list(scope=scope, project=project, include_global=True)
    if not records:
        console.print("[yellow]No environment variables stored.[/yellow]")
        return

    table = Table(title=f"Agent-Seshat Environment Variables ({len(records)})", border_style="cyan")
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")
    table.add_column("Scope", style="magenta")
    table.add_column("Secret", justify="center")
    table.add_column("Description", style="dim")

    for r in records:
        val_display = r.display_value(reveal_secret=show_secrets)
        secret_cell = "[bold red]secret[/bold red]" if r.is_secret else "[dim]no[/dim]"
        table.add_row(r.key, val_display, r.scope, secret_cell, r.description or "-")
    console.print(table)


@env_app.command("unset")
def env_unset_cmd(
    key: Annotated[str, typer.Argument(help="Environment variable name to delete.")],
    scope: Annotated[
        str,
        typer.Option("--scope", "-s", help="Scope from which to remove the variable."),
    ] = "global",
) -> None:
    """Remove an environment variable from a scope."""
    store = EnvStore(settings=get_settings())
    if not store.unset(key, scope=scope):
        console.print(f"[yellow]Variable `{key}` not found in scope `{scope}`.[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"[green]Removed[/green] `{key}` from scope `{scope}`.")


@env_app.command("import")
def env_import_cmd(
    dotenv_file: Annotated[Path, typer.Argument(help="Path to `.env` file to import.")],
    scope: Annotated[
        str,
        typer.Option("--scope", "-s", help="Target scope ('global', 'project:<repo>', 'mcp:<server>')."),
    ] = "global",
    secret: Annotated[
        Optional[bool],
        typer.Option("--secret/--no-secret", help="Force all imported variables to be secret or non-secret."),
    ] = None,
) -> None:
    """Import environment variables from a `.env` file into Agent-Seshat."""
    store = EnvStore(settings=get_settings())
    try:
        imported = store.import_dotenv(dotenv_file, scope=scope, secret=secret)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[bold green]Imported {len(imported)} variable(s)[/bold green] from `{dotenv_file}` into scope `[magenta]{scope}[/magenta]`."
    )


@env_app.command("export")
def env_export_cmd(
    scope: Annotated[
        Optional[str],
        typer.Option("--scope", "-s", help="Filter by scope ('global', 'project:<repo>', 'mcp:<server>')."),
    ] = None,
    project: Annotated[
        str,
        typer.Option("--project", "-p", help="Merge `global` + `project:<name>` variables."),
    ] = "",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Write `.env` output to a file instead of stdout."),
    ] = None,
    mask_secrets: Annotated[
        bool,
        typer.Option("--mask-secrets", help="Mask secret values in exported `.env` output."),
    ] = False,
) -> None:
    """Export stored environment variables in `.env` format."""
    store = EnvStore(settings=get_settings())
    content = store.export_dotenv(scope=scope, project=project, mask_secrets=mask_secrets)
    if output is not None:
        out_path = output.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        console.print(f"[green]Exported `.env` to `{out_path}`.[/green]")
    else:
        typer.echo(content, nl=False)
