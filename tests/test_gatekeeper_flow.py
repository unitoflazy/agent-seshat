"""Tests for v2.0 Interactive Rule Gatekeeper & In-Chat Pinning (`gatekeeper.py`)."""

from __future__ import annotations

from pathlib import Path

from agent_seshat.config import Settings
from agent_seshat.indexer import SkillIndex
from agent_seshat.server import SeshatService


def _write_rule(
    skills_dir: Path,
    slug: str,
    *,
    description: str,
    tags: list[str],
    scope: str,
    source_agent: str,
    body: str,
    pinned: bool = False,
    pinned_scope: str = "none",
    file_globs: str = "",
) -> None:
    d = skills_dir / slug
    d.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"name: {slug}",
        f"description: {description}",
        f"tags: [{', '.join(tags)}]",
        f"scope: {scope}",
        f"source_agent: {source_agent}",
    ]
    if file_globs:
        lines.append(f'file_globs: "{file_globs}"')
    if pinned:
        lines.append("pinned: true")
        lines.append(f"pinned_scope: {pinned_scope}")
    lines.extend(["---", "", body.strip(), ""])
    (d / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")


def test_gatekeeper_propose_and_activate_rules_flow(tmp_path: Path) -> None:
    seshat_home = tmp_path / ".agent-seshat"
    skills_dir = seshat_home / "skills"

    # 1. Global pinned rule (Always-Choose across all workspaces)
    _write_rule(
        skills_dir,
        "global-security-baseline",
        description="Mandatory OWASP input validation and secret handling baseline",
        tags=["security", "global", "owasp"],
        scope="global",
        source_agent="claude",
        body="## Global Security Baseline\n\nNever log bearer tokens or raw PII in structured logs.",
        pinned=True,
        pinned_scope="global",
    )

    # 2. Project-scoped pinned rule for `payments-api`
    _write_rule(
        skills_dir,
        "payments-api-idempotency",
        description="Idempotency key enforcement for all payments-api write endpoints",
        tags=["payments", "idempotency", "golang", "rest"],
        scope="project:payments-api",
        source_agent="cursor",
        body="## Payments Idempotency\n\nRequire `Idempotency-Key` header on every POST/PUT/PATCH handler.",
        pinned=True,
        pinned_scope="project:payments-api",
    )

    # 3. Pinned rule belonging to a DIFFERENT project (`billing-worker`) - must NOT auto-load for `payments-api`
    _write_rule(
        skills_dir,
        "billing-worker-cron",
        description="Billing worker cron schedule conventions",
        tags=["billing", "cron"],
        scope="project:billing-worker",
        source_agent="codex",
        body="## Billing Cron\n\nUse UTC cron schedules with advisory locks.",
        pinned=True,
        pinned_scope="project:billing-worker",
    )

    # 4. Unpinned candidate rules relevant to building a Go REST endpoint
    _write_rule(
        skills_dir,
        "go-chi-router",
        description="Building REST API endpoints in Go with chi router, middleware, and OpenAPI validation",
        tags=["golang", "go", "rest", "endpoint", "chi", "http", "router"],
        scope="project:payments-api",
        source_agent="cursor",
        body="## Go Chi Router Conventions\n\nUse `chi.NewRouter()`, attach request ID + timeout middleware, and return RFC 7807 Problem JSON.",
        pinned=False,
    )

    _write_rule(
        skills_dir,
        "go-structured-logging",
        description="Structured JSON logging with log/slog in Go HTTP services",
        tags=["golang", "go", "logging", "slog", "http", "rest"],
        scope="global",
        source_agent="codex",
        body="## Go Structured Logging\n\nUse `log/slog` with request-scoped attributes.",
        pinned=False,
    )

    idx = SkillIndex(Settings(home=seshat_home))
    idx.index()
    service = SeshatService(idx)

    # Step 1: Call propose_session_rules for payments-api
    proposal = service.propose_session_rules(
        user_prompt_summary="build a REST endpoint in Go",
        workspace_path="/work/payments-api",
    )

    # Must contain full body of global + project:payments-api pinned rules under AUTO-ACTIVE
    assert "## [AUTO-ACTIVE: ALWAYS-USE RULES]" in proposal
    assert "global-security-baseline" in proposal
    assert "Never log bearer tokens or raw PII in structured logs." in proposal
    assert "payments-api-idempotency" in proposal
    assert "Require `Idempotency-Key` header on every POST/PUT/PATCH handler." in proposal

    # Must NOT auto-load billing-worker's pinned rule
    assert "Use UTC cron schedules with advisory locks." not in proposal

    # Must present unpinned candidate rules (`go-chi-router`) with mandatory confirmation prompt
    assert "### Mandatory Pre-Task Rule Confirmation" in proposal
    assert "`[go-chi-router]`" in proposal
    assert "Reply with the numbers to activate for this task" in proposal

    # Step 2: Engineer activates `go-chi-router` and pins it permanently for `project:payments-api`
    activation = service.activate_rules(
        load_slugs=["go-chi-router"],
        pin_always_slugs=["go-chi-router"],
        pin_scope="project:payments-api",
    )

    assert "Go Chi Router Conventions" in activation
    assert "chi.NewRouter()" in activation
    assert "Persisted as Always-Choose" in activation

    # Verify SQLite state updated to pinned = 1 for future sessions
    rec = idx.get_skill("go-chi-router")
    assert rec is not None
    assert rec.pinned == 1
    assert rec.pinned_scope == "project:payments-api"

    # Subsequent propose_session_rules call now auto-loads `go-chi-router` under ALWAYS-USE RULES
    proposal_next = service.propose_session_rules(
        user_prompt_summary="build a REST endpoint in Go",
        workspace_path="/work/payments-api",
    )
    auto_section = proposal_next.split("### Mandatory Pre-Task Rule Confirmation")[0]
    assert "go-chi-router" in auto_section
    assert "chi.NewRouter()" in auto_section
    idx.close()
