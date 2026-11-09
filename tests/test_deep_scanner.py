"""Tests for v2.0 Deep Discovery & Bidirectional Sync (`deep_scanner.py`)."""

from __future__ import annotations

from pathlib import Path

from agent_seshat.config import Settings
from agent_seshat.deep_scanner import deep_scan, deep_sync
from agent_seshat.gatekeeper import GATEKEEPER_END, GATEKEEPER_START
from agent_seshat.indexer import SkillIndex


def test_deep_scanner_discovers_multi_agent_rules_and_auto_pins_mdc(tmp_path: Path) -> None:
    seshat_home = tmp_path / ".agent-seshat"
    fake_user_home = tmp_path / "user_home"
    fake_user_home.mkdir()

    # Global Codex instructions (~/.codex/instructions.md) and global Claude command
    codex_home = fake_user_home / ".codex"
    codex_home.mkdir(parents=True)
    codex_instr = codex_home / "instructions.md"
    codex_instr.write_text(
        "# Global Codex Instructions\n\nAlways prefer deterministic tests and explicit error handling.\n",
        encoding="utf-8",
    )

    claude_cmds = fake_user_home / ".claude" / "commands"
    claude_cmds.mkdir(parents=True)
    claude_cmd_file = claude_cmds / "review-pr.md"
    claude_cmd_file.write_text(
        "# PR Review Checklist\n\nCheck SQL migrations, backward compatibility, and test coverage.\n",
        encoding="utf-8",
    )

    # Workspace repository containing CLAUDE.md, AGENTS.md, .cursor/rules/backend.mdc, and Copilot instructions
    repo = tmp_path / "orders-service"
    (repo / ".git").mkdir(parents=True)
    (repo / ".cursor" / "rules").mkdir(parents=True)
    (repo / ".github" / "instructions").mkdir(parents=True)

    claude_md = repo / "CLAUDE.md"
    claude_md.write_text(
        "# Orders Service Architecture (Claude)\n\n"
        "Use hexagonal architecture and transactional outbox pattern for Kafka events.\n",
        encoding="utf-8",
    )

    agents_md = repo / "AGENTS.md"
    agents_md.write_text(
        "# Orders Service Codex Rules\n\n"
        "Run `go test -race ./...` and `golangci-lint run` before opening pull requests.\n",
        encoding="utf-8",
    )

    backend_mdc = repo / ".cursor" / "rules" / "backend.mdc"
    backend_mdc.write_text(
        "---\n"
        "description: Go backend conventions and context propagation rules\n"
        'globs: ["*.go", "internal/**/*.go"]\n'
        "alwaysApply: true\n"
        "---\n\n"
        "## Go Backend Standards\n\n"
        "Pass `context.Context` as the first argument and wrap errors with `fmt.Errorf(\"...: %w\", err)`.\n",
        encoding="utf-8",
    )

    copilot_md = repo / ".github" / "copilot-instructions.md"
    copilot_md.write_text(
        "# GitHub Copilot Workspace Instructions\n\n"
        "Use table-driven unit tests in Go and avoid global mutable state.\n",
        encoding="utf-8",
    )

    copilot_sql = repo / ".github" / "instructions" / "migrations.instructions.md"
    copilot_sql.write_text(
        "---\n"
        'applyTo: "migrations/*.sql"\n'
        "description: Zero-downtime SQL migration rules\n"
        "---\n\n"
        "Always use `CREATE INDEX CONCURRENTLY` and never rename columns in-place.\n",
        encoding="utf-8",
    )

    settings = Settings(home=seshat_home)
    scanned = deep_scan([repo, fake_user_home], user_home=fake_user_home, settings=settings)
    by_origin = {r.origin_path: r for r in scanned}

    assert claude_md.resolve() in by_origin
    assert agents_md.resolve() in by_origin
    assert backend_mdc.resolve() in by_origin
    assert copilot_md.resolve() in by_origin
    assert copilot_sql.resolve() in by_origin
    assert codex_instr.resolve() in by_origin
    assert claude_cmd_file.resolve() in by_origin

    # Verify source_agent tagging
    assert by_origin[claude_md.resolve()].source_agent == "claude"
    assert by_origin[agents_md.resolve()].source_agent == "codex"
    assert by_origin[backend_mdc.resolve()].source_agent == "cursor"
    assert by_origin[copilot_md.resolve()].source_agent == "copilot"
    assert by_origin[copilot_sql.resolve()].source_agent == "copilot"
    assert by_origin[codex_instr.resolve()].source_agent == "codex"
    assert by_origin[codex_instr.resolve()].scope == "global"

    # Verify Cursor .mdc with alwaysApply: true is auto-pinned to project:orders-service with globs preserved
    mdc_rule = by_origin[backend_mdc.resolve()]
    assert mdc_rule.pinned == 1
    assert mdc_rule.pinned_scope == "project:orders-service"
    assert "*.go" in mdc_rule.file_globs

    # Run full deep_sync with enforce_gatekeeper=True and verify SQLite persistence
    idx = SkillIndex(settings)
    report = deep_sync(
        [repo, fake_user_home],
        user_home=fake_user_home,
        settings=settings,
        index=idx,
        enforce_gatekeeper=True,
    )
    assert report.total_skills == 7

    db_records = {r.origin_path: r for r in idx.list_skills()}
    mdc_rec = db_records[str(backend_mdc.resolve())]
    assert mdc_rec.pinned == 1
    assert mdc_rec.pinned_scope == "project:orders-service"
    assert mdc_rec.source_agent == "cursor"
    assert "*.go" in mdc_rec.file_globs

    sql_rec = db_records[str(copilot_sql.resolve())]
    assert sql_rec.source_agent == "copilot"
    assert sql_rec.file_globs == "migrations/*.sql"

    # Gatekeeper bootstrap block was injected into CLAUDE.md, AGENTS.md, copilot-instructions.md, and 00-seshat-gatekeeper.mdc
    for target in (
        claude_md,
        agents_md,
        copilot_md,
        repo / ".cursor" / "rules" / "00-seshat-gatekeeper.mdc",
    ):
        text = target.read_text(encoding="utf-8")
        assert GATEKEEPER_START in text
        assert GATEKEEPER_END in text
        assert "propose_session_rules" in text

    # Re-running deep_sync after Gatekeeper injection sees all 7 rules as unchanged
    report2 = deep_sync(
        [repo, fake_user_home],
        user_home=fake_user_home,
        settings=settings,
        index=idx,
    )
    assert all(item.status == "unchanged" for item in report2.items)
    idx.close()
