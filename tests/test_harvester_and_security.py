"""Tests for multi-agent harvester, secret scrubbing, path traversal protection, and MCP connector."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import tomlkit

from agent_seshat.config import Settings
from agent_seshat.connector import connect_agents, inspect_connections
from agent_seshat.harvester import STUB_MARKER, harvest
from agent_seshat.indexer import SkillIndex
from agent_seshat.parser import REDACTED, sanitize_secrets
from agent_seshat.server import SeshatService, SkillSecurityError

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SKILLS_DIR = REPO_ROOT / "sample_skills"


def test_sanitize_secrets_masks_api_keys_and_db_urls() -> None:
    raw = (
        "OPENAI_API_KEY=sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456\n"
        "GITHUB_TOKEN=ghp_1234567890abcdefghijklmnopqrstuvwxyzABCD\n"
        "AWS_KEY=AKIAIOSFODNN7EXAMPLE\n"
        "DATABASE_URL=postgres://admin:s3cr3tP4ssw0rd@db.internal.example.com:5432/prod\n"
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secretpayload\n"
    )
    scrubbed, findings = sanitize_secrets(raw)

    assert "sk-proj-" not in scrubbed
    assert "ghp_12345" not in scrubbed
    assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed
    assert "s3cr3tP4ssw0rd" not in scrubbed
    assert f"postgres://admin:{REDACTED}@db.internal.example.com:5432/prod" in scrubbed
    assert REDACTED in scrubbed
    assert "openai_anthropic_key" in findings
    assert "database_url_password" in findings
    assert "github_token" in findings
    assert "aws_access_key" in findings


def test_harvester_redacts_secrets_deduplicates_and_writes_stubs(tmp_path: Path) -> None:
    home = tmp_path / ".agent-seshat"
    settings = Settings(home=home)

    repo = tmp_path / "acme-api"
    (repo / ".git").mkdir(parents=True)
    (repo / ".github").mkdir(parents=True)

    claude_md = repo / "CLAUDE.md"
    claude_md.write_text(
        "# Acme API Architecture\n\n"
        "Use FastAPI with async SQLAlchemy. Never commit keys like sk-live-999999999999999999999999 or "
        "postgres://appuser:leakedpass123@localhost:5432/acme.\n",
        encoding="utf-8",
    )

    # Identical content in .cursorrules should be detected as a duplicate.
    cursorrules = repo / ".cursorrules"
    cursorrules.write_text(claude_md.read_text(encoding="utf-8"), encoding="utf-8")

    # Copilot and Codex instructions.
    (repo / ".github" / "copilot-instructions.md").write_text(
        "# Copilot Guidelines\n\nPrefer pytest fixtures and strict mypy typing across `src/`.\n",
        encoding="utf-8",
    )
    (repo / "AGENTS.md").write_text(
        "# Codex Agent Instructions\n\nRun `uv run pytest` before every commit.\n",
        encoding="utf-8",
    )

    report = harvest([repo], write_stubs=True, settings=settings)
    assert report.secrets_total >= 2

    statuses = {i.source.name: i.status for i in report.items}
    assert statuses["CLAUDE.md"] == "new"
    assert statuses[".cursorrules"] == "duplicate"
    assert statuses["copilot-instructions.md"] == "new"
    assert statuses["AGENTS.md"] == "new"

    # Imported skill file must have secrets redacted and project scope assigned.
    imported = (settings.skills_dir / "acme-api-claude-md" / "SKILL.md").read_text(encoding="utf-8")
    assert "sk-live-" not in imported
    assert "leakedpass123" not in imported
    assert REDACTED in imported
    assert "scope: project:acme-api" in imported

    # Original CLAUDE.md was backed up to CLAUDE.md.bak and replaced with an 8-line Seshat stub.
    backup = repo / "CLAUDE.md.bak"
    assert backup.is_file()
    assert "sk-live-" in backup.read_text(encoding="utf-8")
    stub_lines = claude_md.read_text(encoding="utf-8").strip().splitlines()
    assert len(stub_lines) == 8
    assert STUB_MARKER in stub_lines[0]
    assert "search_skills" in "\n".join(stub_lines)


def test_load_skill_resource_blocks_path_traversal(tmp_path: Path) -> None:
    home = tmp_path / ".agent-seshat"
    shutil.copytree(SAMPLE_SKILLS_DIR, home / "skills")
    idx = SkillIndex(Settings(home=home))
    idx.index()
    service = SeshatService(idx)

    # Valid companion resource succeeds.
    go_code = service.load_skill_resource("golang-concurrency", "worker_pool_template.go")
    assert "package workerpool" in go_code

    # Traversal attacks must raise SkillSecurityError.
    with pytest.raises(SkillSecurityError):
        service.load_skill_resource("golang-concurrency", "../../etc/passwd")

    with pytest.raises(SkillSecurityError):
        service.load_skill_resource("golang-concurrency", "../postgres-indexing/SKILL.md")

    with pytest.raises(SkillSecurityError):
        service.load_skill_resource("golang-concurrency", "/etc/passwd")

    idx.close()


def test_connector_merges_and_backs_up_claude_codex_cursor_copilot_configs(tmp_path: Path) -> None:
    fake_home = tmp_path / "user_home"
    fake_home.mkdir()

    # Pre-populate Claude Code JSON and Codex TOML with existing MCP servers.
    claude_cfg = fake_home / ".claude.json"
    claude_cfg.write_text(
        json.dumps({"mcpServers": {"github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]}}}),
        encoding="utf-8",
    )

    codex_dir = fake_home / ".codex"
    codex_dir.mkdir()
    codex_cfg = codex_dir / "config.toml"
    codex_cfg.write_text(
        'model = "gpt-5-codex"\n\n[mcp_servers.existing]\ncommand = "echo"\n',
        encoding="utf-8",
    )

    results = connect_agents(target="all", mode="auto", user_home=fake_home)
    by_target = {r.target: r for r in results}

    assert by_target["claude"].status == "updated"
    assert by_target["claude"].backup_path == fake_home / ".claude.json.seshat.bak"
    assert by_target["codex"].status == "updated"
    assert by_target["codex"].backup_path == codex_dir / "config.toml.seshat.bak"
    assert by_target["cursor"].status == "created"
    assert by_target["copilot"].status == "created"

    # Existing servers are preserved alongside agent-seshat.
    merged_claude = json.loads(claude_cfg.read_text(encoding="utf-8"))
    assert "github" in merged_claude["mcpServers"]
    assert merged_claude["mcpServers"]["agent-seshat"] == {"command": "sesh", "args": ["mcp"]}

    merged_codex = tomlkit.parse(codex_cfg.read_text(encoding="utf-8"))
    assert merged_codex["model"] == "gpt-5-codex"
    assert "existing" in merged_codex["mcp_servers"]
    assert dict(merged_codex["mcp_servers"]["agent-seshat"]) == {"command": "sesh", "args": ["mcp"]}

    statuses = {c["target"]: c["connected"] for c in inspect_connections(fake_home)}
    assert statuses["claude"] is True
    assert statuses["codex"] is True
    assert statuses["cursor"] is True
    assert statuses["copilot"] is True
