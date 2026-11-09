"""Tests for project-scope isolation (1.35x current repo boost vs 0.40x other repo demotion)."""

from __future__ import annotations

from pathlib import Path

from agent_seshat.config import SCOPE_GLOBAL, SCOPE_OTHER_PROJECT, SCOPE_SAME_PROJECT, Settings
from agent_seshat.indexer import SkillIndex
from agent_seshat.search import HybridSearcher


def _write_skill(skills_dir: Path, slug: str, scope: str, extra: str) -> None:
    skill_dir = skills_dir / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\n"
        f"name: {slug}\n"
        f"description: API authentication and JWT session conventions ({extra})\n"
        f"tags: [auth, jwt, api, security]\n"
        f"scope: {scope}\n"
        f"---\n\n"
        f"## JWT Authentication Conventions\n\n"
        f"Verify JWT access tokens with EdDSA signatures, short 15-minute expiry, and strict audience checks ({extra}).\n",
        encoding="utf-8",
    )


def test_project_scope_isolation_prioritizes_active_project_and_demotes_other(tmp_path: Path) -> None:
    home = tmp_path / ".agent-seshat"
    skills_dir = home / "skills"
    _write_skill(skills_dir, "alpha-jwt-auth", "project:alpha", "alpha service")
    _write_skill(skills_dir, "beta-jwt-auth", "project:beta", "beta service")
    _write_skill(skills_dir, "global-jwt-auth", "global", "shared baseline")

    idx = SkillIndex(Settings(home=home))
    idx.index()
    searcher = HybridSearcher(idx)

    # Searching from project="alpha" ranks alpha #1 (1.35x), global #2 (1.0x), and demotes beta (0.40x).
    alpha_results = searcher.search("JWT authentication access tokens", project="alpha", top_k=5)
    assert [r.slug for r in alpha_results] == ["alpha-jwt-auth", "global-jwt-auth", "beta-jwt-auth"]
    assert alpha_results[0].scope_multiplier == SCOPE_SAME_PROJECT
    assert alpha_results[1].scope_multiplier == SCOPE_GLOBAL
    assert alpha_results[2].scope_multiplier == SCOPE_OTHER_PROJECT
    assert alpha_results[2].scope_label == "[Scope: beta]"

    # Searching from project="beta" flips the project ranking so beta is #1 and alpha is demoted.
    beta_results = searcher.search("JWT authentication access tokens", project="beta", top_k=5)
    assert beta_results[0].slug == "beta-jwt-auth"
    assert beta_results[-1].slug == "alpha-jwt-auth"
    assert beta_results[-1].scope_multiplier == SCOPE_OTHER_PROJECT
    assert beta_results[-1].scope_label == "[Scope: alpha]"
    idx.close()
