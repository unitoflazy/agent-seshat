"""Tests for parser, heading-aware chunking, legacy inference, and incremental SHA-256 caching."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_seshat.config import Settings
from agent_seshat.indexer import SkillIndex
from agent_seshat.parser import chunk_markdown, parse_skill_file, parse_skill_text

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SKILLS_DIR = REPO_ROOT / "sample_skills"


def test_heading_chunker_splits_sections_and_prepends_metadata() -> None:
    path = SAMPLE_SKILLS_DIR / "golang-concurrency" / "SKILL.md"
    skill = parse_skill_file(path)

    assert skill.slug == "golang-concurrency"
    assert skill.name == "golang-concurrency"
    assert "golang" in skill.tags
    assert "worker_pool_template.go" in skill.resources

    pairs = skill.embedding_texts()
    assert len(pairs) >= 5
    headings = [c.heading for c, _ in pairs]
    assert "Golden Rules" in headings
    assert "errgroup with Context Cancellation" in headings
    assert "Preventing Goroutine Leaks" in headings
    assert "Channel Pipelines > Backpressure" in headings

    expected_prefix = f"{skill.name} [{', '.join(skill.tags)}] - Preventing Goroutine Leaks: "
    matching = [emb for c, emb in pairs if c.heading == "Preventing Goroutine Leaks"]
    assert len(matching) == 1
    assert matching[0].startswith(expected_prefix)


def test_chunk_markdown_splits_oversized_section_on_paragraphs() -> None:
    para = " ".join([f"token{i}" for i in range(180)])
    long_section = f"## Massive Section\n\n{para}\n\n{para}\n\n{para}\n"
    chunks = chunk_markdown(long_section, max_words=400, min_words=150)

    assert len(chunks) == 2
    assert chunks[0].heading == "Massive Section (part 1)"
    assert chunks[1].heading == "Massive Section (part 2)"
    for c in chunks:
        assert len(c.text.split()) <= 400


def test_legacy_file_without_frontmatter_infers_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "payments-service"
    (repo / ".git").mkdir(parents=True)
    legacy = repo / ".cursorrules"
    legacy.write_text(
        "# Payments Service Rules\n\n"
        "Always validate idempotency_key on Stripe webhook handlers and use PostgreSQL advisory locks.\n",
        encoding="utf-8",
    )

    skill = parse_skill_file(legacy, infer_scope_from_path=True)
    assert skill.slug == "cursorrules"
    assert skill.description == "Payments Service Rules"
    assert skill.scope == "project:payments-service"
    assert any(t in skill.tags for t in ("idempotency_key", "stripe", "postgresql"))


def test_incremental_indexing_skips_onnx_when_content_hash_unchanged(tmp_path: Path) -> None:
    home = tmp_path / ".agent-seshat"
    skills_dir = home / "skills"
    shutil.copytree(SAMPLE_SKILLS_DIR, skills_dir)

    settings = Settings(home=home)
    idx = SkillIndex(settings)

    # First index embeds all chunks across the 3 sample skills.
    stats1 = idx.index()
    assert stats1.total_skills == 3
    assert len(stats1.added) == 3
    assert stats1.chunks_embedded == stats1.total_chunks
    assert stats1.total_chunks > 10

    # Second index with no file changes must skip ONNX embedding completely.
    with patch.object(idx.embedder, "embed_documents", wraps=idx.embedder.embed_documents) as spy:
        stats2 = idx.index()
        assert spy.call_count == 0
        assert stats2.chunks_embedded == 0
        assert stats2.unchanged == 3
        assert not stats2.changed

    # Removing a skill prunes it from skills_meta, skill_chunks, and skills_fts.
    shutil.rmtree(skills_dir / "python-vector-recsys")
    stats3 = idx.index()
    assert stats3.removed == ["python-vector-recsys"]
    assert stats3.total_skills == 2
    assert idx.get_skill("python-vector-recsys") is None
    fts_rows = idx.conn.execute(
        "SELECT slug FROM skills_fts WHERE skills_fts MATCH '\"faiss\"*'"
    ).fetchall()
    assert fts_rows == []
    idx.close()
