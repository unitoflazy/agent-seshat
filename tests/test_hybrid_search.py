"""Tests for Zero-Miss Hybrid Search (ONNX BGE + SQLite FTS5 BM25 + Exact Boost + RRF + Cutoff)."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from agent_seshat.config import Settings
from agent_seshat.indexer import SkillIndex
from agent_seshat.search import HybridSearcher, hybrid_search
from agent_seshat.server import CATALOG_PREFIX, SeshatService, build_server

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SKILLS_DIR = REPO_ROOT / "sample_skills"


@pytest.fixture(scope="module")
def sample_index(tmp_path_factory: pytest.TempPathFactory) -> SkillIndex:
    home = tmp_path_factory.mktemp("seshat_home")
    shutil.copytree(SAMPLE_SKILLS_DIR, home / "skills")
    idx = SkillIndex(Settings(home=home))
    idx.index()
    return idx


def test_exact_technical_keyword_errgroup_ranks_first(sample_index: SkillIndex) -> None:
    results = hybrid_search("errgroup.WithContext", index=sample_index)
    assert len(results) >= 1
    top = results[0]
    assert top.slug == "golang-concurrency"
    assert top.exact_boost > 0
    assert "errgroup" in top.matched_heading.lower()


def test_exact_technical_keyword_concurrently_ranks_first(sample_index: SkillIndex) -> None:
    results = hybrid_search("CONCURRENTLY", index=sample_index)
    assert len(results) >= 1
    top = results[0]
    assert top.slug == "postgres-indexing"
    assert top.exact_boost > 0


def test_pure_semantic_concept_goroutine_leaks_ranks_first(sample_index: SkillIndex) -> None:
    results = hybrid_search(
        "prevent goroutine leaks when downstream service hangs",
        index=sample_index,
    )
    assert len(results) >= 1
    assert results[0].slug == "golang-concurrency"
    assert "Leak" in results[0].matched_heading


def test_pure_semantic_concept_slow_sql_joins_ranks_first(sample_index: SkillIndex) -> None:
    results = hybrid_search(
        "speed up slow SQL joins on large tables",
        index=sample_index,
    )
    assert len(results) >= 1
    assert results[0].slug == "postgres-indexing"


def test_pure_semantic_concept_recsys_ranks_first(sample_index: SkillIndex) -> None:
    results = hybrid_search(
        "nearest neighbor lookup for user embeddings",
        index=sample_index,
    )
    assert len(results) >= 1
    assert results[0].slug == "python-vector-recsys"


def test_irrelevant_query_cutoff_returns_zero_results(sample_index: SkillIndex) -> None:
    searcher = HybridSearcher(sample_index)
    assert searcher.search("ancient sumerian pottery recipes") == []
    assert searcher.search("how to bake sourdough bread at home") == []


def test_mcp_server_progressive_disclosure_and_dynamic_catalog(sample_index: SkillIndex) -> None:
    service = SeshatService(sample_index)
    mcp = build_server(service)

    tools = asyncio.run(mcp.list_tools())
    tool_names = [t.name for t in tools]
    assert sorted(tool_names) == [
        "activate_rules",
        "call_mcp_tool",
        "get_env_var",
        "list_env_vars",
        "load_skill",
        "load_skill_resource",
        "propose_session_rules",
        "search_mcp_tools",
        "search_skills",
        "set_env_var",
        "unset_env_var",
    ]

    search_tool = next(t for t in tools if t.name == "search_skills")
    assert search_tool.description is not None
    assert search_tool.description.startswith(CATALOG_PREFIX)
    for slug in ("golang-concurrency", "postgres-indexing", "python-vector-recsys"):
        assert slug in search_tool.description

    # Search summary must not return full Markdown bodies.
    md = service.search_markdown("errgroup.WithContext")
    assert "golang-concurrency" in md
    assert "worker_pool_template.go" in md
    assert "func fetchAndStore" not in md

    # No-match message is clear and deterministic.
    empty_md = service.search_markdown("ancient sumerian pottery recipes")
    assert empty_md == "No matching skills found for 'ancient sumerian pottery recipes'."

    # load_skill returns full Markdown body + companion footer.
    loaded = service.load_skill("golang-concurrency")
    assert "## errgroup with Context Cancellation" in loaded
    assert "worker_pool_template.go" in loaded

    # MCP resources include seshat://skill/{slug}.
    resources = asyncio.run(mcp.list_resources())
    uris = {str(r.uri) for r in resources}
    assert "seshat://skill/golang-concurrency" in uris
    assert "seshat://skill/postgres-indexing" in uris
    assert "seshat://skill/python-vector-recsys" in uris
