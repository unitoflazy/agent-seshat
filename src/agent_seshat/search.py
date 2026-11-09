"""Zero-Miss Hybrid Search: chunk max-cosine (ONNX) + weighted BM25 (FTS5) + exact boost + RRF."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import (
    BM25_WEIGHTS,
    EXACT_MATCH_BOOST,
    RRF_K,
    SCOPE_GLOBAL,
    SCOPE_OTHER_PROJECT,
    SCOPE_SAME_PROJECT,
)
from .indexer import SkillIndex

# Stopwords removed from the *lexical* query only (semantic path sees the raw query).
_QUERY_STOPWORDS = frozenset(
    """
    a an and are as at be but by can could do does for from has have how i if in into is it its me my of on
    or our should so than that the their them then there these they this those to too was we were what when
    where which who why will with would you your about best way use using need want help please write make
    get any all some via vs
    """.split()
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._\-/:][A-Za-z0-9]+)*")


@dataclass
class SearchResult:
    slug: str
    name: str
    description: str
    tags: list[str]
    scope: str
    matched_heading: str
    score: float
    resources: list[str] = field(default_factory=list)
    cosine: float | None = None
    bm25: float | None = None
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    exact_boost: float = 0.0
    scope_multiplier: float = 1.0

    @property
    def scope_label(self) -> str:
        if self.scope.startswith("project:"):
            return f"[Scope: {self.scope.split(':', 1)[1]}]"
        return "[Scope: global]"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scope_label"] = self.scope_label
        return d


@dataclass
class SearchDiagnostics:
    query: str
    tokens: list[str]
    fts_query: str
    elapsed_ms: float = 0.0
    refreshed: bool = False


def query_tokens(query: str) -> list[str]:
    """Technical tokens, lowercased, de-duplicated; compounds (``errgroup.WithContext``,
    ``worker-pool``, ``snake_case``) are kept *and* expanded into their parts."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(query):
        tok = raw.lower()
        candidates = [tok]
        parts = [p for p in re.split(r"[._\-/:]", tok) if p]
        if len(parts) > 1:
            candidates.extend(parts)
        for c in candidates:
            if len(c) < 2 or c in _QUERY_STOPWORDS:
                continue
            if c not in out:
                out.append(c)
    return out


def build_fts_query(tokens: list[str]) -> str:
    """``"tok1"* OR "tok2"*`` — quoting makes FTS5 treat compounds as phrases."""
    safe = [t.replace('"', "") for t in tokens if t.replace('"', "")]
    return " OR ".join(f'"{t}"*' for t in safe)


def identifier_tokens(query: str) -> list[str]:
    """Query tokens that look like code identifiers / flags / keywords rather than prose:
    ``errgroup.WithContext``, ``CONCURRENTLY``, ``snake_case``, ``k8s``, ``--force``."""
    out = []
    for raw in _TOKEN_RE.findall(query):
        if len(raw) < 2:
            continue
        code_like = (
            raw != raw.lower()  # CamelCase / ALLCAPS
            or any(ch in raw for ch in "._:/")
            or (any(ch.isdigit() for ch in raw) and any(ch.isalpha() for ch in raw))
        )
        if code_like and raw.lower() not in _QUERY_STOPWORDS:
            out.append(raw)
    return out


def _verbatim(token: str, text: str) -> bool:
    """Whole-token, case-insensitive, *unstemmed* occurrence."""
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text, re.IGNORECASE) is not None


def scope_multiplier(scope: str, project: str) -> float:
    if scope == "global" or not scope.startswith("project:"):
        return SCOPE_GLOBAL
    owner = scope.split(":", 1)[1]
    if not project:
        return SCOPE_GLOBAL
    return SCOPE_SAME_PROJECT if owner.lower() == project.lower() else SCOPE_OTHER_PROJECT


class HybridSearcher:
    def __init__(self, index: SkillIndex):
        self.index = index
        self.last_diagnostics: SearchDiagnostics | None = None

    # -- path 1: dense semantic ---------------------------------------------------

    def _semantic(self, query: str) -> dict[str, tuple[float, str]]:
        cache = self.index.vectors()
        if cache.matrix.shape[0] == 0:
            return {}
        qvec = self.index.embedder.embed_query(query)
        scores = cache.matrix @ qvec  # cosine: both sides L2-normalized
        best: dict[str, tuple[float, str]] = {}
        for i in np.argsort(-scores):
            slug = cache.slugs[i]
            if slug not in best:
                best[slug] = (float(scores[i]), cache.headings[i])
        return best

    # -- path 2: weighted BM25 --------------------------------------------------------

    def _lexical(self, tokens: list[str]) -> list[tuple[str, float]]:
        """Return ``[(slug, bm25)]`` best-first, filtered by an evidence gate so that a
        single stray body word cannot drag an unrelated skill into the fusion."""
        if not tokens:
            return []
        conn = self.index.conn
        w = BM25_WEIGHTS
        try:
            rows = conn.execute(
                f"SELECT slug, bm25(skills_fts, {w[0]}, {w[1]}, {w[2]}, {w[3]}) AS score "
                "FROM skills_fts WHERE skills_fts MATCH ? ORDER BY score LIMIT 200",
                (build_fts_query(tokens),),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        if not rows:
            return []

        # Evidence gate: per-token coverage (any column) and metadata hits (name/tags/description).
        coverage: dict[str, int] = {}
        meta_hit: set[str] = set()
        for tok in tokens:
            phrase = f'"{tok.replace(chr(34), "")}"*'
            try:
                for (slug,) in conn.execute("SELECT slug FROM skills_fts WHERE skills_fts MATCH ?", (phrase,)):
                    coverage[slug] = coverage.get(slug, 0) + 1
                for (slug,) in conn.execute(
                    "SELECT slug FROM skills_fts WHERE skills_fts MATCH ?", (f"{{name tags description}} : {phrase}",)
                ):
                    meta_hit.add(slug)
            except sqlite3.OperationalError:
                continue
        n = len(tokens)
        needed = 1 if n <= 1 else max(2, (n + 1) // 2)
        accepted = []
        for r in rows:
            slug = r["slug"]
            if slug in meta_hit or coverage.get(slug, 0) >= needed:
                accepted.append((slug, float(r["score"])))
        return accepted

    # -- fusion -------------------------------------------------------------------

    def search(self, query: str, project: str = "", top_k: int = 5) -> list[SearchResult]:
        t0 = time.perf_counter()
        refreshed = self.index.ensure_fresh() is not None
        query = (query or "").strip()
        tokens = query_tokens(query)
        diag = SearchDiagnostics(query=query, tokens=tokens, fts_query=build_fts_query(tokens), refreshed=refreshed)
        self.last_diagnostics = diag
        if not query:
            return []

        settings = self.index.settings
        semantic_all = self._semantic(query)
        top_cos = max((v[0] for v in semantic_all.values()), default=0.0)
        floor = max(settings.semantic_cutoff, top_cos - settings.semantic_margin)
        semantic = sorted(
            ((s, v) for s, v in semantic_all.items() if v[0] >= floor), key=lambda kv: kv[1][0], reverse=True
        )
        sem_rank = {slug: i + 1 for i, (slug, _) in enumerate(semantic)}
        with self.index._lock:
            lexical = self._lexical(tokens)
        lex_rank = {slug: i + 1 for i, (slug, _) in enumerate(lexical)}
        lex_score = dict(lexical)

        candidates = set(sem_rank) | set(lex_rank)
        records = self.index.get_skills(candidates)
        token_set = set(tokens)
        identifiers = identifier_tokens(query)

        results: list[SearchResult] = []
        for slug in candidates:
            rec = records.get(slug)
            if rec is None:
                continue
            exact_terms = {slug, *slug.replace("_", "-").split("-"), *rec.tags}
            exact = bool(token_set & {t for t in exact_terms if len(t) >= 2})
            if not exact and identifiers:
                haystack = f"{rec.name}\n{rec.description}\n{rec.body}"
                exact = any(_verbatim(ident, haystack) for ident in identifiers)
            boost = EXACT_MATCH_BOOST if exact else 0.0
            cos, heading = semantic_all.get(slug, (None, ""))
            if slug not in lex_rank and not exact and (cos or 0.0) < settings.semantic_only_cutoff:
                continue  # semantic-only evidence too weak: BGE noise band
            rrf = 0.0
            if slug in sem_rank:
                rrf += 1.0 / (RRF_K + sem_rank[slug])
            if slug in lex_rank:
                rrf += 1.0 / (RRF_K + lex_rank[slug])
            mult = scope_multiplier(rec.scope, project)
            results.append(
                SearchResult(
                    slug=slug,
                    name=rec.name,
                    description=rec.description,
                    tags=rec.tags,
                    scope=rec.scope,
                    matched_heading=heading or rec.name,
                    score=(rrf + boost) * mult,
                    resources=rec.resources,
                    cosine=cos,
                    bm25=lex_score.get(slug),
                    semantic_rank=sem_rank.get(slug),
                    lexical_rank=lex_rank.get(slug),
                    exact_boost=boost,
                    scope_multiplier=mult,
                )
            )
        results.sort(key=lambda r: (-r.score, -(r.cosine or 0.0), r.slug))
        diag.elapsed_ms = (time.perf_counter() - t0) * 1000
        return results[: max(1, top_k)]


_SEARCHERS: dict[int, HybridSearcher] = {}


def hybrid_search(query: str, project: str = "", top_k: int = 5, index: SkillIndex | None = None) -> list[SearchResult]:
    """Convenience wrapper (spec signature). Reuses a process-wide index when none is given."""
    if index is None:
        from .server import get_default_index

        index = get_default_index()
    searcher = _SEARCHERS.get(id(index))
    if searcher is None or searcher.index is not index:
        searcher = _SEARCHERS[id(index)] = HybridSearcher(index)
    return searcher.search(query, project=project, top_k=top_k)
