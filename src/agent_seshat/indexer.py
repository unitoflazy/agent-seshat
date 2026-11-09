"""Incremental SQLite (WAL + FTS5) index with a FastEmbed ONNX chunk-vector cache."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from .config import BGE_QUERY_PREFIX, EMBEDDING_DIM, IGNORED_DIRS, RRF_K, Settings, fastembed_cache_dir, get_settings
from .parser import (
    SKILL_FILENAME,
    Skill,
    compute_content_hash,
    list_resources,
    normalize_scope,
    parse_skill_file,
    split_frontmatter,
)

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS skills_meta (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    tags TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global',
    body TEXT NOT NULL,
    file_path TEXT NOT NULL,
    resources_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    mtime REAL NOT NULL,
    source_agent TEXT NOT NULL DEFAULT 'custom',
    origin_path TEXT DEFAULT '',
    file_globs TEXT DEFAULT '',
    pinned INTEGER NOT NULL DEFAULT 0,
    pinned_scope TEXT NOT NULL DEFAULT 'none'
);

CREATE TABLE IF NOT EXISTS skill_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL REFERENCES skills_meta(slug) ON DELETE CASCADE,
    heading TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skill_chunks_slug ON skill_chunks(slug);

CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
    slug UNINDEXED,
    name,
    tags,
    description,
    body,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS mcp_tools_meta (
    tool_id TEXT PRIMARY KEY,
    server_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    description TEXT NOT NULL,
    input_schema_json TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_server ON mcp_tools_meta(server_name);

CREATE VIRTUAL TABLE IF NOT EXISTS mcp_tools_fts USING fts5(
    tool_id UNINDEXED,
    server_name,
    tool_name,
    description,
    input_schema_json,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS index_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS env_vars (
    key TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global',
    value TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    is_secret INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    PRIMARY KEY (key, scope)
);
CREATE INDEX IF NOT EXISTS idx_env_vars_scope ON env_vars(scope);
"""

_SKILLS_META_MIGRATIONS = (
    ("source_agent", "TEXT NOT NULL DEFAULT 'custom'"),
    ("origin_path", "TEXT DEFAULT ''"),
    ("file_globs", "TEXT DEFAULT ''"),
    ("pinned", "INTEGER NOT NULL DEFAULT 0"),
    ("pinned_scope", "TEXT NOT NULL DEFAULT 'none'"),
)


def _migrate_schema(conn: sqlite3.Connection) -> None:
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(skills_meta)").fetchall()}
    for col_name, col_def in _SKILLS_META_MIGRATIONS:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE skills_meta ADD COLUMN {col_name} {col_def}")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        norm = float(np.linalg.norm(mat))
        return mat / norm if norm > 0 else mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class FastEmbedder:
    """Lazy, thread-safe wrapper around ``fastembed.TextEmbedding`` (CPU ONNX, no PyTorch)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _get(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding  # heavy import, deferred

                    t0 = time.perf_counter()
                    self._model = TextEmbedding(self.model_name, cache_dir=fastembed_cache_dir())
                    log.info("Loaded %s in %.2fs", self.model_name, time.perf_counter() - t0)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        model = self._get()
        vecs = np.vstack([np.asarray(v, dtype=np.float32) for v in model.embed(texts, batch_size=32)])
        return l2_normalize(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        # BGE asymmetric retrieval: queries get the instruction prefix, passages do not.
        model = self._get()
        vec = np.asarray(next(iter(model.embed([f"{BGE_QUERY_PREFIX}{text}"]))), dtype=np.float32)
        return l2_normalize(vec)


_EMBEDDERS: dict[str, FastEmbedder] = {}
_EMBEDDERS_LOCK = threading.Lock()


def get_embedder(model_name: str) -> FastEmbedder:
    with _EMBEDDERS_LOCK:
        if model_name not in _EMBEDDERS:
            _EMBEDDERS[model_name] = FastEmbedder(model_name)
        return _EMBEDDERS[model_name]


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


@dataclass
class IndexStats:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0
    chunks_embedded: int = 0
    total_skills: int = 0
    total_chunks: int = 0
    elapsed_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)


@dataclass
class SkillRecord:
    slug: str
    name: str
    description: str
    tags: list[str]
    scope: str
    body: str
    file_path: Path
    resources: list[str]
    content_hash: str
    mtime: float
    source_agent: str = "custom"
    origin_path: str = ""
    file_globs: str = ""
    pinned: int = 0
    pinned_scope: str = "none"

    @property
    def skill_dir(self) -> Path | None:
        return self.file_path.parent if self.file_path.name == SKILL_FILENAME else None


@dataclass
class VectorCache:
    matrix: np.ndarray  # (M, 384) float32, L2-normalized
    slugs: list[str]
    headings: list[str]


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def discover_skill_files(skills_dir: Path) -> dict[str, Path]:
    """Map ``slug -> skill file``.

    Supported layouts::

        skills/<slug>/SKILL.md            (companion files allowed)
        skills/<slug>.md                  (flat)
        skills/<category>/<slug>/SKILL.md (one level of grouping)
    """
    from .parser import slugify

    found: dict[str, Path] = {}
    if not skills_dir.is_dir():
        return found

    def _add(slug: str, path: Path) -> None:
        slug = slugify(slug)
        if slug in found:
            log.warning("Duplicate skill slug %r: keeping %s, ignoring %s", slug, found[slug], path)
            return
        found[slug] = path

    def _skill_file(d: Path) -> Path | None:
        for name in (SKILL_FILENAME, "skill.md", "Skill.md"):
            p = d / name
            if p.is_file():
                return p
        return None

    with os.scandir(skills_dir) as it:
        entries = sorted(it, key=lambda e: e.name)
    for entry in entries:
        if entry.name.startswith(".") or entry.name in IGNORED_DIRS:
            continue
        path = Path(entry.path)
        if entry.is_dir():
            sf = _skill_file(path)
            if sf is not None:
                _add(entry.name, sf)
                continue
            with os.scandir(path) as sub_it:
                for sub in sorted(sub_it, key=lambda e: e.name):
                    if sub.is_dir() and not sub.name.startswith("."):
                        sf = _skill_file(Path(sub.path))
                        if sf is not None:
                            _add(sub.name, sf)
        elif entry.is_file() and entry.name.lower().endswith(".md") and entry.name.lower() != "readme.md":
            _add(Path(entry.name).stem, path)
    return found


class SkillIndex:
    """Owns the SQLite index for one Seshat home directory."""

    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None):
        self.settings = (settings or get_settings()).ensure_dirs()
        self._embedder = embedder
        self._lock = threading.RLock()
        self._conn = self._open()
        self._fingerprint: tuple | None = None
        self._generation = 0
        self._vector_cache: tuple[tuple[int, int], VectorCache] | None = None
        self.last_check_ms = 0.0

    # -- connection -----------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        db_path = self.settings.db_path
        try:
            conn = connect(db_path)
            conn.executescript(SCHEMA)
            _migrate_schema(conn)
            return conn
        except sqlite3.DatabaseError as exc:
            # The cache is fully derived from skill files: quarantine and rebuild.
            log.warning("Index DB unreadable (%s); rebuilding", exc)
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(db_path) + suffix)
                if p.exists():
                    p.rename(p.with_name(p.name + f".corrupt-{int(time.time())}"))
            conn = connect(db_path)
            conn.executescript(SCHEMA)
            _migrate_schema(conn)
            return conn

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.settings.model_name)
        return self._embedder

    @property
    def generation(self) -> int:
        return self._generation

    # -- change detection -------------------------------------------------------

    def fingerprint(self) -> tuple:
        """Cheap stat-only snapshot of the skills tree (detects add / remove / modify)."""
        skills_dir = self.settings.skills_dir
        items: list[tuple[str, int, int]] = []
        try:
            items.append(("<root>", skills_dir.stat().st_mtime_ns, 0))
        except FileNotFoundError:
            return ()
        for slug, path in discover_skill_files(skills_dir).items():
            try:
                st = path.stat()
                items.append((str(path), st.st_mtime_ns, st.st_size))
                if path.name.lower() == "skill.md":
                    items.append((str(path.parent), path.parent.stat().st_mtime_ns, 0))
            except FileNotFoundError:
                continue
        max_mtime = max((i[1] for i in items), default=0)
        return (len(items), max_mtime, hash(tuple(items)))

    def needs_refresh(self) -> bool:
        t0 = time.perf_counter()
        fp = self.fingerprint()
        self.last_check_ms = (time.perf_counter() - t0) * 1000
        return fp != self._fingerprint

    def ensure_fresh(self) -> IndexStats | None:
        """Run incremental indexing iff the on-disk skills changed since the last check."""
        if not self.needs_refresh():
            return None
        return self.index()

    # -- indexing -----------------------------------------------------------------

    def _meta_get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM index_meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def index(self, force: bool = False) -> IndexStats:
        with self._lock:
            t0 = time.perf_counter()
            stats = IndexStats()
            fingerprint = self.fingerprint()
            files = discover_skill_files(self.settings.skills_dir)

            model_name = getattr(self.embedder, "model_name", self.settings.model_name)
            if self._meta_get("embedding_model") not in (None, model_name):
                force = True

            existing = {
                r["slug"]: (
                    r["content_hash"],
                    r["file_path"],
                    r["mtime"],
                    r["pinned"],
                    r["pinned_scope"],
                )
                for r in self._conn.execute(
                    "SELECT slug, content_hash, file_path, mtime, pinned, pinned_scope FROM skills_meta"
                )
            }

            changed: list[Skill] = []
            touched: list[tuple[str, str, float]] = []
            for slug, path in files.items():
                try:
                    raw = path.read_bytes()
                    resources = list_resources(path)
                    content_hash = compute_content_hash(raw, resources)
                    mtime = path.stat().st_mtime
                except OSError as exc:
                    stats.errors.append(f"{path}: {exc}")
                    continue
                prev = existing.get(slug)
                if not force and prev and prev[0] == content_hash:
                    stats.unchanged += 1
                    if prev[1] != str(path) or prev[2] != mtime:
                        touched.append((str(path), mtime, slug))
                    continue
                try:
                    skill = parse_skill_file(path, slug=slug)
                except Exception as exc:  # malformed file must not break the whole index
                    stats.errors.append(f"{path}: {exc}")
                    continue
                # Preserve existing SQLite pin state if the file itself doesn't explicitly force a pin.
                if prev and prev[3] and not skill.pinned:
                    skill.pinned = int(prev[3])
                    skill.pinned_scope = str(prev[4] or skill.scope)
                skill.content_hash = content_hash
                (stats.updated if prev else stats.added).append(slug)
                changed.append(skill)

            removed = sorted(set(existing) - set(files))
            stats.removed = removed

            # Batch ONNX inference across all changed skills (skipped entirely when nothing changed).
            chunk_rows: list[tuple[Skill, str, str, str]] = []
            for skill in changed:
                for chunk, emb_text in skill.embedding_texts():
                    chunk_rows.append((skill, chunk.heading, chunk.text, emb_text))
            vectors = (
                self.embedder.embed_documents([r[3] for r in chunk_rows])
                if chunk_rows
                else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            )
            stats.chunks_embedded = len(chunk_rows)

            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                for slug in removed:
                    conn.execute("DELETE FROM skills_fts WHERE slug=?", (slug,))
                    conn.execute("DELETE FROM skill_chunks WHERE slug=?", (slug,))
                    conn.execute("DELETE FROM skills_meta WHERE slug=?", (slug,))
                for skill in changed:
                    conn.execute("DELETE FROM skills_fts WHERE slug=?", (skill.slug,))
                    conn.execute("DELETE FROM skill_chunks WHERE slug=?", (skill.slug,))
                    conn.execute(
                        """INSERT INTO skills_meta
                           (slug, name, description, tags, scope, body, file_path, resources_json,
                            content_hash, mtime, source_agent, origin_path, file_globs, pinned, pinned_scope)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(slug) DO UPDATE SET
                             name=excluded.name, description=excluded.description, tags=excluded.tags,
                             scope=excluded.scope, body=excluded.body, file_path=excluded.file_path,
                             resources_json=excluded.resources_json, content_hash=excluded.content_hash,
                             mtime=excluded.mtime, source_agent=excluded.source_agent,
                             origin_path=excluded.origin_path, file_globs=excluded.file_globs,
                             pinned=excluded.pinned, pinned_scope=excluded.pinned_scope""",
                        (
                            skill.slug,
                            skill.name,
                            skill.description,
                            ",".join(t.lower() for t in skill.tags),
                            skill.scope,
                            skill.body,
                            str(skill.file_path),
                            json.dumps(skill.resources),
                            skill.content_hash,
                            skill.mtime,
                            skill.source_agent,
                            skill.origin_path,
                            skill.file_globs,
                            int(skill.pinned),
                            skill.pinned_scope,
                        ),
                    )
                    conn.execute(
                        "INSERT INTO skills_fts (slug, name, tags, description, body) VALUES (?,?,?,?,?)",
                        (
                            skill.slug,
                            f"{skill.name} {skill.slug.replace('-', ' ')}",
                            " ".join(skill.tags),
                            skill.description,
                            skill.body,
                        ),
                    )
                for (skill, heading, text, _), vec in zip(chunk_rows, vectors):
                    conn.execute(
                        "INSERT INTO skill_chunks (slug, heading, chunk_text, embedding) VALUES (?,?,?,?)",
                        (skill.slug, heading, text, np.asarray(vec, dtype=np.float32).tobytes()),
                    )
                for path_s, mtime, slug in touched:
                    conn.execute("UPDATE skills_meta SET file_path=?, mtime=? WHERE slug=?", (path_s, mtime, slug))
                conn.execute(
                    "INSERT INTO index_meta(key, value) VALUES('embedding_model', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (model_name,),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

            if stats.changed or touched:
                self._generation += 1
            self._fingerprint = fingerprint
            stats.total_skills = conn.execute("SELECT COUNT(*) FROM skills_meta").fetchone()[0]
            stats.total_chunks = conn.execute("SELECT COUNT(*) FROM skill_chunks").fetchone()[0]
            stats.elapsed_ms = (time.perf_counter() - t0) * 1000
            if stats.changed:
                log.info(
                    "Indexed: +%d ~%d -%d (%d chunks embedded) in %.0fms",
                    len(stats.added), len(stats.updated), len(stats.removed), stats.chunks_embedded, stats.elapsed_ms,
                )
            return stats

    # -- reads ------------------------------------------------------------------

    def vectors(self) -> VectorCache:
        """All chunk vectors as one ``(M, 384)`` matrix, cached until the DB changes."""
        with self._lock:
            data_version = self._conn.execute("PRAGMA data_version").fetchone()[0]
            key = (self._generation, data_version)
            if self._vector_cache and self._vector_cache[0] == key:
                return self._vector_cache[1]
            rows = self._conn.execute("SELECT slug, heading, embedding FROM skill_chunks ORDER BY chunk_id").fetchall()
            if rows:
                matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            else:
                matrix = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            cache = VectorCache(matrix=matrix, slugs=[r["slug"] for r in rows], headings=[r["heading"] for r in rows])
            self._vector_cache = (key, cache)
            return cache

    @staticmethod
    def _record(row: sqlite3.Row) -> SkillRecord:
        keys = row.keys()
        return SkillRecord(
            slug=row["slug"],
            name=row["name"],
            description=row["description"],
            tags=[t for t in row["tags"].split(",") if t],
            scope=row["scope"],
            body=row["body"],
            file_path=Path(row["file_path"]),
            resources=json.loads(row["resources_json"]),
            content_hash=row["content_hash"],
            mtime=row["mtime"],
            source_agent=row["source_agent"] if "source_agent" in keys and row["source_agent"] else "custom",
            origin_path=row["origin_path"] if "origin_path" in keys and row["origin_path"] else "",
            file_globs=row["file_globs"] if "file_globs" in keys and row["file_globs"] else "",
            pinned=int(row["pinned"]) if "pinned" in keys and row["pinned"] is not None else 0,
            pinned_scope=row["pinned_scope"] if "pinned_scope" in keys and row["pinned_scope"] else "none",
        )

    def get_skill(self, slug: str) -> SkillRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM skills_meta WHERE slug=?", (slug,)).fetchone()
        return self._record(row) if row else None

    def get_skills(self, slugs: Iterable[str]) -> dict[str, SkillRecord]:
        slugs = list(dict.fromkeys(slugs))
        if not slugs:
            return {}
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM skills_meta WHERE slug IN ({','.join('?' * len(slugs))})", slugs
            ).fetchall()
        return {r["slug"]: self._record(r) for r in rows}

    def list_skills(
        self,
        *,
        agent: str | None = None,
        scope: str | None = None,
        pinned_only: bool = False,
    ) -> list[SkillRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if agent:
            clauses.append("LOWER(source_agent) = ?")
            params.append(agent.strip().lower())
        if scope:
            clauses.append("LOWER(scope) = ?")
            params.append(normalize_scope(scope).lower())
        if pinned_only:
            clauses.append("pinned = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(f"SELECT * FROM skills_meta {where} ORDER BY slug", params).fetchall()
        return [self._record(r) for r in rows]

    def set_pin(
        self,
        slug: str,
        pinned: bool,
        pinned_scope: str = "global",
        update_file: bool = True,
    ) -> SkillRecord:
        """Persist `pinned` and `pinned_scope` in SQLite (and optionally in the skill's YAML frontmatter)."""
        import yaml

        self.ensure_fresh()
        rec = self.get_skill(slug)
        if rec is None:
            raise LookupError(f"Unknown skill or rule '{slug}'.")
        norm_scope = normalize_scope(pinned_scope) if pinned else "none"
        pin_int = 1 if pinned else 0
        with self._lock:
            self._conn.execute(
                "UPDATE skills_meta SET pinned=?, pinned_scope=? WHERE slug=?",
                (pin_int, norm_scope, slug),
            )
            if update_file and rec.file_path.is_file():
                try:
                    raw = rec.file_path.read_text(encoding="utf-8", errors="replace")
                    meta, body = split_frontmatter(raw)
                    meta["pinned"] = bool(pinned)
                    meta["pinned_scope"] = norm_scope
                    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000).strip()
                    new_text = f"---\n{front}\n---\n\n{body.strip()}\n"
                    rec.file_path.write_text(new_text, encoding="utf-8")
                    new_hash = compute_content_hash(rec.file_path.read_bytes(), rec.resources)
                    new_mtime = rec.file_path.stat().st_mtime
                    self._conn.execute(
                        "UPDATE skills_meta SET content_hash=?, mtime=? WHERE slug=?",
                        (new_hash, new_mtime, slug),
                    )
                    self._fingerprint = self.fingerprint()
                except OSError:
                    pass
            updated = self.get_skill(slug)
            assert updated is not None
            return updated

    def pin_skill(self, slug: str, scope: str = "global", update_file: bool = True) -> SkillRecord:
        return self.set_pin(slug, pinned=True, pinned_scope=scope, update_file=update_file)

    def unpin_skill(self, slug: str, update_file: bool = True) -> SkillRecord:
        return self.set_pin(slug, pinned=False, pinned_scope="none", update_file=update_file)

    # -- MCP downstream tool index ----------------------------------------------

    def upsert_mcp_tools(self, server_name: str, tools: list[dict[str, Any]]) -> int:
        """Index a downstream MCP server's tools into `mcp_tools_meta` + `mcp_tools_fts` + ONNX."""
        with self._lock:
            texts: list[str] = []
            normalized: list[tuple[str, str, str, str, str]] = []
            for t in tools:
                t_name = str(t.get("name") or t.get("tool_name") or "").strip()
                if not t_name:
                    continue
                desc = str(t.get("description") or "").strip()
                schema = t.get("inputSchema") or t.get("input_schema") or {"type": "object", "properties": {}}
                schema_json = json.dumps(schema, sort_keys=True) if not isinstance(schema, str) else schema
                tool_id = f"{server_name}::{t_name}"
                texts.append(f"{server_name} {t_name}: {desc} {schema_json}")
                normalized.append((tool_id, server_name, t_name, desc, schema_json))

            vecs = (
                self.embedder.embed_documents(texts)
                if texts
                else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            )
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM mcp_tools_fts WHERE server_name=?", (server_name,))
                conn.execute("DELETE FROM mcp_tools_meta WHERE server_name=?", (server_name,))
                for (tool_id, srv, t_name, desc, schema_json), vec in zip(normalized, vecs):
                    conn.execute(
                        """INSERT INTO mcp_tools_meta
                           (tool_id, server_name, tool_name, description, input_schema_json, embedding)
                           VALUES (?,?,?,?,?,?)""",
                        (
                            tool_id,
                            srv,
                            t_name,
                            desc,
                            schema_json,
                            np.asarray(vec, dtype=np.float32).tobytes(),
                        ),
                    )
                    conn.execute(
                        """INSERT INTO mcp_tools_fts
                           (tool_id, server_name, tool_name, description, input_schema_json)
                           VALUES (?,?,?,?,?)""",
                        (tool_id, srv, f"{t_name} {t_name.replace('_', ' ')} {t_name.replace('-', ' ')}", desc, schema_json),
                    )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            return len(normalized)

    def list_mcp_tools(self, server_name: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if server_name:
                rows = self._conn.execute(
                    "SELECT tool_id, server_name, tool_name, description, input_schema_json "
                    "FROM mcp_tools_meta WHERE server_name=? ORDER BY tool_name",
                    (server_name,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT tool_id, server_name, tool_name, description, input_schema_json "
                    "FROM mcp_tools_meta ORDER BY server_name, tool_name"
                ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                schema = json.loads(r["input_schema_json"])
            except Exception:
                schema = {}
            out.append(
                {
                    "tool_id": r["tool_id"],
                    "server_name": r["server_name"],
                    "tool_name": r["tool_name"],
                    "description": r["description"],
                    "input_schema": schema,
                }
            )
        return out

    def search_mcp_tools(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Hybrid search (ONNX cosine + FTS5 BM25 + RRF) over indexed downstream MCP tools."""
        from .search import build_fts_query, query_tokens

        query = (query or "").strip()
        if not query:
            return self.list_mcp_tools()[: max(1, limit)]

        with self._lock:
            rows = self._conn.execute(
                "SELECT tool_id, server_name, tool_name, description, input_schema_json, embedding "
                "FROM mcp_tools_meta"
            ).fetchall()
        if not rows:
            return []

        qvec = self.embedder.embed_query(query)
        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        cos_scores = matrix @ qvec
        sem_order = np.argsort(-cos_scores)
        sem_rank: dict[str, int] = {}
        cos_by_id: dict[str, float] = {}
        for rank_idx, row_idx in enumerate(sem_order, 1):
            tid = rows[int(row_idx)]["tool_id"]
            cos_val = float(cos_scores[int(row_idx)])
            cos_by_id[tid] = cos_val
            if cos_val >= 0.45:
                sem_rank[tid] = rank_idx

        tokens = query_tokens(query)
        lex_rank: dict[str, int] = {}
        if tokens:
            fts_q = build_fts_query(tokens)
            try:
                with self._lock:
                    fts_rows = self._conn.execute(
                        "SELECT tool_id, bm25(mcp_tools_fts, 6.0, 10.0, 5.0, 1.0) AS score "
                        "FROM mcp_tools_fts WHERE mcp_tools_fts MATCH ? ORDER BY score LIMIT 50",
                        (fts_q,),
                    ).fetchall()
                for idx_r, fr in enumerate(fts_rows, 1):
                    lex_rank[fr["tool_id"]] = idx_r
            except sqlite3.OperationalError:
                pass

        token_set = set(tokens)
        by_id = {r["tool_id"]: r for r in rows}
        candidates = set(sem_rank) | set(lex_rank)
        if not candidates and rows:
            # Fall back to top semantic matches if query was very short/generic
            for row_idx in sem_order[:limit]:
                tid = rows[int(row_idx)]["tool_id"]
                if cos_by_id[tid] >= 0.35:
                    candidates.add(tid)
                    sem_rank[tid] = len(sem_rank) + 1

        scored: list[dict[str, Any]] = []
        for tid in candidates:
            r = by_id[tid]
            rrf = 0.0
            if tid in sem_rank:
                rrf += 1.0 / (RRF_K + sem_rank[tid])
            if tid in lex_rank:
                rrf += 1.0 / (RRF_K + lex_rank[tid])
            name_terms = {
                r["server_name"].lower(),
                r["tool_name"].lower(),
                *r["tool_name"].lower().replace("-", "_").split("_"),
            }
            if token_set & name_terms:
                rrf += 0.03
            try:
                schema = json.loads(r["input_schema_json"])
            except Exception:
                schema = {}
            scored.append(
                {
                    "tool_id": tid,
                    "server_name": r["server_name"],
                    "tool_name": r["tool_name"],
                    "description": r["description"],
                    "input_schema": schema,
                    "score": rrf,
                    "cosine": cos_by_id.get(tid, 0.0),
                }
            )
        scored.sort(key=lambda item: (-item["score"], -item["cosine"], item["tool_id"]))
        return scored[: max(1, limit)]

    def counts(self) -> tuple[int, int]:
        with self._lock:
            skills = self._conn.execute("SELECT COUNT(*) FROM skills_meta").fetchone()[0]
            chunks = self._conn.execute("SELECT COUNT(*) FROM skill_chunks").fetchone()[0]
        return skills, chunks

    def db_size_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(self.settings.db_path) + suffix)
            if p.exists():
                total += p.stat().st_size
        return total
