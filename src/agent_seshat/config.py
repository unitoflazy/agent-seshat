"""Central configuration: filesystem layout, model names, and retrieval thresholds.

Everything is resolved lazily from the environment so tests (and users) can point
``AGENT_SESHAT_HOME`` at a different directory at any time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "agent-seshat"
SERVER_NAME = "agent-seshat"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# Retrieval tuning (see search.py for how they are used).
RRF_K = 60
SEMANTIC_CUTOFF = 0.50  # hard floor: below this a chunk never enters semantic ranking
# BGE-small cosine scores are compressed (unrelated text often lands at 0.50-0.60), so two
# extra guards keep noise out of RRF:
SEMANTIC_ONLY_CUTOFF = 0.66  # a skill with *no* lexical/exact evidence must clear this
SEMANTIC_MARGIN = 0.15  # semantic hits must be within this distance of the best hit
EXACT_MATCH_BOOST = 0.025
SCOPE_SAME_PROJECT = 1.35
SCOPE_GLOBAL = 1.0
SCOPE_OTHER_PROJECT = 0.40
BM25_WEIGHTS = (10.0, 8.0, 5.0, 1.0)  # name, tags, description, body

# Chunking.
CHUNK_MAX_WORDS = 400
CHUNK_TARGET_MIN_WORDS = 150

# Directories never descended into by the harvester / skill scanner.
IGNORED_DIRS = frozenset(
    {
        "node_modules",
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        ".gradle",
        ".idea",
        "vendor",
        "Pods",
        "DerivedData",
        "Library",
        ".Trash",
        "AppData",
        "Application Data",
        "Local Settings",
        "site-packages",
        ".cargo",
        ".rustup",
        ".npm",
        ".pnpm-store",
        ".yarn",
        "go",
    }
)

VALID_SOURCE_AGENTS = ("claude", "cursor", "copilot", "codex", "windsurf", "custom")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings."""

    home: Path
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    semantic_cutoff: float = SEMANTIC_CUTOFF
    semantic_only_cutoff: float = SEMANTIC_ONLY_CUTOFF
    semantic_margin: float = SEMANTIC_MARGIN
    model_name: str = EMBEDDING_MODEL
    extra: dict = field(default_factory=dict)

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def db_path(self) -> Path:
        return self.cache_dir / "index.db"

    @property
    def mcp_registry_path(self) -> Path:
        return self.home / "mcp_registry.json"

    @property
    def env_store_path(self) -> Path:
        return self.home / "env_store.json"

    @property
    def daemon_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def ensure_dirs(self) -> "Settings":
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self


def seshat_home() -> Path:
    raw = os.environ.get("AGENT_SESHAT_HOME")
    return Path(raw).expanduser().resolve() if raw else (Path.home() / ".agent-seshat")


def get_settings() -> Settings:
    """Build settings from the current environment (cheap; call freely)."""
    return Settings(
        home=seshat_home(),
        host=os.environ.get("AGENT_SESHAT_HOST", DEFAULT_HOST),
        port=_env_int("AGENT_SESHAT_PORT", DEFAULT_PORT),
        semantic_cutoff=_env_float("AGENT_SESHAT_SEMANTIC_CUTOFF", SEMANTIC_CUTOFF),
        semantic_only_cutoff=_env_float("AGENT_SESHAT_SEMANTIC_ONLY_CUTOFF", SEMANTIC_ONLY_CUTOFF),
        semantic_margin=_env_float("AGENT_SESHAT_SEMANTIC_MARGIN", SEMANTIC_MARGIN),
        model_name=os.environ.get("AGENT_SESHAT_MODEL", EMBEDDING_MODEL),
    )


def fastembed_cache_dir() -> str | None:
    """Where FastEmbed stores ONNX weights (pre-baked to /opt/fastembed_cache in Docker)."""
    raw = os.environ.get("FASTEMBED_CACHE_PATH")
    if raw:
        return raw
    default = Path.home() / ".cache" / "fastembed"
    return str(default)
