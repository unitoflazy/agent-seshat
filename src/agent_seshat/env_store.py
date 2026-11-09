"""Scoped Environment Variable & Secret Store (`env_store.py`).

Stores and retrieves environment variables across scopes:
- `global` (available to all projects and downstream MCP servers)
- `project:<repo-name>` (scoped to a specific repository/workspace, overriding `global`)
- `mcp:<server-name>` (scoped to a specific downstream MCP server in `mcp-hub`)

Persists data in both SQLite (`env_vars` table in `index.db`) and `~/.agent-seshat/env_store.json`
(mode `0600`) so variables survive cache rebuilds.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .indexer import SkillIndex
from .parser import sanitize_secrets

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_SECRET_KEY_PATTERNS = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "APIKEY",
    "PRIVATE_KEY",
    "ACCESS_KEY",
    "CLIENT_SECRET",
    "AUTH_KEY",
    "CREDENTIALS",
    "DATABASE_URL",
    "REDIS_URL",
    "MONGO_URI",
    "WEBHOOK_SECRET",
    "SIGNING_KEY",
)


def validate_env_key(key: str) -> str:
    cleaned = (key or "").strip()
    if not _ENV_KEY_RE.match(cleaned):
        raise ValueError(
            f"Invalid environment variable name {key!r}. Must match [A-Za-z_][A-Za-z0-9_]*."
        )
    return cleaned


def normalize_env_scope(raw: str | None) -> str:
    """Normalize scope into `'global'`, `'project:<name>'`, or `'mcp:<server>'`."""
    if not raw:
        return "global"
    s = str(raw).strip()
    lower = s.lower()
    if lower in ("global", "all", "*"):
        return "global"
    if lower.startswith("project:"):
        name = s.split(":", 1)[1].strip()
        return f"project:{name}" if name else "global"
    if lower.startswith("mcp:"):
        srv = s.split(":", 1)[1].strip().lower()
        return f"mcp:{srv}" if srv else "global"
    return f"project:{s}"


def looks_like_secret(key: str, value: str) -> bool:
    """Heuristic detection for sensitive environment variables when `is_secret` is not specified."""
    upper = key.upper()
    if any(pat in upper for pat in _SECRET_KEY_PATTERNS) or upper.endswith("_KEY"):
        return True
    _, findings = sanitize_secrets(f"{key}={value}")
    return bool(findings)


def mask_value(value: str) -> str:
    """Mask a secret value for safe display in tables and LLM listings."""
    if not value:
        return "********"
    if len(value) <= 8:
        return "********"
    if len(value) <= 14:
        return f"{value[:2]}...{value[-2:]}"
    return f"{value[:4]}...{value[-4:]}"


@dataclass
class EnvVarRecord:
    key: str
    scope: str
    value: str
    description: str = ""
    is_secret: bool = False
    updated_at: float = 0.0

    @property
    def masked_value(self) -> str:
        return mask_value(self.value) if self.is_secret else self.value

    def display_value(self, reveal_secret: bool = False) -> str:
        if self.is_secret and not reveal_secret:
            return self.masked_value
        return self.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_dotenv_text(text: str) -> list[tuple[str, str, str]]:
    """Parse `.env` file content into `[(key, value, description)]`."""
    entries: list[tuple[str, str, str]] = []
    pending_comment: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            pending_comment.clear()
            continue
        if line.startswith("#"):
            comment_text = line.lstrip("#").strip()
            if comment_text:
                pending_comment.append(comment_text)
            continue

        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        if "=" not in line:
            pending_comment.clear()
            continue

        key_part, val_part = line.split("=", 1)
        key = key_part.strip()
        if not _ENV_KEY_RE.match(key):
            pending_comment.clear()
            continue

        val_str = val_part.strip()
        inline_comment = ""

        if val_str.startswith(('"', "'")):
            quote_char = val_str[0]
            end_idx = -1
            escaped = False
            for i in range(1, len(val_str)):
                ch = val_str[i]
                if escaped:
                    escaped = False
                    continue
                if ch == "\\" and quote_char == '"':
                    escaped = True
                    continue
                if ch == quote_char:
                    end_idx = i
                    break
            if end_idx != -1:
                raw_quoted = val_str[1:end_idx]
                if quote_char == '"':
                    raw_quoted = (
                        raw_quoted.replace('\\"', '"')
                        .replace("\\n", "\n")
                        .replace("\\t", "\t")
                        .replace("\\\\", "\\")
                    )
                rest = val_str[end_idx + 1 :].strip()
                if rest.startswith("#"):
                    inline_comment = rest.lstrip("#").strip()
                val_str = raw_quoted
            else:
                val_str = val_str.strip(quote_char)
        else:
            # Unquoted value: check for inline ` # comment`
            if " #" in val_str:
                val_str, comment_part = val_str.split(" #", 1)
                val_str = val_str.strip()
                inline_comment = comment_part.strip()

        desc = inline_comment or (" ".join(pending_comment) if pending_comment else "")
        pending_comment.clear()
        entries.append((key, val_str, desc))

    return entries


def format_dotenv_value(value: str) -> str:
    if value == "":
        return '""'
    if any(ch in value for ch in (' ', '\t', '\n', '"', "'", '#', '$', '=')):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return value


class EnvStore:
    """Manages scoped environment variables in SQLite (`env_vars`) and `env_store.json`."""

    def __init__(self, index: SkillIndex | None = None, settings: Settings | None = None):
        self.settings = (settings or (index.settings if index else get_settings())).ensure_dirs()
        self.index = index or SkillIndex(self.settings)
        self._sync_from_json_if_empty()

    @property
    def json_path(self) -> Path:
        return self.settings.env_store_path

    def _sync_from_json_if_empty(self) -> None:
        """If the SQLite table is empty (e.g. after a DB rebuild) and `env_store.json` exists, restore it."""
        if not self.json_path.is_file():
            return
        with self.index._lock:
            count = self.index.conn.execute("SELECT COUNT(*) FROM env_vars").fetchone()[0]
            if count > 0:
                return
            try:
                raw = json.loads(self.json_path.read_text(encoding="utf-8"))
                items = raw.get("variables", []) if isinstance(raw, dict) else []
            except Exception:
                return
            if not items:
                return
            conn = self.index.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                for item in items:
                    if not isinstance(item, dict) or not item.get("key"):
                        continue
                    conn.execute(
                        """INSERT INTO env_vars (key, scope, value, description, is_secret, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(key, scope) DO UPDATE SET
                             value=excluded.value,
                             description=excluded.description,
                             is_secret=excluded.is_secret,
                             updated_at=excluded.updated_at""",
                        (
                            str(item["key"]),
                            normalize_env_scope(item.get("scope")),
                            str(item.get("value", "")),
                            str(item.get("description", "")),
                            1 if item.get("is_secret") else 0,
                            float(item.get("updated_at") or time.time()),
                        ),
                    )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def _persist_json(self) -> None:
        """Write current `env_vars` table to `~/.agent-seshat/env_store.json` with 0600 permissions."""
        records = self.list_all()
        payload = {
            "version": "2.0",
            "variables": [r.to_dict() for r in records],
        }
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        try:
            os.chmod(self.json_path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _row_to_record(row: Any) -> EnvVarRecord:
        return EnvVarRecord(
            key=row["key"],
            scope=row["scope"],
            value=row["value"],
            description=row["description"] or "",
            is_secret=bool(row["is_secret"]),
            updated_at=float(row["updated_at"]),
        )

    def set(
        self,
        key: str,
        value: str,
        *,
        scope: str = "global",
        description: str = "",
        is_secret: bool | None = None,
    ) -> EnvVarRecord:
        """Create or update an environment variable."""
        norm_key = validate_env_key(key)
        norm_scope = normalize_env_scope(scope)
        secret_flag = looks_like_secret(norm_key, value) if is_secret is None else bool(is_secret)
        now = time.time()

        with self.index._lock:
            if not description:
                existing = self.index.conn.execute(
                    "SELECT description FROM env_vars WHERE key=? AND scope=?",
                    (norm_key, norm_scope),
                ).fetchone()
                if existing and existing["description"]:
                    description = existing["description"]

            self.index.conn.execute(
                """INSERT INTO env_vars (key, scope, value, description, is_secret, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(key, scope) DO UPDATE SET
                     value=excluded.value,
                     description=excluded.description,
                     is_secret=excluded.is_secret,
                     updated_at=excluded.updated_at""",
                (norm_key, norm_scope, value, description.strip(), 1 if secret_flag else 0, now),
            )
            self._persist_json()

        return EnvVarRecord(
            key=norm_key,
            scope=norm_scope,
            value=value,
            description=description.strip(),
            is_secret=secret_flag,
            updated_at=now,
        )

    def get(
        self,
        key: str,
        *,
        scope: str | None = None,
        project: str = "",
        fallback_global: bool = True,
    ) -> EnvVarRecord | None:
        """Retrieve a variable by `key`.

        If `scope` is explicitly provided, looks up that exact scope (plus `global` if `fallback_global`).
        Otherwise, if `project` is provided, checks `project:<project>` first and falls back to `global`.
        """
        norm_key = (key or "").strip()
        if not norm_key:
            return None

        scopes_to_check: list[str] = []
        if scope:
            norm_scope = normalize_env_scope(scope)
            scopes_to_check.append(norm_scope)
            if fallback_global and norm_scope != "global":
                scopes_to_check.append("global")
        else:
            if project:
                proj_scope = normalize_env_scope(f"project:{project}" if not project.startswith("project:") else project)
                scopes_to_check.append(proj_scope)
            if fallback_global or not scopes_to_check:
                scopes_to_check.append("global")

        with self.index._lock:
            for sc in scopes_to_check:
                row = self.index.conn.execute(
                    "SELECT * FROM env_vars WHERE key=? AND LOWER(scope)=LOWER(?)",
                    (norm_key, sc),
                ).fetchone()
                if row is not None:
                    return self._row_to_record(row)
        return None

    def unset(self, key: str, scope: str = "global") -> bool:
        """Delete an environment variable from `scope`. Returns True if a row was deleted."""
        norm_key = (key or "").strip()
        norm_scope = normalize_env_scope(scope)
        with self.index._lock:
            cur = self.index.conn.execute(
                "DELETE FROM env_vars WHERE key=? AND LOWER(scope)=LOWER(?)",
                (norm_key, norm_scope),
            )
            deleted = cur.rowcount > 0
            if deleted:
                self._persist_json()
            return deleted

    def list_all(self) -> list[EnvVarRecord]:
        with self.index._lock:
            rows = self.index.conn.execute(
                "SELECT * FROM env_vars ORDER BY scope, key"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list(
        self,
        *,
        scope: str | None = None,
        project: str = "",
        include_global: bool = True,
    ) -> list[EnvVarRecord]:
        """List stored environment variables filtered by `scope` or `project`."""
        all_vars = self.list_all()
        if scope:
            target_scope = normalize_env_scope(scope).lower()
            allowed = {target_scope}
            if include_global and target_scope != "global":
                allowed.add("global")
            return [r for r in all_vars if r.scope.lower() in allowed]

        if project:
            proj_scope = normalize_env_scope(f"project:{project}" if not project.startswith("project:") else project).lower()
            allowed = {proj_scope}
            if include_global:
                allowed.add("global")
            return [r for r in all_vars if r.scope.lower() in allowed]

        return all_vars

    def resolve_for_project(self, project: str = "") -> dict[str, EnvVarRecord]:
        """Return merged `key -> EnvVarRecord` dictionary where `project:<project>` overrides `global`."""
        merged: dict[str, EnvVarRecord] = {}
        all_vars = self.list_all()
        for r in all_vars:
            if r.scope.lower() == "global":
                merged[r.key] = r
        if project:
            proj_scope = normalize_env_scope(f"project:{project}" if not project.startswith("project:") else project).lower()
            for r in all_vars:
                if r.scope.lower() == proj_scope:
                    merged[r.key] = r
        return dict(sorted(merged.items()))

    def resolve_for_mcp_server(self, server_name: str, project: str = "") -> dict[str, str]:
        """Return merged `key -> value` strings for spawning a downstream MCP server:
        `global` -> `project:<project>` -> `mcp:<server_name>`."""
        merged = {k: rec.value for k, rec in self.resolve_for_project(project).items()}
        mcp_scope = f"mcp:{server_name.strip().lower()}"
        for r in self.list_all():
            if r.scope.lower() == mcp_scope:
                merged[r.key] = r.value
        return merged

    def import_dotenv(
        self,
        dotenv_path: Path | str,
        *,
        scope: str = "global",
        secret: bool | None = None,
    ) -> list[EnvVarRecord]:
        """Import variables from a `.env` file into `scope`."""
        path = Path(dotenv_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f".env file not found: {path}")
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_dotenv_text(raw_text)
        imported: list[EnvVarRecord] = []
        for key, value, desc in parsed:
            rec = self.set(key, value, scope=scope, description=desc, is_secret=secret)
            imported.append(rec)
        return imported

    def export_dotenv(
        self,
        *,
        scope: str | None = None,
        project: str = "",
        include_global: bool = True,
        mask_secrets: bool = False,
    ) -> str:
        """Render matching environment variables as a valid `.env` file."""
        if project and not scope:
            records = list(self.resolve_for_project(project).values())
        elif scope:
            records = self.list(scope=scope, include_global=include_global)
        else:
            records = self.list_all()

        lines: list[str] = ["# Generated by Agent-Seshat (sesh env export)"]
        for r in records:
            if r.description:
                lines.append(f"# {r.description} (scope: {r.scope})")
            val = r.masked_value if (mask_secrets and r.is_secret) else r.value
            lines.append(f"{r.key}={format_dotenv_value(val)}")
        return "\n".join(lines) + "\n"
