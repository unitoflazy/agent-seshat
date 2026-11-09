"""Skill parsing: YAML frontmatter, legacy-file inference, heading chunker, secret scrubber."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import CHUNK_MAX_WORDS, CHUNK_TARGET_MIN_WORDS, user_home_dir

SKILL_FILENAME = "SKILL.md"
REDACTED = "[REDACTED_SECRET]"

# ---------------------------------------------------------------------------
# Secret sanitization
# ---------------------------------------------------------------------------

# (label, compiled pattern, replacement). Order matters: multi-line blocks first.
_SECRET_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
        ),
        REDACTED,
    ),
    (
        "database_url_password",
        re.compile(
            r"\b((?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss|amqps?|mssql|sqlserver|clickhouse)"
            r"(?:\+[a-z0-9]+)?://)([^:/\s@'\"]+):([^@\s/'\"]+)@",
            re.IGNORECASE,
        ),
        r"\1\2:" + REDACTED + "@",
    ),
    ("openai_anthropic_key", re.compile(r"\bsk-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_\-]{20,}"), REDACTED),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), REDACTED),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"), REDACTED),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), REDACTED),
    (
        "aws_secret_key",
        re.compile(r"(?i)(aws_secret_access_key\s*[=:]\s*['\"]?)[A-Za-z0-9/+=]{40}"),
        r"\1" + REDACTED,
    ),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"), REDACTED),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"), REDACTED),
    ("bearer_token", re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9\-._~+/]{16,}=*"), r"\1" + REDACTED),
]


def sanitize_secrets(text: str) -> tuple[str, list[str]]:
    """Mask hard-coded credentials.

    Returns ``(sanitized_text, findings)`` where ``findings`` lists the label of each
    detected secret (one entry per occurrence).
    """
    findings: list[str] = []
    for label, pattern, replacement in _SECRET_RULES:
        def _sub(match: re.Match[str], _label: str = label, _rep: str = replacement) -> str:
            if REDACTED in match.group(0):
                return match.group(0)
            findings.append(_label)
            return match.expand(_rep)

        text = pattern.sub(_sub, text)
    return text, findings


# ---------------------------------------------------------------------------
# Frontmatter & metadata inference
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A\ufeff?---[ \t]*\r?\n(.*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.DOTALL)

_STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being below
    between both but by can could did do does doing done down during each else even ever every few for from
    further get gets got had has have having he her here hers him his how however i if in into is it its
    itself just keep let like may me might more most must my no nor not now of off often on once only or
    other our ours out over own per please rather same see shall she should since so some such than that the
    their theirs them then there these they this those through thus to too under until up upon us use used
    uses using very via was way we well were what when where whether which while who whom why will with
    within without would yes yet you your yours always never instead make makes made new one two three
    file files code example examples e g eg ie etc true false null none
    """.split()
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#._\-]*[A-Za-z0-9+#]|[A-Za-z]{2,}")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return re.sub(r"-{2,}", "-", value).strip("-") or "skill"


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(metadata, body)``. Invalid YAML is treated as no frontmatter."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text.lstrip("\ufeff")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(meta, dict):
        return {}, text
    return meta, text[m.end():]


def normalize_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = re.split(r"[,\s]+", raw.strip("[] "))
    elif isinstance(raw, (list, tuple, set)):
        items = [str(x) for x in raw]
    else:
        items = [str(raw)]
    out: list[str] = []
    for item in items:
        tag = item.strip().strip("'\"").lower()
        tag = re.sub(r"\s+", "-", tag)
        if tag and tag not in out:
            out.append(tag)
    return out


def extract_keywords(text: str, limit: int = 8, exclude: set[str] | None = None) -> list[str]:
    """Cheap TF-based technical keyword extraction (no ML)."""
    exclude = {e.lower() for e in (exclude or set())}
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL) + " " + " ".join(re.findall(r"`([^`]+)`", text))
    counts: Counter[str] = Counter()
    for word in _WORD_RE.findall(text):
        w = word.lower().strip("._-")
        if len(w) < 3 or w in _STOPWORDS or w in exclude or w.isdigit():
            continue
        weight = 2 if (any(c.isupper() for c in word[1:]) or "_" in word or "." in word) else 1
        counts[w] += weight
    return [w for w, _ in counts.most_common(limit)]


def first_description(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("```", "<!--", "---")):
            continue
        stripped = re.sub(r"^#+\s*", "", stripped)
        stripped = re.sub(r"^[-*>]\s*", "", stripped).strip()
        if stripped:
            return stripped[:240]
    return "Untitled skill"


def find_git_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


_GLOBAL_AGENT_DIRS = (
    ".claude",
    ".cursor",
    ".codex",
    ".copilot",
    ".agent-seshat",
    ".codeium",
    ".windsurf",
    ".config/github-copilot",
    ".cline",
    ".roo",
)

_GATEKEEPER_BLOCK_RE = re.compile(
    r"<!--\s*SESHAT-GATEKEEPER-START\s*-->[\s\S]*?<!--\s*SESHAT-GATEKEEPER-END\s*-->\s*",
    re.MULTILINE,
)


def strip_gatekeeper_block(text: str) -> str:
    """Remove any injected Seshat Gatekeeper Protocol block before parsing/hashing."""
    return _GATEKEEPER_BLOCK_RE.sub("", text)


def normalize_globs(raw: Any) -> str:
    """Normalize Cursor `globs` or Copilot `applyTo` into a comma-separated string."""
    if raw is None or isinstance(raw, bool):
        return ""
    if isinstance(raw, (list, tuple, set)):
        parts = [str(x).strip().strip("'\"") for x in raw if str(x).strip()]
    else:
        s = str(raw).strip().strip("[]")
        parts = [p.strip().strip("'\"") for p in re.split(r"[,;]+", s) if p.strip()]
    return ", ".join(dict.fromkeys(p for p in parts if p))


def infer_source_agent(path: Path, meta: dict[str, Any] | None = None) -> str:
    """Infer `source_agent` (`claude`, `cursor`, `copilot`, `codex`, `windsurf`, `custom`)."""
    meta = meta or {}
    explicit = str(meta.get("source_agent") or "").strip().lower()
    if explicit in ("claude", "cursor", "copilot", "codex", "windsurf", "custom"):
        return explicit

    kind = str(meta.get("source_kind") or "").strip().lower()
    if "claude" in kind:
        return "claude"
    if "cursor" in kind:
        return "cursor"
    if "copilot" in kind:
        return "copilot"
    if "codex" in kind or "agents" in kind:
        return "codex"
    if any(k in kind for k in ("windsurf", "cline", "roo")):
        return "windsurf"

    check_path = str(meta.get("origin_path") or meta.get("source") or path)
    lower = check_path.replace("\\", "/").lower()
    name = Path(check_path).name
    if name == "CLAUDE.md" or "/.claude/" in lower:
        return "claude"
    if name in ("AGENTS.md", "CODEX.md") or "/.codex/" in lower:
        return "codex"
    if name == ".cursorrules" or name.endswith(".mdc") or "/.cursor/" in lower:
        return "cursor"
    if "copilot" in lower or name.endswith(".instructions.md"):
        return "copilot"
    if name in (".windsurfrules", ".clinerules", ".roorules") or any(
        seg in lower for seg in ("/.windsurf/", "/.codeium/", "/.cline/", "/.roo/")
    ):
        return "windsurf"
    return "custom"


def infer_scope(path: Path) -> str:
    """``global`` for files in user-level agent dirs, else ``project:<git-repo-name>``."""
    path = path.expanduser().resolve()
    home = user_home_dir()
    for d in _GLOBAL_AGENT_DIRS:
        try:
            if path.is_relative_to(home / d):
                return "global"
        except ValueError:
            pass
    root = find_git_root(path)
    if root is None or root == home:
        return "global"
    return f"project:{root.name}"


def normalize_scope(raw: Any) -> str:
    if not raw:
        return "global"
    scope = str(raw).strip()
    if scope.lower() == "global":
        return "global"
    if scope.lower().startswith("project:"):
        name = scope.split(":", 1)[1].strip()
        return f"project:{name}" if name else "global"
    return f"project:{scope}"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    heading: str
    text: str

    def embedding_text(self, skill_name: str, tags: list[str]) -> str:
        return f"{skill_name} [{', '.join(tags)}] - {self.heading}: {self.text}"


@dataclass
class Skill:
    slug: str
    name: str
    description: str
    tags: list[str]
    scope: str
    body: str
    file_path: Path
    resources: list[str] = field(default_factory=list)
    content_hash: str = ""
    mtime: float = 0.0
    secrets_redacted: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    source_agent: str = "custom"
    origin_path: str = ""
    file_globs: str = ""
    pinned: int = 0
    pinned_scope: str = "none"

    @property
    def skill_dir(self) -> Path | None:
        """Directory holding companion files (only for ``<slug>/SKILL.md`` layout)."""
        return self.file_path.parent if self.file_path.name == SKILL_FILENAME else None

    def chunks(self) -> list[Chunk]:
        return chunk_markdown(self.body, default_heading=self.name)

    def embedding_texts(self) -> list[tuple[Chunk, str]]:
        return [(c, c.embedding_text(self.name, self.tags)) for c in self.chunks()]


def list_resources(skill_file: Path) -> list[str]:
    """Sibling companion files of a ``<slug>/SKILL.md`` skill (recursive, relative paths)."""
    if skill_file.name != SKILL_FILENAME:
        return []
    root = skill_file.parent
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p == skill_file:
            continue
        rel = p.relative_to(root)
        if any(part.startswith(".") or part in {"__pycache__", "node_modules"} for part in rel.parts):
            continue
        out.append(rel.as_posix())
    return out


def compute_content_hash(raw: bytes, resources: list[str]) -> str:
    h = hashlib.sha256()
    h.update(raw)
    h.update(b"\0")
    h.update("\n".join(sorted(resources)).encode())
    return h.hexdigest()


def parse_skill_text(
    text: str,
    *,
    path: Path,
    slug: str | None = None,
    scope_hint: str | None = None,
) -> Skill:
    """Parse markdown text into a :class:`Skill`, inferring missing metadata."""
    cleaned_text = strip_gatekeeper_block(text)
    meta, body = split_frontmatter(cleaned_text)
    body = strip_gatekeeper_block(body)
    body, findings = sanitize_secrets(body.strip("\n"))

    if slug is None:
        slug = path.parent.name if path.name == SKILL_FILENAME else path.stem.lstrip(".")
    slug = slugify(slug)
    name = str(meta.get("name") or slug).strip()
    description = str(meta.get("description") or "").strip() or first_description(body)
    description, desc_findings = sanitize_secrets(" ".join(description.split()))
    findings += desc_findings

    tags = normalize_tags(meta.get("tags") or meta.get("keywords"))
    if not tags:
        tags = extract_keywords(f"{description}\n{body}", limit=8)

    if meta.get("scope"):
        scope = normalize_scope(meta["scope"])
    elif scope_hint:
        scope = normalize_scope(scope_hint)
    else:
        scope = "global"

    source_agent = infer_source_agent(path, meta)
    origin_path = str(meta.get("origin_path") or meta.get("source") or path.resolve())
    file_globs = normalize_globs(meta.get("file_globs") or meta.get("globs") or meta.get("applyTo"))

    always_apply = meta.get("alwaysApply") is True or str(meta.get("alwaysApply", "")).lower() == "true"
    explicit_pin = meta.get("pinned") in (True, 1, "1", "true", "True")
    pinned = 1 if (always_apply or explicit_pin) else 0
    raw_pinned_scope = str(meta.get("pinned_scope") or "").strip()
    if pinned:
        pinned_scope = normalize_scope(raw_pinned_scope) if raw_pinned_scope and raw_pinned_scope != "none" else scope
    else:
        pinned_scope = raw_pinned_scope if raw_pinned_scope else "none"

    return Skill(
        slug=slug,
        name=name,
        description=description,
        tags=tags,
        scope=scope,
        body=body,
        file_path=path,
        secrets_redacted=findings,
        meta=meta,
        source_agent=source_agent,
        origin_path=origin_path,
        file_globs=file_globs,
        pinned=pinned,
        pinned_scope=pinned_scope,
    )


def parse_skill_file(path: Path, *, slug: str | None = None, infer_scope_from_path: bool = False) -> Skill:
    """Parse a skill file on disk (``<slug>/SKILL.md`` or flat ``<slug>.md``)."""
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    scope_hint = infer_scope(path) if infer_scope_from_path else None
    skill = parse_skill_text(text, path=path, slug=slug, scope_hint=scope_hint)
    skill.resources = list_resources(path)
    skill.content_hash = compute_content_hash(raw, skill.resources)
    try:
        skill.mtime = path.stat().st_mtime
    except OSError:
        skill.mtime = 0.0
    return skill


# ---------------------------------------------------------------------------
# Heading-aware chunker
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _word_count(text: str) -> int:
    return len(text.split())


def _split_sections(body: str, default_heading: str) -> list[tuple[str, str]]:
    """Split markdown into ``(heading, text)`` sections on ``##`` / ``###`` (fence-aware)."""
    sections: list[tuple[str, list[str]]] = []
    current_heading = default_heading
    parent_h2: str | None = None
    buf: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in body.splitlines():
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            buf.append(line)
            continue
        m = None if in_fence else _HEADING_RE.match(line)
        if m and len(m.group(1)) in (2, 3):
            sections.append((current_heading, buf))
            title = m.group(2).strip()
            if len(m.group(1)) == 2:
                parent_h2 = title
                current_heading = title
            else:
                current_heading = f"{parent_h2} > {title}" if parent_h2 else title
            buf = []
            continue
        if m and len(m.group(1)) == 1 and current_heading == default_heading and not "".join(buf).strip():
            # A top-level title: don't treat as content, but keep it for context.
            continue
        buf.append(line)
    sections.append((current_heading, buf))
    return [(h, "\n".join(lines).strip()) for h, lines in sections if "\n".join(lines).strip()]


def _split_long(text: str, max_words: int) -> list[str]:
    """Split by paragraph boundaries (never inside fenced code), then by words as a last resort."""
    paragraphs: list[str] = []
    buf: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if buf:
                paragraphs.append("\n".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        paragraphs.append("\n".join(buf))

    pieces: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            pieces.append(para)
        else:
            for i in range(0, len(words), max_words):
                pieces.append(" ".join(words[i : i + max_words]))

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for piece in pieces:
        n = _word_count(piece)
        if current and count + n > max_words:
            chunks.append("\n\n".join(current))
            current, count = [], 0
        current.append(piece)
        count += n
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_markdown(
    body: str,
    *,
    default_heading: str = "Overview",
    max_words: int = CHUNK_MAX_WORDS,
    min_words: int = CHUNK_TARGET_MIN_WORDS,
) -> list[Chunk]:
    """Heading-aware chunking sized for BGE's 512-token window.

    Each ``##``/``###`` section becomes its own chunk (so retrieval can report the exact
    matching heading); sections longer than ``max_words`` are split on paragraph
    boundaries into ~``min_words``-``max_words`` pieces.
    """
    chunks: list[Chunk] = []
    for heading, text in _split_sections(body, default_heading):
        if _word_count(text) <= max_words:
            chunks.append(Chunk(heading=heading, text=text))
            continue
        parts = _split_long(text, max_words)
        # Fold a tiny trailing remainder into the previous part when it still fits.
        if len(parts) > 1 and _word_count(parts[-1]) < min_words // 3:
            merged = parts[-2] + "\n\n" + parts[-1]
            if _word_count(merged) <= int(max_words * 1.15):
                parts = parts[:-2] + [merged]
        for i, part in enumerate(parts, 1):
            chunks.append(Chunk(heading=heading if len(parts) == 1 else f"{heading} (part {i})", text=part))
    if not chunks:
        chunks.append(Chunk(heading=default_heading, text=body.strip() or default_heading))
    return chunks
