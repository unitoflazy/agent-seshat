"""High-speed deep filesystem & config scanner (`sesh deep-sync`) for multi-agent rules and skills.

Discovers, normalizes, deduplicates, and syncs rules/skills across:
- Claude Code & Desktop (`CLAUDE.md`, `.claude/CLAUDE.md`, `.claude/commands/*.md`, `~/.claude/skills/**`)
- OpenAI Codex (`AGENTS.md`, `CODEX.md`, `~/.codex/AGENTS.md`, `~/.codex/instructions.md`, `~/.codex/skills/**`)
- Cursor (`.cursorrules`, `.cursor/rules/*.mdc` with `description`, `globs`, `alwaysApply`, `~/.cursor/rules/**`)
- GitHub Copilot (`.github/copilot-instructions.md`, `.github/instructions/*.instructions.md` with `applyTo`, `~/.config/github-copilot/**`)
- Windsurf / Cline / Roo (`.windsurfrules`, `.clinerules`, `.roorules`, `.windsurf/rules/*.md`)
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import IGNORED_DIRS, Settings, get_settings, user_home_dir
from .harvester import MAX_COMPANION_BYTES, MAX_FILE_BYTES, STUB_MARKER, _copy_companions, render_skill
from .indexer import SkillIndex, SkillRecord
from .parser import (
    SKILL_FILENAME,
    extract_keywords,
    find_git_root,
    first_description,
    normalize_globs,
    normalize_scope,
    normalize_tags,
    sanitize_secrets,
    slugify,
    split_frontmatter,
    strip_gatekeeper_block,
)

AGENT_PRIORITY = {
    "claude": 0,
    "cursor": 1,
    "copilot": 2,
    "codex": 3,
    "windsurf": 4,
    "custom": 5,
}


@dataclass
class ScannedRule:
    origin_path: Path
    source_agent: str  # claude | cursor | copilot | codex | windsurf | custom
    rule_kind: str
    scope: str  # global | project:<repo-name>
    workspace_root: Path | None
    slug: str
    name: str
    description: str
    tags: list[str]
    file_globs: str
    pinned: int
    pinned_scope: str
    body: str
    content_hash: str
    secrets_redacted: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    is_skill_bundle: bool = False


@dataclass
class DeepSyncItem:
    slug: str
    source_agent: str
    rule_kind: str
    scope: str
    origin_path: Path
    status: str  # "added" | "updated" | "unchanged" | "duplicate" | "skipped" | "error"
    pinned: bool = False
    pinned_scope: str = "none"
    file_globs: str = ""
    secrets_redacted: list[str] = field(default_factory=list)
    workspaces: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class DeepSyncReport:
    items: list[DeepSyncItem] = field(default_factory=list)
    gatekeeper_files: list[Path] = field(default_factory=list)
    elapsed_ms: float = 0.0
    total_skills: int = 0
    total_chunks: int = 0

    @property
    def secrets_total(self) -> int:
        return sum(len(i.secrets_redacted) for i in self.items)


def _is_stub_file(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            head = fh.read(512)
        return STUB_MARKER in head or "00-seshat-gatekeeper" in path.name
    except OSError:
        return True


def _looks_like_home_dir(directory: Path, user_home: Path) -> bool:
    if directory == user_home:
        return True
    lower_name = directory.name.lower()
    if lower_name in ("home", "user_home", "fake_home", "users"):
        return True
    # A directory that only holds dot-agent folders (.claude, .codex, .cursor, .config) and no project markers
    project_markers = (".git", ".github", "pyproject.toml", "package.json", "go.mod", "Cargo.toml", "src")
    if any((directory / m).exists() for m in project_markers):
        return False
    agent_folders = (".claude", ".codex", ".cursor", ".copilot", ".config", ".windsurf", ".codeium")
    has_agent_folder = any((directory / a).is_dir() for a in agent_folders)
    has_project_rule = any((directory / f).is_file() for f in ("CLAUDE.md", "AGENTS.md", "CODEX.md", ".cursorrules"))
    return has_agent_folder and not has_project_rule


def classify_rule_file(path: Path, user_home: Path | None = None) -> tuple[str, str, Path | None, str] | None:
    """Classify `path` into `(source_agent, rule_kind, workspace_root, scope)` or `None`."""
    user_home = (user_home or user_home_dir()).expanduser().resolve()
    name = path.name
    parts = path.parts
    parent_name = path.parent.name

    if name == "00-seshat-gatekeeper.mdc":
        return None

    # Helper to resolve workspace_root & scope given a candidate project root dir.
    def _resolve_scope(candidate_root: Path | None, force_global: bool = False) -> tuple[Path | None, str]:
        if force_global:
            return None, "global"
        try:
            if path.is_relative_to(user_home):
                rel = path.relative_to(user_home)
                if rel.parts and rel.parts[0] in (
                    ".claude",
                    ".codex",
                    ".cursor",
                    ".copilot",
                    ".config",
                    ".codeium",
                    ".windsurf",
                    ".cline",
                    ".roo",
                    ".agents",
                ):
                    return None, "global"
        except ValueError:
            pass

        git_root = find_git_root(path)
        if git_root is not None and git_root != user_home:
            return git_root, f"project:{git_root.name}"

        if candidate_root is not None:
            if _looks_like_home_dir(candidate_root, user_home):
                return None, "global"
            return candidate_root, f"project:{candidate_root.name}"
        return None, "global"

    # 1. Claude Code & Desktop
    if name == "CLAUDE.md":
        if parent_name == ".claude":
            ws_root, scope = _resolve_scope(path.parent.parent)
            return ("claude", "claude-md", ws_root, scope)
        ws_root, scope = _resolve_scope(path.parent)
        return ("claude", "claude-md", ws_root, scope)

    if ".claude" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index(".claude")
        claude_parent = Path(*parts[:idx]) if idx > 0 else None
        sub_parts = parts[idx + 1 :]
        if sub_parts and sub_parts[0] == "commands" and name.endswith(".md"):
            ws_root, scope = _resolve_scope(claude_parent)
            return ("claude", "claude-command", ws_root, scope)
        if sub_parts and sub_parts[0] == "skills" and name.lower() == "skill.md":
            ws_root, scope = _resolve_scope(claude_parent)
            return ("claude", "claude-skill", ws_root, scope)

    # 2. OpenAI Codex
    if name in ("AGENTS.md", "CODEX.md"):
        if parent_name == ".codex":
            ws_root, scope = _resolve_scope(path.parent.parent)
            return ("codex", "codex-md", ws_root, scope)
        ws_root, scope = _resolve_scope(path.parent)
        return ("codex", "codex-md", ws_root, scope)

    if ".codex" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index(".codex")
        codex_parent = Path(*parts[:idx]) if idx > 0 else None
        sub_parts = parts[idx + 1 :]
        if name == "instructions.md":
            return ("codex", "codex-instructions", None, "global")
        if sub_parts and sub_parts[0] == "skills" and name.lower() == "skill.md":
            ws_root, scope = _resolve_scope(codex_parent)
            return ("codex", "codex-skill", ws_root, scope)

    # 3. Cursor
    if name == ".cursorrules":
        ws_root, scope = _resolve_scope(path.parent)
        return ("cursor", "cursorrules", ws_root, scope)

    if ".cursor" in parts and "rules" in parts:
        if name.endswith((".mdc", ".md")):
            idx = len(parts) - 1 - list(reversed(parts)).index(".cursor")
            cursor_parent = Path(*parts[:idx]) if idx > 0 else None
            ws_root, scope = _resolve_scope(cursor_parent)
            return ("cursor", "cursor-mdc", ws_root, scope)

    # 4. GitHub Copilot
    if name == "copilot-instructions.md" and parent_name in (".github", ".copilot", "github-copilot"):
        ws_Candidate = path.parent.parent if parent_name == ".github" else None
        ws_root, scope = _resolve_scope(ws_Candidate, force_global=(parent_name != ".github"))
        return ("copilot", "copilot-instructions", ws_root, scope)

    if name.endswith(".instructions.md") and (".github" in parts or "github-copilot" in parts or ".copilot" in parts):
        if ".github" in parts:
            idx = len(parts) - 1 - list(reversed(parts)).index(".github")
            gh_parent = Path(*parts[:idx]) if idx > 0 else None
            ws_root, scope = _resolve_scope(gh_parent)
        else:
            ws_root, scope = None, "global"
        return ("copilot", "copilot-path-instructions", ws_root, scope)

    if "github-copilot" in parts and name.endswith(".md"):
        return ("copilot", "copilot-global", None, "global")

    # 5. Windsurf / Cline / Roo
    if name in (".windsurfrules", ".clinerules", ".roorules") and path.is_file():
        ws_root, scope = _resolve_scope(path.parent)
        return ("windsurf", name.lstrip("."), ws_root, scope)

    if parent_name in (".clinerules", ".roorules") and name.endswith(".md"):
        ws_root, scope = _resolve_scope(path.parent.parent)
        return ("windsurf", parent_name.lstrip("."), ws_root, scope)

    if ".windsurf" in parts and "rules" in parts and name.endswith((".md", ".mdc")):
        idx = len(parts) - 1 - list(reversed(parts)).index(".windsurf")
        ws_parent = Path(*parts[:idx]) if idx > 0 else None
        ws_root, scope = _resolve_scope(ws_parent)
        return ("windsurf", "windsurf-rule", ws_root, scope)

    # 6. Generic SKILL.md inside skills/
    if name == SKILL_FILENAME and "skills" in parts:
        ws_root, scope = _resolve_scope(path.parent.parent)
        return ("custom", "agent-skill", ws_root, scope)

    return None


def _fast_walk(root: Path, out: list[Path], library_dir: Path) -> None:
    """Fast `os.scandir` recursive walker with strict directory pruning (<2s on large trees)."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            if current.resolve().is_relative_to(library_dir):
                continue
        except OSError:
            continue
        try:
            with os.scandir(current) as it:
                entries = list(it)
        except OSError:
            continue

        for entry in entries:
            ename = entry.name
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                continue

            if is_dir:
                if ename in IGNORED_DIRS:
                    continue
                if ename.startswith(".") and ename not in (
                    ".claude",
                    ".codex",
                    ".cursor",
                    ".github",
                    ".copilot",
                    ".config",
                    ".windsurf",
                    ".codeium",
                    ".clinerules",
                    ".roorules",
                    ".agents",
                ):
                    continue
                stack.append(Path(entry.path))
            elif is_file:
                if ename in (
                    "CLAUDE.md",
                    "AGENTS.md",
                    "CODEX.md",
                    ".cursorrules",
                    ".windsurfrules",
                    ".clinerules",
                    ".roorules",
                    "copilot-instructions.md",
                    "instructions.md",
                    SKILL_FILENAME,
                ) or ename.endswith((".mdc", ".instructions.md", ".md")):
                    out.append(Path(entry.path))


def default_deep_scan_roots(user_home: Path | None = None) -> list[Path]:
    home = (user_home or user_home_dir()).expanduser().resolve()
    roots: list[Path] = [Path.cwd().resolve()]
    for candidate in (
        home / ".claude",
        home / ".codex",
        home / ".cursor",
        home / ".copilot",
        home / ".config" / "github-copilot",
        home / ".windsurf",
        home / ".codeium" / "windsurf",
        home / ".clinerules",
        home / ".roorules",
    ):
        if candidate.exists():
            roots.append(candidate)
    return roots


def _propose_rule_slug(
    origin_path: Path,
    source_agent: str,
    rule_kind: str,
    scope: str,
    meta: dict[str, Any],
) -> str:
    if meta.get("name"):
        return slugify(str(meta["name"]))

    repo = scope.split(":", 1)[1] if scope.startswith("project:") else "global"
    stem = origin_path.name
    for suf in (".instructions.md", ".mdc", ".md"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    stem = stem.lstrip(".")

    if rule_kind in ("claude-skill", "codex-skill", "agent-skill"):
        return slugify(origin_path.parent.name)
    if rule_kind in ("cursor-mdc", "claude-command", "copilot-path-instructions", "windsurf-rule"):
        prefix = f"{repo}-" if repo != "global" else f"{source_agent}-"
        return slugify(f"{prefix}{stem}")
    if rule_kind == "codex-instructions":
        return "codex-global-instructions"
    return slugify(f"{repo}-{stem}")


def parse_discovered_rule(
    origin_path: Path,
    source_agent: str,
    rule_kind: str,
    scope: str,
    workspace_root: Path | None,
    *,
    always_pin: bool = False,
) -> ScannedRule | None:
    try:
        if origin_path.stat().st_size > MAX_FILE_BYTES:
            return None
        raw = origin_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    cleaned = strip_gatekeeper_block(raw)
    meta, body = split_frontmatter(cleaned)
    body = strip_gatekeeper_block(body).strip()
    if not body:
        return None

    body, findings = sanitize_secrets(body)
    effective_scope = normalize_scope(meta["scope"]) if meta.get("scope") else scope
    slug = _propose_rule_slug(origin_path, source_agent, rule_kind, effective_scope, meta)
    name = str(meta.get("name") or slug).strip()

    desc = str(meta.get("description") or "").strip()
    if not desc:
        desc = first_description(body)
    desc, desc_findings = sanitize_secrets(" ".join(desc.split()))
    findings += desc_findings

    file_globs = normalize_globs(meta.get("file_globs") or meta.get("globs") or meta.get("applyTo"))
    always_apply = meta.get("alwaysApply") is True or str(meta.get("alwaysApply", "")).lower() == "true"
    explicit_pin = meta.get("pinned") in (True, 1, "1", "true", "True") or always_pin
    pinned = 1 if (always_apply or explicit_pin) else 0
    if pinned:
        raw_ps = str(meta.get("pinned_scope") or "").strip()
        pinned_scope = normalize_scope(raw_ps) if raw_ps and raw_ps != "none" else effective_scope
    else:
        pinned_scope = "none"

    tags = normalize_tags(meta.get("tags") or meta.get("keywords"))
    if not tags:
        tags = extract_keywords(f"{desc}\n{body}", limit=8)
    if source_agent not in tags:
        tags.append(source_agent)
    if effective_scope.startswith("project:"):
        repo_tag = effective_scope.split(":", 1)[1].lower()
        if repo_tag not in tags:
            tags.append(repo_tag)

    norm_body = "\n".join(line.rstrip() for line in body.splitlines())
    content_hash = hashlib.sha256(norm_body.encode("utf-8")).hexdigest()

    return ScannedRule(
        origin_path=origin_path.resolve(),
        source_agent=source_agent,
        rule_kind=rule_kind,
        scope=effective_scope,
        workspace_root=workspace_root,
        slug=slug,
        name=name,
        description=desc,
        tags=tags,
        file_globs=file_globs,
        pinned=pinned,
        pinned_scope=pinned_scope,
        body=body,
        content_hash=content_hash,
        secrets_redacted=findings,
        meta=meta,
        is_skill_bundle=(origin_path.name == SKILL_FILENAME),
    )


def deep_scan(
    paths: list[Path] | None = None,
    *,
    user_home: Path | None = None,
    settings: Settings | None = None,
    always_pin: bool = False,
) -> list[ScannedRule]:
    """Scan directories for all supported multi-agent rule and skill files."""
    settings = settings or get_settings()
    library_dir = settings.home.resolve()
    roots = [Path(p).expanduser().resolve() for p in (paths or default_deep_scan_roots(user_home))]

    candidate_files: list[Path] = []
    for root in roots:
        if root.is_file():
            candidate_files.append(root)
        elif root.is_dir():
            _fast_walk(root, candidate_files, library_dir)

    seen_paths: set[Path] = set()
    discovered: list[ScannedRule] = []
    for file_path in candidate_files:
        try:
            real = file_path.resolve()
        except OSError:
            continue
        if real in seen_paths or _is_stub_file(real):
            continue
        seen_paths.add(real)
        classified = classify_rule_file(real, user_home=user_home)
        if classified is None:
            continue
        source_agent, rule_kind, ws_root, scope = classified
        rule = parse_discovered_rule(
            real,
            source_agent,
            rule_kind,
            scope,
            ws_root,
            always_pin=always_pin,
        )
        if rule is not None:
            discovered.append(rule)

    discovered.sort(key=lambda r: (AGENT_PRIORITY.get(r.source_agent, 99), str(r.origin_path)))
    return discovered


def deep_sync(
    paths: list[Path] | None = None,
    *,
    user_home: Path | None = None,
    settings: Settings | None = None,
    index: SkillIndex | None = None,
    enforce_gatekeeper: bool = False,
    always_pin: bool = False,
    dry_run: bool = False,
) -> DeepSyncReport:
    """Discover all agent rules/skills, deduplicate by content_hash, write to library, and index in SQLite+ONNX."""
    t0 = time.perf_counter()
    settings = (settings or (index.settings if index else get_settings())).ensure_dirs()
    idx = index or SkillIndex(settings)
    report = DeepSyncReport()

    rules = deep_scan(paths, user_home=user_home, settings=settings, always_pin=always_pin)

    # Also re-check any previously synced rules via their `origin_path` when no explicit paths filter is given.
    existing_records = {r.slug: r for r in idx.list_skills()}
    if not paths:
        scanned_origins = {r.origin_path for r in rules}
        for rec in existing_records.values():
            if rec.origin_path:
                op = Path(rec.origin_path)
                if op.is_file() and op.resolve() not in scanned_origins:
                    classified = classify_rule_file(op.resolve(), user_home=user_home)
                    if classified:
                        sa, rk, wr, sc = classified
                        parsed = parse_discovered_rule(op.resolve(), sa, rk, sc, wr, always_pin=always_pin)
                        if parsed:
                            rules.append(parsed)
                            scanned_origins.add(op.resolve())

    by_hash: dict[str, DeepSyncItem] = {}
    used_slugs: dict[str, Path] = {
        slug: Path(rec.origin_path) if rec.origin_path else rec.file_path
        for slug, rec in existing_records.items()
    }
    workspace_roots: set[Path] = set()

    for rule in rules:
        if rule.workspace_root:
            workspace_roots.add(rule.workspace_root)

        if rule.content_hash in by_hash:
            primary = by_hash[rule.content_hash]
            if rule.scope not in primary.workspaces:
                primary.workspaces.append(rule.scope)
            # If the duplicate is pinned (e.g. alwaysApply: true), propagate pin to primary
            if rule.pinned and not primary.pinned:
                primary.pinned = True
                primary.pinned_scope = rule.pinned_scope
                if not dry_run:
                    idx.pin_skill(primary.slug, scope=rule.pinned_scope)
            report.items.append(
                DeepSyncItem(
                    slug=primary.slug,
                    source_agent=rule.source_agent,
                    rule_kind=rule.rule_kind,
                    scope=rule.scope,
                    origin_path=rule.origin_path,
                    status="duplicate",
                    pinned=bool(rule.pinned),
                    pinned_scope=rule.pinned_scope,
                    file_globs=rule.file_globs,
                    secrets_redacted=rule.secrets_redacted,
                    workspaces=list(primary.workspaces),
                    note=f"deduplicated into {primary.slug}",
                )
            )
            continue

        slug = rule.slug
        if slug in used_slugs and used_slugs[slug].resolve() != rule.origin_path.resolve():
            base = slugify(f"{rule.source_agent}-{slug}")
            candidate, n = base, 2
            while candidate in used_slugs and used_slugs[candidate].resolve() != rule.origin_path.resolve():
                candidate, n = f"{base}-{n}", n + 1
            slug = candidate
        used_slugs[slug] = rule.origin_path

        prev_rec: SkillRecord | None = existing_records.get(slug)
        # Preserve existing manual pin if rule file wasn't explicitly pinned
        pinned = 1 if (rule.pinned or (prev_rec and prev_rec.pinned)) else 0
        pinned_scope = (
            rule.pinned_scope
            if rule.pinned
            else (prev_rec.pinned_scope if (prev_rec and prev_rec.pinned) else "none")
        )

        target_dir = settings.skills_dir / slug
        target_file = target_dir / SKILL_FILENAME
        frontmatter: dict[str, Any] = {
            "name": rule.name or slug,
            "description": rule.description,
            "tags": rule.tags,
            "scope": rule.scope,
            "source_agent": rule.source_agent,
            "origin_path": str(rule.origin_path),
            "source_kind": rule.rule_kind,
        }
        if rule.file_globs:
            frontmatter["file_globs"] = rule.file_globs
        if pinned:
            frontmatter["pinned"] = True
            frontmatter["pinned_scope"] = pinned_scope

        rendered = render_skill(frontmatter, rule.body)
        status = "added" if prev_rec is None else "updated"
        if target_file.is_file():
            try:
                if target_file.read_text(encoding="utf-8") == rendered:
                    status = "unchanged"
            except OSError:
                pass

        if not dry_run and status in ("added", "updated"):
            target_dir.mkdir(parents=True, exist_ok=True)
            if rule.is_skill_bundle:
                _, more_secrets = _copy_companions(rule.origin_path.parent, target_dir, rule.origin_path)
                rule.secrets_redacted.extend(more_secrets)
            target_file.write_text(rendered, encoding="utf-8")

        item = DeepSyncItem(
            slug=slug,
            source_agent=rule.source_agent,
            rule_kind=rule.rule_kind,
            scope=rule.scope,
            origin_path=rule.origin_path,
            status=status,
            pinned=bool(pinned),
            pinned_scope=pinned_scope,
            file_globs=rule.file_globs,
            secrets_redacted=rule.secrets_redacted,
            workspaces=[rule.scope],
        )
        by_hash[rule.content_hash] = item
        report.items.append(item)

    if not dry_run:
        stats = idx.index()
        report.total_skills = stats.total_skills
        report.total_chunks = stats.total_chunks
        if enforce_gatekeeper:
            from .connector import enforce_gatekeeper_protocol

            for ws in sorted(workspace_roots) or [Path.cwd().resolve()]:
                report.gatekeeper_files.extend(
                    enforce_gatekeeper_protocol(workspace=ws, user_home=user_home)
                )

    report.elapsed_ms = (time.perf_counter() - t0) * 1000
    return report
