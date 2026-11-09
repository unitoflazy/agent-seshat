"""Harvest scattered agent rules/skills (Claude, Codex, Cursor, Copilot, Windsurf, Cline) into the library."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import IGNORED_DIRS, Settings, get_settings, user_home_dir
from .parser import (
    SKILL_FILENAME,
    extract_keywords,
    find_git_root,
    first_description,
    infer_scope,
    normalize_tags,
    sanitize_secrets,
    slugify,
    split_frontmatter,
)

STUB_MARKER = "seshat-stub"
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_COMPANION_BYTES = 1024 * 1024

KIND_LABELS = {
    "claude-md": "Claude Code rules (CLAUDE.md)",
    "agent-skill": "Agent skill",
    "agents-md": "Codex / agents rules (AGENTS.md)",
    "copilot-instructions": "GitHub Copilot instructions",
    "copilot-path-instructions": "GitHub Copilot path instructions",
    "cursorrules": "Cursor rules (.cursorrules)",
    "cursor-rule": "Cursor rule (.mdc)",
    "windsurfrules": "Windsurf rules",
    "windsurf-rule": "Windsurf rule",
    "clinerules": "Cline rules",
}
KIND_TAGS = {
    "claude-md": "claude",
    "agent-skill": "skill",
    "agents-md": "codex",
    "copilot-instructions": "copilot",
    "copilot-path-instructions": "copilot",
    "cursorrules": "cursor",
    "cursor-rule": "cursor",
    "windsurfrules": "windsurf",
    "windsurf-rule": "windsurf",
    "clinerules": "cline",
}
KIND_PRIORITY = {
    "claude-md": 0,
    "agent-skill": 1,
    "agents-md": 2,
    "copilot-instructions": 3,
    "copilot-path-instructions": 4,
    "cursorrules": 5,
    "cursor-rule": 6,
    "windsurfrules": 7,
    "windsurf-rule": 8,
    "clinerules": 9,
}
# Monolithic per-repo instruction files that `--write-stubs` replaces with a compact index.
STUBBABLE_KINDS = {"claude-md", "agents-md", "copilot-instructions", "cursorrules", "windsurfrules", "clinerules"}
_SKILL_PARENTS = {".claude", ".codex", ".agents", ".github", ".cursor", ".copilot"}


@dataclass
class Discovered:
    path: Path
    kind: str
    scope: str
    repo_root: Path | None

    @property
    def repo_name(self) -> str:
        return self.scope.split(":", 1)[1] if self.scope.startswith("project:") else "global"


@dataclass
class HarvestItem:
    source: Path
    kind: str
    slug: str
    scope: str
    status: str  # new | updated | unchanged | duplicate | skipped | error | would-create | would-update
    secrets: list[str] = field(default_factory=list)
    target: Path | None = None
    note: str = ""
    companions: int = 0


@dataclass
class HarvestReport:
    items: list[HarvestItem] = field(default_factory=list)
    stubs: list[tuple[Path, Path]] = field(default_factory=list)  # (file, backup)
    dry_run: bool = False

    @property
    def secrets_total(self) -> int:
        return sum(len(i.secrets) for i in self.items)

    @property
    def written(self) -> list[HarvestItem]:
        return [i for i in self.items if i.status in ("new", "updated")]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def classify(path: Path) -> str | None:
    name = path.name
    parts = path.parts
    parent = path.parent.name
    if name == ".cursorrules":
        return "cursorrules"
    if name == ".windsurfrules":
        return "windsurfrules"
    if name == ".clinerules" and path.is_file():
        return "clinerules"
    if parent == ".clinerules" and name.endswith(".md"):
        return "clinerules"
    if name == "CLAUDE.md":
        return "claude-md"
    if name == "AGENTS.md":
        return "agents-md"
    if name == "copilot-instructions.md" and parent in (".github", ".copilot"):
        return "copilot-instructions"
    if name.endswith(".instructions.md") and ".github" in parts:
        return "copilot-path-instructions"
    if name.endswith(".mdc") and ".cursor" in parts and "rules" in parts:
        return "cursor-rule"
    if name.endswith(".md") and ".windsurf" in parts and "rules" in parts:
        return "windsurf-rule"
    if name == SKILL_FILENAME and "skills" in parts:
        i = len(parts) - 1 - list(reversed(parts)).index("skills")
        if i > 0 and parts[i - 1] in _SKILL_PARENTS:
            return "agent-skill"
    return None


def default_roots() -> list[Path]:
    home = user_home_dir()
    roots = [Path.cwd()]
    for p in (
        home / ".claude" / "CLAUDE.md",
        home / ".claude" / "skills",
        home / ".codex" / "AGENTS.md",
        home / ".codex" / "skills",
        home / ".agents" / "skills",
        home / ".copilot" / "copilot-instructions.md",
        home / ".copilot" / "skills",
        home / ".cursor" / "rules",
        home / ".codeium" / "windsurf" / "memories" / "global_rules.md",
    ):
        if p.exists():
            roots.append(p)
    return roots


def _is_stub(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return STUB_MARKER in fh.read(400)
    except OSError:
        return True


def discover(paths: list[Path] | None = None, settings: Settings | None = None) -> list[Discovered]:
    settings = settings or get_settings()
    library = settings.home.resolve()
    roots = [Path(p).expanduser().resolve() for p in (paths or default_roots())]
    found: dict[Path, Discovered] = {}

    def _consider(path: Path, forced_kind: str | None = None) -> None:
        try:
            real = path.resolve()
        except OSError:
            return
        if real in found or real.is_relative_to(library) or not real.is_file():
            return
        kind = forced_kind or classify(path)
        if kind is None or real.stat().st_size > MAX_FILE_BYTES or _is_stub(real):
            return
        found[real] = Discovered(path=real, kind=kind, scope=infer_scope(real), repo_root=find_git_root(real))

    for root in roots:
        if root.is_file():
            kind = classify(root)
            if kind is None and root.name == "global_rules.md":
                kind = "windsurfrules"
            if kind is None and root.suffix in (".md", ".mdc"):
                kind = "agent-skill" if root.name == SKILL_FILENAME else "claude-md"
            _consider(root, kind)
            continue
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            here = Path(dirpath)
            for fn in sorted(filenames):
                _consider(here / fn)
            # `.clinerules` may be a file or a directory of markdown files.
            if ".clinerules" in dirnames:
                for p in sorted((here / ".clinerules").glob("*.md")):
                    _consider(p, "clinerules")
    return sorted(found.values(), key=lambda d: (KIND_PRIORITY.get(d.kind, 99), str(d.path)))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _strip_suffixes(name: str) -> str:
    for suf in (".instructions.md", ".mdc", ".md"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name.lstrip(".")


def propose_slug(d: Discovered, meta: dict) -> str:
    repo = d.repo_name
    if d.kind == "agent-skill":
        return slugify(str(meta.get("name") or d.path.parent.name))
    if d.kind in ("cursor-rule", "windsurf-rule", "copilot-path-instructions") or (
        d.kind == "clinerules" and d.path.name != ".clinerules"
    ):
        return slugify(f"{repo}-{KIND_TAGS[d.kind]}-{_strip_suffixes(d.path.name)}")
    base = {
        "claude-md": "claude-md",
        "agents-md": "agents-md",
        "copilot-instructions": "copilot-instructions",
        "cursorrules": "cursorrules",
        "windsurfrules": "windsurfrules",
        "clinerules": "clinerules",
    }[d.kind]
    # Nested CLAUDE.md / AGENTS.md in sub-packages keep their relative directory in the slug.
    rel = ""
    if d.repo_root is not None:
        rel_dir = d.path.parent.relative_to(d.repo_root)
        rel_parts = [p for p in rel_dir.parts if p not in (".claude", ".github", ".copilot")]
        rel = "-".join(rel_parts)
    return slugify("-".join(x for x in (repo, rel, base) if x))


def _body_key(body: str) -> str:
    return hashlib.sha256("\n".join(line.rstrip() for line in body.strip().splitlines()).encode()).hexdigest()


def render_skill(meta: dict, body: str) -> str:
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000).strip()
    return f"---\n{front}\n---\n\n{body.strip()}\n"


def normalize(d: Discovered) -> tuple[dict, str, list[str]]:
    raw = d.path.read_text(encoding="utf-8", errors="replace")
    meta, body = split_frontmatter(raw)
    body, findings = sanitize_secrets(body)
    label = KIND_LABELS[d.kind]
    name = str(meta.get("name") or "").strip()
    desc = str(meta.get("description") or "").strip()
    if not desc:
        first = first_description(body)
        desc = first if d.kind == "agent-skill" else f"{label} for {d.repo_name}: {first}"
    desc, more = sanitize_secrets(" ".join(desc.split()))
    findings += more

    tags = normalize_tags(meta.get("tags"))
    if not tags:
        tags = extract_keywords(f"{desc}\n{body}", limit=8)
    for extra in (KIND_TAGS[d.kind], d.repo_name if d.repo_name != "global" else None):
        if extra and extra.lower() not in tags:
            tags.append(extra.lower())

    out = {
        "name": name or None,
        "description": desc,
        "tags": tags,
        "scope": d.scope if not meta.get("scope") else str(meta["scope"]),
        "source": str(d.path),
        "source_kind": d.kind,
    }
    for key in ("globs", "alwaysApply", "applyTo", "trigger", "allowed-tools", "license"):
        if key in meta:
            out[key] = meta[key]
    return out, body, findings


def _copy_companions(src_dir: Path, dst_dir: Path, skill_file: Path) -> tuple[int, list[str]]:
    count, findings = 0, []
    for p in sorted(src_dir.rglob("*")):
        if not p.is_file() or p.resolve() == skill_file.resolve():
            continue
        rel = p.relative_to(src_dir)
        if any(part.startswith(".") or part in IGNORED_DIRS for part in rel.parts):
            continue
        if p.stat().st_size > MAX_COMPANION_BYTES:
            continue
        target = dst_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = p.read_bytes()
        if b"\x00" not in data[:4096]:
            text, found = sanitize_secrets(data.decode("utf-8", errors="replace"))
            findings += found
            target.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(p, target)
        count += 1
    return count, findings


def _existing_library(skills_dir: Path) -> dict[str, tuple[str, str]]:
    """slug -> (body_key, source) for skills already in the library."""
    from .indexer import discover_skill_files

    out: dict[str, tuple[str, str]] = {}
    for slug, path in discover_skill_files(skills_dir).items():
        try:
            meta, body = split_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        out[slug] = (_body_key(body), str(meta.get("source") or ""))
    return out


# ---------------------------------------------------------------------------
# Harvest
# ---------------------------------------------------------------------------


def _backup_path(path: Path) -> Path:
    candidate = path.with_name(path.name + ".bak")
    i = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak.{i}")
        i += 1
    return candidate


def render_stub(file_name: str, backup_name: str, project: str, slugs: list[str]) -> str:
    listed = ", ".join(f"`{s}`" for s in slugs[:12]) + (f" (+{len(slugs) - 12} more)" if len(slugs) > 12 else "")
    proj_arg = f', project="{project}"' if project != "global" else ""
    lines = [
        f"<!-- {STUB_MARKER} v1: original moved to {backup_name} -->",
        f"# {project} agent rules -> Agent-Seshat (House of Books)",
        "Rules & skills for this repo live in the Agent-Seshat MCP library (server `agent-seshat`).",
        f"Harvested skills: {listed}",
        f"1. FIRST call `search_skills(query{proj_arg})` before planning, coding, or running commands.",
        "2. Call `load_skill(slug)` for every relevant hit and follow it exactly.",
        "3. Templates: `load_skill_resource(slug, filename)`. No MCP? Run `sesh search \"<topic>\"`.",
        f"Restore the original with: mv {backup_name} {file_name}",
    ]
    return "\n".join(lines) + "\n"


def harvest(
    paths: list[Path] | None = None,
    *,
    dry_run: bool = False,
    write_stubs: bool = False,
    settings: Settings | None = None,
) -> HarvestReport:
    settings = (settings or get_settings()).ensure_dirs()
    skills_dir = settings.skills_dir
    report = HarvestReport(dry_run=dry_run)
    library = _existing_library(skills_dir)
    body_owner = {key: slug for slug, (key, _) in library.items()}
    seen_run: dict[str, str] = {}
    repo_slugs: dict[str, list[str]] = {}
    stub_candidates: list[Discovered] = []

    for d in discover(paths, settings):
        try:
            meta, body, findings = normalize(d)
        except Exception as exc:
            report.items.append(HarvestItem(d.path, d.kind, "-", d.scope, "error", note=str(exc)))
            continue
        slug = propose_slug(d, meta)
        had_name = bool(meta.get("name"))
        key = _body_key(body)
        item = HarvestItem(source=d.path, kind=d.kind, slug=slug, scope=meta["scope"], status="new", secrets=findings)

        if not body.strip():
            item.status, item.note = "skipped", "empty file"
            report.items.append(item)
            continue

        if key in seen_run:
            item.status, item.slug, item.note = "duplicate", seen_run[key], f"same content as {seen_run[key]}"
            report.items.append(item)
            repo_slugs.setdefault(d.repo_name, []).append(seen_run[key])
            if d.kind in STUBBABLE_KINDS:
                stub_candidates.append(d)
            continue

        owner = body_owner.get(key)
        if owner:
            # Identical content already lives in the library.
            same_source = library[owner][1] == str(d.path)
            item.slug = owner
            item.status = "unchanged" if (owner == slug or same_source) else "duplicate"
            if item.status == "duplicate":
                item.note = f"already in library as {owner}"
        else:
            # Never clobber a skill that came from somewhere else (or was hand-written).
            existing = library.get(slug)
            if existing and existing[1] != str(d.path):
                base = slug if d.repo_name in slug else slugify(f"{d.repo_name}-{slug}")
                candidate, n = base, 2
                while candidate in library and library[candidate][1] != str(d.path):
                    candidate, n = f"{base}-{n}", n + 1
                slug = item.slug = candidate
            item.status = "updated" if slug in library else "new"
        seen_run[key] = item.slug
        repo_slugs.setdefault(d.repo_name, []).append(item.slug)
        if d.kind in STUBBABLE_KINDS:
            stub_candidates.append(d)

        if item.status in ("new", "updated"):
            target_dir = skills_dir / item.slug
            item.target = target_dir / SKILL_FILENAME
            if dry_run:
                item.status = "would-create" if item.status == "new" else "would-update"
                if d.kind == "agent-skill":
                    item.companions = sum(
                        1 for p in d.path.parent.rglob("*") if p.is_file() and p != d.path and not p.name.startswith(".")
                    )
            else:
                target_dir.mkdir(parents=True, exist_ok=True)
                if d.kind == "agent-skill":
                    item.companions, more = _copy_companions(d.path.parent, target_dir, d.path)
                    item.secrets += more
                if not had_name:
                    meta["name"] = item.slug
                item.target.write_text(render_skill(meta, body), encoding="utf-8")
                library[item.slug] = (key, str(d.path))
                body_owner[key] = item.slug
        report.items.append(item)

    if write_stubs:
        for d in stub_candidates:
            slugs = list(dict.fromkeys(repo_slugs.get(d.repo_name, [])))
            backup = _backup_path(d.path)
            if dry_run:
                report.stubs.append((d.path, backup))
                continue
            shutil.copy2(d.path, backup)
            d.path.write_text(render_stub(d.path.name, backup.name, d.repo_name, slugs), encoding="utf-8")
            report.stubs.append((d.path, backup))
    return report
