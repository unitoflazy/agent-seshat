"""Interactive Rule Gatekeeper & 'Always-On' Pinned Rule resolution.

Implements:
- `propose_session_rules`: Mandatory Step 0 pre-flight tool returning auto-active Pinned
  ('Always-Use') rules in full plus ranked candidate rules for user confirmation.
- `activate_rules`: Loads full Markdown instructions for selected slugs and persists any
  `pin_always_slugs` in SQLite for future sessions.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from .indexer import SkillIndex, SkillRecord
from .parser import find_git_root, normalize_scope
from .search import HybridSearcher, query_tokens

GATEKEEPER_START = "<!-- SESHAT-GATEKEEPER-START -->"
GATEKEEPER_END = "<!-- SESHAT-GATEKEEPER-END -->"

GATEKEEPER_PROTOCOL_BLOCK = """<!-- SESHAT-GATEKEEPER-START -->
## Mandatory Pre-Flight Protocol (Agent-Seshat)
1. On EVERY new coding task, your FIRST action MUST be calling the `propose_session_rules` MCP tool with a summary of the user's request and current workspace path.
2. If `propose_session_rules` returns unpinned candidate rules, STOP and ask the user which candidate rules to activate before editing any files.
3. Call `activate_rules` with the user's chosen slugs (and `pin_always_slugs` if they ask to always use a rule).
<!-- SESHAT-GATEKEEPER-END -->"""


def resolve_workspace_project(workspace_path: str = "") -> str:
    """Infer `<repo-name>` from `workspace_path` (works for both real on-disk paths and virtual paths)."""
    raw = (workspace_path or "").strip()
    if raw:
        p = Path(raw)
        if p.exists():
            git_root = find_git_root(p.resolve())
            if git_root is not None:
                return git_root.name
            target = p.resolve() if p.is_dir() else p.resolve().parent
            return target.name
        cleaned = raw.rstrip("/\\")
        if cleaned.lower().startswith("project:"):
            return cleaned.split(":", 1)[1].strip()
        return Path(cleaned).name

    from .server import detect_project

    return detect_project()


def matches_file_globs(file_globs: str, active_files: list[str] | None) -> bool:
    """Return True if any path in `active_files` matches any pattern in `file_globs`."""
    if not file_globs or not active_files:
        return False
    patterns = [p.strip().strip("'\"") for p in re.split(r"[,;]+", file_globs) if p.strip()]
    if not patterns:
        return False
    for fpath in active_files:
        norm_file = fpath.replace("\\", "/").lstrip("./")
        basename = Path(norm_file).name
        for pat in patterns:
            norm_pat = pat.replace("\\", "/").lstrip("./")
            if fnmatch.fnmatch(norm_file, norm_pat) or fnmatch.fnmatch(basename, norm_pat):
                return True
            if norm_pat.startswith("**/") and (
                fnmatch.fnmatch(norm_file, norm_pat[3:]) or fnmatch.fnmatch(basename, norm_pat[3:])
            ):
                return True
    return False


def resolve_pinned_rules(
    index: SkillIndex,
    project: str = "",
    active_files: list[str] | None = None,
) -> list[SkillRecord]:
    """Return all rules that are auto-active for `(project, active_files)`."""
    index.ensure_fresh()
    project_scope = f"project:{project}".lower() if project else ""
    all_skills = index.list_skills()
    pinned_out: list[SkillRecord] = []
    seen: set[str] = set()

    for rec in all_skills:
        p_scope = (rec.pinned_scope or "none").lower()
        r_scope = (rec.scope or "global").lower()
        scope_matches_workspace = r_scope == "global" or (project_scope and r_scope == project_scope)

        is_pinned_here = False
        if rec.pinned:
            if p_scope == "global":
                is_pinned_here = True
            elif project_scope and p_scope == project_scope:
                is_pinned_here = True
            elif p_scope in ("", "none") and scope_matches_workspace:
                is_pinned_here = True

        glob_matched = bool(
            active_files
            and rec.file_globs
            and scope_matches_workspace
            and matches_file_globs(rec.file_globs, active_files)
        )

        if (is_pinned_here or glob_matched) and rec.slug not in seen:
            seen.add(rec.slug)
            pinned_out.append(rec)

    # Sort project-specific pinned rules first, then global, then alphabetically.
    pinned_out.sort(
        key=lambda r: (
            0 if (project_scope and r.scope.lower() == project_scope) else 1,
            r.slug,
        )
    )
    return pinned_out


def propose_session_rules(
    index: SkillIndex,
    user_prompt_summary: str,
    workspace_path: str = "",
    active_files: list[str] | None = None,
    top_k: int = 6,
) -> str:
    """Execute the Mandatory Step 0 Gatekeeper lookup & format the agent handshake directive."""
    index.ensure_fresh()
    repo_name = resolve_workspace_project(workspace_path)
    active_files = active_files or []

    # Step 2: Fetch Pinned / Always-Choose Rules
    pinned_rules = resolve_pinned_rules(index, project=repo_name, active_files=active_files)
    pinned_slugs = {r.slug for r in pinned_rules}

    # Step 3: Hybrid Search for unpinned candidates
    searcher = HybridSearcher(index)
    raw_hits = searcher.search(user_prompt_summary, project=repo_name, top_k=top_k + len(pinned_slugs) + 4)
    candidate_records: list[SkillRecord] = []
    seen_candidates: set[str] = set(pinned_slugs)
    skills_by_slug = {r.slug: r for r in index.list_skills()}

    for hit in raw_hits:
        if hit.slug in seen_candidates:
            continue
        rec = skills_by_slug.get(hit.slug)
        if rec is not None:
            candidate_records.append(rec)
            seen_candidates.add(hit.slug)
            if len(candidate_records) >= top_k:
                break

    # Fallback: if strict hybrid search returned < top_k and there are unpinned workspace/global rules
    # with lexical or semantic overlap, surface them so the engineer can choose.
    if len(candidate_records) < top_k and user_prompt_summary.strip():
        tokens = set(query_tokens(user_prompt_summary))
        sem_scores = searcher._semantic(user_prompt_summary)
        project_scope = f"project:{repo_name}".lower() if repo_name else ""
        fallback_pool: list[tuple[float, SkillRecord]] = []
        for slug, rec in skills_by_slug.items():
            if slug in seen_candidates:
                continue
            r_scope = rec.scope.lower()
            if r_scope.startswith("project:") and project_scope and r_scope != project_scope:
                continue
            cos = sem_scores.get(slug, (0.0, ""))[0]
            rec_terms = set(query_tokens(f"{rec.slug} {rec.name} {rec.description} {' '.join(rec.tags)}"))
            overlap = len(tokens & rec_terms)
            if cos >= 0.45 or overlap > 0:
                score = cos + (0.15 * overlap) + (0.20 if (project_scope and r_scope == project_scope) else 0.0)
                fallback_pool.append((score, rec))
        fallback_pool.sort(key=lambda item: (-item[0], item[1].slug))
        for _, rec in fallback_pool:
            if len(candidate_records) >= top_k:
                break
            candidate_records.append(rec)
            seen_candidates.add(rec.slug)

    # Step 4: Format output
    sections: list[str] = []
    ws_label = f"project:{repo_name}" if repo_name else "global"
    sections.append(f"# Agent-Seshat Pre-Task Gatekeeper (Workspace: `{ws_label}`)")
    sections.append("")

    sections.append("## [AUTO-ACTIVE: ALWAYS-USE RULES]")
    if pinned_rules:
        for rec in pinned_rules:
            sections.append(
                f"### `{rec.slug}` (Scope: {rec.scope}, Pinned: {rec.pinned_scope}, Source: {rec.source_agent})"
            )
            if rec.file_globs:
                sections.append(f"- **File Globs:** `{rec.file_globs}`")
            sections.append(f"- **Description:** {rec.description}")
            sections.append("")
            sections.append(rec.body.strip())
            sections.append("")
    else:
        sections.append("_No pinned ('Always-Use') rules configured for this workspace._")
        sections.append("")

    pinned_list_str = ", ".join(f"`{r.slug}`" for r in pinned_rules) if pinned_rules else "`none`"

    sections.append("### Mandatory Pre-Task Rule Confirmation")
    if candidate_records:
        sections.append("Instruct the user with the following exact prompt before proceeding:")
        sections.append(
            f'"Before I proceed, I have auto-loaded your **Pinned Rules** ({pinned_list_str}). '
            f'Based on your prompt, I also found these relevant skills/rules in **Agent-Seshat**:'
        )
        for i, rec in enumerate(candidate_records, 1):
            sections.append(
                f"{i}. `[{rec.slug}]` (Scope: {rec.scope}, Source: {rec.source_agent}) — *{rec.description}*"
            )
        sections.append("")
        sections.append(
            'Reply with the numbers to activate for this task (e.g., `1, 2`), `all`, `none`, '
            'or `always 1` to pin a rule permanently."'
        )
    else:
        sections.append(
            f"All matching rules for this task are already active via Pinned Rules ({pinned_list_str}). "
            "You may proceed while strictly following the `[AUTO-ACTIVE: ALWAYS-USE RULES]` above."
        )

    return "\n".join(sections)


def activate_rules(
    index: SkillIndex,
    load_slugs: list[str] | None = None,
    pin_always_slugs: list[str] | None = None,
    pin_scope: str = "global",
) -> str:
    """Load the full Markdown instructions for `load_slugs` and persist `pin_always_slugs` in SQLite."""
    index.ensure_fresh()
    load_slugs = [s.strip().strip("[]`") for s in (load_slugs or []) if s and s.strip()]
    pin_always_slugs = [s.strip().strip("[]`") for s in (pin_always_slugs or []) if s and s.strip()]
    normalized_pin_scope = normalize_scope(pin_scope)

    pinned_now: list[str] = []
    not_found: list[str] = []

    for slug in pin_always_slugs:
        try:
            index.pin_skill(slug, scope=normalized_pin_scope)
            pinned_now.append(slug)
        except LookupError:
            if slug not in not_found:
                not_found.append(slug)

    all_to_load: list[str] = []
    for slug in [*load_slugs, *pin_always_slugs]:
        if slug not in all_to_load:
            all_to_load.append(slug)

    lines: list[str] = ["# Activated Agent-Seshat Rules & Skills", ""]
    if pinned_now:
        lines.append(
            f"**Persisted as Always-Choose (`pinned = 1`, scope = `{normalized_pin_scope}`):** "
            + ", ".join(f"`{s}`" for s in pinned_now)
        )
        lines.append("")

    loaded_count = 0
    for slug in all_to_load:
        rec = index.get_skill(slug)
        if rec is None:
            if slug not in not_found:
                not_found.append(slug)
            continue
        loaded_count += 1
        lines.append(f"## Rule: `{rec.slug}` ({rec.name})")
        lines.append(
            f"- **Scope:** `{rec.scope}` | **Source Agent:** `{rec.source_agent}` | "
            f"**Pinned:** `{'yes (' + rec.pinned_scope + ')' if rec.pinned else 'no'}`"
        )
        if rec.file_globs:
            lines.append(f"- **File Globs:** `{rec.file_globs}`")
        lines.append(f"- **Description:** {rec.description}")
        lines.append("")
        lines.append(rec.body.strip())
        if rec.resources:
            lines.append("")
            lines.append(
                f"Companion resources (load via `load_skill_resource(\"{rec.slug}\", \"<filename>\")`): "
                + ", ".join(f"`{r}`" for r in rec.resources)
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    if not_found:
        lines.append(f"**Unknown rule slugs (skipped):** {', '.join(f'`{s}`' for s in not_found)}")
    if loaded_count == 0 and not pinned_now and not not_found:
        lines.append("No additional rules requested for activation.")

    return "\n".join(lines).strip()
