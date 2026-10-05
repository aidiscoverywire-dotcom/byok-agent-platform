"""SEARCH/REPLACE edit blocks, Aider-style.

Formats the coder model emits::

    <<<<<<< SEARCH
    <exact original text>
    =======
    <replacement text>
    >>>>>>> REPLACE

Whole-file rewrites (for files under 400 lines)::

    <<<<<<< WHOLE FILE
    <entire new file content>
    >>>>>>> END

:func:`apply_edit` is a pure function: exact match first, then a
whitespace-insensitive fuzzy fallback. It NEVER partially applies — on
ambiguity or no match it raises, and the caller (coder node) asks the
model to retry.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: Whole-file rewrites are only allowed for small files; large files must
#: use SEARCH/REPLACE so the model can't silently drop half the file.
WHOLE_FILE_LINE_LIMIT = 400


class EditApplyError(Exception):
    """An edit block could not be applied cleanly."""


class AmbiguousEditError(EditApplyError):
    """The SEARCH text matched more than one location."""


@dataclass(frozen=True)
class EditBlock:
    """One parsed edit: replace ``search`` with ``replace``."""

    search: str
    replace: str
    whole_file: bool = False


_SEARCH_RE = re.compile(
    r"^<<<<<<<\s*SEARCH\s*\n(?P<search>.*?)^=======\s*\n(?P<replace>.*?)^>>>>>>>\s*REPLACE\s*$",
    re.MULTILINE | re.DOTALL,
)
_WHOLE_RE = re.compile(
    r"^<<<<<<<\s*WHOLE FILE\s*\n(?P<content>.*?)^>>>>>>>\s*END\s*$",
    re.MULTILINE | re.DOTALL,
)


def parse_edit_blocks(text: str) -> list[EditBlock]:
    """Extract all edit blocks from model output, in order."""
    blocks: list[EditBlock] = []
    for m in _SEARCH_RE.finditer(text):
        blocks.append(
            EditBlock(
                search=m.group("search").strip("\n"),
                replace=m.group("replace").strip("\n"),
            )
        )
    for m in _WHOLE_RE.finditer(text):
        blocks.append(
            EditBlock(search="", replace=m.group("content").strip("\n"), whole_file=True)
        )
    return blocks


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces (fuzzy matching)."""
    return re.sub(r"\s+", " ", text).strip()


def apply_edit(original: str, search: str, replace: str) -> str:
    """Apply one SEARCH/REPLACE against ``original``.

    1. Exact match: must occur EXACTLY once, else
       :class:`AmbiguousEditError` (0 matches -> fall through to fuzzy).
    2. Fuzzy fallback: whitespace-insensitive match, must be unique.
    3. Otherwise :class:`EditApplyError` — no silent partial applies.

    Returns the new file content.
    """
    if not search.strip():
        raise EditApplyError("empty SEARCH block: use WHOLE FILE for full rewrites")

    count = original.count(search)
    if count == 1:
        return original.replace(search, replace, 1)
    if count > 1:
        raise AmbiguousEditError(
            f"SEARCH text matches {count} locations; make it unique "
            "(include more surrounding context)"
        )

    # Fuzzy fallback: find the unique whitespace-insensitive match, then
    # splice at the ORIGINAL offsets so formatting outside the block is
    # preserved as much as possible.
    norm_search = _normalize_ws(search)
    norm_orig = _normalize_ws(original)
    idx = norm_orig.find(norm_search)
    if idx == -1:
        # Last resort: difflib closest-line alignment for a helpful error.
        raise EditApplyError(
            "SEARCH text not found (exact or fuzzy). "
            "The file may have changed — re-read it and retry."
        )
    if norm_orig.find(norm_search, idx + 1) != -1:
        raise AmbiguousEditError(
            "fuzzy SEARCH text matches multiple locations; "
            "make it unique with more context"
        )

    # Map the normalized match back to original offsets. The anchor MUST be
    # located at a LINE boundary: the old `original.find(anchor)` found the
    # stripped text mid-line (after the original's indentation), so the
    # replacement's own indentation was then doubled ("    " + "    a = 1"
    # -> "        a = 1", an IndentationError in Python). We scan line by
    # line and take the line's start offset, so the original line —
    # indentation included — is replaced wholesale by `replace`.
    # A bare substring search could also match inside an unrelated longer
    # line; the line-boundary scan avoids that too.
    first_line = next((ln for ln in search.splitlines() if ln.strip()), "")
    anchor = first_line.strip()
    anchor_norm = _normalize_ws(anchor)
    start = -1
    for m in re.finditer(r"^.*$", original, re.MULTILINE):
        line = m.group(0)
        if not line.strip():
            continue
        if line.strip() == anchor or _normalize_ws(line) == anchor_norm:
            start = m.start()
            break
    if start == -1:
        raise EditApplyError("fuzzy anchor could not be located in original")

    # Extend from the anchor to cover the whole search block: walk forward
    # in the original matching normalized lines of the search block.
    search_lines = [ln for ln in search.splitlines()]
    orig_lines = original.splitlines(keepends=True)
    # Find the anchor's line index.
    line_no = original.count("\n", 0, start)
    end_line = line_no
    si = 0
    # Skip leading blank search lines.
    while si < len(search_lines) and not search_lines[si].strip():
        si += 1
    oi = line_no
    while si < len(search_lines) and oi < len(orig_lines):
        if not search_lines[si].strip():
            si += 1
            oi += 1
            continue
        if _normalize_ws(search_lines[si]) != _normalize_ws(orig_lines[oi]):
            break
        si += 1
        oi += 1
        end_line = oi
    if si < len([ln for ln in search_lines if ln.strip()]):
        raise EditApplyError(
            "fuzzy match diverged mid-block; refusing partial apply"
        )
    end = sum(len(l) for l in orig_lines[:end_line])
    return original[:start] + replace + original[end:]


def apply_whole_file(original: str, new_content: str) -> str:
    """Replace the whole file — only for files under the line limit."""
    lines = original.count("\n") + (0 if original.endswith("\n") or not original else 1)
    if original and lines > WHOLE_FILE_LINE_LIMIT:
        raise EditApplyError(
            f"whole-file rewrite refused for {lines}-line file "
            f"(limit {WHOLE_FILE_LINE_LIMIT}); use SEARCH/REPLACE"
        )
    return new_content if new_content.endswith("\n") else new_content + "\n"


def render_diff(before: str, after: str, path: str) -> str:
    """Unified diff for review/promotion display."""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
