"""Repository map in Aider's proven style.

Pipeline:

1. Walk the repo, collect source files (skip dot-dirs, ``node_modules``,
   ``__pycache__``, ``.venv``, ``target/``, ``build/``, ``dist/``).
2. Extract definitions (classes/functions/methods with signatures) per
   file — via tree-sitter when importable, else a regex fallback per
   language (Python/JS/TS/Go/Rust/Java).
3. Build a reference graph: definition A in file F references definition
   B if B's name textually appears in F (excluding self-references).
4. Personalized PageRank over that graph, personalized on the definitions
   of ``changed_files`` (the files the task touches rank highest).
5. Render signature-only lines, highest-ranked first, within
   ``token_budget`` tokens (~4 chars/token). Changed files always come
   first, even at low rank.

If nothing parses, the map degrades to a plain file tree — never empty,
never an exception.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SOURCE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".tox",
    "target", "build", "dist", ".idea", ".vscode", "eggs", ".eggs",
}

SKIP_FILES = {"package-lock.json", "yarn.lock", "poetry.lock"}


@dataclass
class Definition:
    """One extracted definition: name + signature + location."""

    name: str
    signature: str  # one-line signature, e.g. "def run(port: int) -> None"
    file: str  # repo-relative posix path
    lineno: int
    kind: str  # "class" | "function" | "method"


# -- regex fallback extraction -------------------------------------------
# Good enough for ranking; tree-sitter is preferred when importable.

_PY_DEF = re.compile(
    r"^(?P<indent>\s*)(?P<kind>async\s+def|def|class)\s+(?P<name>[A-Za-z_]\w*)"
    r"(?P<sig>\([^)]*\)(?:\s*->\s*[^:]+)?)?\s*:",
    re.MULTILINE,
)
_JS_DEF = re.compile(
    r"(?P<kind>class|function)\s+(?P<name>[A-Za-z_]\w*)"
    r"|(?P<const>const|let|var)\s+(?P<cname>[A-Za-z_]\w*)\s*=\s*(?:async\s*)?\("
    r"|(?P<arrow>(?:[A-Za-z_]\w*))\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
)
_GO_DEF = re.compile(
    r"^func\s+(?:\(\w+\s+[\w\*]+\)\s+)?(?P<name>[A-Za-z_]\w*)(?P<sig>\([^)]*\))?",
    re.MULTILINE,
)
_RUST_DEF = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?(?P<kind>fn|struct|enum|trait)\s+"
    r"(?P<name>[A-Za-z_]\w*)(?P<sig>[^\{;]*)",
    re.MULTILINE,
)
_JAVA_DEF = re.compile(
    r"(?P<kind>class|interface|enum)\s+(?P<name>[A-Za-z_]\w*)"
    r"|(?:public|protected|private|static|\s)+[\w<>\[\]]+\s+"
    r"(?P<mname>[A-Za-z_]\w*)\s*(?P<msig>\([^)]*\))",
)


def _regex_definitions(
    language: str, text: str, relfile: str
) -> list[Definition]:
    """Best-effort definition extraction without tree-sitter."""
    defs: list[Definition] = []
    if language == "python":
        for m in _PY_DEF.finditer(text):
            kind = "class" if m.group("kind").strip() == "class" else (
                "method" if m.group("indent") else "function"
            )
            sig = (m.group("sig") or "()").strip()
            lineno = text.count("\n", 0, m.start()) + 1
            defs.append(
                Definition(
                    name=m.group("name"),
                    signature=f"{m.group('kind').strip()} {m.group('name')}{sig}",
                    file=relfile,
                    lineno=lineno,
                    kind=kind,
                )
            )
    elif language in ("javascript", "typescript"):
        for m in _JS_DEF.finditer(text):
            name = m.group("name") or m.group("cname") or m.group("arrow")
            kind = "class" if m.group("kind") == "class" else "function"
            lineno = text.count("\n", 0, m.start()) + 1
            defs.append(
                Definition(
                    name=name,
                    signature=f"{kind} {name}(...)",
                    file=relfile,
                    lineno=lineno,
                    kind=kind,
                )
            )
    elif language == "go":
        for m in _GO_DEF.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            defs.append(
                Definition(
                    name=m.group("name"),
                    signature=f"func {m.group('name')}{m.group('sig') or '(...)'}",
                    file=relfile,
                    lineno=lineno,
                    kind="function",
                )
            )
    elif language == "rust":
        for m in _RUST_DEF.finditer(text):
            kind = m.group("kind")
            lineno = text.count("\n", 0, m.start()) + 1
            defs.append(
                Definition(
                    name=m.group("name"),
                    signature=f"{kind} {m.group('name')}{(m.group('sig') or '').strip()}",
                    file=relfile,
                    lineno=lineno,
                    kind="class" if kind in ("struct", "enum", "trait") else "function",
                )
            )
    elif language == "java":
        for m in _JAVA_DEF.finditer(text):
            name = m.group("name") or m.group("mname")
            if not name or name in {"if", "for", "while", "switch", "return", "new"}:
                continue
            kind = "class" if m.group("kind") else "method"
            lineno = text.count("\n", 0, m.start()) + 1
            defs.append(
                Definition(
                    name=name,
                    signature=f"{name}{(m.group('msig') or '').strip()}",
                    file=relfile,
                    lineno=lineno,
                    kind=kind,
                )
            )
    return defs


def _treesitter_definitions(
    language: str, text: str, relfile: str
) -> list[Definition] | None:
    """Use tree-sitter when importable; None if unavailable/unsupported."""
    try:
        import tree_sitter  # noqa: F401
    except ImportError:
        return None
    # Grammar wheels (tree_sitter_python, ...) are optional extras; the
    # regex path covers us when they are absent. Only Python is attempted
    # here to keep the dependency surface small.
    if language != "python":
        return None
    try:
        import tree_sitter_python
        from tree_sitter import Language, Parser

        lang = Language(tree_sitter_python.language())
        parser = Parser(lang)
        tree = parser.parse(text.encode("utf-8"))
    except Exception as exc:
        log.debug("tree-sitter parse failed for %s: %s", relfile, exc)
        return None

    defs: list[Definition] = []

    def _walk(node: Any, depth: int = 0) -> None:
        if node.type in ("function_definition", "class_definition"):
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = text[name_node.start_byte : name_node.end_byte]
                params = node.child_by_field_name("parameters")
                sig = text[params.start_byte : params.end_byte] if params else ""
                kind = (
                    "class"
                    if node.type == "class_definition"
                    else ("method" if depth > 0 else "function")
                )
                defs.append(
                    Definition(
                        name=name,
                        signature=f"{'class' if kind == 'class' else 'def'} {name}{sig}",
                        file=relfile,
                        lineno=node.start_point[0] + 1,
                        kind=kind,
                    )
                )
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                _walk(child, depth + 1)
            else:
                _walk(child, depth)

    _walk(tree.root_node)
    return defs


# -- reference graph + personalized PageRank ------------------------------

_WORD = re.compile(r"[A-Za-z_]\w*")


def score_definitions(
    defs: list[Definition],
    file_texts: dict[str, str],
    changed_files: list[str] | None = None,
    *,
    damping: float = 0.85,
    iterations: int = 25,
) -> dict[int, float]:
    """Personalized PageRank over the textual reference graph.

    Personalization mass sits on definitions from ``changed_files`` (the
    files the task touches); with no changed files it is uniform.
    Returns {def_index: score}.
    """
    n = len(defs)
    if n == 0:
        return {}

    by_name: dict[str, list[int]] = {}
    for i, d in enumerate(defs):
        by_name.setdefault(d.name, []).append(i)

    # Outgoing edges: def i references def j if j's name appears as a word
    # in i's file text (excluding self-file self-name trivialities is
    # unnecessary — PageRank absorbs them).
    out: list[set[int]] = [set() for _ in range(n)]
    file_words: dict[str, set[str]] = {}
    for relfile, text in file_texts.items():
        file_words[relfile] = set(_WORD.findall(text))
    for i, d in enumerate(defs):
        words = file_words.get(d.file, set())
        for name, js in by_name.items():
            if name in words and name != d.name:
                out[i].update(j for j in js if j != i)

    # Personalization: changed files' defs share the mass.
    changed = set(changed_files or [])
    personal = [1.0 if defs[i].file in changed else 0.0 for i in range(n)]
    if sum(personal) == 0:
        personal = [1.0 / n] * n
    else:
        s = sum(personal)
        personal = [p / s for p in personal]

    rank = list(personal)
    for _ in range(iterations):
        new = [(1.0 - damping) * personal[i] for i in range(n)]
        dangling = 0.0
        for i in range(n):
            if not out[i]:
                dangling += rank[i]
            else:
                share = damping * rank[i] / len(out[i])
                for j in out[i]:
                    new[j] += share
        if dangling:
            for i in range(n):
                new[i] += damping * dangling * personal[i]
        rank = new
    return {i: rank[i] for i in range(n)}


# -- the map ---------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class RepoMap:
    """Build a signature-only, ranked, token-budgeted map of a repo."""

    def __init__(self, *, token_budget: int = 1024) -> None:
        self._token_budget = token_budget

    def _collect_files(self, root: Path) -> list[Path]:
        files: list[Path] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue
            rel = path.relative_to(root)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if path.name in SKIP_FILES:
                continue
            # Extra safety: skip anything under a hidden dir.
            if any(part.startswith(".") for part in rel.parts):
                continue
            try:
                if path.stat().st_size > 500_000:
                    continue
            except OSError:
                continue
            files.append(path)
        return files

    def build(
        self, root_dir: str | Path, changed_files: list[str] | None = None
    ) -> str:
        """Return the repo map string for ``root_dir``.

        Never raises for weird repos: unparseable files are skipped, and a
        repo with no extractable definitions degrades to a file tree.
        """
        root = Path(root_dir)
        if not root.is_dir():
            return f"(repo map: {root_dir} is not a directory)"

        files = self._collect_files(root)
        if not files:
            return "(repo map: no source files found)"

        file_texts: dict[str, str] = {}
        defs: list[Definition] = []
        for path in files:
            rel = path.relative_to(root).as_posix()
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            file_texts[rel] = text
            language = SOURCE_EXTENSIONS[path.suffix.lower()]
            extracted = _treesitter_definitions(language, text, rel)
            if extracted is None:
                extracted = _regex_definitions(language, text, rel)
            defs.extend(extracted)

        if not defs:
            # Graceful degradation: plain file tree.
            lines = ["(repo map: definitions unavailable — file tree)"]
            lines.extend(f"  {p.relative_to(root).as_posix()}" for p in files)
            return "\n".join(lines)

        scores = score_definitions(defs, file_texts, changed_files)
        changed = set(changed_files or [])

        # Order: changed files' defs first (by score), then the rest by score.
        def _key(i: int) -> tuple[int, float]:
            d = defs[i]
            return (0 if d.file in changed else 1, -scores.get(i, 0.0))

        ordered = sorted(range(len(defs)), key=_key)

        lines: list[str] = []
        used = 0
        current_file: str | None = None
        for i in ordered:
            d = defs[i]
            if d.file != current_file:
                header = f"{d.file}:"
                if used + _estimate_tokens(header) > self._token_budget:
                    break
                lines.append(header)
                used += _estimate_tokens(header)
                current_file = d.file
            line = f"  {d.signature}"
            cost = _estimate_tokens(line)
            if used + cost > self._token_budget:
                # Budget exhausted: stop cleanly (headers already emitted
                # stay; we never emit a half file section... actually we may
                # have emitted the header — acceptable, still valid).
                break
            lines.append(line)
            used += cost

        lines.append(f"(map budget: ~{used}/{self._token_budget} tokens)")
        return "\n".join(lines)
