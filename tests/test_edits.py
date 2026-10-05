"""Phase 1 tests: edit applier (exact / fuzzy / failure modes). No network."""
from __future__ import annotations

import pytest

from byok_agent.edits import (
    AmbiguousEditError,
    EditApplyError,
    apply_edit,
    apply_whole_file,
    parse_edit_blocks,
)


ORIGINAL = """def greet(name):
    return f"Hello, {name}!"

def farewell(name):
    return f"Goodbye, {name}!"
"""


def test_parse_search_replace_blocks():
    text = """<<<<<<< SEARCH
def greet(name):
    return f"Hello, {name}!"
=======
def greet(name):
    return f"Hi, {name}!"
>>>>>>> REPLACE
"""
    blocks = parse_edit_blocks(text)
    assert len(blocks) == 1
    assert not blocks[0].whole_file
    assert "Hello" in blocks[0].search
    assert "Hi" in blocks[0].replace


def test_parse_whole_file_block():
    text = "<<<<<<<\nWHOLE FILE\nprint('x')\n>>>>>>>\nEND\n".replace(
        "<<<<<<<\n", "<<<<<<< "
    )
    blocks = parse_edit_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].whole_file
    assert blocks[0].replace == "print('x')"


def test_apply_exact_match():
    out = apply_edit(
        ORIGINAL,
        '    return f"Hello, {name}!"',
        '    return f"Hi, {name}!"',
    )
    assert 'Hi, {name}!' in out
    assert 'Goodbye' in out  # untouched


def test_apply_ambiguous_raises():
    with pytest.raises(AmbiguousEditError):
        apply_edit(
            ORIGINAL,
            "(name):",  # appears twice
            "(name):  # x",
        )


def test_apply_no_match_raises():
    with pytest.raises(EditApplyError):
        apply_edit(ORIGINAL, "def nonexistent():\n    pass", "x")


def test_apply_fuzzy_whitespace():
    # SEARCH uses different indentation than the file; fuzzy still applies.
    out = apply_edit(
        ORIGINAL,
        'def greet(name):\n        return f"Hello, {name}!"',
        'def greet(name):\n    return f"Yo, {name}!"',
    )
    assert "Yo" in out
    assert out.count("def greet") == 1


def test_apply_empty_search_raises():
    with pytest.raises(EditApplyError):
        apply_edit(ORIGINAL, "   \n  ", "x")


def test_whole_file_small_ok():
    out = apply_whole_file(ORIGINAL, "print('new')")
    assert out == "print('new')\n"


def test_whole_file_large_refused():
    big = "".join(f"line {i}\n" for i in range(500))
    with pytest.raises(EditApplyError):
        apply_whole_file(big, "print('new')")


def test_apply_fuzzy_no_indentation_doubling():
    # The fuzzy anchor must resolve at a LINE boundary. The old
    # original.find(anchor) matched mid-line (after the indentation), so
    # the replacement's indentation doubled: "    " + "    a = 20"
    # -> "        a = 20" (IndentationError).
    # (Trailing spaces in SEARCH force the fuzzy path: without them the
    # SEARCH is a substring of the file and takes the exact path.)
    original = "def f():\n    a = 10\n    return a\n"
    out = apply_edit(original, "    a = 10   ", "    a = 20\n")
    assert out == "def f():\n    a = 20\n    return a\n"
    compile(out, "<test>", "exec")  # must stay valid Python
