"""Phase 1 tests: repo map (signatures, budget, changed-file ranking). No network."""
from __future__ import annotations

from pathlib import Path

from byok_agent.codemap import RepoMap

FIXTURES = Path(__file__).parent / "fixtures" / "sample_repo"


def test_map_contains_signatures():
    m = RepoMap(token_budget=2048).build(FIXTURES)
    assert "create_app" in m
    assert "User" in m
    assert "load_config" in m
    # Signature-only: bodies must not leak into the map.
    assert "Hello" not in m
    assert "serving on" not in m


def test_map_respects_token_budget():
    budget = 200
    m = RepoMap(token_budget=budget).build(FIXTURES)
    assert len(m) // 4 <= budget + 50  # ~tokens, with header slack


def test_changed_files_ranked_first():
    m = RepoMap(token_budget=2048).build(FIXTURES, changed_files=["models.py"])
    lines = m.splitlines()
    first_header = next(i for i, ln in enumerate(lines) if ln.endswith(":"))
    assert lines[first_header].startswith("models.py")


def test_graceful_on_empty_dir(tmp_path):
    m = RepoMap().build(tmp_path)
    assert "no source files" in m


def test_graceful_on_non_dir(tmp_path):
    m = RepoMap().build(tmp_path / "missing")
    assert "not a directory" in m


def test_unparseable_file_degrades_to_tree(tmp_path):
    # A .py file with binary garbage: regex finds nothing -> tree fallback
    # only if NO file yields definitions; here one good file exists so the
    # map still renders (bad file simply contributes nothing).
    (tmp_path / "bad.py").write_bytes(b"\x00\x01\x02\xff\xfe" * 100)
    (tmp_path / "good.py").write_text("def ok():\n    return 1\n")
    m = RepoMap().build(tmp_path)
    assert "def ok" in m


def test_all_binary_repo_gives_tree(tmp_path):
    (tmp_path / "bad.py").write_bytes(b"\x00\x01\x02" * 100)
    m = RepoMap().build(tmp_path)
    assert "file tree" in m
    assert "bad.py" in m
