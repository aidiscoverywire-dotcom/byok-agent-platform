"""Graph + tools tests: thread ids, ReAct parsing, tool sandboxing."""
from __future__ import annotations

import pytest

from byok_agent.agent.graph import (
    extract_answer,
    make_thread_id,
    parse_action,
    validate_thread_id,
)
from byok_agent.agent import tools as toolmod


def test_make_thread_id_format():
    assert make_thread_id("u1", "t1") == "user:u1:task:t1"


@pytest.mark.parametrize(
    "bad",
    [
        "user:u1",  # missing task segment
        "user:u1:task:",  # empty tid
        "user::task:t1",  # empty uid
        "user:u1:task:t1:extra",  # extra segment
        "",
        "user:u 1:task:t1",  # whitespace
        "user:u1:task:t/1",  # slash
    ],
)
def test_thread_id_rejects_malformed(bad):
    with pytest.raises(ValueError):
        validate_thread_id(bad)


def test_make_thread_id_rejects_colon_in_parts():
    with pytest.raises(ValueError):
        make_thread_id("u:1", "t1")
    with pytest.raises(ValueError):
        make_thread_id("u1", "")


def test_parse_action_block():
    text = (
        "THOUGHT: I need the time.\n"
        "ACTION: get_current_time\n"
        "ARGS: {}\n"
    )
    assert parse_action(text) == ("get_current_time", {})


def test_parse_action_bad_json():
    text = "ACTION: read_file\nARGS: {not json}\n"
    name, args = parse_action(text)
    assert name == "read_file"
    assert "_raw" in args


def test_parse_action_none_without_action():
    assert parse_action("ANSWER: done") is None


def test_extract_answer():
    assert extract_answer("THOUGHT: x\nANSWER: final text") == "final text"
    assert extract_answer("no answer here") is None


def test_read_file_blocked_outside_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("top secret")
    assert "ERROR" in toolmod.read_file("../secret.txt")
    assert "ERROR" in toolmod.read_file("/etc/passwd")


def test_read_file_inside_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    (tmp_path / "notes.txt").write_text("hello workspace")
    assert toolmod.read_file("notes.txt") == "hello workspace"
    assert "ERROR" in toolmod.read_file("missing.txt")


def test_read_file_symlink_escape_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    outside = tmp_path.parent / "real.txt"
    outside.write_text("nope")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unsupported here")
    assert "ERROR" in toolmod.read_file("link.txt")


def test_unknown_tool_errors():
    assert "ERROR" in toolmod.run_tool("rm_rf", {})


def test_web_search_stub_is_honest():
    out = toolmod.run_tool("web_search_stub", {"query": "x"})
    assert "Phase 3" in out


def test_get_current_time_iso():
    assert "T" in toolmod.get_current_time()
