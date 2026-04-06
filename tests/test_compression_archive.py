"""Tests for summarization/compression_archive.py — JSONL archive read/write."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from middlewares.summarization.compression_archive import archive_messages, read_archived_messages


def test_archive_creates_jsonl(tmp_path):
    msgs = [
        AIMessage(content="分析结果", id="ai-1"),
        ToolMessage(content="tool output", tool_call_id="call-1", name="search", id="tool-1"),
    ]
    result = archive_messages("thread-abc", msgs, base_dir=tmp_path)

    assert "ai-1" in result
    assert "tool-1" in result
    assert result["ai-1"]["line"] == 0
    assert result["tool-1"]["line"] == 1

    archive_path = tmp_path / "thread-abc" / "archive.jsonl"
    assert archive_path.exists()
    lines = archive_path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_archive_appends(tmp_path):
    msg1 = [AIMessage(content="first", id="ai-1")]
    msg2 = [AIMessage(content="second", id="ai-2")]

    r1 = archive_messages("t1", msg1, base_dir=tmp_path)
    r2 = archive_messages("t1", msg2, base_dir=tmp_path)

    assert r1["ai-1"]["line"] == 0
    assert r2["ai-2"]["line"] == 1

    lines = (tmp_path / "t1" / "archive.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2


def test_read_archived_messages(tmp_path):
    msgs = [
        AIMessage(content="hello", id="ai-1"),
        ToolMessage(content='{"data": "big"}', tool_call_id="call-1", name="mcp_tool", id="tool-1"),
    ]
    pointers = archive_messages("t1", msgs, base_dir=tmp_path)

    restored = read_archived_messages("t1", pointers, base_dir=tmp_path)
    assert len(restored) == 2
    assert restored[0].id == "ai-1"
    assert restored[0].content == "hello"
    assert restored[1].id == "tool-1"
    assert restored[1].content == '{"data": "big"}'


def test_read_partial(tmp_path):
    msgs = [
        AIMessage(content="a", id="ai-1"),
        AIMessage(content="b", id="ai-2"),
        AIMessage(content="c", id="ai-3"),
    ]
    pointers = archive_messages("t1", msgs, base_dir=tmp_path)

    partial = read_archived_messages("t1", {"ai-2": pointers["ai-2"]}, base_dir=tmp_path)
    assert len(partial) == 1
    assert partial[0].content == "b"


def test_archive_empty_list(tmp_path):
    result = archive_messages("t1", [], base_dir=tmp_path)
    assert result == {}
