"""Tests for summarization/compression_strategies.py — per-type compression."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from summarization.compression_strategies import (
    CompressionConfig,
    compress_message,
)


# ---------------------------------------------------------------------------
# HumanMessage / SystemMessage — never compress
# ---------------------------------------------------------------------------


def test_human_message_not_compressed():
    msg = HumanMessage(content="用户的问题", id="h1")
    result = compress_message(msg)
    assert result is None


def test_system_message_not_compressed():
    msg = SystemMessage(content="system prompt", id="s1")
    result = compress_message(msg)
    assert result is None


# ---------------------------------------------------------------------------
# AIMessage with tool_calls — keep tool_calls, clear content
# ---------------------------------------------------------------------------


def test_ai_with_tool_calls_strips_content():
    msg = AIMessage(
        content="让我分析一下这个字段的显隐逻辑，首先查看 reaction_rules...",
        tool_calls=[{"name": "locate_field_schema", "args": {"field_name": "价格"}, "id": "call-1"}],
        id="ai-1",
    )
    result = compress_message(msg)
    assert result is not None
    assert result.id == "ai-1"
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "locate_field_schema"


def test_ai_with_tool_calls_empty_content_not_compressed():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {}, "id": "call-1"}],
        id="ai-1",
    )
    result = compress_message(msg)
    assert result is None


# ---------------------------------------------------------------------------
# AIMessage with thinking blocks — strip thinking, keep text
# ---------------------------------------------------------------------------


def test_ai_strip_thinking_blocks():
    msg = AIMessage(
        content=[
            {"type": "thinking", "text": "让我想想这个问题...首先需要考虑..."},
            {"type": "text", "text": "该字段由 reaction_rules 控制显隐。"},
        ],
        id="ai-2",
    )
    result = compress_message(msg)
    assert result is not None
    assert result.id == "ai-2"
    assert isinstance(result.content, list)
    assert len(result.content) == 1
    assert result.content[0]["type"] == "text"


def test_ai_no_thinking_blocks_not_compressed():
    msg = AIMessage(
        content=[{"type": "text", "text": "只有文本"}],
        id="ai-3",
    )
    result = compress_message(msg)
    assert result is None


# ---------------------------------------------------------------------------
# AIMessage with plain text — returns "needs_llm" sentinel
# ---------------------------------------------------------------------------


def test_ai_plain_text_returns_needs_llm():
    msg = AIMessage(
        content="这是一段很长的纯文本分析，包含推理过程和结论..." * 20,
        id="ai-4",
    )
    result = compress_message(msg)
    assert result == "needs_llm"


def test_ai_short_plain_text_not_compressed():
    msg = AIMessage(content="OK", id="ai-5")
    result = compress_message(msg)
    assert result is None


# ---------------------------------------------------------------------------
# ToolMessage — threshold-based
# ---------------------------------------------------------------------------


def test_tool_small_not_compressed():
    msg = ToolMessage(content="small result", tool_call_id="c1", name="search", id="t1")
    result = compress_message(msg)
    assert result is None


def test_tool_large_truncated():
    big_content = "x" * 5000
    msg = ToolMessage(content=big_content, tool_call_id="c1", name="mcp_tool", id="t1")
    config = CompressionConfig(tool_message_threshold=2048, tool_truncate_head=500, tool_truncate_tail=200)
    result = compress_message(msg, config)
    assert result is not None
    assert result.id == "t1"
    assert len(result.content) < len(big_content)
    assert "[archived" in result.content
    assert result.content.startswith("x" * 500)
    assert result.content.endswith("x" * 200)


def test_tool_error_not_compressed():
    msg = ToolMessage(content="Error: connection refused", tool_call_id="c1", name="search", id="t1")
    config = CompressionConfig(tool_message_threshold=10)
    result = compress_message(msg, config)
    assert result is None


def test_tool_exactly_at_threshold():
    content = "x" * 2048
    msg = ToolMessage(content=content, tool_call_id="c1", name="t", id="t1")
    config = CompressionConfig(tool_message_threshold=2048)
    result = compress_message(msg, config)
    assert result is None
