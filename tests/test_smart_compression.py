"""Tests for summarization/smart_compression.py — SmartCompressionMiddleware."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from summarization.smart_compression import SmartCompressionMiddleware


def _make_state(messages, compressed_messages=None):
    return {"messages": messages, "compressed_messages": compressed_messages or {}}


def _make_runtime(thread_id="test-thread"):
    runtime = MagicMock()
    runtime.context = {"thread_id": thread_id}
    return runtime


# ---------------------------------------------------------------------------
# Trigger conditions
# ---------------------------------------------------------------------------


def test_no_compression_below_trigger():
    mw = SmartCompressionMiddleware(trigger=[("messages", 100)], keep=("messages", 3))
    msgs = [HumanMessage(content=f"msg-{i}", id=f"h{i}") for i in range(5)]
    result = mw.before_model(_make_state(msgs), _make_runtime())
    assert result is None


def test_compression_triggers_on_message_count(tmp_path):
    """10 messages with compressible content, trigger at 5, keep 3."""
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 5)],
        keep=("messages", 3),
        archive_base_dir=tmp_path,
    )
    msgs = []
    for i in range(5):
        msgs.append(HumanMessage(content=f"q{i}", id=f"h{i}"))
        msgs.append(AIMessage(
            content=f"let me analyze {i}...",
            tool_calls=[{"name": "search", "args": {}, "id": f"c{i}"}],
            id=f"ai{i}",
        ))
    state = _make_state(msgs)
    result = mw.before_model(state, _make_runtime())
    assert result is not None
    assert len(result["messages"]) == 10
    assert "compressed_messages" in result


# ---------------------------------------------------------------------------
# Per-type compression
# ---------------------------------------------------------------------------


def test_human_messages_preserved(tmp_path):
    """HumanMessages in compress zone stay unchanged."""
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 3)],
        keep=("messages", 1),
        archive_base_dir=tmp_path,
    )
    msgs = [
        HumanMessage(content="question-1", id="h1"),
        AIMessage(content="reasoning...", tool_calls=[{"name": "t", "args": {}, "id": "c1"}], id="ai1"),
        ToolMessage(content="result", tool_call_id="c1", name="t", id="t1"),
        HumanMessage(content="question-2", id="h2"),
    ]
    result = mw.before_model(_make_state(msgs), _make_runtime())
    assert result is not None
    h1 = next(m for m in result["messages"] if m.id == "h1")
    assert h1.content == "question-1"


def test_ai_tool_calls_compressed(tmp_path):
    """AIMessage with tool_calls has content cleared, tool_calls preserved."""
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 3)],
        keep=("messages", 1),
        archive_base_dir=tmp_path,
    )
    msgs = [
        HumanMessage(content="q", id="h1"),
        AIMessage(
            content="let me analyze the schema...",
            tool_calls=[{"name": "locate_field_schema", "args": {"f": "x"}, "id": "c1"}],
            id="ai1",
        ),
        ToolMessage(content="small", tool_call_id="c1", name="locate_field_schema", id="t1"),
        HumanMessage(content="thanks", id="h2"),
    ]
    result = mw.before_model(_make_state(msgs), _make_runtime())
    assert result is not None
    ai1 = next(m for m in result["messages"] if m.id == "ai1")
    assert ai1.content == ""
    assert len(ai1.tool_calls) == 1
    assert ai1.tool_calls[0]["name"] == "locate_field_schema"


def test_large_tool_message_truncated(tmp_path):
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 3)],
        keep=("messages", 1),
        archive_base_dir=tmp_path,
    )
    big_content = "x" * 5000
    msgs = [
        HumanMessage(content="q", id="h1"),
        AIMessage(content="thinking...", tool_calls=[{"name": "mcp", "args": {}, "id": "c1"}], id="ai1"),
        ToolMessage(content=big_content, tool_call_id="c1", name="mcp", id="t1"),
        HumanMessage(content="ok", id="h2"),
    ]
    result = mw.before_model(_make_state(msgs), _make_runtime())
    assert result is not None
    t1 = next(m for m in result["messages"] if m.id == "t1")
    assert len(t1.content) < 5000
    assert "[archived" in t1.content


def test_tool_error_not_compressed(tmp_path):
    """Error ToolMessage preserved, but AIMessage content still compressed."""
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 3)],
        keep=("messages", 1),
        archive_base_dir=tmp_path,
    )
    msgs = [
        HumanMessage(content="q", id="h1"),
        AIMessage(content="let me try...", tool_calls=[{"name": "t", "args": {}, "id": "c1"}], id="ai1"),
        ToolMessage(content="Error: connection refused", tool_call_id="c1", name="t", id="t1"),
        HumanMessage(content="ok", id="h2"),
    ]
    result = mw.before_model(_make_state(msgs), _make_runtime())
    assert result is not None
    t1 = next(m for m in result["messages"] if m.id == "t1")
    assert t1.content == "Error: connection refused"  # Preserved
    ai1 = next(m for m in result["messages"] if m.id == "ai1")
    assert ai1.content == ""  # AI content was compressed


# ---------------------------------------------------------------------------
# Archive & pointers
# ---------------------------------------------------------------------------


def test_archive_written(tmp_path):
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 3)],
        keep=("messages", 1),
        archive_base_dir=tmp_path,
    )
    msgs = [
        HumanMessage(content="q", id="h1"),
        AIMessage(content="analysis...", tool_calls=[{"name": "t", "args": {}, "id": "c1"}], id="ai1"),
        ToolMessage(content="x" * 5000, tool_call_id="c1", name="t", id="t1"),
        HumanMessage(content="ok", id="h2"),
    ]
    mw.before_model(_make_state(msgs), _make_runtime())
    archive = tmp_path / "test-thread" / "archive.jsonl"
    assert archive.exists()


def test_pointers_returned(tmp_path):
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 3)],
        keep=("messages", 1),
        archive_base_dir=tmp_path,
    )
    msgs = [
        HumanMessage(content="q", id="h1"),
        AIMessage(content="reasoning...", tool_calls=[{"name": "t", "args": {}, "id": "c1"}], id="ai1"),
        ToolMessage(content="x" * 5000, tool_call_id="c1", name="t", id="t1"),
        HumanMessage(content="ok", id="h2"),
    ]
    result = mw.before_model(_make_state(msgs), _make_runtime())
    assert result is not None
    cm = result["compressed_messages"]
    assert "ai1" in cm  # Content cleared
    assert "t1" in cm   # Truncated
    assert "line" in cm["ai1"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_ai_tool_pair_not_split(tmp_path):
    """Cutoff adjusted to not split AI+Tool pair."""
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 3)],
        keep=("messages", 2),
        archive_base_dir=tmp_path,
    )
    msgs = [
        HumanMessage(content="q1", id="h1"),
        AIMessage(content="analyzing...", tool_calls=[{"name": "t", "args": {}, "id": "c1"}], id="ai1"),
        ToolMessage(content="x" * 5000, tool_call_id="c1", name="t", id="t1"),
        HumanMessage(content="q2", id="h2"),
        AIMessage(content="done", id="ai2"),
    ]
    result = mw.before_model(_make_state(msgs), _make_runtime())
    assert result is not None
    assert len(result["messages"]) == 5


def test_needs_llm_without_model_skips(tmp_path):
    """Plain text AI + no model → skip that message, but compress others."""
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 3)],
        keep=("messages", 1),
        archive_base_dir=tmp_path,
        model=None,
    )
    long_text = "This is a long analysis. " * 50
    msgs = [
        HumanMessage(content="q", id="h1"),
        AIMessage(content=long_text, id="ai1"),
        AIMessage(content="reasoning...", tool_calls=[{"name": "t", "args": {}, "id": "c1"}], id="ai2"),
        ToolMessage(content="result", tool_call_id="c1", name="t", id="t2"),
        HumanMessage(content="ok", id="h2"),
    ]
    result = mw.before_model(_make_state(msgs), _make_runtime())
    assert result is not None
    ai1 = next(m for m in result["messages"] if m.id == "ai1")
    assert ai1.content == long_text  # Preserved — no model for LLM summary
    ai2 = next(m for m in result["messages"] if m.id == "ai2")
    assert ai2.content == ""  # This one was compressed (had tool_calls)


def test_nothing_compressible_returns_none(tmp_path):
    """All HumanMessages → nothing to compress → returns None."""
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 3)],
        keep=("messages", 1),
        archive_base_dir=tmp_path,
    )
    msgs = [HumanMessage(content=f"q{i}", id=f"h{i}") for i in range(5)]
    result = mw.before_model(_make_state(msgs), _make_runtime())
    assert result is None  # Correct — nothing to compress
