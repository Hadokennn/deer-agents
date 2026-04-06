# SmartCompressionMiddleware Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe, per-type compression middleware that archives originals to JSONL before replacing messages in state, as a drop-in alternative to LangChain's SummarizationMiddleware.

**Architecture:** Three modules in `summarization/`: archive (JSONL read/write), strategies (pure compression functions), middleware (orchestrator). ThreadState gets a `compressed_messages` field for archive pointers. SummarizationConfig gets a `use_smart_compression` switch. Integration via `_create_summarization_middleware()` in lead_agent.

**Tech Stack:** LangChain AgentMiddleware, LangGraph add_messages (same-ID replace), langchain_core message serialization

**Spec:** `docs/superpowers/specs/2026-04-06-smart-compression-middleware-design.md`

---

## File Structure

```
summarization/
    __init__.py
    compression_archive.py    # JSONL archive read/write (pure IO)
    compression_strategies.py # Per-type compression (pure functions)
    smart_compression.py      # SmartCompressionMiddleware

tests/
    test_compression_archive.py
    test_compression_strategies.py
    test_smart_compression.py

# Modifications to deer-flow harness:
deer-flow/backend/packages/harness/deerflow/agents/thread_state.py
deer-flow/backend/packages/harness/deerflow/config/summarization_config.py
deer-flow/backend/packages/harness/deerflow/agents/lead_agent/agent.py
```

---

### Task 1: JSONL Archive

**Files:**
- Create: `summarization/compression_archive.py`
- Create: `summarization/__init__.py`
- Test: `tests/test_compression_archive.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_compression_archive.py
"""Tests for summarization/compression_archive.py — JSONL archive read/write."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from summarization.compression_archive import archive_messages, read_archived_messages


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

    # Verify JSONL file
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

    # Read only ai-2
    partial = read_archived_messages("t1", {"ai-2": pointers["ai-2"]}, base_dir=tmp_path)
    assert len(partial) == 1
    assert partial[0].content == "b"


def test_archive_empty_list(tmp_path):
    result = archive_messages("t1", [], base_dir=tmp_path)
    assert result == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_compression_archive.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# summarization/__init__.py
# (empty)

# summarization/compression_archive.py
"""JSONL archive for original messages before compression.

Append-only. Each line is one LangChain message serialized via dumpd().
"""

import json
import logging
from pathlib import Path

from langchain_core.load import dumpd, load
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

ARCHIVE_FILENAME = "archive.jsonl"


def _archive_path(thread_id: str, base_dir: Path) -> Path:
    return base_dir / thread_id / ARCHIVE_FILENAME


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def archive_messages(
    thread_id: str,
    messages: list[BaseMessage],
    base_dir: Path = Path(".deer-flow/threads"),
) -> dict[str, dict]:
    """Archive messages to JSONL. Returns {msg_id: {"line": N}} map.

    Appends to existing file. Returns empty dict if messages is empty.
    """
    if not messages:
        return {}

    path = _archive_path(thread_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    start_line = _count_lines(path)
    pointers: dict[str, dict] = {}

    with path.open("a", encoding="utf-8") as f:
        for i, msg in enumerate(messages):
            line = json.dumps(dumpd(msg), ensure_ascii=False)
            f.write(line + "\n")
            pointers[msg.id] = {"line": start_line + i}

    logger.info("Archived %d messages to %s (lines %d-%d)", len(messages), path, start_line, start_line + len(messages) - 1)
    return pointers


def read_archived_messages(
    thread_id: str,
    pointers: dict[str, dict],
    base_dir: Path = Path(".deer-flow/threads"),
) -> list[BaseMessage]:
    """Read archived messages by ID from JSONL. Returns list in pointer order."""
    if not pointers:
        return []

    path = _archive_path(thread_id, base_dir)
    if not path.exists():
        logger.warning("Archive not found: %s", path)
        return []

    # Read needed lines
    needed_lines = {v["line"] for v in pointers.values()}
    line_to_data: dict[int, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in needed_lines:
                line_to_data[i] = line.strip()
            if i > max(needed_lines):
                break

    # Deserialize in pointer order
    result = []
    for msg_id, meta in pointers.items():
        raw = line_to_data.get(meta["line"])
        if raw:
            msg = load(json.loads(raw))
            result.append(msg)
        else:
            logger.warning("Line %d not found in archive for msg %s", meta["line"], msg_id)

    return result
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_compression_archive.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add summarization/__init__.py summarization/compression_archive.py tests/test_compression_archive.py
git commit -m "feat(summarization): JSONL archive for original messages"
```

---

### Task 2: Compression Strategies

**Files:**
- Create: `summarization/compression_strategies.py`
- Test: `tests/test_compression_strategies.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_compression_strategies.py
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
    assert result is None  # None = no compression needed


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
    assert result is None  # Already minimal


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
    assert result is None  # No thinking blocks to strip


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
    assert result is None  # Too short to bother


# ---------------------------------------------------------------------------
# ToolMessage — threshold-based
# ---------------------------------------------------------------------------


def test_tool_small_not_compressed():
    msg = ToolMessage(content="small result", tool_call_id="c1", name="search", id="t1")
    result = compress_message(msg)
    assert result is None  # Under threshold


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
    config = CompressionConfig(tool_message_threshold=10)  # Low threshold
    result = compress_message(msg, config)
    assert result is None  # Errors never compressed


def test_tool_exactly_at_threshold():
    content = "x" * 2048
    msg = ToolMessage(content=content, tool_call_id="c1", name="t", id="t1")
    config = CompressionConfig(tool_message_threshold=2048)
    result = compress_message(msg, config)
    assert result is None  # At threshold, not over
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_compression_strategies.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# summarization/compression_strategies.py
"""Per-type message compression strategies.

All functions are pure — no IO, no LLM, independently testable.
compress_message() returns:
  - None: no compression needed
  - "needs_llm": plain text AIMessage, caller should use LLM summary
  - BaseMessage: compressed copy (same ID, reduced content)
"""

from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


@dataclass
class CompressionConfig:
    tool_message_threshold: int = 2048  # bytes
    tool_truncate_head: int = 500
    tool_truncate_tail: int = 200
    ai_min_text_length: int = 200  # Don't LLM-summarize short texts


def compress_message(
    msg: BaseMessage,
    config: CompressionConfig = CompressionConfig(),
) -> BaseMessage | str | None:
    """Compress a single message.

    Returns:
        None — no compression needed
        "needs_llm" — plain text AI, caller should LLM-summarize
        BaseMessage — compressed copy with same ID
    """
    if isinstance(msg, (HumanMessage, SystemMessage)):
        return None

    if isinstance(msg, AIMessage):
        return _compress_ai(msg, config)

    if isinstance(msg, ToolMessage):
        return _compress_tool(msg, config)

    return None


def _compress_ai(msg: AIMessage, config: CompressionConfig) -> AIMessage | str | None:
    """Compress AIMessage based on content structure."""
    # Case 1: Has tool_calls — keep calls, clear content
    if msg.tool_calls:
        if not msg.content:
            return None  # Already minimal
        return AIMessage(
            content="",
            tool_calls=msg.tool_calls,
            id=msg.id,
        )

    # Case 2: Content is block list — strip thinking blocks
    if isinstance(msg.content, list):
        has_thinking = any(
            isinstance(block, dict) and block.get("type") == "thinking"
            for block in msg.content
        )
        if not has_thinking:
            return None
        kept = [
            block for block in msg.content
            if not (isinstance(block, dict) and block.get("type") == "thinking")
        ]
        return AIMessage(content=kept, id=msg.id)

    # Case 3: Plain text string
    if isinstance(msg.content, str):
        if len(msg.content) < config.ai_min_text_length:
            return None  # Too short to bother
        return "needs_llm"

    return None


def _compress_tool(msg: ToolMessage, config: CompressionConfig) -> ToolMessage | None:
    """Compress ToolMessage if over threshold and not an error."""
    if not isinstance(msg.content, str):
        return None

    if _is_tool_error(msg.content):
        return None

    content_bytes = len(msg.content.encode("utf-8"))
    if content_bytes <= config.tool_message_threshold:
        return None

    head = msg.content[:config.tool_truncate_head]
    tail = msg.content[-config.tool_truncate_tail:]
    truncated = f"{head}\n\n[archived, {content_bytes}B total]\n\n{tail}"

    return ToolMessage(
        content=truncated,
        tool_call_id=msg.tool_call_id,
        name=msg.name,
        id=msg.id,
    )


def _is_tool_error(content: str) -> bool:
    """Check if tool content is an error message."""
    return content.startswith("Error:")
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_compression_strategies.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add summarization/compression_strategies.py tests/test_compression_strategies.py
git commit -m "feat(summarization): per-type compression strategies"
```

---

### Task 3: ThreadState + Config Changes

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/thread_state.py`
- Modify: `deer-flow/backend/packages/harness/deerflow/config/summarization_config.py`

- [ ] **Step 1: Add `compressed_messages` to ThreadState**

In `deer-flow/backend/packages/harness/deerflow/agents/thread_state.py`, add the reducer and field:

```python
# Add after merge_viewed_images function (around line 45):

def merge_compressed_messages(
    existing: dict[str, dict] | None, new: dict[str, dict] | None
) -> dict[str, dict]:
    """Reducer for compressed_messages — merge new pointers into existing."""
    if existing is None:
        return new or {}
    if new is None:
        return existing
    return {**existing, **new}
```

In the `ThreadState` class, add after `viewed_images`:

```python
    compressed_messages: Annotated[dict[str, dict], merge_compressed_messages]
```

- [ ] **Step 2: Add `use_smart_compression` + `CompressionConfig` to SummarizationConfig**

In `deer-flow/backend/packages/harness/deerflow/config/summarization_config.py`, add after imports:

```python
class CompressionSettings(BaseModel):
    """Settings for SmartCompressionMiddleware."""
    tool_message_threshold: int = Field(default=2048, description="Bytes threshold for ToolMessage compression")
    tool_truncate_head: int = Field(default=500, description="Bytes to keep from start of truncated ToolMessage")
    tool_truncate_tail: int = Field(default=200, description="Bytes to keep from end of truncated ToolMessage")
    ai_summary_model: str | None = Field(default=None, description="Model for AI text summarization (None = use summarization model)")
```

In the `SummarizationConfig` class, add two fields:

```python
    use_smart_compression: bool = Field(
        default=False,
        description="Use SmartCompressionMiddleware instead of default SummarizationMiddleware",
    )
    compression: CompressionSettings = Field(
        default_factory=CompressionSettings,
        description="Settings for SmartCompressionMiddleware (only when use_smart_compression=True)",
    )
```

- [ ] **Step 3: Verify existing tests still pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All 81 existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add deer-flow/backend/packages/harness/deerflow/agents/thread_state.py deer-flow/backend/packages/harness/deerflow/config/summarization_config.py
git commit -m "feat(summarization): add compressed_messages state + smart compression config"
```

---

### Task 4: SmartCompressionMiddleware

**Files:**
- Create: `summarization/smart_compression.py`
- Test: `tests/test_smart_compression.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_smart_compression.py
"""Tests for summarization/smart_compression.py — SmartCompressionMiddleware."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from summarization.smart_compression import SmartCompressionMiddleware


def _make_state(messages, compressed_messages=None):
    return {
        "messages": messages,
        "compressed_messages": compressed_messages or {},
    }


def _make_runtime():
    return MagicMock()


# ---------------------------------------------------------------------------
# Trigger logic
# ---------------------------------------------------------------------------


def test_no_compression_below_trigger():
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 100)],
        keep=("messages", 5),
    )
    state = _make_state([HumanMessage(content="hi", id="h1")])
    result = mw.before_model(state, _make_runtime())
    assert result is None


def test_compression_triggers_on_message_count(tmp_path):
    msgs = [HumanMessage(content=f"msg {i}", id=f"h{i}") for i in range(10)]
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 5)],
        keep=("messages", 3),
        archive_base_dir=str(tmp_path),
    )
    state = _make_state(msgs)
    result = mw.before_model(state, _make_runtime())

    assert result is not None
    assert "messages" in result
    # All 10 messages returned (some compressed, some preserved)
    assert len(result["messages"]) == 10
    # compressed_messages should have pointers for archived messages
    assert "compressed_messages" in result


# ---------------------------------------------------------------------------
# Per-type strategy application
# ---------------------------------------------------------------------------


def test_human_messages_preserved(tmp_path):
    msgs = [
        HumanMessage(content=f"question {i}", id=f"h{i}") for i in range(10)
    ]
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 5)],
        keep=("messages", 3),
        archive_base_dir=str(tmp_path),
    )
    state = _make_state(msgs)
    result = mw.before_model(state, _make_runtime())

    # HumanMessages are never compressed — returned as-is
    for msg in result["messages"]:
        assert msg.content.startswith("question")


def test_ai_tool_calls_compressed(tmp_path):
    msgs = [
        HumanMessage(content="help", id="h1"),
        AIMessage(content="let me think...", tool_calls=[{"name": "search", "args": {}, "id": "call-1"}], id="ai-1"),
        ToolMessage(content="small result", tool_call_id="call-1", name="search", id="t1"),
        HumanMessage(content="thanks", id="h2"),
        AIMessage(content="you're welcome", id="ai-2"),
    ]
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 3)],
        keep=("messages", 2),
        archive_base_dir=str(tmp_path),
    )
    state = _make_state(msgs)
    result = mw.before_model(state, _make_runtime())

    assert result is not None
    # Find the compressed AI message (ai-1 is in compression zone)
    ai1 = next(m for m in result["messages"] if getattr(m, "id", None) == "ai-1")
    assert ai1.content == ""  # Content cleared
    assert len(ai1.tool_calls) == 1  # tool_calls preserved


def test_large_tool_message_truncated(tmp_path):
    big_content = "x" * 5000
    msgs = [
        HumanMessage(content="q", id="h1"),
        AIMessage(content="", tool_calls=[{"name": "mcp", "args": {}, "id": "call-1"}], id="ai-1"),
        ToolMessage(content=big_content, tool_call_id="call-1", name="mcp", id="t1"),
        HumanMessage(content="ok", id="h2"),
        AIMessage(content="done", id="ai-2"),
    ]
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 3)],
        keep=("messages", 2),
        archive_base_dir=str(tmp_path),
    )
    state = _make_state(msgs)
    result = mw.before_model(state, _make_runtime())

    t1 = next(m for m in result["messages"] if getattr(m, "id", None) == "t1")
    assert "[archived" in t1.content
    assert len(t1.content) < 5000


def test_tool_error_not_compressed(tmp_path):
    msgs = [
        HumanMessage(content="q", id="h1"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c1"}], id="ai-1"),
        ToolMessage(content="Error: " + "x" * 5000, tool_call_id="c1", name="t", id="t1"),
        HumanMessage(content="ok", id="h2"),
        AIMessage(content="done", id="ai-2"),
    ]
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 3)],
        keep=("messages", 2),
        archive_base_dir=str(tmp_path),
    )
    state = _make_state(msgs)
    result = mw.before_model(state, _make_runtime())

    t1 = next(m for m in result["messages"] if getattr(m, "id", None) == "t1")
    assert t1.content.startswith("Error:")  # Unchanged


# ---------------------------------------------------------------------------
# Archive safety
# ---------------------------------------------------------------------------


def test_archive_written_before_state_update(tmp_path):
    msgs = [
        HumanMessage(content="q", id="h1"),
        AIMessage(content="long " * 100, tool_calls=[{"name": "s", "args": {}, "id": "c1"}], id="ai-1"),
        ToolMessage(content="result", tool_call_id="c1", name="s", id="t1"),
        HumanMessage(content="ok", id="h2"),
    ]
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 2)],
        keep=("messages", 1),
        archive_base_dir=str(tmp_path),
    )
    # Use a fixed thread_id via runtime mock
    runtime = _make_runtime()
    runtime.config = {"configurable": {"thread_id": "test-thread"}}
    state = _make_state(msgs)
    result = mw.before_model(state, _make_runtime())

    # Archive file should exist
    # (thread_id comes from state or defaults)
    archive_files = list(tmp_path.rglob("archive.jsonl"))
    assert len(archive_files) >= 1


def test_compressed_messages_pointers_returned(tmp_path):
    msgs = [
        HumanMessage(content="q", id="h1"),
        AIMessage(content="analysis...", tool_calls=[{"name": "s", "args": {}, "id": "c1"}], id="ai-1"),
        ToolMessage(content="x" * 5000, tool_call_id="c1", name="s", id="t1"),
        HumanMessage(content="ok", id="h2"),
    ]
    mw = SmartCompressionMiddleware(
        trigger=[("messages", 2)],
        keep=("messages", 1),
        archive_base_dir=str(tmp_path),
    )
    state = _make_state(msgs)
    result = mw.before_model(state, _make_runtime())

    assert "compressed_messages" in result
    cm = result["compressed_messages"]
    # ai-1 had content stripped, t1 was truncated — both should be archived
    assert "ai-1" in cm
    assert "t1" in cm
    assert "line" in cm["ai-1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_smart_compression.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# summarization/smart_compression.py
"""SmartCompressionMiddleware — safe, per-type message compression.

Drop-in replacement for SummarizationMiddleware (position 8 in middleware chain).
Archives originals to JSONL before replacing with compressed versions in state.
"""

import logging
import uuid
from pathlib import Path
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage

from summarization.compression_archive import archive_messages
from summarization.compression_strategies import CompressionConfig, compress_message

logger = logging.getLogger(__name__)

ContextSize = tuple[str, int | float]


class SmartCompressionMiddleware(AgentMiddleware[AgentState]):
    """Compress messages by type, archive originals to JSONL."""

    def __init__(
        self,
        trigger: ContextSize | list[ContextSize] | None = None,
        keep: ContextSize = ("messages", 20),
        compression_config: CompressionConfig | None = None,
        model: Any = None,
        archive_base_dir: str = ".deer-flow/threads",
    ):
        self._trigger_conditions: list[ContextSize] = []
        if trigger is not None:
            self._trigger_conditions = trigger if isinstance(trigger, list) else [trigger]
        self._keep = keep
        self._config = compression_config or CompressionConfig()
        self._model = model
        self._archive_base_dir = Path(archive_base_dir)

    def _should_compress(self, messages: list[BaseMessage]) -> bool:
        if not self._trigger_conditions:
            return False
        for ctype, cvalue in self._trigger_conditions:
            if ctype == "messages" and len(messages) > cvalue:
                return True
            if ctype == "tokens":
                from langchain.agents.middleware.summarization import count_tokens_approximately
                total = count_tokens_approximately(messages)
                if total > cvalue:
                    return True
        return False

    def _determine_cutoff(self, messages: list[BaseMessage]) -> int:
        """Determine index: messages[:cutoff] = compress zone, [cutoff:] = keep zone."""
        ktype, kvalue = self._keep
        if ktype == "messages":
            cutoff = max(0, len(messages) - int(kvalue))
        elif ktype == "tokens":
            from langchain.agents.middleware.summarization import count_tokens_approximately
            total = 0
            cutoff = len(messages)
            for i in range(len(messages) - 1, -1, -1):
                total += count_tokens_approximately([messages[i]])
                if total > kvalue:
                    cutoff = i + 1
                    break
            else:
                cutoff = 0
        else:
            cutoff = 0

        # Don't split AI+Tool pairs: if cutoff lands between them, move cutoff back
        if cutoff > 0 and cutoff < len(messages):
            msg_at_cutoff = messages[cutoff]
            if hasattr(msg_at_cutoff, "tool_call_id"):
                # This is a ToolMessage — its AIMessage is before it, move cutoff back
                cutoff = max(0, cutoff - 1)

        return cutoff

    def _get_thread_id(self, state: dict, runtime: Any) -> str:
        """Extract thread_id from runtime config or generate one."""
        try:
            config = getattr(runtime, "config", {})
            tid = config.get("configurable", {}).get("thread_id")
            if tid:
                return tid
        except Exception:
            pass
        return f"unknown-{uuid.uuid4().hex[:8]}"

    def _compress_zone(
        self, messages: list[BaseMessage], thread_id: str
    ) -> tuple[list[BaseMessage], dict[str, dict]]:
        """Compress messages in the compression zone.

        Returns (compressed_messages, archive_pointers).
        """
        to_archive = []
        compressed = []

        for msg in messages:
            result = compress_message(msg, self._config)

            if result is None:
                # No compression needed — keep original
                compressed.append(msg)
            elif result == "needs_llm":
                # LLM summary needed — for now, keep original (LLM path deferred)
                if self._model is not None:
                    summary = self._llm_summarize(msg)
                    to_archive.append(msg)
                    compressed.append(summary)
                else:
                    compressed.append(msg)
            else:
                # Rule-based compression succeeded
                to_archive.append(msg)
                compressed.append(result)

        # Archive originals
        pointers = {}
        if to_archive:
            try:
                pointers = archive_messages(thread_id, to_archive, self._archive_base_dir)
            except Exception:
                logger.exception("Archive failed, skipping compression")
                return messages, {}  # Return originals on archive failure

        return compressed, pointers

    def _llm_summarize(self, msg: AIMessage) -> AIMessage:
        """Use LLM to summarize a plain-text AIMessage."""
        prompt = (
            "请压缩以下 AI 回复，保留架构决策和结论，删除推理过程。"
            "只输出压缩后的内容，不要解释。\n\n"
            f"{msg.content}"
        )
        from langchain_core.messages import HumanMessage
        response = self._model.invoke([HumanMessage(content=prompt)])
        return AIMessage(content=response.content, id=msg.id)

    def before_model(self, state: dict, runtime: Any) -> dict[str, Any] | None:
        """Check trigger, archive, compress, return state update."""
        messages = state.get("messages", [])

        # Ensure all messages have IDs
        for msg in messages:
            if msg.id is None:
                msg.id = str(uuid.uuid4())

        if not self._should_compress(messages):
            return None

        cutoff = self._determine_cutoff(messages)
        if cutoff <= 0:
            return None

        compress_zone = messages[:cutoff]
        keep_zone = messages[cutoff:]

        thread_id = self._get_thread_id(state, runtime)
        compressed, pointers = self._compress_zone(compress_zone, thread_id)

        if not pointers:
            return None  # Archive failed or nothing to compress

        logger.info(
            "Compressed %d/%d messages (thread=%s, archived=%d)",
            len(pointers), len(compress_zone), thread_id, len(pointers),
        )

        # Return ALL messages (compressed zone + keep zone) for same-ID replacement
        return {
            "messages": compressed + keep_zone,
            "compressed_messages": pointers,
        }

    async def abefore_model(self, state: dict, runtime: Any) -> dict[str, Any] | None:
        """Async version — delegates to sync (archive IO is fast)."""
        return self.before_model(state, runtime)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_smart_compression.py -v`
Expected: PASS

- [ ] **Step 5: Run all tests for regression**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add summarization/smart_compression.py tests/test_smart_compression.py
git commit -m "feat(summarization): SmartCompressionMiddleware with archive-first safety"
```

---

### Task 5: Integration — Wire Up Switch

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/lead_agent/agent.py`

- [ ] **Step 1: Modify `_create_summarization_middleware()`**

In `deer-flow/backend/packages/harness/deerflow/agents/lead_agent/agent.py`, find `_create_summarization_middleware()` (around line 41) and add the switch:

```python
def _create_summarization_middleware() -> AgentMiddleware | None:
    """Create and configure the summarization middleware from config."""
    config = get_summarization_config()

    if not config.enabled:
        return None

    # --- Smart compression switch ---
    if config.use_smart_compression:
        return _create_smart_compression_middleware(config)

    # --- Original SummarizationMiddleware (default) ---
    # ... existing code unchanged ...
```

Add the new factory function right below:

```python
def _create_smart_compression_middleware(config) -> AgentMiddleware:
    """Create SmartCompressionMiddleware from config."""
    from summarization.smart_compression import SmartCompressionMiddleware
    from summarization.compression_strategies import CompressionConfig

    trigger = None
    if config.trigger is not None:
        if isinstance(config.trigger, list):
            trigger = [t.to_tuple() for t in config.trigger]
        else:
            trigger = [config.trigger.to_tuple()]

    keep = config.keep.to_tuple()

    compression_config = CompressionConfig(
        tool_message_threshold=config.compression.tool_message_threshold,
        tool_truncate_head=config.compression.tool_truncate_head,
        tool_truncate_tail=config.compression.tool_truncate_tail,
    )

    model = None
    model_name = config.compression.ai_summary_model or config.model_name
    if model_name:
        from deerflow.models import create_chat_model
        model = create_chat_model(name=model_name, thinking_enabled=False)

    return SmartCompressionMiddleware(
        trigger=trigger,
        keep=keep,
        compression_config=compression_config,
        model=model,
    )
```

- [ ] **Step 2: Verify default behavior unchanged**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS (use_smart_compression defaults to False, no behavior change)

- [ ] **Step 3: Commit**

```bash
git add deer-flow/backend/packages/harness/deerflow/agents/lead_agent/agent.py
git commit -m "feat(summarization): wire up use_smart_compression switch in lead_agent"
```

---

### Task 6: Config + E2E Smoke Test

- [ ] **Step 1: Enable in deer-flow/config.yaml for testing**

Add to the summarization section in `deer-flow/config.yaml`:

```yaml
summarization:
  enabled: true
  use_smart_compression: true
  trigger:
    - type: messages
      value: 10
  keep:
    type: messages
    value: 5
  compression:
    tool_message_threshold: 2048
```

- [ ] **Step 2: Run e2e test**

Run: `source .venv/bin/activate && export $(cat deer-flow/.env | grep -v '^#' | xargs) && python scripts/e2e_test.py`
Expected: All steps pass — basic chat works with SmartCompressionMiddleware enabled

- [ ] **Step 3: Revert config to default (use_smart_compression: false)**

Set `use_smart_compression: false` in `deer-flow/config.yaml` to keep default behavior.

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add deer-flow/config.yaml
git commit -m "feat(summarization): complete SmartCompressionMiddleware integration"
```

---

## Verification

After all tasks, verify:

1. **Unit tests**: `.venv/bin/python -m pytest tests/test_compression_archive.py tests/test_compression_strategies.py tests/test_smart_compression.py -v` — all PASS
2. **No regression**: `.venv/bin/python -m pytest tests/ -v` — all PASS
3. **Default behavior**: `use_smart_compression: false` → original SummarizationMiddleware, zero change
4. **Archive created**: After a compressed conversation, `.deer-flow/threads/{tid}/archive.jsonl` exists with original messages
5. **Same-ID replacement**: Compressed messages in state have same IDs as originals (not new messages)
