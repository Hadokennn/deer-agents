# LoopDetectionMiddleware Rewind V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `LoopDetectionMiddleware` with `wrap_model_call` view-layer patching that excises repeated tool-call loops from the LLM-visible message view (without modifying state), inserts a rule-based "ruled out" hint, and removes the now-superseded warn tier.

**Architecture:** Middleware gains a `wrap_model_call` hook that scans `request.messages` for tool-call hash repetitions ≥ `rewind_threshold`, builds a structured hint per merged loop region, and substitutes the loop region with the hint via `request.override(messages=patched)`. State is never touched. Hard-stop in `after_model` remains as a final safety net. Observability adds first-detection dedup so each unique loop reports exactly once per thread.

**Tech Stack:** Python 3.12, LangGraph, LangChain agents middleware, pytest 8+, uv workspace.

**Spec reference:** `docs/superpowers/specs/2026-04-16-loop-detection-rewind-design.md`

---

## File Structure

```
deer-flow/backend/packages/harness/deerflow/agents/middlewares/
    loop_hash.py                    # NEW: pure hash + arg utilities (extracted)
    loop_hint_builder.py            # NEW: rule-based hint generation (pure functions)
    loop_detection_middleware.py    # MODIFIED: wrap_model_call patching, hard_stop only,
                                    #           observability dedup, drop warn tier

deer-flow/backend/tests/
    test_loop_hash.py               # NEW: unit tests for hash utilities
    test_loop_hint_builder.py       # NEW: unit tests for hint generation
    test_loop_detection_middleware.py   # MODIFIED: drop warn tests, add wrap_model_call tests
```

**Responsibilities:**

| File | Responsibility |
|------|---------------|
| `loop_hash.py` | Compute stable hash of tool-call multisets; salient-arg extraction; arg normalization |
| `loop_hint_builder.py` | Build human-readable rule-based "ruled out" hint from loop region; extract original intent; format failure groups |
| `loop_detection_middleware.py` | Detect loops in messages; merge overlapping regions; apply patches; hard-stop fallback; observability dedup |

---

## Test Conventions

- All tests live in `deer-flow/backend/tests/`
- Run a single test file: `cd deer-flow/backend && uv run pytest tests/test_X.py -v`
- Run a single test: `cd deer-flow/backend && uv run pytest tests/test_X.py::TestClass::test_method -v`
- Mock helpers: existing tests use `unittest.mock.MagicMock` for Runtime/handler
- Reuse helper patterns from `tests/test_dangling_tool_call_middleware.py` (similar `wrap_model_call` middleware)

---

## Task 1: Extract pure hash utilities into new module

**Files:**
- Create: `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_hash.py`
- Create: `deer-flow/backend/tests/test_loop_hash.py`

- [ ] **Step 1: Write the failing test for `_hash_tool_calls` import from new module**

Create `deer-flow/backend/tests/test_loop_hash.py`:

```python
"""Tests for loop_hash utility module."""

from deerflow.agents.middlewares.loop_hash import (
    hash_tool_calls,
    normalize_tool_call_args,
    stable_tool_key,
)


def _bash_call(cmd="ls"):
    return {"name": "bash", "id": f"call_{cmd}", "args": {"command": cmd}}


class TestHashToolCalls:
    def test_same_calls_same_hash(self):
        assert hash_tool_calls([_bash_call("ls")]) == hash_tool_calls([_bash_call("ls")])

    def test_different_calls_different_hash(self):
        assert hash_tool_calls([_bash_call("ls")]) != hash_tool_calls([_bash_call("pwd")])

    def test_order_independent(self):
        a = hash_tool_calls([_bash_call("ls"), {"name": "read_file", "args": {"path": "/tmp"}}])
        b = hash_tool_calls([{"name": "read_file", "args": {"path": "/tmp"}}, _bash_call("ls")])
        assert a == b


class TestNormalizeArgs:
    def test_dict_passthrough(self):
        result, fallback = normalize_tool_call_args({"a": 1})
        assert result == {"a": 1}
        assert fallback is None

    def test_string_json_parsed(self):
        result, fallback = normalize_tool_call_args('{"a": 1}')
        assert result == {"a": 1}
        assert fallback is None

    def test_invalid_json_string_to_fallback(self):
        result, fallback = normalize_tool_call_args("not-json")
        assert result == {}
        assert fallback == "not-json"


class TestStableToolKey:
    def test_read_file_buckets_lines(self):
        a = stable_tool_key("read_file", {"path": "/x.py", "start_line": 1, "end_line": 10}, None)
        b = stable_tool_key("read_file", {"path": "/x.py", "start_line": 5, "end_line": 50}, None)
        assert a == b   # same 200-line bucket

    def test_write_file_uses_full_args(self):
        a = stable_tool_key("write_file", {"path": "/x.py", "content": "v1"}, None)
        b = stable_tool_key("write_file", {"path": "/x.py", "content": "v2"}, None)
        assert a != b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_hash.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deerflow.agents.middlewares.loop_hash'`

- [ ] **Step 3: Create `loop_hash.py` by extracting from `loop_detection_middleware.py`**

Create `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_hash.py`:

```python
"""Pure utilities for computing stable hashes of tool calls.

Extracted from loop_detection_middleware.py for reuse across multiple
loop detectors (V1 hash-based + V2 detectors planned in design spec).
"""

import hashlib
import json


def normalize_tool_call_args(raw_args: object) -> tuple[dict, str | None]:
    """Normalize tool call args to a dict plus an optional fallback key.

    Some providers serialize ``args`` as a JSON string instead of a dict.
    Parses defensively; returns (dict, None) on success or (empty_dict, fallback_str).
    """
    if isinstance(raw_args, dict):
        return raw_args, None

    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}, raw_args
        if isinstance(parsed, dict):
            return parsed, None
        return {}, json.dumps(parsed, sort_keys=True, default=str)

    if raw_args is None:
        return {}, None

    return {}, json.dumps(raw_args, sort_keys=True, default=str)


def stable_tool_key(name: str, args: dict, fallback_key: str | None) -> str:
    """Derive a stable key from salient args without overfitting to noise."""
    if name == "read_file" and fallback_key is None:
        path = args.get("path") or ""
        start_line = args.get("start_line")
        end_line = args.get("end_line")

        bucket_size = 200
        try:
            start_line = int(start_line) if start_line is not None else 1
        except (TypeError, ValueError):
            start_line = 1
        try:
            end_line = int(end_line) if end_line is not None else start_line
        except (TypeError, ValueError):
            end_line = start_line

        start_line, end_line = sorted((start_line, end_line))
        bucket_start = max(start_line, 1)
        bucket_end = max(end_line, 1)
        bucket_start = (bucket_start - 1) // bucket_size
        bucket_end = (bucket_end - 1) // bucket_size
        return f"{path}:{bucket_start}-{bucket_end}"

    if name in {"write_file", "str_replace"}:
        if fallback_key is not None:
            return fallback_key
        return json.dumps(args, sort_keys=True, default=str)

    salient_fields = ("path", "url", "query", "command", "pattern", "glob", "cmd")
    stable_args = {field: args[field] for field in salient_fields if args.get(field) is not None}
    if stable_args:
        return json.dumps(stable_args, sort_keys=True, default=str)

    if fallback_key is not None:
        return fallback_key

    return json.dumps(args, sort_keys=True, default=str)


def hash_tool_calls(tool_calls: list[dict]) -> str:
    """Deterministic hash of a tool-call multiset (order-independent).

    The same multiset always produces the same hash regardless of input order.
    """
    normalized: list[str] = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args, fallback_key = normalize_tool_call_args(tc.get("args", {}))
        key = stable_tool_key(name, args, fallback_key)
        normalized.append(f"{name}:{key}")

    normalized.sort()
    blob = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


# Compatibility aliases (legacy underscore-prefixed names kept for transition)
_normalize_tool_call_args = normalize_tool_call_args
_stable_tool_key = stable_tool_key
_hash_tool_calls = hash_tool_calls
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_hash.py -v`
Expected: PASS — 8 tests pass

- [ ] **Step 5: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_hash.py deer-flow/backend/tests/test_loop_hash.py && git commit -m "$(cat <<'EOF'
refactor(loop-detection): extract hash utilities into loop_hash module

Pulls _normalize_tool_call_args, _stable_tool_key, _hash_tool_calls out of
loop_detection_middleware.py into a dedicated module so V2 detectors
(no-progress, periodic, tool-spam) can reuse the same hash logic.
Public names dropped underscore prefix; aliases kept for transition.
EOF
)"
```

---

## Task 2: Switch middleware to import from `loop_hash`

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py:14-123`

- [ ] **Step 1: Replace inline definitions with imports**

In `loop_detection_middleware.py`, change top of file:

```python
"""Middleware to detect and break repetitive tool call loops.

(Existing docstring preserved.)
"""

import logging
import threading
from collections import OrderedDict, defaultdict
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.loop_hash import (
    hash_tool_calls as _hash_tool_calls,
    normalize_tool_call_args as _normalize_tool_call_args,
    stable_tool_key as _stable_tool_key,
)

logger = logging.getLogger(__name__)
```

Then **delete** the original inline definitions of `_normalize_tool_call_args`, `_stable_tool_key`, and `_hash_tool_calls` (lines 36-123 in the old file).

- [ ] **Step 2: Run all middleware tests to verify nothing broke**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py tests/test_loop_hash.py -v`
Expected: PASS — all existing loop tests still pass + new loop_hash tests pass

- [ ] **Step 3: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py && git commit -m "refactor(loop-detection): import hash utilities from loop_hash module"
```

---

## Task 3: Skeleton `loop_hint_builder.py` with `_extract_text` helper

**Files:**
- Create: `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_hint_builder.py`
- Create: `deer-flow/backend/tests/test_loop_hint_builder.py`

- [ ] **Step 1: Write failing tests for `_extract_text`**

Create `deer-flow/backend/tests/test_loop_hint_builder.py`:

```python
"""Tests for loop_hint_builder module."""

from deerflow.agents.middlewares.loop_hint_builder import _extract_text


class TestExtractText:
    def test_str_content(self):
        assert _extract_text("hello world") == "hello world"

    def test_none_content(self):
        assert _extract_text(None) == ""

    def test_empty_str(self):
        assert _extract_text("") == ""

    def test_list_content_with_text_blocks(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": " second"},
        ]
        assert _extract_text(content) == "first second"

    def test_list_with_thinking_block_skipped(self):
        content = [
            {"type": "thinking", "thinking": "internal monologue"},
            {"type": "text", "text": "external answer"},
        ]
        assert _extract_text(content) == "external answer"

    def test_list_with_non_dict_items(self):
        content = [{"type": "text", "text": "ok"}, "loose-string", 42]
        assert _extract_text(content) == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_hint_builder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `loop_hint_builder.py` with `_extract_text` only**

Create `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_hint_builder.py`:

```python
"""Build rule-based "ruled out" hints from loop regions.

Pure-function module; no LangGraph runtime dependency. Used by
LoopDetectionMiddleware.wrap_model_call to construct the HumanMessage
that replaces a loop region in the LLM-visible message view.
"""


def _extract_text(content) -> str:
    """Extract concatenated text from AIMessage.content.

    Handles three shapes:
      - str: returned as-is
      - None: returned as empty string
      - list (Anthropic block format): concatenates {"type": "text", ...} blocks,
        ignores other block types (e.g., thinking, tool_use)
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_hint_builder.py -v`
Expected: PASS — 6 tests pass

- [ ] **Step 5: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_hint_builder.py deer-flow/backend/tests/test_loop_hint_builder.py && git commit -m "feat(loop-detection): add loop_hint_builder skeleton with _extract_text"
```

---

## Task 4: Add `_salient_args` and `_extract_intent` helpers

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_hint_builder.py`
- Modify: `deer-flow/backend/tests/test_loop_hint_builder.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_loop_hint_builder.py`:

```python
from langchain_core.messages import AIMessage

from deerflow.agents.middlewares.loop_hint_builder import (
    _extract_intent,
    _salient_args,
)


class TestSalientArgs:
    def test_extracts_whitelisted_fields(self):
        assert _salient_args({"path": "/a", "command": "ls", "noise": "x"}) == \
            "path='/a', command='ls'"

    def test_falls_back_to_str_when_no_whitelisted(self):
        result = _salient_args({"unknown_field": 12345})
        assert "unknown_field" in result or "12345" in result

    def test_empty_dict_returns_empty_marker(self):
        assert _salient_args({}) == "{}"


class TestExtractIntent:
    def test_short_content_returns_none(self):
        ai = AIMessage(content="ok", tool_calls=[])
        assert _extract_intent(ai) is None

    def test_meaningful_content_returned_truncated(self):
        long = "I need to check if foo.py has the bug because the stack trace points there."
        ai = AIMessage(content=long, tool_calls=[])
        result = _extract_intent(ai)
        assert result is not None
        assert "check if foo.py" in result
        assert len(result) <= 120

    def test_list_content_concatenated_then_extracted(self):
        ai = AIMessage(
            content=[
                {"type": "text", "text": "Looking at the codebase to understand the bug pattern."},
            ],
            tool_calls=[],
        )
        assert _extract_intent(ai) == "Looking at the codebase to understand the bug pattern."

    def test_empty_content_returns_none(self):
        ai = AIMessage(content="", tool_calls=[])
        assert _extract_intent(ai) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_hint_builder.py -v`
Expected: FAIL with `ImportError: cannot import name '_extract_intent'`

- [ ] **Step 3: Add `_salient_args` and `_extract_intent` to module**

Append to `loop_hint_builder.py`:

```python
_SALIENT_FIELDS = ("path", "url", "query", "command", "pattern", "glob", "cmd")

_INTENT_MIN_CHARS = 20
_INTENT_MAX_CHARS = 120


def _salient_args(args: dict) -> str:
    """Format args dict for hint display, keeping only whitelisted fields.

    Whitelist matches stable_tool_key in loop_hash to avoid drift.
    """
    items = [f"{k}={args[k]!r}" for k in _SALIENT_FIELDS if args.get(k) is not None]
    if items:
        return ", ".join(items)
    if not args:
        return "{}"
    return str(args)[:40]


def _extract_intent(ai_message) -> str | None:
    """Extract original intent from an AIMessage.content as a single short string.

    Returns None if content is shorter than _INTENT_MIN_CHARS (likely just filler).
    Otherwise returns text truncated to _INTENT_MAX_CHARS.
    """
    text = _extract_text(ai_message.content).strip()
    if len(text) < _INTENT_MIN_CHARS:
        return None
    if len(text) > _INTENT_MAX_CHARS:
        return text[:_INTENT_MAX_CHARS].rstrip() + "..."
    return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_hint_builder.py -v`
Expected: PASS — all tests pass

- [ ] **Step 5: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_hint_builder.py deer-flow/backend/tests/test_loop_hint_builder.py && git commit -m "feat(loop-detection): add _salient_args and _extract_intent helpers"
```

---

## Task 5: Implement `build_rule_hint` for single-loop case

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_hint_builder.py`
- Modify: `deer-flow/backend/tests/test_loop_hint_builder.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_loop_hint_builder.py`:

```python
from langchain_core.messages import ToolMessage

from deerflow.agents.middlewares.loop_hint_builder import build_rule_hint


def _ai(intent="", tool_calls=()):
    return AIMessage(content=intent, tool_calls=list(tool_calls))


def _tc(name, args, tc_id):
    return {"name": name, "args": args, "id": tc_id}


def _tool_msg(content, tc_id, name="bash"):
    return ToolMessage(content=content, tool_call_id=tc_id, name=name)


class TestBuildRuleHint:
    def test_includes_ruled_out_header(self):
        msgs = [
            _ai("Long-enough intent description for extraction here", [_tc("read_file", {"path": "/a.py"}, "c1")]),
            _tool_msg("Error: file not found", "c1", "read_file"),
            _ai("", [_tc("read_file", {"path": "/a.py"}, "c2")]),
            _tool_msg("Error: file not found", "c2", "read_file"),
            _ai("", [_tc("read_file", {"path": "/a.py"}, "c3")]),
            _tool_msg("Error: file not found", "c3", "read_file"),
        ]
        hint = build_rule_hint(msgs, start=0, end=5)
        assert "[LOOP RECOVERY]" in hint
        assert "ruled out" in hint.lower()

    def test_includes_original_intent_when_present(self):
        msgs = [
            _ai("Long-enough intent description for extraction here", [_tc("read_file", {"path": "/a.py"}, "c1")]),
            _tool_msg("Error: not found", "c1"),
        ]
        hint = build_rule_hint(msgs, start=0, end=1)
        assert "Original intent" in hint
        assert "intent description" in hint

    def test_omits_intent_when_short(self):
        msgs = [
            _ai("ok", [_tc("read_file", {"path": "/a.py"}, "c1")]),
            _tool_msg("Error: not found", "c1"),
        ]
        hint = build_rule_hint(msgs, start=0, end=1)
        assert "Original intent" not in hint

    def test_groups_errors_separately_from_unhelpful(self):
        msgs = [
            _ai("", [_tc("read_file", {"path": "/a.py"}, "c1")]),
            _tool_msg("Error: not found", "c1"),
            _ai("", [_tc("read_file", {"path": "/b.py"}, "c2")]),
            _tool_msg("", "c2"),  # unhelpful (empty)
        ]
        hint = build_rule_hint(msgs, start=0, end=3)
        assert "Failed with errors" in hint
        assert "Returned unhelpful" in hint

    def test_dedupes_by_tool_args(self):
        msgs = [
            _ai("", [_tc("read_file", {"path": "/a.py"}, "c1")]),
            _tool_msg("Error: x", "c1"),
            _ai("", [_tc("read_file", {"path": "/a.py"}, "c2")]),
            _tool_msg("Error: x", "c2"),  # dup of c1
        ]
        hint = build_rule_hint(msgs, start=0, end=3)
        # Should appear only once in the list
        assert hint.count("/a.py") == 1

    def test_fallback_when_no_pairs_found(self):
        msgs = [_ai("", [_tc("read_file", {"path": "/a.py"}, "c1")])]   # no ToolMessage
        hint = build_rule_hint(msgs, start=0, end=0)
        # Fallback hint is shorter generic warning, no per-tool listing
        assert "LOOP" in hint
        assert "/a.py" not in hint
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_hint_builder.py::TestBuildRuleHint -v`
Expected: FAIL with `ImportError: cannot import name 'build_rule_hint'`

- [ ] **Step 3: Implement `build_rule_hint`**

Append to `loop_hint_builder.py`:

```python
from langchain_core.messages import AIMessage, ToolMessage

_FALLBACK_HINT = (
    "[LOOP RECOVERY] Repeated tool calls detected. Stop calling tools and produce "
    "your final answer using whatever information you have so far."
)


def build_rule_hint(messages: list, start: int, end: int) -> str:
    """Build a human-readable rule-based hint for a loop region.

    Args:
        messages: full message list (the patched view's source)
        start: inclusive start index of the merged loop region
        end: inclusive end index (last message belonging to the region,
             including paired ToolMessages)

    Returns:
        Hint string. Falls back to _FALLBACK_HINT when no AIMessage/ToolMessage
        pairs can be extracted from the region.
    """
    region = messages[start : end + 1]

    # Walk region, pair each AIMessage.tool_call with its ToolMessage by id
    pending: dict[str, tuple[str, str]] = {}   # tool_call_id -> (name, salient_args_str)
    attempts: list[tuple[str, str, str, bool]] = []   # (name, args, result_preview, is_error)

    for msg in region:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", None) or []:
                tc_id = tc.get("id")
                if tc_id is None:
                    continue
                name = tc.get("name", "?")
                args = tc.get("args", {}) if isinstance(tc.get("args"), dict) else {}
                pending[tc_id] = (name, _salient_args(args))
        elif isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", None)
            if tc_id and tc_id in pending:
                name, args = pending.pop(tc_id)
                content = msg.content if msg.content is not None else ""
                preview = (str(content)[:120]).strip() if content else "(empty)"
                is_error = isinstance(content, str) and content.startswith("Error:")
                attempts.append((name, args, preview, is_error))

    if not attempts:
        return _FALLBACK_HINT

    # Dedupe by (name, args)
    seen: dict[tuple[str, str], tuple[str, bool]] = {}
    for name, args, preview, is_err in attempts:
        seen.setdefault((name, args), (preview, is_err))

    errors = [(k, v) for k, v in seen.items() if v[1]]
    unhelpful = [(k, v) for k, v in seen.items() if not v[1]]

    lines: list[str] = []
    intent = _extract_intent(messages[start]) if isinstance(messages[start], AIMessage) else None
    if intent:
        lines.append("[LOOP RECOVERY] Original intent at the start of this loop region:")
        lines.append(f'  "{intent}"')
        lines.append("")
        lines.append("These tool-call paths have been ruled out:")
    else:
        lines.append("[LOOP RECOVERY] These tool-call paths have been ruled out:")

    if errors:
        lines.append("")
        lines.append("Failed with errors:")
        for (name, args), (preview, _) in errors:
            lines.append(f"  ✗ {name}({args}) → {preview[:80]}")

    if unhelpful:
        lines.append("")
        lines.append("Returned unhelpful results:")
        for (name, args), (preview, _) in unhelpful:
            lines.append(f"  ○ {name}({args}) → {preview[:80]}")

    lines.append("")
    lines.append("Do NOT retry the ruled-out paths. Reassess your approach toward the original")
    lines.append("intent. Choose:")
    lines.append("  (a) a different tool")
    lines.append("  (b) different arguments")
    lines.append("  (c) produce a final answer using partial information.")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_hint_builder.py -v`
Expected: PASS — all tests pass

- [ ] **Step 5: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_hint_builder.py deer-flow/backend/tests/test_loop_hint_builder.py && git commit -m "feat(loop-detection): implement build_rule_hint for single-loop region"
```

---

## Task 6: Add `_detect_all_loops` to middleware

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py`
- Modify: `deer-flow/backend/tests/test_loop_detection_middleware.py`

- [ ] **Step 1: Write failing test for `_detect_all_loops`**

Add to `tests/test_loop_detection_middleware.py`:

```python
from langchain_core.messages import AIMessage, ToolMessage


def _ai(content="", tool_calls=()):
    return AIMessage(content=content, tool_calls=list(tool_calls))


def _tc(name, path, tc_id):
    return {"name": name, "args": {"path": path}, "id": tc_id}


def _tm(content, tc_id, name="read_file"):
    return ToolMessage(content=content, tool_call_id=tc_id, name=name)


class TestDetectAllLoops:
    def test_no_loop_returns_empty(self):
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = [
            _ai("", [_tc("read_file", "/a", "c1")]),
            _tm("ok", "c1"),
        ]
        assert mw._detect_all_loops(msgs) == []

    def test_single_loop_at_threshold_returns_one(self):
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = []
        for i in range(3):
            msgs.append(_ai("", [_tc("read_file", "/a", f"c{i}")]))
            msgs.append(_tm("err", f"c{i}"))
        loops = mw._detect_all_loops(msgs)
        assert len(loops) == 1
        # (hash, first_idx, last_idx) — first AIMessage at idx 0, last at idx 4
        h, first, last = loops[0]
        assert first == 0
        assert last == 4

    def test_two_disjoint_loops_returns_both(self):
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = []
        for i in range(3):
            msgs.append(_ai("", [_tc("read_file", "/a", f"a{i}")]))
            msgs.append(_tm("err", f"a{i}"))
        msgs.append(_ai("intermediate non-loop", []))
        for i in range(3):
            msgs.append(_ai("", [_tc("read_file", "/b", f"b{i}")]))
            msgs.append(_tm("err", f"b{i}"))
        loops = mw._detect_all_loops(msgs)
        assert len(loops) == 2

    def test_below_threshold_not_detected(self):
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = [
            _ai("", [_tc("read_file", "/a", "c1")]),
            _tm("ok", "c1"),
            _ai("", [_tc("read_file", "/a", "c2")]),
            _tm("ok", "c2"),
        ]
        assert mw._detect_all_loops(msgs) == []
```

Note: this test instantiates `LoopDetectionMiddleware(rewind_threshold=3)` which requires the new constructor signature. We'll add that signature in this task.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py::TestDetectAllLoops -v`
Expected: FAIL — either constructor doesn't accept `rewind_threshold`, or `_detect_all_loops` doesn't exist

- [ ] **Step 3: Add `rewind_threshold` to constructor and implement `_detect_all_loops`**

In `loop_detection_middleware.py`, modify `__init__` to accept `rewind_threshold` (keep existing params for now to avoid breaking existing tests):

```python
# Add at top of class, near existing _DEFAULT_* constants
_DEFAULT_REWIND_THRESHOLD = 3


class LoopDetectionMiddleware(AgentMiddleware[AgentState]):
    def __init__(
        self,
        warn_threshold: int = _DEFAULT_WARN_THRESHOLD,
        rewind_threshold: int = _DEFAULT_REWIND_THRESHOLD,
        hard_limit: int = _DEFAULT_HARD_LIMIT,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        max_tracked_threads: int = _DEFAULT_MAX_TRACKED_THREADS,
    ):
        super().__init__()
        self.warn_threshold = warn_threshold
        self.rewind_threshold = rewind_threshold
        self.hard_limit = hard_limit
        self.window_size = window_size
        self.max_tracked_threads = max_tracked_threads
        self._lock = threading.Lock()
        self._history: OrderedDict[str, list[str]] = OrderedDict()
        self._warned: dict[str, set[str]] = defaultdict(set)
```

Add the new `_detect_all_loops` method:

```python
    def _detect_all_loops(self, messages: list) -> list[tuple[str, int, int]]:
        """Find all tool-call hashes that meet rewind_threshold in messages.

        Returns list of (hash, first_ai_idx, last_ai_idx) sorted by first_ai_idx.
        Indices reference AIMessage positions; ToolMessage absorption happens later.
        """
        from langchain_core.messages import AIMessage as _AIMessage

        hash_counts: dict[str, int] = {}
        hash_first_idx: dict[str, int] = {}
        hash_last_idx: dict[str, int] = {}

        for i, msg in enumerate(messages):
            if not isinstance(msg, _AIMessage):
                continue
            tcs = getattr(msg, "tool_calls", None)
            if not tcs:
                continue
            try:
                h = _hash_tool_calls(tcs)
            except Exception:
                logger.warning("Failed to hash tool_calls at index %d, skipping", i)
                continue
            hash_counts[h] = hash_counts.get(h, 0) + 1
            hash_first_idx.setdefault(h, i)
            hash_last_idx[h] = i

        loops = [
            (h, hash_first_idx[h], hash_last_idx[h])
            for h, count in hash_counts.items()
            if count >= self.rewind_threshold
        ]
        return sorted(loops, key=lambda t: t[1])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py::TestDetectAllLoops -v`
Expected: PASS — 4 tests pass

- [ ] **Step 5: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py deer-flow/backend/tests/test_loop_detection_middleware.py && git commit -m "feat(loop-detection): add _detect_all_loops method and rewind_threshold param"
```

---

## Task 7: Implement `_merge_overlapping`

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py`
- Modify: `deer-flow/backend/tests/test_loop_detection_middleware.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_loop_detection_middleware.py`:

```python
class TestMergeOverlapping:
    def test_empty(self):
        mw = LoopDetectionMiddleware()
        assert mw._merge_overlapping([]) == []

    def test_single_region(self):
        mw = LoopDetectionMiddleware()
        result = mw._merge_overlapping([("h1", 0, 5)])
        assert result == [({"h1"}, 0, 5)]

    def test_disjoint_regions_unmerged(self):
        mw = LoopDetectionMiddleware()
        result = mw._merge_overlapping([("h1", 0, 3), ("h2", 10, 15)])
        assert result == [({"h1"}, 0, 3), ({"h2"}, 10, 15)]

    def test_overlapping_regions_merged(self):
        mw = LoopDetectionMiddleware()
        result = mw._merge_overlapping([("h1", 0, 8), ("h2", 5, 12)])
        assert result == [({"h1", "h2"}, 0, 12)]

    def test_adjacent_regions_merged(self):
        mw = LoopDetectionMiddleware()
        result = mw._merge_overlapping([("h1", 0, 5), ("h2", 6, 10)])
        # adjacent (start <= prev_end + 1) → merged
        assert result == [({"h1", "h2"}, 0, 10)]

    def test_nested_regions_merged(self):
        mw = LoopDetectionMiddleware()
        result = mw._merge_overlapping([("h1", 0, 20), ("h2", 5, 10)])
        assert result == [({"h1", "h2"}, 0, 20)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py::TestMergeOverlapping -v`
Expected: FAIL with `AttributeError: 'LoopDetectionMiddleware' object has no attribute '_merge_overlapping'`

- [ ] **Step 3: Implement `_merge_overlapping`**

Add to `LoopDetectionMiddleware` class:

```python
    def _merge_overlapping(
        self, regions: list[tuple[str, int, int]]
    ) -> list[tuple[set[str], int, int]]:
        """Merge overlapping or adjacent regions, aggregating their hash sets.

        Two regions [a, b] and [c, d] (a <= c) merge if c <= b + 1.
        """
        if not regions:
            return []
        merged: list[tuple[set[str], int, int]] = []
        for h, start, end in regions:
            if merged and start <= merged[-1][2] + 1:
                hashes, m_start, m_end = merged[-1]
                merged[-1] = (hashes | {h}, m_start, max(m_end, end))
            else:
                merged.append(({h}, start, end))
        return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py::TestMergeOverlapping -v`
Expected: PASS — 6 tests pass

- [ ] **Step 5: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py deer-flow/backend/tests/test_loop_detection_middleware.py && git commit -m "feat(loop-detection): add _merge_overlapping for multi-loop support"
```

---

## Task 8: Implement `_expand_for_tool_messages`

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py`
- Modify: `deer-flow/backend/tests/test_loop_detection_middleware.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_loop_detection_middleware.py`:

```python
class TestExpandForToolMessages:
    def test_no_trailing_tool_messages(self):
        mw = LoopDetectionMiddleware()
        msgs = [_ai("", [_tc("read_file", "/a", "c1")])]
        # last AIMessage at idx 0; no following ToolMessages → end stays 0
        assert mw._expand_for_tool_messages(msgs, hashes_in_region={}, region_end=0) == 0

    def test_absorbs_immediate_tool_messages(self):
        mw = LoopDetectionMiddleware()
        msgs = [
            _ai("", [_tc("read_file", "/a", "c1")]),
            _tm("err", "c1"),
            _tm("err2", "c1"),  # second response (rare but possible)
        ]
        # tool_call_ids in region: {"c1"}
        result = mw._expand_for_tool_messages(msgs, tool_call_ids={"c1"}, region_end=0)
        assert result == 2

    def test_stops_at_unrelated_message(self):
        mw = LoopDetectionMiddleware()
        msgs = [
            _ai("", [_tc("read_file", "/a", "c1")]),
            _tm("err", "c1"),
            _ai("unrelated", [_tc("ls", "/x", "c2")]),
        ]
        result = mw._expand_for_tool_messages(msgs, tool_call_ids={"c1"}, region_end=0)
        assert result == 1   # stops before the unrelated AIMessage at idx 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py::TestExpandForToolMessages -v`
Expected: FAIL with `AttributeError: ... no attribute '_expand_for_tool_messages'`

- [ ] **Step 3: Implement `_expand_for_tool_messages`**

Add to `LoopDetectionMiddleware`:

```python
    def _expand_for_tool_messages(
        self,
        messages: list,
        tool_call_ids: set[str],
        region_end: int,
    ) -> int:
        """Walk forward from region_end absorbing ToolMessages whose tool_call_id
        is in tool_call_ids. Stops at the first non-matching message.

        Returns the new (inclusive) end index.
        """
        from langchain_core.messages import ToolMessage as _ToolMessage

        i = region_end + 1
        while i < len(messages):
            msg = messages[i]
            if isinstance(msg, _ToolMessage) and getattr(msg, "tool_call_id", None) in tool_call_ids:
                i += 1
                continue
            break
        return i - 1
```

- [ ] **Step 4: Update test signature and run**

The test in step 1 uses parameter `tool_call_ids={"c1"}` and `region_end=0`. Implementation matches. Update the first test (which had `hashes_in_region={}`) to also use `tool_call_ids=set()`:

```python
    def test_no_trailing_tool_messages(self):
        mw = LoopDetectionMiddleware()
        msgs = [_ai("", [_tc("read_file", "/a", "c1")])]
        assert mw._expand_for_tool_messages(msgs, tool_call_ids=set(), region_end=0) == 0
```

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py::TestExpandForToolMessages -v`
Expected: PASS — 3 tests pass

- [ ] **Step 5: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py deer-flow/backend/tests/test_loop_detection_middleware.py && git commit -m "feat(loop-detection): add _expand_for_tool_messages for patch_end calculation"
```

---

## Task 9: Implement `_apply_all_patches`

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py`
- Modify: `deer-flow/backend/tests/test_loop_detection_middleware.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_loop_detection_middleware.py`:

```python
class TestApplyAllPatches:
    def test_no_loops_returns_unchanged(self):
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = [_ai("only one", [_tc("read_file", "/a", "c1")]), _tm("ok", "c1")]
        patched = mw._apply_all_patches(msgs)
        assert patched == msgs   # no loops → identical

    def test_single_loop_replaced_with_hint(self):
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = []
        for i in range(3):
            msgs.append(_ai("Searching for the bug in foo.py based on stack trace.", 
                            [_tc("read_file", "/a", f"c{i}")]))
            msgs.append(_tm("Error: not found", f"c{i}"))
        patched = mw._apply_all_patches(msgs)
        # Original 6 messages → patched into 1 HumanMessage
        assert len(patched) == 1
        assert "[LOOP RECOVERY]" in patched[0].content
        assert "ruled out" in patched[0].content.lower()

    def test_two_disjoint_loops_two_patches(self):
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = []
        for i in range(3):
            msgs.append(_ai("", [_tc("read_file", "/a", f"a{i}")]))
            msgs.append(_tm("err", f"a{i}"))
        msgs.append(_ai("intermediate", []))
        for i in range(3):
            msgs.append(_ai("", [_tc("read_file", "/b", f"b{i}")]))
            msgs.append(_tm("err", f"b{i}"))
        patched = mw._apply_all_patches(msgs)
        # Two HumanMessage hints + the intermediate AIMessage
        assert len(patched) == 3
        assert "/a" in patched[0].content   # first hint
        assert patched[1].content == "intermediate"
        assert "/b" in patched[2].content   # second hint

    def test_idempotence(self):
        """Applying patch twice yields identical result."""
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = []
        for i in range(3):
            msgs.append(_ai("", [_tc("read_file", "/a", f"c{i}")]))
            msgs.append(_tm("err", f"c{i}"))
        once = mw._apply_all_patches(msgs)
        twice = mw._apply_all_patches(once)
        assert [m.content for m in once] == [m.content for m in twice]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py::TestApplyAllPatches -v`
Expected: FAIL with `AttributeError: ... no attribute '_apply_all_patches'`

- [ ] **Step 3: Implement `_apply_all_patches`**

Add imports at top of `loop_detection_middleware.py`:

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.agents.middlewares.loop_hint_builder import build_rule_hint
```

Add to `LoopDetectionMiddleware`:

```python
    def _collect_tool_call_ids_in_range(
        self, messages: list, start: int, end: int
    ) -> set[str]:
        """Collect tool_call_id values from AIMessages in [start, end]."""
        ids: set[str] = set()
        for i in range(start, end + 1):
            msg = messages[i]
            if isinstance(msg, AIMessage):
                for tc in getattr(msg, "tool_calls", None) or []:
                    tc_id = tc.get("id")
                    if tc_id:
                        ids.add(tc_id)
        return ids

    def _apply_all_patches(self, messages: list) -> list:
        """Detect all loops, merge overlapping, apply patches from end to start.

        Returns a new list (does not mutate input).
        """
        loops = self._detect_all_loops(messages)
        if not loops:
            return list(messages)

        merged = self._merge_overlapping(loops)

        # Process from end to start to keep earlier indices valid
        patched = list(messages)
        for hashes, start, end in sorted(merged, key=lambda r: -r[1]):
            tc_ids = self._collect_tool_call_ids_in_range(patched, start, end)
            expanded_end = self._expand_for_tool_messages(patched, tc_ids, end)
            hint = build_rule_hint(patched, start, expanded_end)
            patched = patched[:start] + [HumanMessage(content=hint)] + patched[expanded_end + 1 :]

        return patched
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py::TestApplyAllPatches -v`
Expected: PASS — 4 tests pass

- [ ] **Step 5: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py deer-flow/backend/tests/test_loop_detection_middleware.py && git commit -m "feat(loop-detection): add _apply_all_patches end-to-end pipeline"
```

---

## Task 10: Wire `wrap_model_call` hook (sync + async)

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py`
- Modify: `deer-flow/backend/tests/test_loop_detection_middleware.py`

- [ ] **Step 1: Write failing tests for `wrap_model_call`**

Add to `tests/test_loop_detection_middleware.py`:

```python
from langchain.agents.middleware.types import ModelRequest


def _model_request(messages):
    """Build a minimal ModelRequest for wrap_model_call testing."""
    req = MagicMock(spec=ModelRequest)
    req.messages = messages
    captured = {}

    def fake_override(messages):
        captured["messages"] = messages
        return req
    req.override = MagicMock(side_effect=fake_override)
    req._captured = captured
    return req


class TestWrapModelCall:
    def test_no_loop_passes_request_unchanged(self):
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = [_ai("", [_tc("read_file", "/a", "c1")]), _tm("ok", "c1")]
        req = _model_request(msgs)
        handler = MagicMock(return_value="response")

        result = mw.wrap_model_call(req, handler)

        req.override.assert_not_called()
        handler.assert_called_once_with(req)
        assert result == "response"

    def test_loop_triggers_patching(self):
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = []
        for i in range(3):
            msgs.append(_ai("", [_tc("read_file", "/a", f"c{i}")]))
            msgs.append(_tm("err", f"c{i}"))
        req = _model_request(msgs)
        handler = MagicMock(return_value="response")

        mw.wrap_model_call(req, handler)

        req.override.assert_called_once()
        patched = req._captured["messages"]
        # 6 original → 1 hint
        assert len(patched) == 1
        assert "[LOOP RECOVERY]" in patched[0].content
        handler.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py::TestWrapModelCall -v`
Expected: FAIL with `AttributeError: 'LoopDetectionMiddleware' object has no attribute 'wrap_model_call'`

- [ ] **Step 3: Implement `wrap_model_call` (sync) and `awrap_model_call` (async)**

Add imports at top of file:

```python
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
```

Add to `LoopDetectionMiddleware`:

```python
    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        patched = self._apply_all_patches(request.messages)
        if patched is not request.messages and patched != request.messages:
            request = request.override(messages=patched)
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        patched = self._apply_all_patches(request.messages)
        if patched is not request.messages and patched != request.messages:
            request = request.override(messages=patched)
        return await handler(request)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py::TestWrapModelCall -v`
Expected: PASS — 2 tests pass

- [ ] **Step 5: Run full middleware test suite to confirm no regression in existing tests**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py -v`
Expected: All tests pass (existing warn / hard_stop tests still green; new wrap_model_call / detection / patching tests green)

- [ ] **Step 6: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py deer-flow/backend/tests/test_loop_detection_middleware.py && git commit -m "feat(loop-detection): wire wrap_model_call hook for view-layer patching"
```

---

## Task 11: Remove warn tier from `after_model`

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py`
- Modify: `deer-flow/backend/tests/test_loop_detection_middleware.py`

The warn tier (additive HumanMessage on count >= warn_threshold) is now superseded by `wrap_model_call` patching. Remove it but keep `hard_stop`.

- [ ] **Step 1: Identify and delete tests covering warn-only behavior**

Find tests in `tests/test_loop_detection_middleware.py` whose names or assertions specifically reference the warn tier (e.g., test functions named `test_warn_*` or asserting on `_WARNING_MSG` injection). Examples to delete or update:

```python
# Delete: tests asserting that count==3 produces an additive HumanMessage with _WARNING_MSG
# Delete: tests asserting that warn is suppressed on second hit of same hash
# Update: tests using warn_threshold parameter — switch to rewind_threshold
```

Open the test file and look for `_WARNING_MSG` usages. Remove tests whose entire purpose was to verify warn injection. Update parameter usages: `LoopDetectionMiddleware(warn_threshold=3)` → `LoopDetectionMiddleware(rewind_threshold=3)`.

- [ ] **Step 2: Run remaining tests to confirm they still apply**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py -v`
Expected: warn-related tests deleted; remaining tests still pass

- [ ] **Step 3: Simplify `_track_and_check` to only return hard_stop signal**

In `loop_detection_middleware.py`, replace the `_track_and_check` body to only check hard_limit:

```python
    def _track_and_check(self, state: AgentState, runtime: Runtime) -> tuple[str | None, bool]:
        """Track tool calls and return (hard_stop_msg_or_none, should_hard_stop).

        Note: warn tier removed in favor of wrap_model_call view patching.
        Only the hard_stop safety net remains in after_model.
        """
        messages = state.get("messages", [])
        if not messages:
            return None, False

        last_msg = messages[-1]
        if getattr(last_msg, "type", None) != "ai":
            return None, False

        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return None, False

        thread_id = self._get_thread_id(runtime)
        try:
            call_hash = _hash_tool_calls(tool_calls)
        except Exception:
            return None, False

        with self._lock:
            if thread_id in self._history:
                self._history.move_to_end(thread_id)
            else:
                self._history[thread_id] = []
                self._evict_if_needed()

            history = self._history[thread_id]
            history.append(call_hash)
            if len(history) > self.window_size:
                history[:] = history[-self.window_size :]

            count = history.count(call_hash)
            tool_names = [tc.get("name", "?") for tc in tool_calls]

            if count >= self.hard_limit:
                logger.error(
                    "Loop hard limit reached - forcing stop",
                    extra={
                        "thread_id": thread_id,
                        "call_hash": call_hash,
                        "count": count,
                        "tools": tool_names,
                    },
                )
                return _HARD_STOP_MSG, True

        return None, False
```

Then simplify `_apply` to only handle hard_stop (delete the `if warning:` branch):

```python
    def _apply(self, state: AgentState, runtime: Runtime) -> dict | None:
        warning, hard_stop = self._track_and_check(state, runtime)

        if hard_stop:
            messages = state.get("messages", [])
            last_msg = messages[-1]
            stripped_msg = last_msg.model_copy(
                update={
                    "tool_calls": [],
                    "content": self._append_text(last_msg.content, _HARD_STOP_MSG),
                }
            )
            return {"messages": [stripped_msg]}

        return None
```

Remove unused `_warned` field from `__init__` and `reset` (kept `_history` because hard_stop tracking still uses sliding window).

- [ ] **Step 4: Run tests to verify hard_stop still works and warn no longer fires**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py -v`
Expected: hard_stop tests pass; no warn tests remain; new patching tests pass

- [ ] **Step 5: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py deer-flow/backend/tests/test_loop_detection_middleware.py && git commit -m "refactor(loop-detection): remove warn tier (superseded by wrap_model_call patching)"
```

---

## Task 12: Add observability dedup with `_reported_loops`

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py`
- Modify: `deer-flow/backend/tests/test_loop_detection_middleware.py`

- [ ] **Step 1: Write failing test for dedup**

Add to `tests/test_loop_detection_middleware.py`:

```python
class TestObservabilityDedup:
    def test_first_detection_logs_warning(self, caplog):
        import logging
        caplog.set_level(logging.WARNING, logger="deerflow.agents.middlewares.loop_detection_middleware")
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = []
        for i in range(3):
            msgs.append(_ai("", [_tc("read_file", "/a", f"c{i}")]))
            msgs.append(_tm("err", f"c{i}"))
        req = _model_request(msgs)
        handler = MagicMock(return_value="r")

        mw.wrap_model_call(req, handler)

        first_detected = [r for r in caplog.records if "loop.rewind.first_detected" in r.message]
        assert len(first_detected) == 1

    def test_subsequent_calls_silent(self, caplog):
        import logging
        caplog.set_level(logging.WARNING, logger="deerflow.agents.middlewares.loop_detection_middleware")
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = []
        for i in range(3):
            msgs.append(_ai("", [_tc("read_file", "/a", f"c{i}")]))
            msgs.append(_tm("err", f"c{i}"))

        # Set up runtime context so middleware can extract thread_id
        for _ in range(5):
            req = _model_request(msgs)
            req.runtime = _make_runtime(thread_id="t1")
            handler = MagicMock(return_value="r")
            mw.wrap_model_call(req, handler)

        first_detected = [r for r in caplog.records if "loop.rewind.first_detected" in r.message]
        assert len(first_detected) == 1   # only first call logs

    def test_different_threads_each_log_once(self, caplog):
        import logging
        caplog.set_level(logging.WARNING, logger="deerflow.agents.middlewares.loop_detection_middleware")
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = []
        for i in range(3):
            msgs.append(_ai("", [_tc("read_file", "/a", f"c{i}")]))
            msgs.append(_tm("err", f"c{i}"))

        for thread_id in ("t1", "t2"):
            req = _model_request(msgs)
            req.runtime = _make_runtime(thread_id=thread_id)
            handler = MagicMock(return_value="r")
            mw.wrap_model_call(req, handler)

        first_detected = [r for r in caplog.records if "loop.rewind.first_detected" in r.message]
        assert len(first_detected) == 2
```

Update `_model_request` helper to also set a default runtime:

```python
def _model_request(messages, thread_id="test-thread"):
    req = MagicMock(spec=ModelRequest)
    req.messages = messages
    req.runtime = _make_runtime(thread_id=thread_id)
    captured = {}

    def fake_override(messages):
        captured["messages"] = messages
        return req
    req.override = MagicMock(side_effect=fake_override)
    req._captured = captured
    return req
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py::TestObservabilityDedup -v`
Expected: FAIL — no `loop.rewind.first_detected` log records produced

- [ ] **Step 3: Implement dedup cache + first_detected logging**

Add to `LoopDetectionMiddleware.__init__`:

```python
        # Observation-only dedup cache: tracks (thread_id, loop_hash) reported once.
        # Pure optimization — clearing it only causes some loops to be re-reported,
        # never affects patching behavior.
        self._reported_loops: dict[str, set[str]] = defaultdict(set)
```

Add a helper to extract thread_id from a ModelRequest:

```python
    def _get_thread_id_from_request(self, request: ModelRequest) -> str:
        """Extract thread_id from request.runtime.context, fallback to 'default'."""
        runtime = getattr(request, "runtime", None)
        if runtime is None:
            return "default"
        ctx = getattr(runtime, "context", None)
        if not ctx:
            return "default"
        return ctx.get("thread_id", "default")
```

Modify `wrap_model_call` to log first_detected events:

```python
    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        loops = self._detect_all_loops(request.messages)
        if not loops:
            return handler(request)

        merged = self._merge_overlapping(loops)
        thread_id = self._get_thread_id_from_request(request)

        # Emit first_detected log per (thread_id, loop_hash)
        for hashes, start, end in merged:
            for h in hashes:
                if h not in self._reported_loops[thread_id]:
                    self._reported_loops[thread_id].add(h)
                    logger.warning(
                        "loop.rewind.first_detected",
                        extra={
                            "thread_id": thread_id,
                            "loop_hash": h,
                            "loop_start_idx": start,
                            "loop_end_idx": end,
                        },
                    )

        # Apply patches (always, regardless of whether this is first detection)
        patched = list(request.messages)
        for hashes, start, end in sorted(merged, key=lambda r: -r[1]):
            tc_ids = self._collect_tool_call_ids_in_range(patched, start, end)
            expanded_end = self._expand_for_tool_messages(patched, tc_ids, end)
            hint = build_rule_hint(patched, start, expanded_end)
            patched = patched[:start] + [HumanMessage(content=hint)] + patched[expanded_end + 1 :]

        request = request.override(messages=patched)
        return handler(request)
```

Apply identical changes to `awrap_model_call`.

(Note: `_apply_all_patches` is now inlined into `wrap_model_call` for the logging hookpoint. Remove `_apply_all_patches` if no other test depends on it, or keep as private helper called from both sync/async variants.)

Refactor option: Extract the patching loop back into `_apply_all_patches` for clarity:

```python
    def _apply_patches_to_merged(self, messages: list, merged) -> list:
        patched = list(messages)
        for hashes, start, end in sorted(merged, key=lambda r: -r[1]):
            tc_ids = self._collect_tool_call_ids_in_range(patched, start, end)
            expanded_end = self._expand_for_tool_messages(patched, tc_ids, end)
            hint = build_rule_hint(patched, start, expanded_end)
            patched = patched[:start] + [HumanMessage(content=hint)] + patched[expanded_end + 1 :]
        return patched
```

Then both sync and async `wrap_model_call` call `_apply_patches_to_merged(request.messages, merged)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py::TestObservabilityDedup -v`
Expected: PASS — 3 dedup tests pass

- [ ] **Step 5: Run full middleware test suite**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py tests/test_loop_hash.py tests/test_loop_hint_builder.py -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py deer-flow/backend/tests/test_loop_detection_middleware.py && git commit -m "feat(loop-detection): add observability dedup via _reported_loops cache"
```

---

## Task 13: Cleanup deprecated parameters and confirm signature

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py`
- Modify: `deer-flow/backend/tests/test_loop_detection_middleware.py`

The `warn_threshold` and `_warned` field are no longer used. Clean up.

- [ ] **Step 1: Remove `warn_threshold` parameter from `__init__`**

Update `LoopDetectionMiddleware.__init__`:

```python
    def __init__(
        self,
        rewind_threshold: int = _DEFAULT_REWIND_THRESHOLD,
        hard_limit: int = _DEFAULT_HARD_LIMIT,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        max_tracked_threads: int = _DEFAULT_MAX_TRACKED_THREADS,
    ):
        super().__init__()
        self.rewind_threshold = rewind_threshold
        self.hard_limit = hard_limit
        self.window_size = window_size
        self.max_tracked_threads = max_tracked_threads
        self._lock = threading.Lock()
        self._history: OrderedDict[str, list[str]] = OrderedDict()
        self._reported_loops: dict[str, set[str]] = defaultdict(set)
```

Remove `_DEFAULT_WARN_THRESHOLD` constant from the file (search for it; should be one declaration near the top of the class section).

Remove `self._warned` initialization and any remaining references.

- [ ] **Step 2: Update `reset` method**

```python
    def reset(self, thread_id: str | None = None) -> None:
        """Clear tracking state. If thread_id given, clear only that thread."""
        with self._lock:
            if thread_id:
                self._history.pop(thread_id, None)
                self._reported_loops.pop(thread_id, None)
            else:
                self._history.clear()
                self._reported_loops.clear()
```

- [ ] **Step 3: Search for any remaining `warn_threshold` callers**

Run: `cd /Users/bytedance/Documents/aime/deer-agents && grep -rn "warn_threshold" --include="*.py" --include="*.yaml" 2>/dev/null`
Expected: zero matches (other than docstrings/comments referring to history). If matches found, update each call site to drop the parameter.

- [ ] **Step 4: Run all middleware tests**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py tests/test_loop_hash.py tests/test_loop_hint_builder.py -v`
Expected: all pass

- [ ] **Step 5: Run broader test suite to catch any callers we missed**

Run: `cd deer-flow/backend && uv run pytest tests/ -v --ignore=tests/test_create_deerflow_agent.py 2>&1 | tail -30`
Expected: no failures from `LoopDetectionMiddleware` signature mismatch

If `test_create_deerflow_agent.py` references warn_threshold, update it too:

Run: `cd deer-flow/backend && uv run pytest tests/test_create_deerflow_agent.py -v 2>&1 | grep -i "warn_threshold\|loop"`
Expected: no failures

- [ ] **Step 6: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add -A deer-flow/backend && git commit -m "refactor(loop-detection): drop warn_threshold parameter (warn tier removed)"
```

---

## Task 14: End-to-end sanity test with realistic loop scenario

**Files:**
- Modify: `deer-flow/backend/tests/test_loop_detection_middleware.py`

- [ ] **Step 1: Add an end-to-end integration test**

```python
class TestEndToEndPatching:
    def test_realistic_oncall_loop_scenario(self):
        """Simulates oncall agent reading the same file 3 times with errors."""
        mw = LoopDetectionMiddleware(rewind_threshold=3)

        # Build a 10-message history: 1 user query + 3 loop iterations + 3 prior unrelated tool calls
        msgs = [
            HumanMessage(content="Why is the foo service slow?"),
            _ai("Let me check the deploy log first.", [_tc("read_file", "/log", "p1")]),
            _tm("Recent deploys: ...", "p1"),
            _ai("Now I'll inspect the service code.", [_tc("read_file", "/foo.py", "p2")]),
            _tm("Error: file not found", "p2"),
            _ai("Try absolute path.", [_tc("read_file", "/foo.py", "p3")]),
            _tm("Error: file not found", "p3"),
            _ai("Hmm, try once more.", [_tc("read_file", "/foo.py", "p4")]),
            _tm("Error: file not found", "p4"),
        ]

        req = _model_request(msgs)
        handler = MagicMock(return_value="next-response")
        mw.wrap_model_call(req, handler)

        patched = req._captured["messages"]
        # Pre-loop messages preserved (HumanMessage + first 2 AI/Tool pair)
        assert isinstance(patched[0], HumanMessage)
        assert "foo service" in patched[0].content
        assert isinstance(patched[1], AIMessage)
        assert "deploy log" in patched[1].content
        assert isinstance(patched[2], ToolMessage)
        # Loop region (msgs[3..8]) collapsed to single hint
        hint_msg = patched[3]
        assert isinstance(hint_msg, HumanMessage)
        assert "[LOOP RECOVERY]" in hint_msg.content
        assert "/foo.py" in hint_msg.content
        # Total length: 3 preserved + 1 hint = 4
        assert len(patched) == 4
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_detection_middleware.py::TestEndToEndPatching -v`
Expected: PASS

- [ ] **Step 3: Run all loop-related tests one final time**

Run: `cd deer-flow/backend && uv run pytest tests/test_loop_hash.py tests/test_loop_hint_builder.py tests/test_loop_detection_middleware.py -v --tb=short`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents && git add deer-flow/backend/tests/test_loop_detection_middleware.py && git commit -m "test(loop-detection): add end-to-end realistic oncall loop scenario"
```

---

## Self-Review Notes

Reviewed against `docs/superpowers/specs/2026-04-16-loop-detection-rewind-design.md`:

**Spec coverage map:**

| Spec section | Covered by |
|--------------|-----------|
| Core mechanism (wrap_model_call) | Tasks 9, 10 |
| Three-tier (warn removed, rewind, hard_stop) | Tasks 10, 11 |
| Detection algorithm (`_detect_all_loops`) | Task 6 |
| Merge overlapping | Task 7 |
| Patch region definition + expand_for_tool_messages | Tasks 8, 9 |
| Hint format (intent, errors, unhelpful, fallback) | Tasks 4, 5 |
| Multi-hash combined hint | Task 5 (test) + 9 (apply) |
| Interface changes (constructor) | Tasks 6, 13 |
| Invariants I1-I10 | Tested across Tasks 6-12 |
| Observability (dedup, first_detected) | Task 12 |
| Test matrix T1-T19 | Distributed across Tasks 5-14 |
| File structure (loop_hash, loop_hint_builder, middleware) | Tasks 1-3 |
| KV cache friendliness | Implicit (deterministic hint, stable patching) |

**Out of scope (V2, not in this plan):**
- V2.1 no-progress detector
- V2.2 periodic detector
- V2.3 judge-based hint
- V2.5 tool-spam detector
- V2.4 detector chain orchestration
- Metrics emission (logging is in scope; Prometheus counters defer)

**Placeholder scan:** No TBD/TODO/vague-handling steps. Each step has either runnable test code, runnable implementation code, or an exact command + expected output.

**Type consistency:** 
- `_detect_all_loops` returns `list[tuple[str, int, int]]` consistently across Tasks 6, 7, 9, 10, 12
- `_merge_overlapping` returns `list[tuple[set[str], int, int]]` consistently
- `build_rule_hint(messages, start, end)` signature consistent across Tasks 5, 9, 12
- `_expand_for_tool_messages(messages, tool_call_ids, region_end)` consistent

**Known cleanup deferred:**
- `_history` sliding window is still used by hard_stop tracking (Task 11); could be cleaned up later if hard_stop also becomes stateless
- `_DEFAULT_WARN_THRESHOLD` constant deletion happens in Task 13

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-16-loop-detection-rewind-implementation.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
