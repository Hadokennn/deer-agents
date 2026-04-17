"""Tests for LoopDetectionMiddleware."""

import copy
from unittest.mock import MagicMock

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deerflow.agents.middlewares.loop_detection_middleware import (
    _HARD_STOP_MSG,
    LoopDetectionMiddleware,
    _hash_tool_calls,
)
from deerflow.agents.middlewares.loop_hint_builder import build_no_progress_hint


def _ai(content="", tool_calls=()):
    return AIMessage(content=content, tool_calls=list(tool_calls))


def _tc(name, path, tc_id):
    return {"name": name, "args": {"path": path}, "id": tc_id}


def _tm(content, tc_id, name="read_file"):
    return ToolMessage(content=content, tool_call_id=tc_id, name=name)


def _make_runtime(thread_id="test-thread"):
    """Build a minimal Runtime mock with context."""
    runtime = MagicMock()
    runtime.context = {"thread_id": thread_id}
    return runtime


def _make_state(tool_calls=None, content=""):
    """Build a minimal AgentState dict with an AIMessage.

    Deep-copies *content* when it is mutable (e.g. list) so that
    successive calls never share the same object reference.
    """
    safe_content = copy.deepcopy(content) if isinstance(content, list) else content
    msg = AIMessage(content=safe_content, tool_calls=tool_calls or [])
    return {"messages": [msg]}


def _bash_call(cmd="ls"):
    return {"name": "bash", "id": f"call_{cmd}", "args": {"command": cmd}}


def _model_request(messages, thread_id="test-thread"):
    """Build a minimal ModelRequest for wrap_model_call testing."""
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


class TestHashToolCalls:
    def test_same_calls_same_hash(self):
        a = _hash_tool_calls([_bash_call("ls")])
        b = _hash_tool_calls([_bash_call("ls")])
        assert a == b

    def test_different_calls_different_hash(self):
        a = _hash_tool_calls([_bash_call("ls")])
        b = _hash_tool_calls([_bash_call("pwd")])
        assert a != b

    def test_order_independent(self):
        a = _hash_tool_calls([_bash_call("ls"), {"name": "read_file", "args": {"path": "/tmp"}}])
        b = _hash_tool_calls([{"name": "read_file", "args": {"path": "/tmp"}}, _bash_call("ls")])
        assert a == b

    def test_empty_calls(self):
        h = _hash_tool_calls([])
        assert isinstance(h, str)
        assert len(h) > 0

    def test_stringified_dict_args_match_dict_args(self):
        dict_call = {
            "name": "read_file",
            "args": {"path": "/tmp/demo.py", "start_line": "1", "end_line": "150"},
        }
        string_call = {
            "name": "read_file",
            "args": '{"path":"/tmp/demo.py","start_line":"1","end_line":"150"}',
        }

        assert _hash_tool_calls([dict_call]) == _hash_tool_calls([string_call])

    def test_reversed_read_file_range_matches_forward_range(self):
        forward_call = {
            "name": "read_file",
            "args": {"path": "/tmp/demo.py", "start_line": 10, "end_line": 300},
        }
        reversed_call = {
            "name": "read_file",
            "args": {"path": "/tmp/demo.py", "start_line": 300, "end_line": 10},
        }

        assert _hash_tool_calls([forward_call]) == _hash_tool_calls([reversed_call])

    def test_stringified_non_dict_args_do_not_crash(self):
        non_dict_json_call = {"name": "bash", "args": '"echo hello"'}
        plain_string_call = {"name": "bash", "args": "echo hello"}

        json_hash = _hash_tool_calls([non_dict_json_call])
        plain_hash = _hash_tool_calls([plain_string_call])

        assert isinstance(json_hash, str)
        assert isinstance(plain_hash, str)
        assert json_hash
        assert plain_hash

    def test_grep_pattern_affects_hash(self):
        grep_foo = {"name": "grep", "args": {"path": "/tmp", "pattern": "foo"}}
        grep_bar = {"name": "grep", "args": {"path": "/tmp", "pattern": "bar"}}

        assert _hash_tool_calls([grep_foo]) != _hash_tool_calls([grep_bar])

    def test_glob_pattern_affects_hash(self):
        glob_py = {"name": "glob", "args": {"path": "/tmp", "pattern": "*.py"}}
        glob_ts = {"name": "glob", "args": {"path": "/tmp", "pattern": "*.ts"}}

        assert _hash_tool_calls([glob_py]) != _hash_tool_calls([glob_ts])

    def test_write_file_content_affects_hash(self):
        v1 = {"name": "write_file", "args": {"path": "/tmp/a.py", "content": "v1"}}
        v2 = {"name": "write_file", "args": {"path": "/tmp/a.py", "content": "v2"}}
        assert _hash_tool_calls([v1]) != _hash_tool_calls([v2])

    def test_str_replace_content_affects_hash(self):
        a = {
            "name": "str_replace",
            "args": {"path": "/tmp/a.py", "old_str": "foo", "new_str": "bar"},
        }
        b = {
            "name": "str_replace",
            "args": {"path": "/tmp/a.py", "old_str": "foo", "new_str": "baz"},
        }
        assert _hash_tool_calls([a]) != _hash_tool_calls([b])


class TestLoopDetection:
    def test_no_tool_calls_returns_none(self):
        mw = LoopDetectionMiddleware()
        runtime = _make_runtime()
        state = {"messages": [AIMessage(content="hello")]}
        result = mw._apply(state, runtime)
        assert result is None

    def test_below_threshold_returns_none(self):
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        # First two identical calls — no warning
        for _ in range(2):
            result = mw._apply(_make_state(tool_calls=call), runtime)
            assert result is None

    def test_hard_stop_at_limit(self):
        mw = LoopDetectionMiddleware(rewind_threshold=2, hard_limit=4)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        for _ in range(3):
            mw._apply(_make_state(tool_calls=call), runtime)

        # Fourth call triggers hard stop
        result = mw._apply(_make_state(tool_calls=call), runtime)
        assert result is not None
        msgs = result["messages"]
        assert len(msgs) == 1
        # Hard stop strips tool_calls
        assert isinstance(msgs[0], AIMessage)
        assert msgs[0].tool_calls == []
        assert _HARD_STOP_MSG in msgs[0].content

    def test_different_calls_dont_trigger(self):
        mw = LoopDetectionMiddleware(rewind_threshold=2)
        runtime = _make_runtime()

        # Each call is different
        for i in range(10):
            result = mw._apply(_make_state(tool_calls=[_bash_call(f"cmd_{i}")]), runtime)
            assert result is None

    def test_window_sliding(self):
        mw = LoopDetectionMiddleware(rewind_threshold=3, window_size=5)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        # Fill with 2 identical calls
        mw._apply(_make_state(tool_calls=call), runtime)
        mw._apply(_make_state(tool_calls=call), runtime)

        # Push them out of the window with different calls
        for i in range(5):
            mw._apply(_make_state(tool_calls=[_bash_call(f"other_{i}")]), runtime)

        # Now the original call should be fresh again — no warning
        result = mw._apply(_make_state(tool_calls=call), runtime)
        assert result is None

    def test_reset_clears_state(self):
        mw = LoopDetectionMiddleware(rewind_threshold=2)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        mw._apply(_make_state(tool_calls=call), runtime)
        mw._apply(_make_state(tool_calls=call), runtime)

        # Would trigger warning, but reset first
        mw.reset()
        result = mw._apply(_make_state(tool_calls=call), runtime)
        assert result is None

    def test_non_ai_message_ignored(self):
        mw = LoopDetectionMiddleware()
        runtime = _make_runtime()
        state = {"messages": [SystemMessage(content="hello")]}
        result = mw._apply(state, runtime)
        assert result is None

    def test_empty_messages_ignored(self):
        mw = LoopDetectionMiddleware()
        runtime = _make_runtime()
        result = mw._apply({"messages": []}, runtime)
        assert result is None

    def test_lru_eviction(self):
        """Old threads should be evicted when max_tracked_threads is exceeded."""
        mw = LoopDetectionMiddleware(rewind_threshold=2, max_tracked_threads=3)
        call = [_bash_call("ls")]

        # Fill up 3 threads
        for i in range(3):
            runtime = _make_runtime(f"thread-{i}")
            mw._apply(_make_state(tool_calls=call), runtime)

        # Add a 4th thread — should evict thread-0
        runtime_new = _make_runtime("thread-new")
        mw._apply(_make_state(tool_calls=call), runtime_new)

        assert "thread-0" not in mw._history
        assert "thread-new" in mw._history
        assert len(mw._history) == 3

    def test_thread_safe_mutations(self):
        """Verify lock is used for mutations (basic structural test)."""
        mw = LoopDetectionMiddleware()
        # The middleware should have a lock attribute
        assert hasattr(mw, "_lock")
        assert isinstance(mw._lock, type(mw._lock))

    def test_fallback_thread_id_when_missing(self):
        """When runtime context has no thread_id, should use 'default'."""
        mw = LoopDetectionMiddleware(rewind_threshold=2)
        runtime = MagicMock()
        runtime.context = {}
        call = [_bash_call("ls")]

        mw._apply(_make_state(tool_calls=call), runtime)
        assert "default" in mw._history


class TestAppendText:
    """Unit tests for LoopDetectionMiddleware._append_text."""

    def test_none_content_returns_text(self):
        result = LoopDetectionMiddleware._append_text(None, "hello")
        assert result == "hello"

    def test_str_content_concatenates(self):
        result = LoopDetectionMiddleware._append_text("existing", "appended")
        assert result == "existing\n\nappended"

    def test_empty_str_content_concatenates(self):
        result = LoopDetectionMiddleware._append_text("", "appended")
        assert result == "\n\nappended"

    def test_list_content_appends_text_block(self):
        """List content (e.g. Anthropic thinking mode) should get a new text block."""
        content = [
            {"type": "thinking", "text": "Let me think..."},
            {"type": "text", "text": "Here is my answer"},
        ]
        result = LoopDetectionMiddleware._append_text(content, "stop msg")
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == content[0]
        assert result[1] == content[1]
        assert result[2] == {"type": "text", "text": "\n\nstop msg"}

    def test_empty_list_content_appends_text_block(self):
        result = LoopDetectionMiddleware._append_text([], "stop msg")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == {"type": "text", "text": "\n\nstop msg"}

    def test_unexpected_type_coerced_to_str(self):
        """Unexpected content types should be coerced to str as a fallback."""
        result = LoopDetectionMiddleware._append_text(42, "stop msg")
        assert isinstance(result, str)
        assert result == "42\n\nstop msg"

    def test_list_content_not_mutated_in_place(self):
        """_append_text must not modify the original list."""
        original = [{"type": "text", "text": "hello"}]
        result = LoopDetectionMiddleware._append_text(original, "appended")
        assert len(original) == 1  # original unchanged
        assert len(result) == 2  # new list has the appended block


class TestHardStopWithListContent:
    """Regression tests: hard stop must not crash when AIMessage.content is a list."""

    def test_hard_stop_with_list_content(self):
        """Hard stop on list content should not raise TypeError (regression)."""
        mw = LoopDetectionMiddleware(rewind_threshold=2, hard_limit=4)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        # Build state with list content (e.g. Anthropic thinking mode)
        list_content = [
            {"type": "thinking", "text": "Let me think..."},
            {"type": "text", "text": "I'll run ls"},
        ]

        for _ in range(3):
            mw._apply(_make_state(tool_calls=call, content=list_content), runtime)

        # Fourth call triggers hard stop — must not raise TypeError
        result = mw._apply(_make_state(tool_calls=call, content=list_content), runtime)
        assert result is not None
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert msg.tool_calls == []
        # Content should remain a list with the stop message appended
        assert isinstance(msg.content, list)
        assert len(msg.content) == 3
        assert msg.content[2]["type"] == "text"
        assert _HARD_STOP_MSG in msg.content[2]["text"]

    def test_hard_stop_with_none_content(self):
        """Hard stop on None content should produce a plain string."""
        mw = LoopDetectionMiddleware(rewind_threshold=2, hard_limit=4)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        for _ in range(3):
            mw._apply(_make_state(tool_calls=call), runtime)

        # Fourth call with default empty-string content
        result = mw._apply(_make_state(tool_calls=call), runtime)
        assert result is not None
        msg = result["messages"][0]
        assert isinstance(msg.content, str)
        assert _HARD_STOP_MSG in msg.content

    def test_hard_stop_with_str_content(self):
        """Hard stop on str content should concatenate the stop message."""
        mw = LoopDetectionMiddleware(rewind_threshold=2, hard_limit=4)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        for _ in range(3):
            mw._apply(_make_state(tool_calls=call, content="thinking..."), runtime)

        result = mw._apply(_make_state(tool_calls=call, content="thinking..."), runtime)
        assert result is not None
        msg = result["messages"][0]
        assert isinstance(msg.content, str)
        assert msg.content.startswith("thinking...")
        assert _HARD_STOP_MSG in msg.content


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


class TestExpandForToolMessages:
    def test_no_trailing_tool_messages(self):
        mw = LoopDetectionMiddleware()
        msgs = [_ai("", [_tc("read_file", "/a", "c1")])]
        # last AIMessage at idx 0; no following ToolMessages → end stays 0
        assert mw._expand_for_tool_messages(msgs, tool_call_ids=set(), region_end=0) == 0

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


class TestObservabilityDedup:
    def test_first_detection_logs_warning(self, caplog):
        import logging
        caplog.set_level(
            logging.WARNING,
            logger="deerflow.agents.middlewares.loop_detection_middleware",
        )
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = []
        for i in range(3):
            msgs.append(_ai("", [_tc("read_file", "/a", f"c{i}")]))
            msgs.append(_tm("err", f"c{i}"))
        req = _model_request(msgs)
        handler = MagicMock(return_value="r")

        mw.wrap_model_call(req, handler)

        first_detected = [
            r for r in caplog.records if "loop.rewind.first_detected" in r.message
        ]
        assert len(first_detected) == 1

    def test_subsequent_calls_silent(self, caplog):
        import logging
        caplog.set_level(
            logging.WARNING,
            logger="deerflow.agents.middlewares.loop_detection_middleware",
        )
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

        first_detected = [
            r for r in caplog.records if "loop.rewind.first_detected" in r.message
        ]
        assert len(first_detected) == 1   # only first call logs

    def test_different_threads_each_log_once(self, caplog):
        import logging
        caplog.set_level(
            logging.WARNING,
            logger="deerflow.agents.middlewares.loop_detection_middleware",
        )
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

        first_detected = [
            r for r in caplog.records if "loop.rewind.first_detected" in r.message
        ]
        assert len(first_detected) == 2


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


class TestDetectNoProgress:
    def test_below_min_window_returns_none(self):
        mw = LoopDetectionMiddleware()
        # Only 5 AIMessages — below default min_window_to_trigger=10
        msgs = []
        for i in range(5):
            msgs.append(_ai("", [_tc("read_file", f"/f{i}", f"c{i}")]))
            msgs.append(_tm("ok", f"c{i}"))
        assert mw._detect_no_progress(msgs) is None

    def test_meaningful_thinking_not_detected(self):
        mw = LoopDetectionMiddleware()
        msgs = []
        for i in range(15):
            content = f"Analyzing file {i}: the pattern suggests a race condition in the worker pool based on stack trace."
            msgs.append(_ai(content, [_tc("read_file", f"/f{i}", f"c{i}")]))
            msgs.append(_tm("ok", f"c{i}"))
        assert mw._detect_no_progress(msgs) is None

    def test_tool_only_no_progress_detected(self):
        mw = LoopDetectionMiddleware()
        msgs = []
        for i in range(15):
            msgs.append(_ai("", [_tc("read_file", f"/f{i}", f"c{i}")]))   # empty content
            msgs.append(_tm("ok", f"c{i}"))
        region = mw._detect_no_progress(msgs)
        assert region is not None
        start, end = region
        # Should cover the AIMessage range (indices 0, 2, 4, ..., 28)
        assert start == 0
        assert end == 28

    def test_mixed_but_mostly_no_progress(self):
        mw = LoopDetectionMiddleware(
            # allow triggering with 15 window + 0.85 ratio
        )
        msgs = []
        # 13 no-progress + 2 meaningful = 13/15 = 86.6% > 85%
        for i in range(13):
            msgs.append(_ai("", [_tc("read_file", f"/f{i}", f"c{i}")]))
            msgs.append(_tm("ok", f"c{i}"))
        for i in range(2):
            meaningful = f"Based on the data seen so far, the bug appears to be in module X because the stack trace clearly shows the failure point at line 42."
            msgs.append(_ai(meaningful, [_tc("read_file", f"/g{i}", f"g{i}")]))
            msgs.append(_tm("ok", f"g{i}"))
        region = mw._detect_no_progress(msgs)
        assert region is not None


class TestDetectorPriority:
    def test_hash_loop_takes_priority_over_no_progress(self):
        """When both V1 and V2.1 would trigger, V1 wins (more specific)."""
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = []
        # V1 hash loop: 3 identical tool calls
        for i in range(3):
            msgs.append(_ai("", [_tc("read_file", "/same", f"c{i}")]))
            msgs.append(_tm("err", f"c{i}"))
        # Extend to trigger V2.1 as well: more empty-content tool-call AIMessages
        for i in range(12):
            msgs.append(_ai("", [_tc("grep", {"path": f"/f{i}"}, f"g{i}")]))
            msgs.append(_tm("ok", f"g{i}"))

        req = _model_request(msgs)
        handler = MagicMock(return_value="r")
        mw.wrap_model_call(req, handler)

        # V1 hint (has "[LOOP RECOVERY]") should appear, not V2.1 hint ("[NO PROGRESS]")
        patched = req._captured["messages"]
        combined = "\n".join(
            m.content for m in patched if isinstance(m, HumanMessage)
        )
        assert "[LOOP RECOVERY]" in combined
        assert "[NO PROGRESS]" not in combined

    def test_no_progress_fires_when_no_hash_loop(self):
        """V2.1 fallback triggers when V1 detects nothing."""
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = []
        # 15 distinct tool calls, no hash loop, no meaningful thinking
        for i in range(15):
            msgs.append(_ai("", [_tc("read_file", f"/f{i}", f"c{i}")]))
            msgs.append(_tm("ok", f"c{i}"))

        req = _model_request(msgs)
        handler = MagicMock(return_value="r")
        mw.wrap_model_call(req, handler)

        patched = req._captured["messages"]
        combined = "\n".join(
            m.content for m in patched if isinstance(m, HumanMessage)
        )
        assert "[NO PROGRESS]" in combined
        assert "[LOOP RECOVERY]" not in combined

    def test_neither_fires_on_healthy_thread(self):
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = []
        for i in range(5):
            content = (
                f"Looking at step {i}: the investigation suggests a latency spike in the "
                f"queue worker based on profiler data from the previous call."
            )
            msgs.append(_ai(content, [_tc("read_file", f"/f{i}", f"c{i}")]))
            msgs.append(_tm("ok", f"c{i}"))

        req = _model_request(msgs)
        handler = MagicMock(return_value="r")
        mw.wrap_model_call(req, handler)

        # handler called with original request (no patching)
        req.override.assert_not_called()


class TestEndToEndNoProgress:
    def test_realistic_aimless_exploration(self):
        """Simulates oncall agent reading many different files without forming a hypothesis."""
        mw = LoopDetectionMiddleware(rewind_threshold=3)

        msgs = [
            HumanMessage(content="Investigate the 500 error in the payment service."),
        ]
        # 12 different read_file calls with only mechanical narration — no real thinking
        for i in range(12):
            msgs.append(_ai(f"Let me check file {i}.", [_tc("read_file", {"path": f"/svc/file{i}.py"}, f"c{i}")]))
            msgs.append(_tm(f"# contents of file{i}\npass", f"c{i}"))

        req = _model_request(msgs)
        handler = MagicMock(return_value="response")
        mw.wrap_model_call(req, handler)

        patched = req._captured["messages"]
        # Pre-message user query preserved
        assert isinstance(patched[0], HumanMessage)
        assert "500 error" in patched[0].content
        # Rest collapsed to one hint
        assert isinstance(patched[1], HumanMessage)
        assert "[NO PROGRESS]" in patched[1].content
        assert len(patched) == 2

    def test_hybrid_scenario_hash_loop_dominates(self):
        """Mix of hash loop + no-progress signal — hash takes priority, no-progress deferred."""
        mw = LoopDetectionMiddleware(rewind_threshold=3)
        msgs = [HumanMessage(content="Find the bug")]
        # Hash loop: 3 identical reads
        for i in range(3):
            msgs.append(_ai("", [_tc("read_file", "/dup.py", f"d{i}")]))
            msgs.append(_tm("Error: not found", f"d{i}"))
        # Plus more empty-content tool calls (would trigger V2.1 alone)
        for i in range(12):
            msgs.append(_ai("", [_tc("grep", {"path": f"/f{i}"}, f"g{i}")]))
            msgs.append(_tm("", f"g{i}"))

        req = _model_request(msgs)
        handler = MagicMock(return_value="r")
        mw.wrap_model_call(req, handler)

        patched = req._captured["messages"]
        combined = "\n".join(m.content for m in patched if isinstance(m, HumanMessage))
        # V1 hint wins
        assert "[LOOP RECOVERY]" in combined
        assert "[NO PROGRESS]" not in combined
