"""Tests for loop_hint_builder module."""

from langchain_core.messages import AIMessage, ToolMessage

from deerflow.agents.middlewares.loop_hint_builder import (
    _extract_intent,
    _extract_text,
    _has_meaningful_text,
    _salient_args,
    build_no_progress_hint,
    build_rule_hint,
)


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


class TestHasMeaningfulText:
    def test_short_text_not_meaningful(self):
        assert _has_meaningful_text("ok") is False
        assert _has_meaningful_text("") is False
        assert _has_meaningful_text(None) is False

    def test_text_over_80_chars_meaningful(self):
        text = "The stack trace points to a missing import. Let me check the module structure carefully."
        assert _has_meaningful_text(text) is True

    def test_filler_only_not_meaningful(self):
        text = "Let me try this. I'll check the next file. Let me see."
        assert _has_meaningful_text(text) is False

    def test_filler_with_real_content_meaningful(self):
        text = "Let me check the imports. The stack points at line 42 which suggests a race condition in the worker pool."
        assert _has_meaningful_text(text) is True

    def test_list_content_evaluated(self):
        content = [
            {"type": "text", "text": "The root cause appears to be a mismatched schema between service A and service B after the deploy."},
        ]
        assert _has_meaningful_text(content) is True


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


class TestBuildNoProgressHint:
    def test_contains_no_progress_header(self):
        msgs = []
        for i in range(10):
            msgs.append(_ai("", [_tc("read_file", {"path": f"/f{i}"}, f"c{i}")]))
            msgs.append(_tool_msg("ok", f"c{i}"))
        hint = build_no_progress_hint(msgs, start=0, end=len(msgs) - 1)
        assert "[NO PROGRESS]" in hint
        assert "tool calls" in hint

    def test_contains_three_choices(self):
        msgs = [_ai("", [_tc("read_file", {"path": "/a"}, "c1")])]
        hint = build_no_progress_hint(msgs, start=0, end=0)
        assert "(a)" in hint
        assert "(b)" in hint
        assert "(c)" in hint

    def test_mentions_tool_call_count(self):
        msgs = []
        for i in range(7):
            msgs.append(_ai("", [_tc("read_file", {"path": f"/f{i}"}, f"c{i}")]))
            msgs.append(_tool_msg("ok", f"c{i}"))
        hint = build_no_progress_hint(msgs, start=0, end=len(msgs) - 1)
        # Should mention 7 tool calls in the region
        assert "7" in hint
