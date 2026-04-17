"""Tests for loop_hint_builder module."""

from langchain_core.messages import AIMessage

from deerflow.agents.middlewares.loop_hint_builder import (
    _extract_intent,
    _extract_text,
    _salient_args,
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
