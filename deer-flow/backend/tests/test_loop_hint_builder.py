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
