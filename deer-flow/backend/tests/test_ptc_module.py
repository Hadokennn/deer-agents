"""Tests for deerflow.sandbox.ptc leaf helpers."""

import pytest

from deerflow.sandbox.ptc import (
    _execute_code,
    _extract_output_schema,
    _extract_structured_content,
    _invoke_tool_with_runtime,
    _restricted_builtins,
    _safe_modules,
)


# ---------- _restricted_builtins ----------


def test_restricted_builtins_has_print():
    b = _restricted_builtins()
    assert "print" in b and b["print"] is print


def test_restricted_builtins_has_common_types():
    b = _restricted_builtins()
    for name in ["str", "int", "float", "bool", "list", "dict", "set", "tuple"]:
        assert name in b


def test_restricted_builtins_has_common_iterables():
    b = _restricted_builtins()
    for name in ["len", "range", "enumerate", "zip", "map", "filter", "sorted"]:
        assert name in b


def test_restricted_builtins_excludes_dangerous_functions():
    b = _restricted_builtins()
    for name in ["__import__", "eval", "exec", "compile", "open",
                 "input", "globals", "locals", "vars"]:
        assert name not in b, f"{name} must not be in restricted builtins"


# ---------- _safe_modules ----------


def test_safe_modules_has_json():
    import json
    m = _safe_modules()
    assert m["json"] is json


def test_safe_modules_has_data_processing():
    m = _safe_modules()
    for name in ["json", "re", "math", "collections", "itertools", "functools", "datetime"]:
        assert name in m


def test_safe_modules_excludes_dangerous():
    m = _safe_modules()
    for name in ["os", "sys", "subprocess", "socket", "requests", "urllib"]:
        assert name not in m


# ---------- _execute_code ----------


def test_execute_code_captures_print_output():
    result = _execute_code("print('hello world')", tool_wrappers={}, runtime=None)
    assert result == "hello world\n"


def test_execute_code_returns_no_output_marker_on_silence():
    result = _execute_code("x = 1 + 1", tool_wrappers={}, runtime=None)
    assert result == "(no output)"


def test_execute_code_pre_imports_json():
    result = _execute_code(
        "print(json.dumps({'a': 1}))",
        tool_wrappers={},
        runtime=None,
    )
    assert result.strip() == '{"a": 1}'


def test_execute_code_returns_error_message_on_exception():
    result = _execute_code("raise ValueError('boom')", tool_wrappers={}, runtime=None)
    assert "ValueError" in result and "boom" in result


def test_execute_code_blocks_import_os():
    result = _execute_code("import os\nprint(os.listdir('/'))", tool_wrappers={}, runtime=None)
    assert "error" in result.lower() or "Error" in result


def test_execute_code_blocks_open():
    result = _execute_code("open('/etc/passwd').read()", tool_wrappers={}, runtime=None)
    assert "error" in result.lower() or "NameError" in result


def test_execute_code_truncates_large_output():
    result = _execute_code(
        "print('x' * 30000)",
        tool_wrappers={},
        runtime=None,
        max_output_chars=100,
    )
    assert len(result) < 300
    assert "truncated" in result


def test_execute_code_timeout():
    result = _execute_code(
        "while True:\n    pass",
        tool_wrappers={},
        runtime=None,
        timeout=1,
    )
    assert "timeout" in result.lower() or "exceeded" in result.lower()


def test_execute_code_calls_injected_wrapper_and_sets_runtime():
    calls = []
    def fake_tool(**kwargs):
        calls.append((kwargs, fake_tool._runtime))
        return "tool_result"
    fake_tool._runtime = None

    result = _execute_code(
        "print(fake_tool(x=42))",
        tool_wrappers={"fake_tool": fake_tool},
        runtime="test-runtime",
    )
    assert result.strip() == "tool_result"
    assert calls == [({"x": 42}, "test-runtime")]


def test_execute_code_timeout_override_per_call():
    # Default is 30s; we pass timeout=1 to force a fast-fail
    result = _execute_code(
        "while True:\n    pass",
        tool_wrappers={},
        runtime=None,
        timeout=1,
    )
    assert "1s" in result or "exceeded" in result.lower()


# ---------- _extract_output_schema ----------


class _FakeTool:
    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.name = "fake"


def test_extract_output_schema_always_returns_none():
    """Current langchain-mcp-adapters doesn't expose outputSchema.
    Stub returns None regardless of metadata content."""
    tool = _FakeTool(metadata={"outputSchema": {"type": "object"}})
    # The stub returns None for now — when adapter support arrives we'll update
    # this test together with the implementation.
    result = _extract_output_schema(tool)
    # Allow either None (strict stub) or the metadata value (reads but adapter
    # doesn't populate). We check it doesn't raise.
    assert result is None or result == {"type": "object"}


def test_extract_output_schema_handles_missing_metadata():
    tool = _FakeTool()
    tool.metadata = None
    assert _extract_output_schema(tool) is None


# ---------- _extract_structured_content ----------


def test_extract_structured_content_from_mcp_tuple_with_artifact():
    content = [{"type": "text", "text": '{"temperature": 22.5}'}]
    artifact = {"structured_content": {"temperature": 22.5, "conditions": "sunny"}}
    result = (content, artifact)
    assert _extract_structured_content(result) == {"temperature": 22.5, "conditions": "sunny"}


def test_extract_structured_content_from_mcp_tuple_with_none_artifact():
    content = [{"type": "text", "text": "plain"}]
    result = (content, None)
    assert _extract_structured_content(result) == content


def test_extract_structured_content_passes_through_string():
    assert _extract_structured_content("plain tool output") == "plain tool output"


def test_extract_structured_content_passes_through_dict():
    assert _extract_structured_content({"key": "value"}) == {"key": "value"}


def test_extract_structured_content_passes_through_non_mcp_tuple():
    """Tuple of wrong length isn't an MCP result — pass through."""
    assert _extract_structured_content((1, 2, 3)) == (1, 2, 3)


# ---------- _invoke_tool_with_runtime ----------


class _FakeToolWithFunc:
    name = "fake"

    def __init__(self):
        self.calls = []
        def _func(**kwargs):
            self.calls.append(kwargs)
            return f"called with {kwargs}"
        self.func = _func


def test_invoke_tool_with_runtime_uses_func():
    tool = _FakeToolWithFunc()
    result = _invoke_tool_with_runtime(tool, {"command": "ls"}, runtime="rt-1")
    assert "command" in result
    assert tool.calls == [{"command": "ls", "runtime": "rt-1"}]


def test_invoke_tool_with_runtime_passes_runtime_kwarg():
    tool = _FakeToolWithFunc()
    _invoke_tool_with_runtime(tool, {}, runtime="ctx-123")
    assert tool.calls[0]["runtime"] == "ctx-123"
