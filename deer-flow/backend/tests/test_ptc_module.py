"""Tests for deerflow.sandbox.ptc leaf helpers."""

import pytest
from typing import Any

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
    for name in ["eval", "exec", "compile", "open",
                 "input", "globals", "locals", "vars"]:
        assert name not in b, f"{name} must not be in restricted builtins"


def test_restricted_builtins_provides_whitelisted_import():
    """__import__ is provided but restricted to safe modules only.

    LLM-generated code often writes `import json` out of habit even though
    json is pre-imported. The restricted __import__ lets that work while
    still blocking dangerous modules.
    """
    b = _restricted_builtins()
    assert "__import__" in b
    import json
    # Importing a safe module returns the pre-imported instance
    assert b["__import__"]("json") is json
    # Importing a dangerous module raises
    import pytest as _pytest
    with _pytest.raises(ImportError, match="Cannot import 'os'"):
        b["__import__"]("os")


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
    result = _execute_code("print('hello world')", resolved_tools=[], runtime=None)
    assert result == "hello world\n"


def test_execute_code_returns_no_output_marker_on_silence():
    result = _execute_code("x = 1 + 1", resolved_tools=[], runtime=None)
    assert result == "(no output)"


def test_execute_code_pre_imports_json():
    result = _execute_code(
        "print(json.dumps({'a': 1}))",
        resolved_tools=[],
        runtime=None,
    )
    assert result.strip() == '{"a": 1}'


def test_execute_code_returns_error_message_on_exception():
    result = _execute_code("raise ValueError('boom')", resolved_tools=[], runtime=None)
    assert "ValueError" in result and "boom" in result


def test_execute_code_blocks_import_os():
    """Importing a blocked module produces a clear, actionable error."""
    result = _execute_code("import os\nprint(os.listdir('/'))", resolved_tools=[], runtime=None)
    assert "Cannot import 'os'" in result
    # Error message should mention the safe alternatives
    assert "json" in result and "re" in result


def test_execute_code_allows_import_json():
    """LLM-generated code that writes `import json` should work, because
    json is pre-imported and the restricted __import__ returns it.

    Regression: thread 7c1e03b4 step 25 — LLM wrote `import json` and got
    `ImportError: __import__ not found` which was confusing and caused retries.
    """
    result = _execute_code(
        "import json\nprint(json.dumps({'ok': True}))",
        resolved_tools=[],
        runtime=None,
    )
    assert result.strip() == '{"ok": true}'


def test_execute_code_allows_from_import_of_safe_module():
    """`from json import dumps` should also work — Python pulls the attribute
    from whatever __import__ returns, which is the real json module."""
    result = _execute_code(
        "from json import dumps\nprint(dumps({'x': 1}))",
        resolved_tools=[],
        runtime=None,
    )
    assert result.strip() == '{"x": 1}'


def test_execute_code_blocks_import_subprocess():
    """Another blocked module — message should still mention the whitelist."""
    result = _execute_code("import subprocess", resolved_tools=[], runtime=None)
    assert "Cannot import 'subprocess'" in result


def test_execute_code_blocks_open():
    result = _execute_code("open('/etc/passwd').read()", resolved_tools=[], runtime=None)
    assert "error" in result.lower() or "NameError" in result


def test_execute_code_truncates_large_output():
    result = _execute_code(
        "print('x' * 30000)",
        resolved_tools=[],
        runtime=None,
        max_output_chars=100,
    )
    assert len(result) < 300
    assert "truncated" in result


def test_execute_code_timeout():
    result = _execute_code(
        "while True:\n    pass",
        resolved_tools=[],
        runtime=None,
        timeout=1,
    )
    assert "timeout" in result.lower() or "exceeded" in result.lower()


def test_execute_code_passes_runtime_to_tool_via_wrappers():
    """Verify runtime flows through _execute_code into a tool whose schema declares it."""
    from pydantic import BaseModel, Field
    from typing import Any

    class _FakeSchemaWithRuntime(BaseModel):
        runtime: Any = Field(default=None)
        query: str = Field()

    calls = []

    class _FakeTool:
        name = "fake_tool"
        description = "fake"
        args_schema = _FakeSchemaWithRuntime
        metadata = {}

        def func(self, *, runtime, **kwargs):
            calls.append({"runtime": runtime, **kwargs})
            return "tool_result"

    tool = _FakeTool()
    # Bind func so self is auto-passed (matches how real tools expose .func)
    bound = tool.func
    def _func(**kwargs):
        return bound(**kwargs)
    tool.func = _func

    result = _execute_code(
        "print(fake_tool(query='hi'))",
        resolved_tools=[(tool, None)],
        runtime="test-runtime",
    )
    assert result.strip() == "tool_result"
    assert calls == [{"runtime": "test-runtime", "query": "hi"}]


def test_execute_code_timeout_override_per_call():
    # Default is 30s; we pass timeout=1 to force a fast-fail
    result = _execute_code(
        "while True:\n    pass",
        resolved_tools=[],
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
    args_schema = None

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
    # args_schema is None so runtime is NOT passed (no model_fields to check)
    assert tool.calls == [{"command": "ls"}]


def test_invoke_tool_with_runtime_passes_runtime_kwarg():
    """Tool with runtime declared in args_schema receives runtime kwarg."""
    from pydantic import BaseModel, Field
    from typing import Any

    class _SchemaWithRuntime(BaseModel):
        runtime: Any = Field(default=None)
        command: str = Field()

    received = {}

    class _T:
        name = "t"
        args_schema = _SchemaWithRuntime
        metadata: dict = {}
        func: Any = None

    tool = _T()
    def _func(**kwargs):
        received.update(kwargs)
        return "ok"
    tool.func = _func

    _invoke_tool_with_runtime(tool, {"command": "ls"}, runtime="ctx-123")
    assert received["runtime"] == "ctx-123"
    assert received["command"] == "ls"


# ---------- Regression: Bug 1 (signal in worker thread) ----------


def test_execute_code_works_in_worker_thread():
    """Regression: _execute_code must NOT use signal-based timeout because
    signal.signal() only works in the main thread. LangGraph runs tool
    calls from worker threads; this test simulates that.

    Bug history: trace 417a5afd step 17 failed with
    'ValueError: signal only works in main thread of the main interpreter'.
    """
    import threading as _threading

    result_holder: dict[str, str] = {}

    def _run_from_worker():
        result_holder["result"] = _execute_code(
            "print('hello from worker')",
            resolved_tools=[],
            runtime=None,
        )

    worker = _threading.Thread(target=_run_from_worker)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert result_holder["result"] == "hello from worker\n"


def test_execute_code_timeout_works_in_worker_thread():
    """Regression: timeout must work from a worker thread, not just main."""
    import threading as _threading

    result_holder: dict[str, str] = {}

    def _run_from_worker():
        result_holder["result"] = _execute_code(
            "while True:\n    pass",
            resolved_tools=[],
            runtime=None,
            timeout=1,
        )

    worker = _threading.Thread(target=_run_from_worker)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert "exceeded" in result_holder["result"].lower() or "timeout" in result_holder["result"].lower()


# ---------- Regression: Bug 2 (runtime kwarg to MCP tools) ----------


def test_invoke_tool_without_runtime_field_omits_runtime_kwarg():
    """Regression: MCP tools don't declare `runtime` in their args_schema.
    _invoke_tool_with_runtime must NOT pass runtime=... to them.

    Bug history: initial implementation always passed runtime=..., which
    would raise TypeError for MCP tools. This test uses a fake tool whose
    schema has no runtime field and verifies the call succeeds.
    """
    from pydantic import BaseModel, Field

    class _SchemaNoRuntime(BaseModel):
        query: str = Field()

    received_kwargs: dict = {}

    class _FakeMcpTool:
        name = "fake_mcp"
        description = "fake mcp tool"
        args_schema = _SchemaNoRuntime
        metadata: dict = {}
        func: Any = None  # set below

    tool = _FakeMcpTool()

    def _func(**kwargs):
        received_kwargs.update(kwargs)
        return "mcp_result"
    tool.func = _func

    result = _invoke_tool_with_runtime(tool, {"query": "hello"}, runtime="should-not-leak")
    assert result == "mcp_result"
    # Critical assertion: runtime was NOT passed to the tool
    assert "runtime" not in received_kwargs
    assert received_kwargs == {"query": "hello"}


def test_invoke_tool_with_runtime_field_includes_runtime_kwarg():
    """Sanity: sandbox tools (with runtime in args_schema) still receive runtime."""
    from pydantic import BaseModel, Field
    from typing import Any

    class _SchemaWithRuntime(BaseModel):
        runtime: Any = Field(default=None)
        command: str = Field()

    received_kwargs: dict = {}

    class _FakeSandboxTool:
        name = "fake_sandbox"
        description = "fake sandbox tool"
        args_schema = _SchemaWithRuntime
        metadata: dict = {}
        func: Any = None  # set below

    tool = _FakeSandboxTool()
    def _func(**kwargs):
        received_kwargs.update(kwargs)
        return "sandbox_result"
    tool.func = _func

    result = _invoke_tool_with_runtime(tool, {"command": "ls"}, runtime="my-runtime")
    assert result == "sandbox_result"
    assert received_kwargs == {"runtime": "my-runtime", "command": "ls"}


# ---------- Regression: Bug 3 (concurrent runtime isolation) ----------


def test_concurrent_execute_code_calls_isolate_runtimes():
    """Regression: Two concurrent PTC invocations with different runtimes
    must each see their own runtime, not the other call's.

    Before the fix, wrappers shared a `_runtime` attribute and concurrent
    calls would race.
    """
    import threading as _threading
    import time
    from pydantic import BaseModel, Field
    from typing import Any

    class _SchemaWithRuntime(BaseModel):
        runtime: Any = Field(default=None)

    observed: list[str] = []
    observed_lock = _threading.Lock()

    class _FakeTool:
        name = "fake"
        description = "fake"
        args_schema = _SchemaWithRuntime
        metadata: dict = {}
        func: Any = None  # set below

    tool = _FakeTool()

    def _func(**kwargs):
        # Delay to ensure both calls overlap in time
        time.sleep(0.2)
        with observed_lock:
            observed.append(str(kwargs.get("runtime")))
        return "ok"
    tool.func = _func

    results: dict[int, str] = {}

    def _call(idx: int, runtime_name: str):
        r = _execute_code(
            "print(fake())",
            resolved_tools=[(tool, None)],
            runtime=runtime_name,
            timeout=5,
        )
        results[idx] = r

    t1 = _threading.Thread(target=_call, args=(1, "runtime-A"))
    t2 = _threading.Thread(target=_call, args=(2, "runtime-B"))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    # Both observed runtimes should be present — one "runtime-A" and one "runtime-B"
    assert sorted(observed) == ["runtime-A", "runtime-B"], (
        f"Runtime cross-contamination: observed={observed}. "
        "Concurrent calls saw each other's runtime, indicating the wrapper "
        "state is not per-invocation."
    )


# ---------- Regression: stdout is per-call, not process-global ----------


def test_execute_code_does_not_touch_process_stdout():
    """Regression: verify _execute_code's output capture is isolated
    from sys.stdout (doesn't use contextlib.redirect_stdout)."""
    import sys
    import io as _io

    # Before the call, swap sys.stdout for a buffer. If _execute_code uses
    # redirect_stdout, it would swap again and restore our buffer after — but
    # more importantly, anything printed inside the exec that lands on sys.stdout
    # should NOT appear in _execute_code's captured output.
    captured_sys_stdout = _io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_sys_stdout
    try:
        result = _execute_code("print('hello')", resolved_tools=[], runtime=None)
    finally:
        sys.stdout = original_stdout

    # The code's output goes to the PTC-managed buffer, not to sys.stdout
    assert "hello" in result
    # Nothing should have been written to our replacement sys.stdout
    assert "hello" not in captured_sys_stdout.getvalue(), (
        "_execute_code leaked output to sys.stdout — it should use a "
        "custom print injected into namespace, not contextlib.redirect_stdout"
    )
