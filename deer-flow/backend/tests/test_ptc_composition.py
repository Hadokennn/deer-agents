"""Tests for PTC composition layer: docs generation, wrappers, factory."""

import pytest
from pydantic import BaseModel, Field
from typing import Any

from deerflow.sandbox.ptc import (
    _build_function_docs,
    _build_tool_wrappers,
    make_ptc_tool,
)


# ---------- Fake schemas / tools for testing ----------


class _SchemaBash(BaseModel):
    command: str = Field(description="The command to run")
    description: str = Field(default="", description="What this command does")


class _SchemaGrep(BaseModel):
    pattern: str = Field()
    path: str = Field()
    glob: str | None = Field(default=None)


class _SchemaNoDesc(BaseModel):
    query: str = Field()


class _FakeTool:
    """Minimal fake tool compatible with _build_function_docs and _build_tool_wrappers."""

    def __init__(self, name, args_schema, description_str="fake tool", run_return: Any = "default"):
        self.name = name
        self.description = description_str
        self.args_schema = args_schema
        self.metadata = {}
        self._run_return = run_return
        self.calls = []

        def _func(**kwargs):
            self.calls.append(kwargs)
            return self._run_return

        self.func = _func


# ---------- _build_function_docs ----------


def test_build_function_docs_lists_tool_name():
    tool = _FakeTool("bash", _SchemaBash, "Execute a bash command")
    docs = _build_function_docs([(tool, None)])
    assert "bash(" in docs


def test_build_function_docs_hides_runtime_and_description_params():
    tool = _FakeTool("bash", _SchemaBash)
    docs = _build_function_docs([(tool, None)])
    assert "description: str" not in docs
    assert "runtime:" not in docs


def test_build_function_docs_shows_business_params():
    tool = _FakeTool("bash", _SchemaBash)
    docs = _build_function_docs([(tool, None)])
    assert "command: str" in docs


def test_build_function_docs_shows_multiple_params():
    tool = _FakeTool("grep", _SchemaGrep)
    docs = _build_function_docs([(tool, None)])
    assert "pattern" in docs
    assert "path" in docs
    assert "glob" in docs


def test_build_function_docs_marks_any_when_no_schema():
    tool = _FakeTool("bash", _SchemaBash)
    docs = _build_function_docs([(tool, None)])
    assert "-> Any" in docs


def test_build_function_docs_marks_structured_when_schema_present():
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    tool = _FakeTool("weather", _SchemaNoDesc)
    docs = _build_function_docs([(tool, schema)])
    assert "-> dict | list" in docs
    assert '"type": "object"' in docs or "type" in docs


def test_build_function_docs_empty_list_returns_stub():
    docs = _build_function_docs([])
    assert isinstance(docs, str) and len(docs) > 0


def test_build_function_docs_lists_multiple_tools():
    tools = [
        (_FakeTool("bash", _SchemaBash), None),
        (_FakeTool("grep", _SchemaGrep), None),
    ]
    docs = _build_function_docs(tools)
    assert "bash(" in docs and "grep(" in docs


# ---------- _build_tool_wrappers ----------


def test_build_tool_wrappers_creates_one_per_tool():
    a = _FakeTool("a", _SchemaNoDesc)
    b = _FakeTool("b", _SchemaNoDesc)
    wrappers = _build_tool_wrappers([(a, None), (b, None)])
    assert "a" in wrappers and "b" in wrappers


def test_wrapper_forwards_kwargs_and_injects_description_runtime():
    tool = _FakeTool("bash", _SchemaBash, run_return="hello")
    wrappers = _build_tool_wrappers([(tool, None)])
    wrapper = wrappers["bash"]
    wrapper._runtime = "test-runtime"

    result = wrapper(command="echo hi")
    assert result == "hello"
    assert tool.calls[0]["command"] == "echo hi"
    assert "description" in tool.calls[0]
    assert tool.calls[0]["runtime"] == "test-runtime"


def test_wrapper_skips_description_when_tool_does_not_accept_it():
    tool = _FakeTool("search", _SchemaNoDesc)
    wrappers = _build_tool_wrappers([(tool, None)])
    wrapper = wrappers["search"]
    wrapper._runtime = None

    wrapper(query="foo")
    assert "description" not in tool.calls[0]
    assert tool.calls[0]["query"] == "foo"


def test_wrapper_unwraps_mcp_tuple_to_structured_content():
    tool = _FakeTool(
        "mcp_tool",
        _SchemaNoDesc,
        run_return=(["text content"], {"structured_content": {"key": "value"}}),
    )
    wrappers = _build_tool_wrappers([(tool, None)])
    wrapper = wrappers["mcp_tool"]
    wrapper._runtime = None

    result = wrapper(query="foo")
    assert result == {"key": "value"}


def test_wrapper_preserves_plain_string_result():
    tool = _FakeTool("bash", _SchemaBash, run_return="raw output")
    wrappers = _build_tool_wrappers([(tool, None)])
    wrapper = wrappers["bash"]
    wrapper._runtime = None

    result = wrapper(command="echo hi")
    assert result == "raw output"


# ---------- make_ptc_tool ----------


def test_make_ptc_tool_returns_langchain_tool():
    from langchain_core.tools import BaseTool as LCBaseTool
    from deerflow.config.tool_config import PTCToolConfig, PTCEligibleToolConfig

    config = PTCToolConfig(
        name="ptc_test",
        purpose="Test purpose",
        eligible_tools=[PTCEligibleToolConfig(name="bash")],
    )
    tool = make_ptc_tool(config, [])
    assert isinstance(tool, LCBaseTool)
    assert tool.name == "ptc_test"


def test_make_ptc_tool_description_includes_purpose():
    from deerflow.config.tool_config import PTCToolConfig, PTCEligibleToolConfig

    config = PTCToolConfig(
        name="ptc_test",
        purpose="This purpose string must appear in the description",
        eligible_tools=[PTCEligibleToolConfig(name="bash")],
    )
    tool = make_ptc_tool(config, [])
    assert "This purpose string must appear in the description" in tool.description


def test_make_ptc_tool_description_lists_eligible_tools():
    from deerflow.config.tool_config import PTCToolConfig, PTCEligibleToolConfig

    bash = _FakeTool("bash", _SchemaBash, "Execute a bash command")
    config = PTCToolConfig(
        name="ptc_test",
        purpose="Test purpose",
        eligible_tools=[PTCEligibleToolConfig(name="bash")],
    )
    tool = make_ptc_tool(config, [(bash, None)])
    assert "bash(" in tool.description


def test_make_ptc_tool_executes_code_via_func():
    from deerflow.config.tool_config import PTCToolConfig, PTCEligibleToolConfig

    echo = _FakeTool("echo", _SchemaNoDesc, "Echo", run_return="hello")
    config = PTCToolConfig(
        name="ptc_test",
        purpose="Test purpose",
        eligible_tools=[PTCEligibleToolConfig(name="echo")],
    )
    tool = make_ptc_tool(config, [(echo, None)])

    result = tool.func(code="print(echo(query='x'))", runtime=None)
    assert result.strip() == "hello"


def test_make_ptc_tool_reports_errors_inline():
    from deerflow.config.tool_config import PTCToolConfig, PTCEligibleToolConfig

    config = PTCToolConfig(
        name="ptc_test",
        purpose="Test purpose",
        eligible_tools=[PTCEligibleToolConfig(name="bash")],
    )
    tool = make_ptc_tool(config, [])

    result = tool.func(code="1 / 0", runtime=None)
    assert "ZeroDivisionError" in result or "error" in result.lower()


def test_make_ptc_tool_honors_timeout_override():
    from deerflow.config.tool_config import PTCToolConfig, PTCEligibleToolConfig

    config = PTCToolConfig(
        name="ptc_test",
        purpose="Test purpose",
        eligible_tools=[PTCEligibleToolConfig(name="bash")],
        timeout_seconds=1,  # force fast timeout
    )
    tool = make_ptc_tool(config, [])
    result = tool.func(code="while True:\n    pass", runtime=None)
    assert "exceeded" in result.lower() or "timeout" in result.lower()


def test_make_ptc_tool_honors_max_output_override():
    from deerflow.config.tool_config import PTCToolConfig, PTCEligibleToolConfig

    config = PTCToolConfig(
        name="ptc_test",
        purpose="Test purpose",
        eligible_tools=[PTCEligibleToolConfig(name="bash")],
        max_output_chars=100,
    )
    tool = make_ptc_tool(config, [])
    result = tool.func(code="print('x' * 1000)", runtime=None)
    assert "truncated" in result
