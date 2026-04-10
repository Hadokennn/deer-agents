"""Tests for codeact/namespace.py — BaseTool → Python callable mapping."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from codeact.namespace import ToolNamespace


class _AdderInput(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")


def _adder_tool() -> StructuredTool:
    def _run(a: int, b: int) -> dict:
        return {"sum": a + b}

    return StructuredTool.from_function(
        func=_run,
        name="adder",
        description="Add two integers",
        args_schema=_AdderInput,
    )


def _greeter_tool() -> StructuredTool:
    def _run(name: str = "world") -> str:
        return f"hello {name}"

    return StructuredTool.from_function(
        func=_run,
        name="greeter",
        description="Greet someone",
    )


def test_build_returns_dict_with_tool_names_as_keys():
    tools = [_adder_tool(), _greeter_tool()]
    ns = ToolNamespace.build(tools)

    assert "adder" in ns
    assert "greeter" in ns
    assert callable(ns["adder"])
    assert callable(ns["greeter"])


def test_build_callable_invokes_tool():
    tools = [_adder_tool()]
    ns = ToolNamespace.build(tools)

    result = ns["adder"](a=2, b=3)
    assert result == {"sum": 5}


def test_build_callable_works_with_default_args():
    tools = [_greeter_tool()]
    ns = ToolNamespace.build(tools)

    assert ns["greeter"](name="alice") == "hello alice"


def test_build_empty_tools_returns_empty_dict():
    assert ToolNamespace.build([]) == {}


def test_render_signatures_includes_tool_name_and_doc():
    tools = [_adder_tool(), _greeter_tool()]
    output = ToolNamespace.render_signatures(tools)

    assert "def adder(" in output
    assert "Add two integers" in output
    assert "def greeter(" in output
    assert "Greet someone" in output


def test_render_signatures_includes_argument_types():
    tools = [_adder_tool()]
    output = ToolNamespace.render_signatures(tools)

    assert "a: int" in output
    assert "b: int" in output


def test_namespace_does_not_share_state_between_calls():
    """Each tool call should be independent."""
    ns = ToolNamespace.build([_adder_tool()])
    r1 = ns["adder"](a=1, b=1)
    r2 = ns["adder"](a=10, b=20)
    assert r1 == {"sum": 2}
    assert r2 == {"sum": 30}
