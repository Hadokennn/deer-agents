"""Tests for pipelines/registry.py — name → tool lookup."""

from langchain_core.tools import StructuredTool

from pipelines.registry import ToolRegistry


def _make_tool(name: str) -> StructuredTool:
    def _run(value: str = "") -> dict:
        return {"echoed": value}

    return StructuredTool.from_function(
        func=_run,
        name=name,
        description=f"Test tool {name}",
    )


def test_register_and_get():
    registry = ToolRegistry()
    tool = _make_tool("foo")
    registry.register(tool)
    assert registry.get("foo") is tool


def test_get_missing_returns_none():
    registry = ToolRegistry()
    assert registry.get("nonexistent") is None


def test_register_many():
    registry = ToolRegistry()
    tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
    registry.register_many(tools)
    for t in tools:
        assert registry.get(t.name) is t


def test_from_tools_classmethod():
    tools = [_make_tool("x"), _make_tool("y")]
    registry = ToolRegistry.from_tools(tools)
    assert registry.get("x") is tools[0]
    assert registry.get("y") is tools[1]


def test_register_overwrites_same_name():
    registry = ToolRegistry()
    first = _make_tool("dup")
    second = _make_tool("dup")
    registry.register(first)
    registry.register(second)
    assert registry.get("dup") is second


def test_names_property_lists_registered():
    registry = ToolRegistry.from_tools([_make_tool("a"), _make_tool("b")])
    assert sorted(registry.names()) == ["a", "b"]
