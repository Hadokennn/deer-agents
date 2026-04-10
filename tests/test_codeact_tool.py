"""Tests for codeact/code_act_tool.py — LangChain StructuredTool wrapper."""

from langchain_core.tools import StructuredTool

from codeact import make_code_act_tool
from codeact.sandbox import CodeExecutionSandbox


def _adder_tool() -> StructuredTool:
    def _run(a: int, b: int) -> dict:
        return {"sum": a + b}
    return StructuredTool.from_function(func=_run, name="adder", description="Add a and b")


def _greeter_tool() -> StructuredTool:
    def _run(name: str = "world") -> str:
        return f"hello {name}"
    return StructuredTool.from_function(func=_run, name="greeter", description="Greet")


def test_make_code_act_tool_returns_structured_tool():
    tool = make_code_act_tool(available_tools=[_adder_tool()])
    assert isinstance(tool, StructuredTool)
    assert tool.name == "code_execute"


def test_description_includes_available_tool_signatures():
    tool = make_code_act_tool(available_tools=[_adder_tool(), _greeter_tool()])
    assert "def adder(" in tool.description
    assert "def greeter(" in tool.description
    assert "Add a and b" in tool.description
    assert "Greet" in tool.description


def test_invoke_runs_simple_code():
    tool = make_code_act_tool(available_tools=[])
    result = tool.invoke({"code": "result = 1 + 2"})

    assert result["success"] is True
    assert result["return_value"] == 3
    assert result["exception"] is None


def test_invoke_can_call_provided_tools():
    tool = make_code_act_tool(available_tools=[_adder_tool()])
    code = """
sum1 = adder(a=1, b=2)
sum2 = adder(a=10, b=20)
result = {"first": sum1["sum"], "second": sum2["sum"]}
"""
    result = tool.invoke({"code": code})

    assert result["success"] is True
    assert result["return_value"] == {"first": 3, "second": 30}


def test_invoke_with_failing_code_returns_error_in_result():
    tool = make_code_act_tool(available_tools=[])
    result = tool.invoke({"code": "result = 1 / 0"})

    assert result["success"] is False
    assert "ZeroDivisionError" in result["exception"]
    assert result["traceback"] is not None


def test_invoke_captures_stdout_alongside_result():
    tool = make_code_act_tool(available_tools=[])
    code = """
print("step 1")
print("step 2")
result = 42
"""
    result = tool.invoke({"code": code})

    assert result["success"] is True
    assert "step 1" in result["stdout"]
    assert "step 2" in result["stdout"]
    assert result["return_value"] == 42


def test_custom_name_and_sandbox_are_used():
    custom_sandbox = CodeExecutionSandbox(timeout_seconds=5.0)
    tool = make_code_act_tool(
        available_tools=[],
        sandbox=custom_sandbox,
        name="run_python",
    )
    assert tool.name == "run_python"
    result = tool.invoke({"code": "result = 'ok'"})
    assert result["return_value"] == "ok"
