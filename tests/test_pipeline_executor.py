"""Tests for pipelines/executor.py — step-by-step pipeline execution."""

import pytest
from langchain_core.tools import StructuredTool

from pipelines.errors import PipelineStepError
from pipelines.executor import PipelineExecutor
from pipelines.parser import Pipeline, PipelineStep
from pipelines.registry import ToolRegistry


def _echo_tool() -> StructuredTool:
    def _run(value: str = "") -> dict:
        return {"echoed": value, "length": len(value)}

    return StructuredTool.from_function(func=_run, name="echo", description="Echo input")


def _adder_tool() -> StructuredTool:
    def _run(a: int = 0, b: int = 0) -> dict:
        return {"sum": a + b}

    return StructuredTool.from_function(func=_run, name="adder", description="Add a and b")


def _failing_tool() -> StructuredTool:
    def _run(**_kwargs) -> dict:
        raise RuntimeError("intentional failure")

    return StructuredTool.from_function(func=_run, name="failer", description="Always fails")


def _make_executor(tools) -> PipelineExecutor:
    return PipelineExecutor(ToolRegistry.from_tools(tools))


def test_execute_single_step():
    pipeline = Pipeline(
        name="test",
        description="single step",
        input_schema={"text": {"type": "str"}},
        steps=[
            PipelineStep(id="echo", tool="echo", input={"value": "${input.text}"}),
        ],
        output_template={"result": "${echo.echoed}"},
    )
    executor = _make_executor([_echo_tool()])
    result = executor.execute(pipeline, {"text": "hello"})
    assert result == {"result": "hello"}


def test_execute_chained_steps_passes_data():
    pipeline = Pipeline(
        name="chain",
        description="step b uses step a output",
        input_schema={},
        steps=[
            PipelineStep(id="a", tool="adder", input={"a": 1, "b": 2}),
            PipelineStep(id="b", tool="adder", input={"a": "${a.sum}", "b": 10}),
        ],
        output_template={"final": "${b.sum}"},
    )
    executor = _make_executor([_adder_tool()])
    result = executor.execute(pipeline, {})
    assert result == {"final": 13}


def test_execute_when_condition_skips_step():
    pipeline = Pipeline(
        name="skip",
        description="conditional skip",
        input_schema={"go": {"type": "bool"}},
        steps=[
            PipelineStep(
                id="echo",
                tool="echo",
                input={"value": "ran"},
                when="${input.go}",
            ),
        ],
        output_template={"result": '${echo.echoed | "skipped"}'},
    )
    executor = _make_executor([_echo_tool()])
    assert executor.execute(pipeline, {"go": True}) == {"result": "ran"}
    assert executor.execute(pipeline, {"go": False}) == {"result": "skipped"}


def test_execute_unless_condition_inverts_skip():
    pipeline = Pipeline(
        name="unless",
        description="inverse condition",
        input_schema={"flag": {"type": "bool"}},
        steps=[
            PipelineStep(
                id="echo",
                tool="echo",
                input={"value": "ran"},
                unless="${input.flag}",
            ),
        ],
        output_template={"result": '${echo.echoed | "skipped"}'},
    )
    executor = _make_executor([_echo_tool()])
    assert executor.execute(pipeline, {"flag": False}) == {"result": "ran"}
    assert executor.execute(pipeline, {"flag": True}) == {"result": "skipped"}


def test_execute_optional_step_failure_continues():
    pipeline = Pipeline(
        name="optional",
        description="optional failing step",
        input_schema={},
        steps=[
            PipelineStep(id="bad", tool="failer", input={}, optional=True),
            PipelineStep(id="ok", tool="echo", input={"value": "after"}),
        ],
        output_template={
            "bad_result": '${bad.echoed | "none"}',
            "ok_result": "${ok.echoed}",
        },
    )
    executor = _make_executor([_echo_tool(), _failing_tool()])
    result = executor.execute(pipeline, {})
    assert result == {"bad_result": "none", "ok_result": "after"}


def test_execute_required_step_failure_raises():
    pipeline = Pipeline(
        name="required_fail",
        description="non-optional failure",
        input_schema={},
        steps=[
            PipelineStep(id="bad", tool="failer", input={}),
        ],
        output_template={},
    )
    executor = _make_executor([_failing_tool()])
    with pytest.raises(PipelineStepError, match="step 'bad' failed"):
        executor.execute(pipeline, {})


def test_execute_missing_tool_raises():
    pipeline = Pipeline(
        name="missing_tool",
        description="tool not in registry",
        input_schema={},
        steps=[
            PipelineStep(id="x", tool="nonexistent", input={}),
        ],
        output_template={},
    )
    executor = _make_executor([])
    with pytest.raises(PipelineStepError, match="tool 'nonexistent' not found"):
        executor.execute(pipeline, {})


def test_execute_output_template_resolves_with_all_steps():
    pipeline = Pipeline(
        name="output",
        description="output uses multiple steps",
        input_schema={},
        steps=[
            PipelineStep(id="a", tool="adder", input={"a": 1, "b": 2}),
            PipelineStep(id="b", tool="adder", input={"a": 5, "b": 5}),
        ],
        output_template={"sum_a": "${a.sum}", "sum_b": "${b.sum}", "label": "ok"},
    )
    executor = _make_executor([_adder_tool()])
    result = executor.execute(pipeline, {})
    assert result == {"sum_a": 3, "sum_b": 10, "label": "ok"}


def test_execute_skipped_step_sets_none_in_context():
    pipeline = Pipeline(
        name="skipped_in_context",
        description="downstream step references skipped step",
        input_schema={"go": {"type": "bool"}},
        steps=[
            PipelineStep(
                id="maybe",
                tool="echo",
                input={"value": "ran"},
                when="${input.go}",
            ),
            PipelineStep(
                id="downstream",
                tool="echo",
                input={"value": '${maybe.echoed | "fallback"}'},
            ),
        ],
        output_template={"result": "${downstream.echoed}"},
    )
    executor = _make_executor([_echo_tool()])
    assert executor.execute(pipeline, {"go": False}) == {"result": "fallback"}
