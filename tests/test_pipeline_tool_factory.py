"""Tests for pipelines/tool_factory.py — Pipeline to LangChain StructuredTool."""

from typing import cast

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from pipelines.executor import PipelineExecutor
from pipelines.parser import Pipeline, PipelineStep
from pipelines.registry import ToolRegistry
from pipelines.tool_factory import PipelineToolFactory


def _echo_tool() -> StructuredTool:
    def _run(value: str = "") -> dict:
        return {"echoed": value}
    return StructuredTool.from_function(func=_run, name="echo", description="Echo")


def _build_pipeline() -> Pipeline:
    return Pipeline(
        name="greet",
        description="Greet a user by name",
        input_schema={
            "name": {"type": "str", "description": "User name"},
            "loud": {"type": "bool", "description": "Whether to shout"},
        },
        steps=[
            PipelineStep(id="echo", tool="echo", input={"value": "${input.name}"}),
        ],
        output_template={"greeting": "${echo.echoed}", "loud": "${input.loud}"},
    )


def test_to_tool_returns_structured_tool():
    pipeline = _build_pipeline()
    executor = PipelineExecutor(ToolRegistry.from_tools([_echo_tool()]))
    tool = PipelineToolFactory.to_tool(pipeline, executor)

    assert isinstance(tool, StructuredTool)
    assert tool.name == "greet"
    assert tool.description == "Greet a user by name"


def test_to_tool_args_schema_has_declared_fields():
    pipeline = _build_pipeline()
    executor = PipelineExecutor(ToolRegistry.from_tools([_echo_tool()]))
    tool = PipelineToolFactory.to_tool(pipeline, executor)

    schema_fields = cast(type[BaseModel], tool.args_schema).model_fields
    assert "name" in schema_fields
    assert "loud" in schema_fields
    assert schema_fields["name"].annotation is str
    assert schema_fields["loud"].annotation is bool


def test_invoking_tool_runs_pipeline():
    pipeline = _build_pipeline()
    executor = PipelineExecutor(ToolRegistry.from_tools([_echo_tool()]))
    tool = PipelineToolFactory.to_tool(pipeline, executor)

    result = tool.invoke({"name": "alice", "loud": False})
    assert result == {"greeting": "alice", "loud": False}


def test_to_tool_supports_int_and_dict_types():
    pipeline = Pipeline(
        name="multi_type",
        description="Pipeline with int and dict inputs",
        input_schema={
            "count": {"type": "int", "description": "A number"},
            "meta": {"type": "dict", "description": "Metadata"},
        },
        steps=[],
        output_template={"got_count": "${input.count}", "got_meta": "${input.meta}"},
    )
    executor = PipelineExecutor(ToolRegistry())
    tool = PipelineToolFactory.to_tool(pipeline, executor)

    schema_fields = cast(type[BaseModel], tool.args_schema).model_fields
    assert schema_fields["count"].annotation is int
    assert schema_fields["meta"].annotation is dict

    result = tool.invoke({"count": 42, "meta": {"k": "v"}})
    assert result == {"got_count": 42, "got_meta": {"k": "v"}}


def test_to_tool_default_str_for_unknown_type():
    pipeline = Pipeline(
        name="unknown_type",
        description="Falls back to str for unknown types",
        input_schema={
            "value": {"type": "unknown_kind", "description": "?"},
        },
        steps=[],
        output_template={"v": "${input.value}"},
    )
    executor = PipelineExecutor(ToolRegistry())
    tool = PipelineToolFactory.to_tool(pipeline, executor)
    assert cast(type[BaseModel], tool.args_schema).model_fields["value"].annotation is str
