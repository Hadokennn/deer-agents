"""PipelineToolFactory: wrap a Pipeline as a LangChain StructuredTool."""

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from pipelines.executor import PipelineExecutor
from pipelines.parser import Pipeline

_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "dict": dict,
    "list": list,
}


class PipelineToolFactory:
    """Builds LangChain StructuredTools from Pipeline definitions."""

    @staticmethod
    def to_tool(pipeline: Pipeline, executor: PipelineExecutor) -> StructuredTool:
        input_model = PipelineToolFactory._build_input_model(pipeline)

        def _run(**kwargs: Any) -> dict:
            return executor.execute(pipeline, kwargs)

        return StructuredTool.from_function(
            func=_run,
            name=pipeline.name,
            description=pipeline.description,
            args_schema=input_model,
        )

    @staticmethod
    def _build_input_model(pipeline: Pipeline) -> type[BaseModel]:
        fields: dict[str, Any] = {}
        for param_name, spec in pipeline.input_schema.items():
            type_name = (spec or {}).get("type", "str")
            py_type = _TYPE_MAP.get(type_name, str)
            description = (spec or {}).get("description", "")
            fields[param_name] = (py_type, Field(..., description=description))

        model_name = f"{pipeline.name}Input"
        return create_model(model_name, **fields)
