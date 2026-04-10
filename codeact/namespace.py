"""ToolNamespace — wraps BaseTool list as callable Python functions.

Used by CodeAct: the sandbox needs each tool to look like a regular Python
function so the LLM can call it from generated code (e.g. `result = adder(a=1, b=2)`).
"""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel


class ToolNamespace:
    """Maps BaseTool instances to Python-callable wrappers."""

    @staticmethod
    def build(tools: Sequence[BaseTool]) -> dict[str, Callable[..., Any]]:
        """Build a {name: callable} dict suitable for use as exec globals."""
        return {tool.name: ToolNamespace._wrap_tool(tool) for tool in tools}

    @staticmethod
    def _wrap_tool(tool: BaseTool) -> Callable[..., Any]:
        def wrapper(**kwargs: Any) -> Any:
            return tool.invoke(kwargs)
        wrapper.__name__ = tool.name
        wrapper.__doc__ = tool.description
        return wrapper

    @staticmethod
    def render_signatures(tools: Sequence[BaseTool]) -> str:
        """Render tool signatures as Python function stubs for LLM prompts.

        Output is a multi-line string like:

            def adder(a: int, b: int) -> Any:
                '''Add two integers'''
                ...

            def greeter(name: str) -> Any:
                '''Greet someone'''
                ...
        """
        blocks: list[str] = []
        for tool in tools:
            params = ToolNamespace._render_params(tool)
            doc = (tool.description or "").strip().replace("'''", "\\'\\'\\'")
            blocks.append(
                f"def {tool.name}({params}) -> Any:\n"
                f"    '''{doc}'''\n"
                f"    ..."
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _render_params(tool: BaseTool) -> str:
        schema = tool.args_schema
        if schema is None or not isinstance(schema, type) or not issubclass(schema, BaseModel):
            return ""
        parts: list[str] = []
        for name, field in schema.model_fields.items():
            type_name = ToolNamespace._type_name(field.annotation)
            parts.append(f"{name}: {type_name}")
        return ", ".join(parts)

    @staticmethod
    def _type_name(annotation: Any) -> str:
        if annotation is None:
            return "Any"
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        return str(annotation)
