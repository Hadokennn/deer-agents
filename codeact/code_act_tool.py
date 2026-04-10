"""make_code_act_tool — wraps sandbox + namespace as a LangChain StructuredTool."""

from collections.abc import Sequence

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from codeact.namespace import ToolNamespace
from codeact.prompt import CODE_ACT_DESCRIPTION_TEMPLATE
from codeact.sandbox import CodeExecutionSandbox


class CodeActInput(BaseModel):
    code: str = Field(
        ...,
        description=(
            "Python code to execute. Set `result` to mark the return value. "
            "Use print() for diagnostic output."
        ),
    )


def make_code_act_tool(
    available_tools: Sequence[BaseTool],
    sandbox: CodeExecutionSandbox | None = None,
    name: str = "code_execute",
    description: str | None = None,
) -> StructuredTool:
    """Build a LangChain StructuredTool that runs LLM-generated code in the sandbox."""
    sandbox = sandbox or CodeExecutionSandbox()
    namespace = ToolNamespace.build(available_tools)

    final_description = description or CODE_ACT_DESCRIPTION_TEMPLATE.format(
        timeout=sandbox.timeout_seconds,
        tool_signatures=ToolNamespace.render_signatures(available_tools) or "(no tools available)",
    )

    def _run(code: str) -> dict:
        result = sandbox.execute(code, namespace)
        return {
            "success": result.success,
            "stdout": result.stdout,
            "return_value": result.return_value,
            "exception": result.exception,
            "traceback": result.traceback,
        }

    return StructuredTool.from_function(
        func=_run,
        name=name,
        description=final_description,
        args_schema=CodeActInput,
    )
