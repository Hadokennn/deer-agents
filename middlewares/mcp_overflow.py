"""MCP Overflow Middleware — write oversized tool responses to sandbox files."""

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


class McpOverflowMiddleware(AgentMiddleware[AgentState]):
    """Intercept oversized tool responses and write them to sandbox files.

    When a tool response exceeds max_response_size bytes, the content is
    written to a file in sandbox_path and the ToolMessage content is replaced
    with a pointer instructing the agent to use read_file.
    """

    def __init__(self, max_response_size: int = 8192, sandbox_path: str = "/tmp/mcp_responses/"):
        self.max_response_size = max_response_size
        self.sandbox_path = sandbox_path
        Path(self.sandbox_path).mkdir(parents=True, exist_ok=True)

    def process_tool_response(self, content: Any, tool_call_id: str) -> Any:
        """Core logic: check size and replace if needed. Returns new content."""
        if not isinstance(content, str):
            return content
        if len(content) <= self.max_response_size:
            return content

        # Write to file
        file_path = Path(self.sandbox_path) / f"{tool_call_id}.txt"
        file_path.write_text(content)

        size_kb = len(content) // 1024
        logger.info("MCP response overflow: %dKB written to %s", size_kb, file_path)

        return (
            f"Response too large ({size_kb}KB), saved to {file_path}\n"
            f"Use read_file tool to inspect specific sections."
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = handler(request)
        if isinstance(result, ToolMessage):
            result.content = self.process_tool_response(
                result.content,
                tool_call_id=str(request.tool_call.get("id", "unknown")),
            )
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)
        if isinstance(result, ToolMessage):
            result.content = self.process_tool_response(
                result.content,
                tool_call_id=str(request.tool_call.get("id", "unknown")),
            )
        return result
