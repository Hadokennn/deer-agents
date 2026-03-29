"""Programmatic Tool Processing Middleware.

Intercepts tool responses and processes them in the execution environment,
keeping intermediate data out of LLM context to reduce token consumption.

Two processing modes:
1. Structured extraction — for known tool types (web_search, web_fetch, log_search),
   extract key information and return a compact summary
2. Truncation fallback — for unknown tools with large responses,
   keep head + tail with size indicator
"""

import json
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


class ToolResponseProcessorMiddleware(AgentMiddleware[AgentState]):
    """Process tool responses in execution environment before returning to LLM.

    Instead of sending raw 50KB tool output to LLM context, process it locally:
    - Extract structured information (search results → title + snippet)
    - Trim redundant content (HTML boilerplate, repeated headers)
    - Keep only what LLM needs to make decisions
    - Save full response to sandbox for LLM to read_file if needed
    """

    def __init__(
        self,
        max_response_size: int = 4096,
        sandbox_path: str = "/tmp/mcp_responses/",
        extractors: dict[str, str] | None = None,
    ):
        self.max_response_size = max_response_size
        self.sandbox_path = sandbox_path
        Path(self.sandbox_path).mkdir(parents=True, exist_ok=True)

        # Tool name → extractor method name
        self._extractors = extractors or {}

    def _save_full(self, content: str, tool_call_id: str) -> str:
        """Save full content to sandbox, return path."""
        file_path = Path(self.sandbox_path) / f"{tool_call_id}.txt"
        file_path.write_text(content)
        return str(file_path)

    def _extract_search_results(self, content: str) -> str:
        """Extract key info from web_search results."""
        # Tavily/DuckDuckGo search results are typically formatted as:
        # multiple blocks with title, url, snippet
        lines = content.split("\n")
        results = []
        current = {}

        for line in lines:
            line = line.strip()
            if not line:
                if current:
                    results.append(current)
                    current = {}
                continue
            # Common patterns in search result formatting
            if line.startswith("Title:") or line.startswith("title:"):
                current["title"] = line.split(":", 1)[1].strip()
            elif line.startswith("URL:") or line.startswith("url:") or line.startswith("Source:"):
                current["url"] = line.split(":", 1)[1].strip()
            elif line.startswith("Content:") or line.startswith("content:") or line.startswith("Snippet:"):
                current["snippet"] = line.split(":", 1)[1].strip()[:200]
            elif not current:
                # First non-empty line might be a title
                current["title"] = line[:100]
            elif "snippet" not in current:
                current["snippet"] = line[:200]

        if current:
            results.append(current)

        if not results:
            # Fallback: couldn't parse structure, return truncated
            return None

        # Build compact summary
        parts = []
        for i, r in enumerate(results[:5], 1):  # Top 5 results
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            url = r.get("url", "")
            parts.append(f"{i}. {title}")
            if snippet:
                parts.append(f"   {snippet}")
            if url:
                parts.append(f"   {url}")

        return "\n".join(parts)

    def _extract_log_content(self, content: str) -> str:
        """Extract key patterns from log search results."""
        lines = content.split("\n")

        # Find error/warning lines
        key_lines = []
        for i, line in enumerate(lines):
            if re.search(r"(ERROR|FATAL|WARN|Exception|Traceback|panic)", line, re.IGNORECASE):
                # Include context: 1 line before, the match, 2 lines after
                start = max(0, i - 1)
                end = min(len(lines), i + 3)
                key_lines.extend(lines[start:end])
                key_lines.append("---")

        if key_lines:
            # Stats header
            error_count = sum(1 for l in lines if re.search(r"(ERROR|FATAL)", l, re.IGNORECASE))
            warn_count = sum(1 for l in lines if re.search(r"WARN", l, re.IGNORECASE))
            header = f"[{len(lines)} lines total, {error_count} errors, {warn_count} warnings]\n"
            return header + "\n".join(key_lines[:50])  # Cap at 50 key lines

        return None  # No patterns found, use fallback

    def _truncate_smart(self, content: str, tool_call_id: str) -> str:
        """Smart truncation: head + tail + full saved to sandbox."""
        full_path = self._save_full(content, tool_call_id)
        size_kb = len(content) // 1024

        # Keep first 2KB + last 1KB
        head = content[:2048]
        tail = content[-1024:]

        return (
            f"{head}\n"
            f"\n... [{size_kb}KB total, truncated. Full content: {full_path}] ...\n\n"
            f"{tail}"
        )

    def process_tool_response(self, content: Any, tool_call_id: str, tool_name: str = "") -> Any:
        """Process tool response in execution environment.

        Returns compact result for LLM context. Saves full data to sandbox.
        """
        if not isinstance(content, str):
            return content
        if len(content) <= self.max_response_size:
            return content

        original_size = len(content)
        processed = None

        # Try structured extraction based on tool name
        # Order matters: check more specific patterns first
        if "log" in tool_name.lower():
            processed = self._extract_log_content(content)
        elif "search" in tool_name.lower():
            processed = self._extract_search_results(content)

        # Custom extractor from config
        extractor_name = self._extractors.get(tool_name)
        if extractor_name and hasattr(self, extractor_name):
            processed = getattr(self, extractor_name)(content)

        if processed and len(processed) <= self.max_response_size:
            # Save full for reference, return processed
            full_path = self._save_full(content, tool_call_id)
            processed_size = len(processed)
            ratio = f"{processed_size}/{original_size} chars ({processed_size * 100 // original_size}%)"
            logger.info("Tool response processed: %s %s → %s", tool_name, original_size, ratio)
            return f"{processed}\n\n[Processed {ratio}. Full: {full_path}]"

        # Fallback: smart truncation
        logger.info("Tool response truncated: %s %d chars", tool_name, original_size)
        return self._truncate_smart(content, tool_call_id)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = handler(request)
        if isinstance(result, ToolMessage):
            tool_name = request.tool_call.get("name", "")
            result.content = self.process_tool_response(
                result.content,
                tool_call_id=str(request.tool_call.get("id", "unknown")),
                tool_name=tool_name,
            )
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)
        if isinstance(result, ToolMessage):
            tool_name = request.tool_call.get("name", "")
            result.content = self.process_tool_response(
                result.content,
                tool_call_id=str(request.tool_call.get("id", "unknown")),
                tool_name=tool_name,
            )
        return result


# Keep backward compatibility alias
McpOverflowMiddleware = ToolResponseProcessorMiddleware
