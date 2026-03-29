"""Code Index Middleware — enrich code search with symbol index.

When the LLM calls bash(grep/find) or read_file on a code_repo,
intercept and first consult the symbol index to narrow the search space.

Two modes:
1. Search enrichment: bash grep → prepend index results as structured context
2. Targeted read: read_file on a large file → append relevant symbol locations
"""

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


class CodeIndexMiddleware(AgentMiddleware[AgentState]):
    """Enrich code search tool calls with symbol index data.

    Intercepts grep/search-like bash commands targeting code_repos,
    prepends symbol index results so LLM gets precise locations first.
    """

    def __init__(self, index_dir: str = "", code_repos: list[dict] | None = None):
        self._indexes = {}  # repo_name → RepoIndex (lazy loaded)
        self._index_dir = index_dir
        self._code_repos = code_repos or []
        self._repo_paths = {r["path"]: r["name"] for r in self._code_repos}

    def _get_index(self, repo_name: str):
        """Lazy load index for a repo."""
        if repo_name in self._indexes:
            return self._indexes[repo_name]

        from scripts.index_repo import load_index
        index = load_index(repo_name)
        self._indexes[repo_name] = index
        return index

    def _extract_search_query(self, tool_call: dict) -> tuple[str, str] | None:
        """Extract search keyword and target repo from a bash grep/find command.

        Returns (query, repo_name) or None if not a code search.
        """
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})

        if name == "bash":
            command = args.get("command", "")
            # Match: grep "keyword" /path/to/repo or rg "keyword" /path/to/repo
            for repo_path, repo_name in self._repo_paths.items():
                if repo_path in command:
                    # Extract grep pattern
                    m = re.search(r'(?:grep|rg|ag)\s+(?:-[^\s]*\s+)*["\']?([^"\']+)["\']?', command)
                    if m:
                        return m.group(1).strip(), repo_name
                    # Extract find -name pattern
                    m = re.search(r'find\s+.*-name\s+["\']?([^"\']+)["\']?', command)
                    if m:
                        return m.group(1).strip().replace("*", ""), repo_name

        return None

    def _format_index_results(self, results: list[dict], query: str) -> str:
        """Format index search results as concise context."""
        if not results:
            return ""

        lines = [f"[Symbol index: {len(results)} matches for \"{query}\"]"]
        for r in results[:10]:
            exp = "export " if r["exported"] else ""
            lines.append(f"  {exp}{r['kind']} {r['name']}  →  {r['file']}:{r['line']} ({r['span']} lines)")

        return "\n".join(lines)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        # Check if this is a code search
        search_info = self._extract_search_query(request.tool_call)

        if search_info:
            query, repo_name = search_info
            index = self._get_index(repo_name)

            if index:
                from scripts.index_repo import search_index
                results = search_index(index, query, limit=10)

                if results:
                    index_context = self._format_index_results(results, query)
                    logger.info("Code index enriched search: %s → %d symbols", query, len(results))

                    # Execute original tool
                    result = handler(request)

                    # Prepend index results to tool output
                    if isinstance(result, ToolMessage) and isinstance(result.content, str):
                        result.content = f"{index_context}\n\n---\n\n{result.content}"
                    return result

        # No enrichment, pass through
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        search_info = self._extract_search_query(request.tool_call)

        if search_info:
            query, repo_name = search_info
            index = self._get_index(repo_name)

            if index:
                from scripts.index_repo import search_index
                results = search_index(index, query, limit=10)

                if results:
                    index_context = self._format_index_results(results, query)
                    result = await handler(request)
                    if isinstance(result, ToolMessage) and isinstance(result.content, str):
                        result.content = f"{index_context}\n\n---\n\n{result.content}"
                    return result

        return await handler(request)
