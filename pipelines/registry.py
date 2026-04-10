"""Tool registry: name → BaseTool lookup."""

from collections.abc import Iterable

from langchain_core.tools import BaseTool


class ToolRegistry:
    """Maps tool names to BaseTool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def register_many(self, tools: Iterable[BaseTool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    @classmethod
    def from_tools(cls, tools: Iterable[BaseTool]) -> "ToolRegistry":
        registry = cls()
        registry.register_many(tools)
        return registry
