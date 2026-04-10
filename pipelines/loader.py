"""Agent integration helper.

Convention: each agent has its pipelines under `agents/<name>/pipelines/*.yaml`.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from langchain_core.tools import BaseTool

from pipelines.executor import PipelineExecutor
from pipelines.parser import PipelineParser
from pipelines.registry import ToolRegistry
from pipelines.tool_factory import PipelineToolFactory

logger = logging.getLogger(__name__)

PIPELINE_SUBDIR = "pipelines"


def load_pipelines_for_agent(
    agent_dir: Path,
    base_tools: Sequence[BaseTool],
) -> list[BaseTool]:
    """Load all pipelines under <agent_dir>/pipelines/ and return base + pipeline tools."""
    pipeline_dir = Path(agent_dir) / PIPELINE_SUBDIR
    pipelines = PipelineParser.parse_dir(pipeline_dir)

    if not pipelines:
        return list(base_tools)

    registry = ToolRegistry.from_tools(base_tools)
    executor = PipelineExecutor(registry)

    pipeline_tools: list[BaseTool] = []
    for pipeline in pipelines:
        tool = PipelineToolFactory.to_tool(pipeline, executor)
        pipeline_tools.append(tool)
        registry.register(tool)
        logger.info("Loaded pipeline tool: %s", pipeline.name)

    return list(base_tools) + pipeline_tools
