"""PipelineExecutor: runs a Pipeline step by step."""

import logging
from typing import Any

from pipelines.errors import PipelineStepError
from pipelines.parser import Pipeline, PipelineStep
from pipelines.registry import ToolRegistry
from pipelines.resolver import VariableResolver

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes pipelines against a tool registry."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, pipeline: Pipeline, inputs: dict[str, Any]) -> dict[str, Any]:
        context: dict[str, Any] = {"input": dict(inputs)}
        for step in pipeline.steps:
            self._run_step(step, context)
        return VariableResolver(context).resolve(pipeline.output_template)

    def _run_step(self, step: PipelineStep, context: dict[str, Any]) -> None:
        resolver = VariableResolver(context)

        if not self._condition_passes(step, resolver):
            logger.debug("step '%s' skipped by condition", step.id)
            context[step.id] = None
            return

        try:
            tool_input = resolver.resolve(step.input)
        except Exception as e:
            if step.optional:
                logger.debug("step '%s' input resolution failed (optional): %s", step.id, e)
                context[step.id] = None
                return
            raise PipelineStepError(
                f"step '{step.id}' input resolution failed: {e}"
            ) from e

        tool = self.registry.get(step.tool)
        if tool is None:
            if step.optional:
                logger.debug("step '%s' tool '%s' missing (optional)", step.id, step.tool)
                context[step.id] = None
                return
            raise PipelineStepError(
                f"step '{step.id}': tool '{step.tool}' not found in registry"
            )

        try:
            result = tool.invoke(tool_input)
        except Exception as e:
            if step.optional:
                logger.debug("step '%s' tool failed (optional): %s", step.id, e)
                context[step.id] = None
                return
            raise PipelineStepError(f"step '{step.id}' failed: {e}") from e

        context[step.id] = result
        logger.debug("step '%s' completed", step.id)

    @staticmethod
    def _condition_passes(step: PipelineStep, resolver: VariableResolver) -> bool:
        if step.when is not None:
            if not resolver.resolve(step.when):
                return False
        if step.unless is not None:
            if resolver.resolve(step.unless):
                return False
        return True
