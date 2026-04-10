"""Custom exceptions for the Pipeline Engine."""


class PipelineError(Exception):
    """Base class for all Pipeline Engine errors."""


class PipelineDefinitionError(PipelineError):
    """Raised when a pipeline YAML is malformed or missing required fields."""


class PipelineStepError(PipelineError):
    """Raised when a pipeline step fails at execution time."""
