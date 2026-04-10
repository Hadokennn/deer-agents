"""CodeAct Executor — generative tool orchestration (Mode 2).

Public API:
    ExecutionResult — execution outcome dataclass
    CodeExecutionSandbox — restricted Python execution
    CodeActError, SandboxTimeoutError, RestrictedImportError — exceptions
"""

from codeact.errors import (
    CodeActError,
    RestrictedImportError,
    SandboxTimeoutError,
)
from codeact.sandbox import CodeExecutionSandbox, ExecutionResult

__all__ = [
    "ExecutionResult",
    "CodeExecutionSandbox",
    "CodeActError",
    "SandboxTimeoutError",
    "RestrictedImportError",
]
