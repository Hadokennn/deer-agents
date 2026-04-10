"""Custom exceptions for the CodeAct Executor."""


class CodeActError(Exception):
    """Base class for all CodeAct Executor errors."""


class SandboxTimeoutError(CodeActError):
    """Raised when sandbox execution exceeds the configured timeout."""


class RestrictedImportError(CodeActError):
    """Raised when sandboxed code tries to import a non-whitelisted module."""
