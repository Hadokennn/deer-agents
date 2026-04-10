"""CodeExecutionSandbox — restricted Python execution for CodeAct.

Threat model: LLM is a trusted but error-prone code author. The sandbox protects
against accidental misuse (writing to /etc/passwd, infinite loops, importing os),
not against truly adversarial code.

Mechanisms:
- Restricted __builtins__ (only safe functions)
- Custom __import__ enforcing a whitelist
- signal.SIGALRM-based timeout
- stdout capture via redirect_stdout
- Output size limit
- Convention: user code assigns to `result` to mark return value
"""

import builtins
import contextlib
import io
import signal
import sys
import traceback
from dataclasses import dataclass
from typing import Any

from codeact.errors import RestrictedImportError, SandboxTimeoutError

DEFAULT_ALLOWED_IMPORTS: frozenset[str] = frozenset({
    "json", "re", "math", "datetime", "collections",
    "itertools", "functools", "operator", "typing", "string",
})

_SAFE_BUILTIN_NAMES: tuple[str, ...] = (
    "print", "len", "range", "enumerate", "zip",
    "map", "filter", "sorted", "reversed",
    "list", "dict", "tuple", "set", "frozenset",
    "str", "int", "float", "bool", "bytes",
    "isinstance", "issubclass", "type", "id",
    "min", "max", "sum", "abs", "round", "pow", "divmod",
    "any", "all", "iter", "next",
    "getattr", "hasattr", "setattr", "delattr",
    "repr", "format", "ord", "chr", "hex", "oct", "bin",
    "True", "False", "None",
    "Exception", "ValueError", "TypeError", "KeyError",
    "IndexError", "RuntimeError", "AttributeError", "ZeroDivisionError",
    "ArithmeticError", "LookupError", "StopIteration",
)


@dataclass
class ExecutionResult:
    """Captures everything from a sandbox execution."""

    stdout: str = ""
    return_value: Any = None
    exception: str | None = None
    traceback: str | None = None

    @property
    def success(self) -> bool:
        return self.exception is None


class CodeExecutionSandbox:
    """Runs Python code in a restricted environment."""

    def __init__(
        self,
        allowed_imports: frozenset[str] | set[str] | None = None,
        timeout_seconds: float = 10.0,
        max_stdout_bytes: int = 50_000,
    ):
        self.allowed_imports = frozenset(allowed_imports) if allowed_imports else DEFAULT_ALLOWED_IMPORTS
        self.timeout_seconds = timeout_seconds
        self.max_stdout_bytes = max_stdout_bytes

    def execute(self, code: str, namespace: dict[str, Any]) -> ExecutionResult:
        """Run code in the sandbox. Returns ExecutionResult (never raises)."""
        result = ExecutionResult()
        restricted_globals = {
            **namespace,
            "__builtins__": self._build_restricted_builtins(),
        }
        stdout_buffer = io.StringIO()

        try:
            with self._timeout_context():
                with contextlib.redirect_stdout(stdout_buffer):
                    exec(code, restricted_globals)
        except SandboxTimeoutError as e:
            result.exception = f"SandboxTimeoutError: {e}"
            result.traceback = traceback.format_exc()
        except Exception as e:
            result.exception = f"{type(e).__name__}: {e}"
            result.traceback = traceback.format_exc()

        captured = stdout_buffer.getvalue()
        result.stdout = captured[: self.max_stdout_bytes]
        if "result" in restricted_globals:
            result.return_value = restricted_globals["result"]
        return result

    def _build_restricted_builtins(self) -> dict[str, Any]:
        safe = {
            name: getattr(builtins, name)
            for name in _SAFE_BUILTIN_NAMES
            if hasattr(builtins, name)
        }
        safe["__import__"] = self._make_safe_import()
        return safe

    def _make_safe_import(self):
        allowed = self.allowed_imports

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            base = name.split(".")[0]
            if base not in allowed:
                raise RestrictedImportError(
                    f"import of '{name}' not allowed in sandbox "
                    f"(allowed: {sorted(allowed)})"
                )
            return __import__(name, globals, locals, fromlist, level)
        return safe_import

    def _timeout_context(self):
        if sys.platform == "win32" or self.timeout_seconds <= 0:
            return contextlib.nullcontext()
        return _SignalAlarmContext(self.timeout_seconds)


class _SignalAlarmContext:
    """Context manager that raises SandboxTimeoutError after N seconds."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._previous_handler = None

    def __enter__(self):
        self._previous_handler = signal.signal(signal.SIGALRM, self._handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self._previous_handler)
        return False

    def _handler(self, signum, frame):
        raise SandboxTimeoutError(
            f"sandbox execution exceeded {self.seconds}s"
        )
