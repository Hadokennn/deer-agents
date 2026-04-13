"""Purpose-scoped Programmatic Tool Calling (PTC).

Each PTC tool is a LangChain tool that accepts a `code: str` argument,
executes it in a restricted Python namespace where a pre-declared set of
"eligible" tools are available as callable functions, and returns the
printed output. The tool's return values stay inside the code scope;
only print() output enters the model context.

See: docs/superpowers/specs/2026-04-13-purpose-scoped-ptc-design.md
"""

from __future__ import annotations

import builtins
import contextlib
import io
import signal
import traceback
from typing import Any

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_OUTPUT = 20000


# ---------- Safety primitives ----------


def _restricted_builtins() -> dict:
    """Return a mapping of safe Python built-ins for the exec namespace.

    Excludes dangerous functions like __import__, eval, exec, open, which
    would let LLM-generated code escape the sandbox restrictions.
    """
    allowed = {
        # I/O
        "print", "repr", "format",
        # Iteration / collections
        "len", "range", "enumerate", "zip", "map", "filter",
        "sorted", "reversed", "min", "max", "sum", "any", "all",
        # Numeric
        "abs", "round", "divmod", "pow",
        # Type introspection
        "isinstance", "issubclass", "type", "callable", "hasattr", "getattr",
        # Type constructors
        "str", "int", "float", "bool", "list", "dict", "set", "tuple",
        "bytes", "frozenset", "complex",
        # Constants
        "None", "True", "False",
        # Characters
        "chr", "ord",
        # Exceptions (so try/except works inside the namespace)
        "StopIteration", "ValueError", "TypeError", "KeyError", "IndexError",
        "AttributeError", "RuntimeError", "Exception",
    }
    return {k: getattr(builtins, k) for k in allowed if hasattr(builtins, k)}


def _safe_modules() -> dict:
    """Return a mapping of safe stdlib modules pre-imported into the exec namespace.

    Covers data processing needs (JSON parsing, regex, math, iteration
    helpers) without exposing network, filesystem, or process control.
    """
    import collections
    import datetime
    import functools
    import itertools
    import json
    import math
    import re

    return {
        "json": json,
        "re": re,
        "math": math,
        "collections": collections,
        "itertools": itertools,
        "functools": functools,
        "datetime": datetime,
    }


# ---------- Core executor ----------


def _execute_code(
    code: str,
    tool_wrappers: dict[str, Any],
    runtime: Any,
    timeout: int = _DEFAULT_TIMEOUT,
    max_output_chars: int = _DEFAULT_MAX_OUTPUT,
) -> str:
    """Execute LLM-generated Python code in a restricted namespace.

    Args:
        code: Python source to execute.
        tool_wrappers: Mapping of function name → callable injected into namespace.
                       Each wrapper with a `_runtime` attribute gets it set to
                       `runtime` before execution.
        runtime: The ToolRuntime (duck-typed) injected into wrappers.
        timeout: Max wall-clock seconds.
        max_output_chars: Max chars of stdout returned; beyond this, output
                          is truncated with a marker.

    Returns:
        Captured stdout (or an error / timeout message string).
    """
    # Inject runtime into each wrapper's private slot
    for wrapper in tool_wrappers.values():
        if hasattr(wrapper, "_runtime"):
            wrapper._runtime = runtime

    namespace: dict[str, Any] = {
        **tool_wrappers,
        **_safe_modules(),
        "__builtins__": _restricted_builtins(),
    }

    stdout = io.StringIO()

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution exceeded {timeout}s limit")

    # SIGALRM is Unix-only. Windows would need threading-based timeout.
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, namespace)
    except TimeoutError as e:
        return f"Error: {e}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Code execution error: {type(e).__name__}: {e}\n{tb}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    output = stdout.getvalue() or "(no output)"

    if len(output) > max_output_chars:
        output = output[:max_output_chars] + f"\n... (truncated, {len(output)} total chars)"

    return output


# ---------- MCP compatibility helpers ----------


def _extract_output_schema(tool: Any) -> dict | None:
    """Extract MCP outputSchema from a LangChain BaseTool.

    **Current state (2026-04-13, langchain-mcp-adapters 0.2.2):**
    The adapter does NOT expose MCP outputSchema — source inspection found
    zero references to "outputSchema" in the adapter codebase. So this
    function reads `tool.metadata["outputSchema"]` but will always get None.

    Kept as a forward-compatible stub: when a future adapter version supports
    MCP 2025-06's outputSchema, only this function needs updating. Other PTC
    code paths already handle the "schema present" branch.

    Args:
        tool: A LangChain BaseTool instance.

    Returns:
        JSON Schema dict if the adapter ever starts exposing it, else None.
    """
    metadata = getattr(tool, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    schema = metadata.get("outputSchema")
    return schema if isinstance(schema, dict) else None


def _extract_structured_content(tool_result: Any) -> Any:
    """Extract structured content from a tool result.

    MCP tools via langchain-mcp-adapters return a `(content, artifact)`
    tuple (StructuredTool with response_format="content_and_artifact").
    The artifact is a dict with `structured_content` holding parsed data.

    Shape handling:
    - `(content, artifact_dict_with_structured_content)` → return structured_content
    - `(content, None)` or artifact without the key → return content
    - Any other tuple shape (wrong length) → pass through as-is
    - Non-tuple (str, dict, list, object) → pass through as-is

    Args:
        tool_result: Raw return value from a tool invocation.

    Returns:
        Structured content (dict/list) if available, else the original value.
    """
    if isinstance(tool_result, tuple) and len(tool_result) == 2:
        content, artifact = tool_result
        if isinstance(artifact, dict):
            structured = artifact.get("structured_content")
            if structured is not None:
                return structured
        return content
    return tool_result


# ---------- Tool invocation glue ----------


def _invoke_tool_with_runtime(tool: Any, kwargs: dict, runtime: Any) -> Any:
    """Invoke a LangChain @tool-decorated function with runtime injected.

    Uses the underlying `.func()` attribute to bypass pydantic ToolRuntime
    validation that `.invoke()` and `._run()` would perform. This lets PTC
    pass a duck-typed runtime (an object with `.state` and `.context`
    attributes) instead of constructing a full ToolRuntime with all 6
    required fields.

    Why `.func()` is the right call:
    - `.invoke()` / `._run()` validate `runtime` against ToolRuntime via pydantic
    - `.func()` is the underlying Python function — accepts any duck-typed object
    - Matches the pattern used in test_sandbox_tools_security.py with SimpleNamespace
    - For MCP tools patched via _make_sync_tool_wrapper, `.func` is the sync
      wrapper and accepts the same runtime kwarg

    Args:
        tool: A LangChain BaseTool instance.
        kwargs: Business parameters to pass to the tool (no "runtime" key).
        runtime: A duck-typed runtime object.

    Returns:
        Whatever the tool's underlying function returns (str for sandbox
        tools, (content, artifact) tuple for MCP tools via adapter).
    """
    return tool.func(runtime=runtime, **kwargs)
