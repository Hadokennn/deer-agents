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


# ---------- Composition layer ----------


_SKIP_PARAMS = {"runtime", "description"}


def _build_function_docs(resolved_tools: list[tuple[Any, dict | None]]) -> str:
    """Generate concise function-signature docs for a PTC tool's description.

    Each entry in `resolved_tools` is a `(tool, output_schema_or_None)` tuple.
    The generated docs list each tool as a Python function signature with
    business parameters only (runtime/description are hidden). Return types
    are marked as `dict | list` when a schema is present, else `Any`.

    Args:
        resolved_tools: Pairs of (tool, schema) as resolved from PTCToolConfig.

    Returns:
        Multi-line string for inclusion in the PTC tool's description.
    """
    if not resolved_tools:
        return "No tools are currently exposed to this PTC tool."

    import json as _json

    docs = []
    for tool, output_schema in resolved_tools:
        params = []
        if tool.args_schema is not None:
            for name, field in tool.args_schema.model_fields.items():
                if name in _SKIP_PARAMS:
                    continue
                annotation = field.annotation
                type_name = (
                    annotation.__name__
                    if hasattr(annotation, "__name__")
                    else str(annotation)
                )
                if field.default is not None and field.default is not ...:
                    default = f" = {field.default!r}"
                else:
                    default = ""
                params.append(f"{name}: {type_name}{default}")

        if output_schema is not None:
            return_type = "dict | list"
            schema_hint = (
                f"\n  Returns structured data. Schema: "
                f"{_json.dumps(output_schema, ensure_ascii=False)}"
            )
        else:
            return_type = "Any"
            schema_hint = "\n  Returns raw tool result (may be str, dict, or tuple)"

        sig = f"{tool.name}({', '.join(params)}) -> {return_type}"
        desc = tool.description.split("\n")[0]
        docs.append(f"- {sig}\n  {desc}{schema_hint}")

    return (
        "Available functions (full input schemas are in the tool list above):\n\n"
        + "\n\n".join(docs)
    )


def _build_tool_wrappers(resolved_tools: list[tuple[Any, dict | None]]) -> dict[str, Any]:
    """Build name → callable mapping for the PTC exec namespace.

    Each wrapper:
    - Hides `runtime` and `description` parameters from the LLM.
    - Auto-injects `description="called from ptc tool"` if the tool accepts it.
    - Invokes via `_invoke_tool_with_runtime` with the wrapper's `_runtime` slot.
    - Runs the result through `_extract_structured_content` to unwrap MCP tuples.

    `_runtime` must be set on each wrapper before the code runs (done in
    `_execute_code`).
    """
    wrappers: dict[str, Any] = {}

    for tool, _schema in resolved_tools:
        accepts_description = (
            tool.args_schema is not None
            and "description" in tool.args_schema.model_fields
        )

        def _make_wrapper(tool_ref=tool, needs_desc=accepts_description):
            def wrapper(**kwargs):
                if needs_desc and "description" not in kwargs:
                    kwargs["description"] = "called from ptc tool"
                result = _invoke_tool_with_runtime(tool_ref, kwargs, wrapper._runtime)
                return _extract_structured_content(result)

            wrapper._runtime = None
            return wrapper

        wrappers[tool.name] = _make_wrapper()

    return wrappers


# ---------- Factory ----------


def make_ptc_tool(
    ptc_config,  # PTCToolConfig — avoid import cycle
    resolved_tools: list[tuple[Any, dict | None]],
):
    """Factory: build a purpose-scoped PTC tool from config + resolved eligible tools.

    The returned tool is a LangChain `BaseTool` whose:
    - `name` matches `ptc_config.name`
    - `description` starts with `ptc_config.purpose` and lists all eligible
      functions as Python signatures
    - `func(code, runtime)` runs the code in a restricted namespace where each
      eligible tool is available as a callable

    Args:
        ptc_config: A PTCToolConfig declaring name, purpose, and overrides.
        resolved_tools: List of (tool, output_schema_or_None) pairs resolved
                        from ptc_config.eligible_tools.

    Returns:
        A LangChain BaseTool ready to be appended to the agent's tool list.
    """
    # Local import to avoid circular dependency
    from langchain_core.tools import tool as lc_tool_decorator

    func_docs = _build_function_docs(resolved_tools)
    tool_wrappers = _build_tool_wrappers(resolved_tools)

    timeout = ptc_config.timeout_seconds or _DEFAULT_TIMEOUT
    max_output = ptc_config.max_output_chars or _DEFAULT_MAX_OUTPUT

    description = (
        f"{ptc_config.purpose}\n\n"
        f"Write Python code that calls the available functions and processes "
        f"results. Only print() output is returned to your context — tool "
        f"return values stay inside the code scope.\n\n"
        f"{func_docs}\n\n"
        f"Pre-imported modules: json, re, math, collections, itertools, "
        f"functools, datetime\n"
        f"Use print() to output anything you want to see.\n\n"
        f"The `code` argument is the Python source to execute."
    )

    @lc_tool_decorator(ptc_config.name, parse_docstring=False)
    def ptc_tool(code: str, runtime: Any = None) -> str:
        """Execute Python code programmatically within a purpose-scoped PTC environment."""
        return _execute_code(
            code,
            tool_wrappers=tool_wrappers,
            runtime=runtime,
            timeout=timeout,
            max_output_chars=max_output,
        )

    # Override the auto-generated description with our dynamic purpose-prefixed one
    ptc_tool.description = description
    return ptc_tool
