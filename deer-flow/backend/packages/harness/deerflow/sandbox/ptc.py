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
import io
import threading
import traceback
from typing import Any

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_OUTPUT = 20000


# ---------- Safety primitives ----------


def _restricted_import(
    name: str,
    globals=None,
    locals=None,
    fromlist=(),
    level: int = 0,
):
    """Whitelisted __import__ for PTC code.

    Only modules in `_safe_modules()` can be imported. Everything else
    raises ImportError with a clear, actionable message pointing the LLM
    at the pre-imported modules.

    This lets LLM code use either style interchangeably:
    - `import json` → works, `json` already in namespace but import is idempotent
    - `from json import dumps` → works, pulls attribute from the returned module
    - `import os` → ImportError with helpful message
    """
    safe = _safe_modules()
    # Accept top-level names only (e.g. "json"); reject submodule imports
    # like "json.decoder" since we don't pre-import those.
    root = name.split(".", 1)[0]
    if root in safe:
        return safe[root]
    raise ImportError(
        f"Cannot import '{name}' in PTC code execution. "
        f"Pre-imported and directly usable (no import needed): "
        f"{', '.join(sorted(safe))}. "
        f"Filesystem / network / process modules are not available."
    )


def _restricted_builtins() -> dict:
    """Return a mapping of safe Python built-ins for the exec namespace.

    Excludes dangerous functions like eval, exec, open, which would let
    LLM-generated code escape the sandbox restrictions.

    `__import__` is provided as a whitelisted wrapper (`_restricted_import`)
    so `import json` and friends work for pre-imported safe modules; any
    other import raises a clear ImportError.
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
        "AttributeError", "RuntimeError", "Exception", "ImportError",
    }
    result = {k: getattr(builtins, k) for k in allowed if hasattr(builtins, k)}
    # Whitelisted __import__ so `import json` works for pre-imported safe modules
    result["__import__"] = _restricted_import
    return result


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
    resolved_tools: list[tuple[Any, dict | None]],
    runtime: Any,
    timeout: int = _DEFAULT_TIMEOUT,
    max_output_chars: int = _DEFAULT_MAX_OUTPUT,
) -> str:
    """Execute LLM-generated Python code in a restricted namespace.

    Thread-safe: each invocation builds fresh wrappers closing over the
    call's runtime, and captures stdout via a per-call StringIO injected
    as a custom `print` function (not via process-global redirect_stdout).

    Args:
        code: Python source to execute.
        resolved_tools: List of (tool, schema) pairs resolved from PTCToolConfig.
        runtime: The ToolRuntime (duck-typed) to bind into tool wrappers.
        timeout: Max wall-clock seconds. Enforced via daemon-thread join.
        max_output_chars: Max chars of captured stdout returned.

    Returns:
        Captured stdout (or error / timeout message).
    """
    # Build per-call wrappers with runtime baked into each closure
    tool_wrappers = _build_tool_wrappers(resolved_tools, runtime)

    # Per-call stdout buffer (thread-safe, isolated from sys.stdout).
    # We do NOT use contextlib.redirect_stdout because it swaps sys.stdout
    # globally — in a threaded context another thread's print() would leak
    # into our StringIO. Instead we inject a custom print into the namespace.
    stdout = io.StringIO()

    def _captured_print(*args, **kwargs):
        # Prevent the namespace code from overriding `file=`
        kwargs.pop("file", None)
        import builtins as _builtins
        _builtins.print(*args, file=stdout, **kwargs)

    safe_builtins = _restricted_builtins()
    safe_builtins["print"] = _captured_print

    namespace: dict[str, Any] = {
        **tool_wrappers,
        **_safe_modules(),
        "__builtins__": safe_builtins,
    }

    # Run exec in a daemon thread with a join-based timeout.
    # We use threading instead of signal.SIGALRM because SIGALRM only works
    # in the main thread of the main interpreter. LangGraph runs tool calls
    # from worker threads, which caused:
    #   ValueError: signal only works in main thread of the main interpreter
    # (production trace 417a5afd step 17)
    #
    # Limitation: Python has no safe way to forcibly terminate a thread.
    # If the thread is still alive after join(timeout), we return a timeout
    # error immediately but the runaway thread continues until it finishes
    # or the process exits. daemon=True ensures it won't block process exit.
    error_box: dict[str, Any] = {"type": None, "msg": None, "tb": None}

    def _run():
        try:
            exec(code, namespace)
        except Exception as e:
            error_box["type"] = type(e).__name__
            error_box["msg"] = str(e)
            error_box["tb"] = traceback.format_exc()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        return f"Error: Code execution exceeded {timeout}s limit"

    if error_box["type"] is not None:
        return f"Code execution error: {error_box['type']}: {error_box['msg']}\n{error_box['tb']}"

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
    """Extract structured content from a tool result for ergonomic LLM access.

    MCP tools via langchain-mcp-adapters return a `(content, artifact)`
    tuple (StructuredTool with response_format="content_and_artifact").

    Three extraction paths, in priority order:

    1. **outputSchema path** (forward-compat): if the artifact has
       `structured_content` (because the MCP server declared outputSchema),
       return it directly. **Currently never fires** because adapter 0.2.2
       doesn't expose outputSchema, but kept for future versions.

    2. **JSON text auto-parse** (common case): MCP tools that don't declare
       outputSchema typically return a SINGLE TextContent block whose text
       is a JSON string. We auto-parse it so the LLM gets the structured
       data directly instead of a confusing list-of-dict-of-string-of-JSON.
       If the text isn't valid JSON, return it as a plain string.

    3. **Fallback**: multi-block content or other shapes are returned as-is.

    For non-tuple results (sandbox tool strings, plain dicts/lists), return
    the value unchanged.

    Args:
        tool_result: Raw return value from a tool invocation.

    Returns:
        Structured content (dict/list/str) when extractable, else the
        original value unchanged.
    """
    if isinstance(tool_result, tuple) and len(tool_result) == 2:
        content, artifact = tool_result

        # Path 1: outputSchema-declared structured content
        if isinstance(artifact, dict):
            structured = artifact.get("structured_content")
            if structured is not None:
                return structured

        # Path 2: single TextContent block — auto-parse JSON or return text
        if isinstance(content, list) and len(content) == 1:
            block = content[0]
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                try:
                    import json as _json
                    return _json.loads(text)
                except (ValueError, Exception):
                    # Not JSON — return the raw text string
                    return text

        # Path 3: multi-block or non-text content — return content list as-is
        return content
    return tool_result


# ---------- Tool invocation glue ----------


def _invoke_tool_with_runtime(tool: Any, kwargs: dict, runtime: Any) -> Any:
    """Invoke a LangChain @tool-decorated function with runtime injected if declared.

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

    Runtime injection policy:
    - Sandbox tools (bash, read_file, etc.) declare `runtime: ToolRuntime[...]`
      in their args_schema — runtime IS passed.
    - MCP tools (via langchain-mcp-adapters) don't declare `runtime` in their
      args_schema — runtime is NOT passed to avoid TypeError.

    Args:
        tool: A LangChain BaseTool instance.
        kwargs: Business parameters to pass to the tool (no "runtime" key).
        runtime: A duck-typed runtime object.

    Returns:
        Whatever the tool's underlying function returns (str for sandbox
        tools, (content, artifact) tuple for MCP tools via adapter).
    """
    # Check args_schema.model_fields to determine if this tool declares runtime.
    # This is more reliable than inspect.signature because:
    # - MCP tools via langchain-mcp-adapters may have **kwargs in their func
    #   signature (which inspect would see as accepting runtime), but the schema
    #   doesn't declare runtime — passing it would cause unexpected behavior.
    # - Sandbox tools explicitly declare runtime in their Pydantic schema.
    declares_runtime = (
        tool.args_schema is not None
        and hasattr(tool.args_schema, "model_fields")
        and "runtime" in tool.args_schema.model_fields
    )

    func = getattr(tool, "func", None)
    if func is not None:
        if declares_runtime:
            return func(runtime=runtime, **kwargs)
        return func(**kwargs)
    return tool.invoke(kwargs)


# ---------- Composition layer ----------


_SKIP_PARAMS = {"runtime", "description"}


def _safe_python_name(name: str) -> str:
    """Convert a tool name to a valid Python identifier.

    Replaces dashes and dots with underscores so MCP tool names like
    'bytedance-mcp-ace_ai_ace_ai_locate_template' become callable
    Python function names in the PTC exec namespace.
    """
    return name.replace("-", "_").replace(".", "_")


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
        schema = tool.args_schema
        if schema is not None:
            if hasattr(schema, "model_fields"):
                for name, field in schema.model_fields.items():
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
            elif isinstance(schema, dict):
                _JSON_TYPE_MAP = {"string": "str", "integer": "int", "number": "float",
                                  "boolean": "bool", "array": "list", "object": "dict"}
                required = set(schema.get("required", []))
                for name, prop in schema.get("properties", {}).items():
                    if name in _SKIP_PARAMS:
                        continue
                    type_name = _JSON_TYPE_MAP.get(prop.get("type", ""), "Any")
                    default = "" if name in required else " = None"
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

        safe_name = _safe_python_name(tool.name)
        sig = f"{safe_name}({', '.join(params)}) -> {return_type}"
        desc = tool.description.split("\n")[0]
        docs.append(f"- {sig}\n  {desc}{schema_hint}")

    return (
        "Available functions (full input schemas are in the tool list above):\n\n"
        + "\n\n".join(docs)
    )


def _build_tool_wrappers(
    resolved_tools: list[tuple[Any, dict | None]],
    runtime: Any,
) -> dict[str, Any]:
    """Build name → callable mapping for a SINGLE PTC invocation.

    Each wrapper closes over `runtime` at creation time, so concurrent
    invocations with different runtimes get isolated wrapper sets.

    Each wrapper:
    - Hides `runtime` and `description` parameters from the LLM.
    - Auto-injects `description="called from ptc tool"` if the tool accepts it.
    - Invokes via `_invoke_tool_with_runtime` with `runtime` captured in closure.
    - Runs the result through `_extract_structured_content` to unwrap MCP tuples.

    Args:
        resolved_tools: List of (tool, schema) pairs from PTCToolConfig resolution.
        runtime: The ToolRuntime for this specific invocation. Captured in each
                 wrapper closure — NOT set as a mutable attribute.
    """
    wrappers: dict[str, Any] = {}

    for tool, _schema in resolved_tools:
        schema = tool.args_schema
        if schema is not None and hasattr(schema, "model_fields"):
            accepts_description = "description" in schema.model_fields
        elif isinstance(schema, dict):
            accepts_description = "description" in schema.get("properties", {})
        else:
            accepts_description = False

        def _make_wrapper(tool_ref=tool, needs_desc=accepts_description):
            # `runtime` is captured from the enclosing scope (single value per
            # _build_tool_wrappers call), so each call to make_ptc_tool gets a
            # fresh, isolated set of wrappers with no shared mutable state.
            def wrapper(**kwargs):
                if needs_desc and "description" not in kwargs:
                    kwargs["description"] = "called from ptc tool"
                result = _invoke_tool_with_runtime(tool_ref, kwargs, runtime)
                return _extract_structured_content(result)

            return wrapper

        wrappers[_safe_python_name(tool.name)] = _make_wrapper()

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

    Thread safety: wrappers are built fresh per invocation inside _execute_code,
    so concurrent calls with different runtimes don't share wrapper state.

    Args:
        ptc_config: A PTCToolConfig declaring name, purpose, and overrides.
        resolved_tools: List of (tool, output_schema_or_None) pairs resolved
                        from ptc_config.eligible_tools.

    Returns:
        A LangChain BaseTool ready to be appended to the agent's tool list.
    """
    func_docs = _build_function_docs(resolved_tools)
    # NOTE: _build_tool_wrappers is NOT called here at factory time.
    # It is called per-invocation inside _execute_code so that each call
    # gets its own wrapper set with runtime captured in the closure.
    # This prevents the concurrent-call race on shared wrapper._runtime state.

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

    # Local import to avoid circular dependency
    from langchain_core.tools import tool as lc_tool_decorator

    @lc_tool_decorator(ptc_config.name, parse_docstring=False)
    def ptc_tool(code: str, runtime: Any = None) -> str:
        """Execute Python code programmatically within a purpose-scoped PTC environment."""
        return _execute_code(
            code,
            resolved_tools=resolved_tools,
            runtime=runtime,
            timeout=timeout,
            max_output_chars=max_output,
        )

    # Override the auto-generated description with our dynamic purpose-prefixed one
    ptc_tool.description = description
    return ptc_tool
