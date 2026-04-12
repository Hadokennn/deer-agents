# Programmatic Tool Calling (PTC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 deer-flow 中新增 `code_execution` tool，允许 LLM 生成 Python 代码批量调用其他 tools，代码中的 tool 返回值不进入 model context，只有 `print()` 输出返回，显著降低批量操作的 token 消耗。

**Architecture:** 在 deer-flow 的 tool 加载层（`get_available_tools()`）末尾动态构建 `code_execution_tool`。该 tool 接受 Python 代码作为输入，在受限 namespace 中 `exec()` 执行，namespace 注入其他 ptc_eligible tools 的 wrapper 函数。兼容 MCP 2025-06-18 的 `outputSchema` / `structuredContent` 规范。

**Tech Stack:** Python 3.12, LangChain, LangGraph, pydantic, pytest, MCP 协议

**Spec:** `docs/superpowers/specs/2026-04-12-programmatic-tool-calling-design.md`

**Phase 1 Scope:**
- PTC 支持 `config.yaml` 中 `ptc_eligible: true` 的 tools（sandbox + 其他配置 tools）
- MCP tools：通过 `ptc.include_mcp: true` 全量开启（不支持单个挑选）
- 当 PTC 对 MCP 生效时，自动取消 tool_search 的 deferred（保证 LLM 能看到 input schema）
- MCP outputSchema 和 structuredContent 兼容处理
- 执行环境：in-process `exec()` + 受限 namespace + 超时保护

**Out of Scope（Phase 2+）：**
- 单个 MCP tool 粒度开启 PTC
- Wrapper 侧 schema 自动发现缓存
- Windows 超时支持（当前仅 Unix `SIGALRM`）

---

## File Structure

| 文件 | 动作 | 职责 |
|------|------|------|
| `deer-flow/backend/packages/harness/deerflow/config/tool_config.py` | 修改 | 给 `ToolConfig` 加 `ptc_eligible` 字段 |
| `deer-flow/backend/packages/harness/deerflow/config/app_config.py` | 修改 | 给 `AppConfig` 加 `ptc` section（`enabled`, `include_mcp`, `timeout_seconds`, `max_output_chars`） |
| `deer-flow/backend/packages/harness/deerflow/config/ptc_config.py` | 创建 | 新建 `PtcConfig` pydantic 模型 |
| `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py` | 创建 | 所有 PTC 逻辑：工厂、wrappers、exec、schema 提取。独立文件避免膨胀 `sandbox/tools.py` |
| `deer-flow/backend/packages/harness/deerflow/tools/tools.py` | 修改 | `get_available_tools()` 末尾集成 code_execution 构建 |
| `deer-flow/config.example.yaml` | 修改 | 示例配置：给 sandbox tools 加 `ptc_eligible: true`，新增 `ptc` section |
| `deer-flow/backend/tests/test_code_execution_tool.py` | 创建 | PTC 所有单元测试 + 集成测试 |
| `deer-flow/backend/tests/test_ptc_integration.py` | 创建 | 端到端测试（与 `get_available_tools` 集成） |
| `deer-flow/backend/CLAUDE.md` | 修改 | 文档更新：说明 PTC 机制 |

**为什么新建 `sandbox/code_execution.py` 而不是放进 `sandbox/tools.py`？** `sandbox/tools.py` 已经 1300+ 行，承载了 bash/ls/read_file/write_file/glob/grep/str_replace 等多个 tool 实现和虚拟路径翻译逻辑。PTC 是一个独立的子系统（包含 10+ 个辅助函数），放进去会让文件更难维护。独立文件符合"文件过大就拆分"原则。

---

## Task 0: Investigate LangChain Tool Invocation Mechanism

**Goal：** 实现前确认 LangChain `@tool` 装饰器的底层调用接口，避免写代码时反复踩坑。

**Files:**
- Create (throwaway): `deer-flow/backend/tests/scratch_ptc_invoke.py`

- [ ] **Step 1: 写一个探针脚本**

Create `deer-flow/backend/tests/scratch_ptc_invoke.py`:

```python
"""Scratch: verify how to invoke a LangChain @tool directly with runtime."""

from deerflow.sandbox.tools import bash_tool, read_file_tool

# Check 1: tool.args_schema 的字段结构
print("=" * 60)
print("bash_tool.name:", bash_tool.name)
print("bash_tool.description (first 200):", bash_tool.description[:200])
print("bash_tool.args_schema:", bash_tool.args_schema)
if bash_tool.args_schema is not None:
    print("bash_tool.args_schema.model_fields:")
    for name, field in bash_tool.args_schema.model_fields.items():
        print(f"  {name}: annotation={field.annotation!r}, default={field.default!r}")

# Check 2: 可用的调用接口
print("\n" + "=" * 60)
print("bash_tool attributes related to invocation:")
for attr in ["func", "coroutine", "_run", "_arun", "invoke", "ainvoke", "run"]:
    val = getattr(bash_tool, attr, None)
    print(f"  {attr}: {type(val).__name__ if val is not None else None}")

# Check 3: 尝试 invoke()
print("\n" + "=" * 60)
print("Try bash_tool.invoke() with dummy runtime:")
try:
    # LangChain tools expect args as a dict
    result = bash_tool.invoke({"description": "probe", "command": "echo hello"})
    print(f"  invoke result: {result[:200]!r}")
except Exception as e:
    print(f"  invoke error: {type(e).__name__}: {e}")
```

- [ ] **Step 2: 执行探针脚本，记录结果**

Run: `cd deer-flow/backend && PYTHONPATH=packages/harness uv run python tests/scratch_ptc_invoke.py`

期待输出三块信息：
1. `args_schema.model_fields` 中每个字段的 annotation / default
2. `bash_tool` 上有哪些可用的调用方法（`func`, `coroutine`, `_run`, `invoke` 等）
3. `invoke({...})` 的调用是否成功、错误类型是什么

**关键问题：**
- `bash_tool.invoke({...})` 是否能在没有 runtime 的情况下工作？（runtime 由 LangChain 内部注入）
- 如果需要 runtime，接口是什么？（`config` 参数？显式传入？）
- `_run(**kwargs)` 的签名是否直接接受 runtime？

- [ ] **Step 3: 记录结论，删除脚本**

根据探针输出，确定后续任务中 `_invoke_tool_with_runtime()` 的实现策略。把关键发现写进 commit message 里：

```bash
git rm deer-flow/backend/tests/scratch_ptc_invoke.py
git commit -m "chore(ptc): investigate LangChain tool invocation mechanism

Findings:
- bash_tool.invoke({kwargs}) [success/failure: ...]
- Preferred call path: [_run / invoke / func]
- Runtime injection: [how]

This informs _invoke_tool_with_runtime() implementation in later tasks."
```

---

## Task 1: Investigate MCP outputSchema Exposure

**Goal：** 确认 `langchain-mcp-adapters` 把 MCP `outputSchema` 暴露在 LangChain `BaseTool` 的哪个属性上。

**Files:**
- Create (throwaway): `deer-flow/backend/tests/scratch_mcp_schema.py`

- [ ] **Step 1: 检查 langchain-mcp-adapters 版本和 Tool 结构**

```bash
cd deer-flow/backend && uv run pip show langchain-mcp-adapters
```

记录版本号，查看 [langchain-mcp-adapters GitHub](https://github.com/langchain-ai/langchain-mcp-adapters) 源码确认 outputSchema 暴露方式。

- [ ] **Step 2: 写探针脚本**

Create `deer-flow/backend/tests/scratch_mcp_schema.py`:

```python
"""Scratch: verify how langchain-mcp-adapters exposes MCP outputSchema."""

import asyncio
from deerflow.mcp.tools import get_mcp_tools

async def probe():
    tools = await get_mcp_tools()
    if not tools:
        print("No MCP tools loaded. Make sure an MCP server is configured in extensions_config.json")
        return

    for t in tools[:3]:  # probe first 3
        print("=" * 60)
        print(f"Tool name: {t.name}")
        print(f"Type: {type(t).__name__}")
        print(f"description (first 100): {t.description[:100]}")
        # Candidate locations for outputSchema
        for attr in ["metadata", "response_format", "output_schema",
                     "_tool", "tool_info", "args_schema"]:
            val = getattr(t, attr, None)
            print(f"  {attr}: {type(val).__name__ if val is not None else None}")
            if attr == "metadata" and isinstance(val, dict):
                print(f"    keys: {list(val.keys())}")
                if "outputSchema" in val:
                    print(f"    outputSchema: {val['outputSchema']}")

        # Check the tool result structure
        print("\n  Invoking tool with empty args to see error/structure...")
        try:
            result = await t.ainvoke({})
            print(f"  result type: {type(result).__name__}")
            print(f"  result (first 200): {str(result)[:200]}")
        except Exception as e:
            print(f"  error: {type(e).__name__}: {e}")

asyncio.run(probe())
```

- [ ] **Step 3: 执行探针**

Run: `cd deer-flow/backend && PYTHONPATH=packages/harness uv run python tests/scratch_mcp_schema.py`

**关键问题：**
- MCP Tool 的 `outputSchema` 暴露在哪个属性？（`metadata['outputSchema']`? `response_format`? 不暴露？）
- `tool.ainvoke()` 或 `tool.invoke()` 返回值的类型是什么？（字符串？dict？CallToolResult？）
- `structuredContent` 是否直接可访问？

- [ ] **Step 4: 记录结论，删除脚本**

如果发现 adapter 不暴露 `outputSchema`，在 commit message 中记录，plan 后续实现要处理"字段不存在"的 fallback。

```bash
git rm deer-flow/backend/tests/scratch_mcp_schema.py
git commit -m "chore(ptc): investigate MCP outputSchema exposure in langchain-mcp-adapters

Adapter version: [X.Y.Z]
Findings:
- outputSchema location: [metadata / not exposed / ...]
- Tool invoke return type: [str / dict / CallToolResult]
- structuredContent access: [direct / wrapped / parse from text]

Implementation implications documented in _extract_output_schema() task."
```

---

## Task 2: Add `ptc_eligible` field to `ToolConfig`

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/config/tool_config.py`
- Test: `deer-flow/backend/tests/test_tool_config.py` (may need creation)

- [ ] **Step 1: 写失败测试**

Create or add to `deer-flow/backend/tests/test_tool_config.py`:

```python
from deerflow.config.tool_config import ToolConfig


def test_tool_config_defaults_ptc_eligible_to_false():
    """ptc_eligible defaults to False when not specified."""
    cfg = ToolConfig(name="bash", group="sandbox", use="deerflow.sandbox.tools:bash_tool")
    assert cfg.ptc_eligible is False


def test_tool_config_accepts_ptc_eligible_true():
    """ptc_eligible can be explicitly set to True."""
    cfg = ToolConfig(
        name="bash",
        group="sandbox",
        use="deerflow.sandbox.tools:bash_tool",
        ptc_eligible=True,
    )
    assert cfg.ptc_eligible is True


def test_tool_config_rejects_invalid_ptc_eligible_type():
    """ptc_eligible only accepts bool-compatible values."""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        ToolConfig(
            name="bash",
            group="sandbox",
            use="deerflow.sandbox.tools:bash_tool",
            ptc_eligible="not-a-bool",  # type: ignore
        )
```

Add at top if not present:
```python
import pytest
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd deer-flow/backend && make test -- tests/test_tool_config.py -v`
或: `PYTHONPATH=packages/harness uv run pytest tests/test_tool_config.py -v`

Expected: AttributeError 或 ValidationError (field not declared)

- [ ] **Step 3: 添加字段实现**

Modify `deer-flow/backend/packages/harness/deerflow/config/tool_config.py`:

```python
from pydantic import BaseModel, ConfigDict, Field


class ToolGroupConfig(BaseModel):
    """Config section for a tool group"""

    name: str = Field(..., description="Unique name for the tool group")
    model_config = ConfigDict(extra="allow")


class ToolConfig(BaseModel):
    """Config section for a tool"""

    name: str = Field(..., description="Unique name for the tool")
    group: str = Field(..., description="Group name for the tool")
    use: str = Field(
        ...,
        description="Variable name of the tool provider(e.g. deerflow.sandbox.tools:bash_tool)",
    )
    ptc_eligible: bool = Field(
        default=False,
        description="If True, this tool is callable from code_execution (PTC) environment",
    )
    model_config = ConfigDict(extra="allow")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_tool_config.py -v`

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/bytedance/Documents/aime/deer-agents/deer-flow
git add backend/packages/harness/deerflow/config/tool_config.py backend/tests/test_tool_config.py
git commit -m "feat(ptc): add ptc_eligible field to ToolConfig

Tools opt in to programmatic tool calling by setting
ptc_eligible: true in config.yaml. Default is False.
Tests verify default and explicit values."
```

---

## Task 3: Add `PtcConfig` to `AppConfig`

**Files:**
- Create: `deer-flow/backend/packages/harness/deerflow/config/ptc_config.py`
- Modify: `deer-flow/backend/packages/harness/deerflow/config/app_config.py`
- Test: `deer-flow/backend/tests/test_ptc_config.py` (create)

- [ ] **Step 1: 写失败测试**

Create `deer-flow/backend/tests/test_ptc_config.py`:

```python
import pytest
from deerflow.config.ptc_config import PtcConfig


def test_ptc_config_all_defaults():
    """PtcConfig has sensible defaults."""
    cfg = PtcConfig()
    assert cfg.enabled is True
    assert cfg.include_mcp is False
    assert cfg.timeout_seconds == 30
    assert cfg.max_output_chars == 20000


def test_ptc_config_can_disable():
    cfg = PtcConfig(enabled=False)
    assert cfg.enabled is False


def test_ptc_config_can_enable_mcp():
    cfg = PtcConfig(include_mcp=True)
    assert cfg.include_mcp is True


def test_ptc_config_custom_timeout():
    cfg = PtcConfig(timeout_seconds=60)
    assert cfg.timeout_seconds == 60


def test_ptc_config_rejects_negative_timeout():
    with pytest.raises(ValueError):
        PtcConfig(timeout_seconds=-1)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_ptc_config.py -v`

Expected: ModuleNotFoundError

- [ ] **Step 3: 创建 PtcConfig**

Create `deer-flow/backend/packages/harness/deerflow/config/ptc_config.py`:

```python
from pydantic import BaseModel, ConfigDict, Field


class PtcConfig(BaseModel):
    """Configuration for Programmatic Tool Calling (PTC).

    PTC adds a `code_execution` tool that lets the LLM write Python code
    calling other tools programmatically. Tool results stay in code scope;
    only print() output enters model context.
    """

    enabled: bool = Field(
        default=True,
        description="Master switch for PTC. If False, code_execution tool is not registered.",
    )
    include_mcp: bool = Field(
        default=False,
        description="If True, all MCP tools are treated as ptc_eligible. Also auto-disables "
                    "tool_search deferred loading for MCP tools to ensure LLM sees their schemas.",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        description="Max wall time for a single code_execution invocation.",
    )
    max_output_chars: int = Field(
        default=20000,
        ge=100,
        description="Max characters of stdout returned to model; beyond this, output is truncated.",
    )

    model_config = ConfigDict(extra="forbid")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_ptc_config.py -v`

Expected: 5 tests PASS

- [ ] **Step 5: 集成到 AppConfig**

Read `deer-flow/backend/packages/harness/deerflow/config/app_config.py` to find where other config sections (like `sandbox`, `memory`) are declared, then add:

```python
from deerflow.config.ptc_config import PtcConfig

# In AppConfig class:
class AppConfig(BaseModel):
    # ... existing fields ...
    ptc: PtcConfig = Field(default_factory=PtcConfig)
```

- [ ] **Step 6: 验证 AppConfig 加载**

Create test in `deer-flow/backend/tests/test_ptc_config.py`:

```python
def test_app_config_has_ptc_section_with_defaults():
    """AppConfig.ptc defaults to PtcConfig() when not specified in YAML."""
    from deerflow.config.app_config import AppConfig
    cfg = AppConfig(models=[], tools=[], sandbox={"use": "deerflow.sandbox.local:LocalSandboxProvider"})
    assert cfg.ptc.enabled is True
    assert cfg.ptc.include_mcp is False
```

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_ptc_config.py -v`

Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add backend/packages/harness/deerflow/config/ptc_config.py \
        backend/packages/harness/deerflow/config/app_config.py \
        backend/tests/test_ptc_config.py
git commit -m "feat(ptc): add PtcConfig with enabled/include_mcp/timeout/max_output

New top-level config section controls PTC behavior.
Defaults: enabled=true, include_mcp=false, timeout=30s, max_output=20000."
```

---

## Task 4: Implement `_restricted_builtins()` and `_safe_modules()`

**Goal：** 构建 `exec()` namespace 的安全约束基础。

**Files:**
- Create: `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`
- Create: `deer-flow/backend/tests/test_code_execution_tool.py`

- [ ] **Step 1: 写失败测试**

Create `deer-flow/backend/tests/test_code_execution_tool.py`:

```python
"""Tests for PTC code_execution tool."""

import pytest
from deerflow.sandbox.code_execution import _restricted_builtins, _safe_modules


def test_restricted_builtins_has_print():
    b = _restricted_builtins()
    assert "print" in b
    assert b["print"] is print


def test_restricted_builtins_has_common_types():
    b = _restricted_builtins()
    for name in ["str", "int", "float", "bool", "list", "dict", "set", "tuple"]:
        assert name in b


def test_restricted_builtins_has_common_iterables():
    b = _restricted_builtins()
    for name in ["len", "range", "enumerate", "zip", "map", "filter", "sorted"]:
        assert name in b


def test_restricted_builtins_excludes_dangerous_functions():
    b = _restricted_builtins()
    for name in ["__import__", "eval", "exec", "compile", "open",
                 "input", "globals", "locals", "vars"]:
        assert name not in b, f"{name} should not be in restricted builtins"


def test_safe_modules_has_json():
    m = _safe_modules()
    import json
    assert m["json"] is json


def test_safe_modules_has_data_processing():
    m = _safe_modules()
    for name in ["json", "re", "math", "collections", "itertools", "functools"]:
        assert name in m


def test_safe_modules_excludes_dangerous():
    m = _safe_modules()
    for name in ["os", "sys", "subprocess", "socket", "requests", "urllib"]:
        assert name not in m
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v`

Expected: ModuleNotFoundError

- [ ] **Step 3: 实现 helpers**

Create `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`:

```python
"""Programmatic Tool Calling (PTC) code_execution tool.

Implements the code_execution tool that executes LLM-generated Python code
in a restricted namespace. Inside the code, other ptc_eligible tools are
callable as Python functions, and only print() output is returned to the
model context.

See: docs/superpowers/specs/2026-04-12-programmatic-tool-calling-design.md
"""

import builtins


def _restricted_builtins() -> dict:
    """Return a mapping of safe Python built-ins for the exec namespace.

    Excludes dangerous functions like __import__, eval, exec, open, which
    would let the LLM-generated code escape the sandbox restrictions.
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
        # Character / ord
        "chr", "ord",
        # Exceptions (so try/except works)
        "StopIteration", "ValueError", "TypeError", "KeyError", "IndexError",
        "AttributeError", "RuntimeError", "Exception",
    }
    return {k: getattr(builtins, k) for k in allowed if hasattr(builtins, k)}


def _safe_modules() -> dict:
    """Return a mapping of safe standard-library modules pre-imported into the namespace.

    Covers data processing needs (JSON parsing, regex, math, iteration helpers)
    without exposing network, filesystem, or process control.
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v`

Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/packages/harness/deerflow/sandbox/code_execution.py \
        backend/tests/test_code_execution_tool.py
git commit -m "feat(ptc): add _restricted_builtins and _safe_modules helpers

Core safety primitives for PTC code execution namespace.
Excludes __import__, eval, exec, open, os, sys, subprocess."
```

---

## Task 5: Implement `_execute_code()`

**Goal：** 在受限 namespace 中执行 Python 代码，捕获 stdout，处理异常和超时。

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`
- Modify: `deer-flow/backend/tests/test_code_execution_tool.py`

- [ ] **Step 1: 写失败测试**

Add to `tests/test_code_execution_tool.py`:

```python
from deerflow.sandbox.code_execution import _execute_code


def test_execute_code_captures_print_output():
    result = _execute_code("print('hello world')", tool_wrappers={}, runtime=None)
    assert result == "hello world\n"


def test_execute_code_returns_no_output_marker_on_silence():
    result = _execute_code("x = 1 + 1", tool_wrappers={}, runtime=None)
    assert result == "(no output)"


def test_execute_code_pre_imports_json_and_re():
    result = _execute_code(
        "print(json.dumps({'a': 1}))",
        tool_wrappers={},
        runtime=None,
    )
    assert result.strip() == '{"a": 1}'


def test_execute_code_returns_error_message_on_exception():
    result = _execute_code("raise ValueError('boom')", tool_wrappers={}, runtime=None)
    assert "ValueError" in result
    assert "boom" in result


def test_execute_code_blocks_import_of_os():
    # restricted builtins excludes __import__, so import statements fail
    result = _execute_code("import os\nprint(os.listdir('/'))", tool_wrappers={}, runtime=None)
    assert "error" in result.lower() or "Error" in result


def test_execute_code_blocks_open():
    result = _execute_code("open('/etc/passwd').read()", tool_wrappers={}, runtime=None)
    assert "error" in result.lower() or "NameError" in result


def test_execute_code_truncates_large_output():
    # Generate output > 20000 chars
    code = "print('x' * 30000)"
    result = _execute_code("print('x' * 30000)", tool_wrappers={}, runtime=None, max_output_chars=100)
    assert len(result) < 300  # truncation marker adds some overhead
    assert "truncated" in result


def test_execute_code_timeout():
    import time
    code = "while True:\n    pass"
    result = _execute_code(code, tool_wrappers={}, runtime=None, timeout=1)
    assert "timeout" in result.lower() or "exceeded" in result.lower()


def test_execute_code_calls_injected_wrapper():
    calls = []
    def fake_tool(**kwargs):
        calls.append(kwargs)
        return "tool_result"

    result = _execute_code(
        "print(fake_tool(x=42))",
        tool_wrappers={"fake_tool": fake_tool},
        runtime=None,
    )
    assert result.strip() == "tool_result"
    assert calls == [{"x": 42}]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py::test_execute_code_captures_print_output -v`

Expected: ImportError

- [ ] **Step 3: 实现 `_execute_code()`**

Add to `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`:

```python
import contextlib
import io
import signal
import traceback
from typing import Any

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_OUTPUT = 20000


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
        tool_wrappers: Mapping of function name -> callable injected into namespace.
                       Each wrapper should have a `_runtime` attribute that this
                       function will set to `runtime` before execution.
        runtime: The LangChain ToolRuntime injected into wrappers.
        timeout: Max wall-clock seconds.
        max_output_chars: Max chars of stdout returned; beyond this, output is truncated.

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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v -k test_execute_code`

Expected: 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/packages/harness/deerflow/sandbox/code_execution.py \
        backend/tests/test_code_execution_tool.py
git commit -m "feat(ptc): implement _execute_code with timeout and restricted namespace

Captures stdout, injects tool wrappers, enforces SIGALRM timeout,
truncates oversized output. Restricted builtins block import/open/exec."
```

---

## Task 6: Implement `_extract_output_schema()` and `_extract_structured_content()`

**Goal：** 屏蔽 MCP 2025-06-18 `outputSchema` / `structuredContent` 在 `langchain-mcp-adapters` 中的暴露差异。

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`
- Modify: `deer-flow/backend/tests/test_code_execution_tool.py`

- [ ] **Step 1: 写失败测试**

Add to `tests/test_code_execution_tool.py`:

```python
from deerflow.sandbox.code_execution import _extract_output_schema, _extract_structured_content


class _FakeTool:
    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.name = "fake"


def test_extract_output_schema_returns_none_when_absent():
    tool = _FakeTool(metadata={})
    assert _extract_output_schema(tool) is None


def test_extract_output_schema_from_metadata():
    schema = {"type": "object", "properties": {"temperature": {"type": "number"}}}
    tool = _FakeTool(metadata={"outputSchema": schema})
    assert _extract_output_schema(tool) == schema


def test_extract_output_schema_when_metadata_is_none():
    tool = _FakeTool(metadata=None)
    tool.metadata = None
    assert _extract_output_schema(tool) is None


def test_extract_structured_content_passes_through_dict():
    result = {"temperature": 22.5, "conditions": "sunny"}
    assert _extract_structured_content(result) == result


def test_extract_structured_content_passes_through_list():
    result = [{"id": 1}, {"id": 2}]
    assert _extract_structured_content(result) == result


def test_extract_structured_content_parses_json_string():
    result = '{"temperature": 22.5}'
    parsed = _extract_structured_content(result)
    assert parsed == {"temperature": 22.5}


def test_extract_structured_content_returns_string_when_not_json():
    result = "plain text not json"
    assert _extract_structured_content(result) == "plain text not json"


def test_extract_structured_content_from_object_with_attribute():
    class FakeResult:
        structuredContent = {"a": 1}
    assert _extract_structured_content(FakeResult()) == {"a": 1}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v -k "extract"`

Expected: ImportError

- [ ] **Step 3: 实现 helpers**

Add to `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`:

```python
import json as _json  # alias to avoid shadowing namespace 'json'
from langchain_core.tools import BaseTool


def _extract_output_schema(tool: BaseTool) -> dict | None:
    """Extract MCP outputSchema from a LangChain BaseTool.

    langchain-mcp-adapters exposes MCP Tool metadata including outputSchema
    on tool.metadata. This function returns None if no schema is declared.

    NOTE: Task 1 (investigation) should confirm the exact attribute path;
    if the adapter version uses a different location, update this function.

    Args:
        tool: A LangChain BaseTool instance.

    Returns:
        JSON Schema dict describing the tool's output, or None if not declared.
    """
    metadata = getattr(tool, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    schema = metadata.get("outputSchema")
    return schema if isinstance(schema, dict) else None


def _extract_structured_content(tool_result: Any) -> Any:
    """Extract structured content from an MCP tool result.

    Handles three shapes:
    1. Dict or list: passed through as-is (already parsed).
    2. JSON string: parsed via json.loads().
    3. Object with `.structuredContent` attribute: that attribute returned.

    Non-JSON strings are returned as-is (the LLM will handle them).

    Args:
        tool_result: Raw return value from a tool wrapper invocation.

    Returns:
        Parsed structured content (dict/list) or the original if parsing fails.
    """
    if isinstance(tool_result, (dict, list)):
        return tool_result
    if isinstance(tool_result, str):
        try:
            return _json.loads(tool_result)
        except _json.JSONDecodeError:
            return tool_result
    if hasattr(tool_result, "structuredContent"):
        return tool_result.structuredContent
    return tool_result
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v -k "extract"`

Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/packages/harness/deerflow/sandbox/code_execution.py \
        backend/tests/test_code_execution_tool.py
git commit -m "feat(ptc): add MCP outputSchema and structuredContent extraction helpers

_extract_output_schema reads tool.metadata['outputSchema'] (per
langchain-mcp-adapters convention; exact path verified in Task 1).
_extract_structured_content handles dict, list, JSON string, and object
with .structuredContent attribute for adapter version compatibility."
```

---

## Task 7: Implement `_invoke_tool_with_runtime()`

**Goal：** 以正确方式调用一个 LangChain `@tool`，把 runtime 透传进去。

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`
- Modify: `deer-flow/backend/tests/test_code_execution_tool.py`

- [ ] **Step 1: 写失败测试**

Add to `tests/test_code_execution_tool.py`:

```python
from deerflow.sandbox.code_execution import _invoke_tool_with_runtime


class _FakeToolWithRun:
    """Simulates a LangChain @tool-decorated function with _run attribute."""
    name = "fake"
    call_log = []

    def _run(self, **kwargs):
        self.call_log.append(kwargs)
        return f"called with {kwargs}"


def test_invoke_tool_with_runtime_calls_underlying_run():
    tool = _FakeToolWithRun()
    tool.call_log = []
    result = _invoke_tool_with_runtime(tool, {"foo": "bar"}, runtime="fake-runtime")
    assert "foo" in result
    assert tool.call_log == [{"foo": "bar", "runtime": "fake-runtime"}]


def test_invoke_tool_with_runtime_propagates_runtime_kwarg():
    tool = _FakeToolWithRun()
    tool.call_log = []
    _invoke_tool_with_runtime(tool, {}, runtime="ctx-123")
    assert tool.call_log[0]["runtime"] == "ctx-123"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v -k "invoke_tool"`

Expected: ImportError

- [ ] **Step 3: 实现 `_invoke_tool_with_runtime()`**

Add to `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`:

```python
def _invoke_tool_with_runtime(tool: BaseTool, kwargs: dict, runtime: Any) -> Any:
    """Invoke a LangChain @tool-decorated function with runtime injected.

    Task 0 investigation should have confirmed the correct call path for
    the current LangChain version. Two candidates:

    1. Direct `_run(**kwargs, runtime=runtime)` — works when ToolRuntime is
       declared as a keyword arg of the underlying function.
    2. `tool.invoke({...}, config={"configurable": {"runtime": runtime}})` —
       uses LangChain's runtime injection mechanism.

    This implementation uses option 1 because deer-flow sandbox tools define
    `runtime: ToolRuntime[...]` as an explicit parameter. If MCP tools require
    option 2, extend this function with type-based dispatch.

    Args:
        tool: LangChain BaseTool instance.
        kwargs: Arguments to pass to the tool (business parameters).
        runtime: The ToolRuntime to inject.

    Returns:
        Tool's return value (before structured content extraction).
    """
    return tool._run(**kwargs, runtime=runtime)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v -k "invoke_tool"`

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/packages/harness/deerflow/sandbox/code_execution.py \
        backend/tests/test_code_execution_tool.py
git commit -m "feat(ptc): add _invoke_tool_with_runtime for explicit runtime injection

Calls tool._run(**kwargs, runtime=runtime) which matches the signature
convention used by deer-flow sandbox tools. MCP tools may need different
dispatch if Task 1 reveals incompatibility."
```

---

## Task 8: Implement `_build_function_docs()`

**Goal：** 为 `code_execution` tool 动态生成其 description 中的函数签名列表。

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`
- Modify: `deer-flow/backend/tests/test_code_execution_tool.py`

- [ ] **Step 1: 写失败测试**

Add to `tests/test_code_execution_tool.py`:

```python
from deerflow.sandbox.code_execution import _build_function_docs
from pydantic import BaseModel, Field


class _FakeSchemaBash(BaseModel):
    command: str = Field(description="The command to run")
    description: str = Field(default="", description="What this command does")


class _FakeToolForDocs:
    """Minimal fake tool for testing _build_function_docs."""
    def __init__(self, name, description, args_schema, metadata=None):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.metadata = metadata or {}


def test_build_function_docs_lists_tool_name():
    tool = _FakeToolForDocs(
        name="bash",
        description="Execute a bash command",
        args_schema=_FakeSchemaBash,
    )
    docs = _build_function_docs([tool])
    assert "bash(" in docs


def test_build_function_docs_hides_runtime_and_description_params():
    tool = _FakeToolForDocs(
        name="bash",
        description="Execute a bash command",
        args_schema=_FakeSchemaBash,
    )
    docs = _build_function_docs([tool])
    # description field is in args_schema but should be hidden
    assert "description: str" not in docs
    # runtime is not in args_schema (it's injected) but also should not appear
    assert "runtime:" not in docs


def test_build_function_docs_shows_business_params():
    tool = _FakeToolForDocs(
        name="bash",
        description="Execute a bash command",
        args_schema=_FakeSchemaBash,
    )
    docs = _build_function_docs([tool])
    assert "command: str" in docs


def test_build_function_docs_marks_return_type_str_without_schema():
    tool = _FakeToolForDocs(
        name="bash",
        description="Execute a bash command",
        args_schema=_FakeSchemaBash,
    )
    docs = _build_function_docs([tool])
    assert "-> str" in docs


def test_build_function_docs_marks_structured_return_with_output_schema():
    output_schema = {"type": "object", "properties": {"temp": {"type": "number"}}}
    tool = _FakeToolForDocs(
        name="get_weather",
        description="Get weather",
        args_schema=_FakeSchemaBash,
        metadata={"outputSchema": output_schema},
    )
    docs = _build_function_docs([tool])
    assert "-> dict | list" in docs
    assert "temp" in docs  # schema hint included


def test_build_function_docs_includes_probe_hint():
    tool = _FakeToolForDocs(
        name="bash",
        description="Execute a bash command",
        args_schema=_FakeSchemaBash,
    )
    docs = _build_function_docs([tool])
    assert "probe" in docs.lower()
    assert "json.dumps" in docs


def test_build_function_docs_empty_list_returns_stub():
    docs = _build_function_docs([])
    assert isinstance(docs, str)
    assert len(docs) > 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v -k "build_function_docs"`

Expected: ImportError

- [ ] **Step 3: 实现 `_build_function_docs()`**

Add to `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`:

```python
_SKIP_PARAMS = {"runtime", "description"}


def _build_function_docs(eligible_tools: list[BaseTool]) -> str:
    """Generate a concise function signature listing for code_execution description.

    LLM has already seen each tool's full input schema in the top-level tool list.
    This function only needs to remind LLM of the function names, business params,
    and how to handle the return value (structured vs raw string).

    Args:
        eligible_tools: List of tools to expose in the code execution environment.

    Returns:
        Multi-line string documenting each tool as a Python function signature,
        plus a probe hint for tools without declared outputSchema.
    """
    if not eligible_tools:
        return "No tools are currently exposed to code_execution."

    docs = []
    for t in eligible_tools:
        params = []
        if t.args_schema is not None:
            for name, field in t.args_schema.model_fields.items():
                if name in _SKIP_PARAMS:
                    continue
                type_name = (
                    field.annotation.__name__
                    if hasattr(field.annotation, "__name__")
                    else str(field.annotation)
                )
                if field.default is not None and field.default is not ...:
                    default = f" = {field.default!r}"
                else:
                    default = ""
                params.append(f"{name}: {type_name}{default}")

        output_schema = _extract_output_schema(t)
        if output_schema is not None:
            return_type = "dict | list"
            schema_hint = (
                f"\n  Returns structured data. Schema: "
                f"{_json.dumps(output_schema, ensure_ascii=False)}"
            )
        else:
            return_type = "str"
            schema_hint = (
                "\n  Returns str (may be JSON — use json.loads() to parse)"
            )

        sig = f"{t.name}({', '.join(params)}) -> {return_type}"
        desc = t.description.split("\n")[0]
        docs.append(f"- {sig}\n  {desc}{schema_hint}")

    probe_hint = (
        "\n\nFor tools returning str without a schema, you can probe the "
        "structure once before batch processing:\n"
        "  sample = json.loads(tool_name(...))\n"
        "  print(json.dumps(sample[0] if isinstance(sample, list) else sample, indent=2))\n"
        "Then use the observed structure in subsequent calls."
    )

    return (
        "Available functions (full input schemas in the tool list above):\n\n"
        + "\n\n".join(docs)
        + probe_hint
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v -k "build_function_docs"`

Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/packages/harness/deerflow/sandbox/code_execution.py \
        backend/tests/test_code_execution_tool.py
git commit -m "feat(ptc): add _build_function_docs for dynamic code_execution description

Generates concise function signatures for eligible tools, marking
structured return types when MCP outputSchema is available, and
appending probe guidance for tools without schemas."
```

---

## Task 9: Implement `_build_tool_wrappers()`

**Goal：** 把每个 eligible tool 包装成一个可在 exec namespace 中调用的 Python 函数。

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`
- Modify: `deer-flow/backend/tests/test_code_execution_tool.py`

- [ ] **Step 1: 写失败测试**

Add to `tests/test_code_execution_tool.py`:

```python
from deerflow.sandbox.code_execution import _build_tool_wrappers


class _FakeSchemaWithDesc(BaseModel):
    command: str = Field()
    description: str = Field(default="")


class _FakeSchemaNoDesc(BaseModel):
    query: str = Field()


class _FakeToolForWrap:
    def __init__(self, name, args_schema, metadata=None, run_return="default"):
        self.name = name
        self.description = f"{name} tool"
        self.args_schema = args_schema
        self.metadata = metadata or {}
        self._run_return = run_return
        self.run_calls = []

    def _run(self, **kwargs):
        self.run_calls.append(kwargs)
        return self._run_return


def test_build_tool_wrappers_creates_one_per_tool():
    tool_a = _FakeToolForWrap("a", _FakeSchemaNoDesc)
    tool_b = _FakeToolForWrap("b", _FakeSchemaNoDesc)
    wrappers = _build_tool_wrappers([tool_a, tool_b])
    assert "a" in wrappers
    assert "b" in wrappers


def test_wrapper_forwards_kwargs_to_tool():
    tool = _FakeToolForWrap("bash", _FakeSchemaWithDesc, run_return="hello")
    wrappers = _build_tool_wrappers([tool])
    wrapper = wrappers["bash"]
    wrapper._runtime = "test-runtime"

    result = wrapper(command="echo hi")
    assert result == "hello"
    # Should have injected description and runtime
    assert tool.run_calls[0]["command"] == "echo hi"
    assert "description" in tool.run_calls[0]
    assert tool.run_calls[0]["runtime"] == "test-runtime"


def test_wrapper_skips_description_when_tool_does_not_accept_it():
    tool = _FakeToolForWrap("search", _FakeSchemaNoDesc)
    wrappers = _build_tool_wrappers([tool])
    wrapper = wrappers["search"]
    wrapper._runtime = None

    wrapper(query="foo")
    assert "description" not in tool.run_calls[0]
    assert tool.run_calls[0]["query"] == "foo"


def test_wrapper_extracts_structured_content_when_output_schema_present():
    tool = _FakeToolForWrap(
        "get_weather",
        _FakeSchemaNoDesc,
        metadata={"outputSchema": {"type": "object"}},
        run_return='{"temperature": 22.5}',  # JSON string
    )
    wrappers = _build_tool_wrappers([tool])
    wrapper = wrappers["get_weather"]
    wrapper._runtime = None

    result = wrapper(query="NY")
    # Should have been parsed because outputSchema is declared
    assert result == {"temperature": 22.5}


def test_wrapper_preserves_raw_string_when_no_output_schema():
    tool = _FakeToolForWrap(
        "bash",
        _FakeSchemaWithDesc,
        run_return="raw output\n",
    )
    wrappers = _build_tool_wrappers([tool])
    wrapper = wrappers["bash"]
    wrapper._runtime = None

    result = wrapper(command="echo hi")
    assert result == "raw output\n"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v -k "wrapper"`

Expected: ImportError

- [ ] **Step 3: 实现 `_build_tool_wrappers()`**

Add to `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`:

```python
def _build_tool_wrappers(eligible_tools: list[BaseTool]) -> dict[str, Any]:
    """Build a mapping of tool name -> Python callable for the exec namespace.

    Each wrapper:
    - Hides `runtime` and `description` parameters (LLM only sees business params).
    - Auto-injects `description="called from code_execution"` if the tool accepts it.
    - Invokes the tool via `_invoke_tool_with_runtime()`.
    - Returns structured content if the tool declares outputSchema, else raw result.

    The `_runtime` attribute on each wrapper is set to None initially and must
    be set to the actual ToolRuntime before code execution (done by _execute_code).
    """
    wrappers: dict[str, Any] = {}

    for t in eligible_tools:
        accepts_description = (
            t.args_schema is not None
            and "description" in t.args_schema.model_fields
        )
        has_output_schema = _extract_output_schema(t) is not None

        def _make_wrapper(
            tool_ref=t,
            needs_desc=accepts_description,
            structured=has_output_schema,
        ):
            def wrapper(**kwargs):
                if needs_desc and "description" not in kwargs:
                    kwargs["description"] = "called from code_execution"
                result = _invoke_tool_with_runtime(tool_ref, kwargs, wrapper._runtime)
                if structured:
                    return _extract_structured_content(result)
                return result

            wrapper._runtime = None
            return wrapper

        wrappers[t.name] = _make_wrapper()

    return wrappers
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v -k "wrapper"`

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/packages/harness/deerflow/sandbox/code_execution.py \
        backend/tests/test_code_execution_tool.py
git commit -m "feat(ptc): add _build_tool_wrappers to bridge tools into exec namespace

Each wrapper auto-injects description (if accepted), propagates runtime,
and unwraps structuredContent when outputSchema is declared."
```

---

## Task 10: Implement `make_code_execution_tool()` factory

**Goal：** 工厂函数，把所有 helpers 组装成一个 LangChain `@tool`。

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`
- Modify: `deer-flow/backend/tests/test_code_execution_tool.py`

- [ ] **Step 1: 写失败测试**

Add to `tests/test_code_execution_tool.py`:

```python
from deerflow.sandbox.code_execution import make_code_execution_tool


def test_make_code_execution_tool_returns_langchain_tool():
    from langchain_core.tools import BaseTool as LCBaseTool
    tool = make_code_execution_tool([])
    assert isinstance(tool, LCBaseTool)
    assert tool.name == "code_execution"


def test_code_execution_tool_description_lists_eligible_tools():
    fake = _FakeToolForWrap("bash", _FakeSchemaWithDesc)
    tool = make_code_execution_tool([fake])
    assert "bash" in tool.description


def test_code_execution_tool_executes_simple_code_via_run():
    fake = _FakeToolForWrap("echo", _FakeSchemaNoDesc, run_return="hello")
    tool = make_code_execution_tool([fake])
    # _run accepts runtime + code; pass None for runtime
    result = tool._run(code="print(echo(query='x'))", runtime=None)
    assert result.strip() == "hello"


def test_code_execution_tool_reports_errors_inline():
    tool = make_code_execution_tool([])
    result = tool._run(code="1 / 0", runtime=None)
    assert "ZeroDivisionError" in result or "error" in result.lower()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v -k "make_code_execution"`

Expected: ImportError

- [ ] **Step 3: 实现 `make_code_execution_tool()`**

Add to `deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py`:

```python
from langchain_core.tools import tool as lc_tool_decorator


def make_code_execution_tool(
    eligible_tools: list[BaseTool],
    timeout_seconds: int = _DEFAULT_TIMEOUT,
    max_output_chars: int = _DEFAULT_MAX_OUTPUT,
) -> BaseTool:
    """Factory: build a `code_execution` LangChain tool from the given eligible tools.

    Args:
        eligible_tools: Tools that should be exposed as callable functions inside
                        the exec namespace. Can be empty — the tool still registers.
        timeout_seconds: Max wall time per exec invocation.
        max_output_chars: Max chars of stdout returned to the model.

    Returns:
        A LangChain BaseTool (named `code_execution`) ready to be added to the
        agent's tool list.
    """
    func_docs = _build_function_docs(eligible_tools)
    tool_wrappers = _build_tool_wrappers(eligible_tools)

    description = (
        "Execute Python code with programmatic tool access.\n\n"
        "Use this when you need to batch-call tools and process results in code,\n"
        "avoiding large tool outputs in your context window. Only print() output\n"
        "is returned to you — tool return values stay inside the code scope.\n\n"
        f"{func_docs}\n\n"
        "Pre-imported modules: json, re, math, collections, itertools, functools, datetime\n"
        "Use print() to output anything you want to see.\n\n"
        "The `code` argument is the Python source to execute."
    )

    @lc_tool_decorator("code_execution", parse_docstring=False)
    def code_execution_tool(code: str, runtime: Any = None) -> str:
        """Execute Python code programmatically."""
        return _execute_code(
            code,
            tool_wrappers=tool_wrappers,
            runtime=runtime,
            timeout=timeout_seconds,
            max_output_chars=max_output_chars,
        )

    # Override the auto-generated description with our dynamic one
    code_execution_tool.description = description
    return code_execution_tool
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v -k "make_code_execution"`

Expected: 4 tests PASS

- [ ] **Step 5: 运行完整测试文件，确认所有测试仍通过**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_code_execution_tool.py -v`

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add backend/packages/harness/deerflow/sandbox/code_execution.py \
        backend/tests/test_code_execution_tool.py
git commit -m "feat(ptc): add make_code_execution_tool factory

Wraps helpers into a LangChain @tool named 'code_execution'.
Dynamic description lists all eligible tools as Python signatures.
The tool executes LLM-generated code in a restricted namespace."
```

---

## Task 11: Integrate `code_execution` into `get_available_tools()`

**Goal：** 在 `get_available_tools()` 末尾根据 `ptc_eligible` 标记和 `ptc.include_mcp` 选项收集 eligible tools，构建并追加 `code_execution_tool`。

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/tools/tools.py`
- Create: `deer-flow/backend/tests/test_ptc_integration.py`

- [ ] **Step 1: 写失败测试（集成层）**

Create `deer-flow/backend/tests/test_ptc_integration.py`:

```python
"""Integration tests for PTC registration in get_available_tools()."""

import pytest
from unittest.mock import MagicMock, patch

from deerflow.tools.tools import get_available_tools


@pytest.fixture
def mock_config_with_ptc_eligible_bash(monkeypatch):
    """Fixture: AppConfig with bash tool marked ptc_eligible."""
    from deerflow.config.app_config import AppConfig
    from deerflow.config.tool_config import ToolConfig
    from deerflow.config.ptc_config import PtcConfig

    fake_config = MagicMock(spec=AppConfig)
    fake_config.tools = [
        ToolConfig(
            name="bash",
            group="sandbox",
            use="deerflow.sandbox.tools:bash_tool",
            ptc_eligible=True,
        )
    ]
    fake_config.models = []
    fake_config.tool_search = MagicMock(enabled=False)
    fake_config.skill_evolution = None
    fake_config.ptc = PtcConfig(enabled=True, include_mcp=False)
    fake_config.get_model_config = MagicMock(return_value=None)
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: fake_config)
    monkeypatch.setattr("deerflow.tools.tools.is_host_bash_allowed", lambda c: True)
    yield fake_config


def test_get_available_tools_adds_code_execution_when_eligible_tools_present(
    mock_config_with_ptc_eligible_bash,
):
    tools = get_available_tools(include_mcp=False)
    tool_names = [t.name for t in tools]
    assert "bash" in tool_names
    assert "code_execution" in tool_names


def test_get_available_tools_skips_code_execution_when_ptc_disabled(monkeypatch):
    from deerflow.config.app_config import AppConfig
    from deerflow.config.tool_config import ToolConfig
    from deerflow.config.ptc_config import PtcConfig

    fake_config = MagicMock(spec=AppConfig)
    fake_config.tools = [
        ToolConfig(
            name="bash",
            group="sandbox",
            use="deerflow.sandbox.tools:bash_tool",
            ptc_eligible=True,
        )
    ]
    fake_config.models = []
    fake_config.tool_search = MagicMock(enabled=False)
    fake_config.skill_evolution = None
    fake_config.ptc = PtcConfig(enabled=False)
    fake_config.get_model_config = MagicMock(return_value=None)
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: fake_config)
    monkeypatch.setattr("deerflow.tools.tools.is_host_bash_allowed", lambda c: True)

    tools = get_available_tools(include_mcp=False)
    tool_names = [t.name for t in tools]
    assert "code_execution" not in tool_names


def test_get_available_tools_skips_code_execution_when_no_eligible_tools(monkeypatch):
    from deerflow.config.app_config import AppConfig
    from deerflow.config.tool_config import ToolConfig
    from deerflow.config.ptc_config import PtcConfig

    fake_config = MagicMock(spec=AppConfig)
    fake_config.tools = [
        ToolConfig(
            name="bash",
            group="sandbox",
            use="deerflow.sandbox.tools:bash_tool",
            ptc_eligible=False,  # not eligible
        )
    ]
    fake_config.models = []
    fake_config.tool_search = MagicMock(enabled=False)
    fake_config.skill_evolution = None
    fake_config.ptc = PtcConfig(enabled=True)
    fake_config.get_model_config = MagicMock(return_value=None)
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: fake_config)
    monkeypatch.setattr("deerflow.tools.tools.is_host_bash_allowed", lambda c: True)

    tools = get_available_tools(include_mcp=False)
    tool_names = [t.name for t in tools]
    assert "code_execution" not in tool_names
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_ptc_integration.py -v`

Expected: FAIL (code_execution not in tool list)

- [ ] **Step 3: 修改 `get_available_tools()`**

Modify `deer-flow/backend/packages/harness/deerflow/tools/tools.py`:

Find the final `return` statement:

```python
    logger.info(f"Total tools loaded: {len(loaded_tools)}, built-in tools: {len(builtin_tools)}, MCP tools: {len(mcp_tools)}, ACP tools: {len(acp_tools)}")
    return loaded_tools + builtin_tools + mcp_tools + acp_tools
```

Replace with:

```python
    logger.info(f"Total tools loaded: {len(loaded_tools)}, built-in tools: {len(builtin_tools)}, MCP tools: {len(mcp_tools)}, ACP tools: {len(acp_tools)}")

    final_tools = loaded_tools + builtin_tools + mcp_tools + acp_tools

    # Build code_execution tool if PTC is enabled and there are eligible tools
    ptc_config = getattr(config, "ptc", None)
    if ptc_config is not None and ptc_config.enabled:
        eligible_tools = _collect_ptc_eligible_tools(
            tool_configs=tool_configs,
            loaded_tools=loaded_tools,
            mcp_tools=mcp_tools,
            include_mcp=ptc_config.include_mcp,
        )
        if eligible_tools:
            from deerflow.sandbox.code_execution import make_code_execution_tool
            code_exec = make_code_execution_tool(
                eligible_tools,
                timeout_seconds=ptc_config.timeout_seconds,
                max_output_chars=ptc_config.max_output_chars,
            )
            final_tools.append(code_exec)
            logger.info(
                f"PTC enabled: code_execution tool added with "
                f"{len(eligible_tools)} eligible tool(s)"
            )

    return final_tools
```

Add the helper function at module level (after `_is_host_bash_tool`):

```python
def _collect_ptc_eligible_tools(
    tool_configs: list,
    loaded_tools: list[BaseTool],
    mcp_tools: list[BaseTool],
    include_mcp: bool,
) -> list[BaseTool]:
    """Collect tools that should be exposed to the code_execution environment.

    Sources:
    - Config-defined tools where tool_config.ptc_eligible is True.
    - All MCP tools if include_mcp is True.
    """
    eligible: list[BaseTool] = []

    # Map tool_configs by position to loaded_tools (same order, 1:1)
    for cfg, tool in zip(tool_configs, loaded_tools):
        if getattr(cfg, "ptc_eligible", False):
            eligible.append(tool)

    if include_mcp:
        eligible.extend(mcp_tools)

    return eligible
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_ptc_integration.py -v`

Expected: 3 tests PASS

- [ ] **Step 5: 运行所有 PTC 相关测试**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_ptc_config.py tests/test_tool_config.py tests/test_code_execution_tool.py tests/test_ptc_integration.py -v`

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add backend/packages/harness/deerflow/tools/tools.py \
        backend/tests/test_ptc_integration.py
git commit -m "feat(ptc): register code_execution tool in get_available_tools

Collects ptc_eligible tools from config plus MCP tools if
ptc.include_mcp is enabled. Skips registration when PTC is
disabled or no eligible tools exist."
```

---

## Task 12: Handle MCP `tool_search` deferred loading conflict

**Goal：** 当 `ptc.include_mcp = true` 时，MCP tools 被标记为 eligible；需要确保 LLM 能看到它们的 input schema（不被 `DeferredToolFilterMiddleware` 过滤掉）。

**Files:**
- Modify: `deer-flow/backend/packages/harness/deerflow/tools/tools.py`
- Modify: `deer-flow/backend/tests/test_ptc_integration.py`

- [ ] **Step 1: 写失败测试**

Add to `tests/test_ptc_integration.py`:

```python
def test_tool_search_disabled_when_ptc_include_mcp_is_true(monkeypatch):
    """When ptc.include_mcp=True, tool_search deferred registry must not be populated."""
    from deerflow.config.app_config import AppConfig
    from deerflow.config.tool_config import ToolConfig
    from deerflow.config.ptc_config import PtcConfig

    fake_config = MagicMock(spec=AppConfig)
    fake_config.tools = []
    fake_config.models = []
    fake_config.tool_search = MagicMock(enabled=True)  # tool_search ON
    fake_config.skill_evolution = None
    fake_config.ptc = PtcConfig(enabled=True, include_mcp=True)  # PTC includes MCP
    fake_config.get_model_config = MagicMock(return_value=None)
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: fake_config)
    monkeypatch.setattr("deerflow.tools.tools.is_host_bash_allowed", lambda c: True)

    fake_mcp_tools = [MagicMock(name="fake_mcp_1"), MagicMock(name="fake_mcp_2")]
    for t in fake_mcp_tools:
        t.metadata = {}
        t.args_schema = None
    monkeypatch.setattr(
        "deerflow.mcp.cache.get_cached_mcp_tools",
        lambda: fake_mcp_tools,
    )
    # Also mock extensions_config to report enabled servers
    from deerflow.config.extensions_config import ExtensionsConfig
    fake_ext = MagicMock()
    fake_ext.get_enabled_mcp_servers = MagicMock(return_value=["fake-server"])
    monkeypatch.setattr(
        "deerflow.config.extensions_config.ExtensionsConfig.from_file",
        classmethod(lambda cls: fake_ext),
    )

    tools = get_available_tools(include_mcp=True)
    tool_names = [getattr(t, "name", str(t)) for t in tools]

    # code_execution should be registered
    assert "code_execution" in tool_names
    # tool_search tool should NOT be in the final list (deferred loading bypassed)
    assert "tool_search" not in tool_names, (
        "tool_search must be absent when PTC include_mcp overrides deferred loading"
    )
    # MCP tools should appear directly in the list (not deferred)
    # Note: MagicMock tools don't have real names, so we check by count
    non_fixture_tool_count = sum(
        1 for t in tools
        if getattr(t, "name", "") not in {"code_execution", "tool_search"}
    )
    assert non_fixture_tool_count >= 2, (
        f"Expected 2 MCP tools to be directly exposed, got {non_fixture_tool_count}"
    )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_ptc_integration.py::test_tool_search_disabled_when_ptc_include_mcp_is_true -v`

Expected: FAIL (deferred registry has tools because tool_search is enabled)

- [ ] **Step 3: 修改 `get_available_tools()` 使 PTC 覆盖 tool_search for MCP**

In `tools.py`, find the tool_search activation block:

```python
                    # When tool_search is enabled, register MCP tools in the
                    # deferred registry and add tool_search to builtin tools.
                    if config.tool_search.enabled:
                        from deerflow.tools.builtins.tool_search import DeferredToolRegistry, set_deferred_registry
                        from deerflow.tools.builtins.tool_search import tool_search as tool_search_tool

                        registry = DeferredToolRegistry()
                        for t in mcp_tools:
                            registry.register(t)
                        set_deferred_registry(registry)
                        builtin_tools.append(tool_search_tool)
                        logger.info(f"Tool search active: {len(mcp_tools)} tools deferred")
```

Replace with:

```python
                    # When tool_search is enabled, register MCP tools in the
                    # deferred registry and add tool_search to builtin tools.
                    # EXCEPT: if PTC is enabled and includes MCP, we need the
                    # LLM to see all MCP tool schemas directly, so tool_search
                    # deferred loading is skipped for MCP tools.
                    ptc_claims_mcp = (
                        getattr(config, "ptc", None) is not None
                        and config.ptc.enabled
                        and config.ptc.include_mcp
                    )
                    if config.tool_search.enabled and not ptc_claims_mcp:
                        from deerflow.tools.builtins.tool_search import DeferredToolRegistry, set_deferred_registry
                        from deerflow.tools.builtins.tool_search import tool_search as tool_search_tool

                        registry = DeferredToolRegistry()
                        for t in mcp_tools:
                            registry.register(t)
                        set_deferred_registry(registry)
                        builtin_tools.append(tool_search_tool)
                        logger.info(f"Tool search active: {len(mcp_tools)} tools deferred")
                    elif config.tool_search.enabled and ptc_claims_mcp:
                        logger.info(
                            "PTC include_mcp=True overrides tool_search deferred loading; "
                            "MCP tools are exposed directly to the LLM"
                        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=packages/harness uv run pytest tests/test_ptc_integration.py -v`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add backend/packages/harness/deerflow/tools/tools.py \
        backend/tests/test_ptc_integration.py
git commit -m "feat(ptc): skip tool_search deferred loading when PTC claims MCP

When ptc.include_mcp=True, MCP tool schemas must be visible to the LLM
so it can write correct code against them. This patch bypasses the
tool_search deferred registry for MCP tools in that mode."
```

---

## Task 13: Update `config.example.yaml` with PTC markers

**Files:**
- Modify: `deer-flow/config.example.yaml`

- [ ] **Step 1: 读取现有 config.example.yaml**

Read `deer-flow/config.example.yaml` and locate the `tools:` section and find where to add a new top-level `ptc:` section (near `sandbox:` or `memory:`).

- [ ] **Step 2: 给 sandbox tools 添加 `ptc_eligible: true`**

For each of the following tools in the `tools:` list, add `ptc_eligible: true`:

- `bash` (`deerflow.sandbox.tools:bash_tool`)
- `read_file` (`deerflow.sandbox.tools:read_file_tool`)
- `write_file` (`deerflow.sandbox.tools:write_file_tool`)
- `grep` (`deerflow.sandbox.tools:grep_tool`)
- `glob` (`deerflow.sandbox.tools:glob_tool`)
- `ls` (`deerflow.sandbox.tools:ls_tool`)
- `str_replace` (`deerflow.sandbox.tools:str_replace_tool`)

Example (find each and add the field):

```yaml
tools:
  - name: bash
    group: sandbox
    use: deerflow.sandbox.tools:bash_tool
    ptc_eligible: true   # ← new

  - name: read_file
    group: sandbox
    use: deerflow.sandbox.tools:read_file_tool
    ptc_eligible: true   # ← new
  # ...etc
```

Note: **do not** mark `ask_clarification` or `present_file` as ptc_eligible (they are interactive / UI tools).

- [ ] **Step 3: 添加顶层 `ptc:` section**

Add near the `sandbox:` or `memory:` section:

```yaml
# Programmatic Tool Calling (PTC) configuration.
# Allows the LLM to write Python code that calls tools programmatically,
# reducing context usage for batch operations.
# See: docs/superpowers/specs/2026-04-12-programmatic-tool-calling-design.md
ptc:
  enabled: true           # Master switch for PTC
  include_mcp: false      # Set to true to expose all MCP tools to code_execution
  timeout_seconds: 30     # Max wall time per code_execution invocation
  max_output_chars: 20000 # Max chars of stdout returned to model
```

- [ ] **Step 4: 验证 YAML 解析正常**

Run: `cd deer-flow/backend && PYTHONPATH=packages/harness uv run python -c "
from deerflow.config.app_config import AppConfig
cfg = AppConfig.from_file('../config.example.yaml')
print('ptc enabled:', cfg.ptc.enabled)
print('ptc_eligible tools:', [t.name for t in cfg.tools if t.ptc_eligible])
"`

Expected: no errors, prints the PTC config and the list of eligible tools.

- [ ] **Step 5: Commit**

```bash
git add deer-flow/config.example.yaml
git commit -m "feat(ptc): add ptc_eligible markers and ptc config section to config.example.yaml

Marks all sandbox tools (bash, read_file, write_file, grep, glob, ls, str_replace)
as ptc_eligible. Interactive tools (ask_clarification, present_file) are excluded.
Adds top-level ptc section documenting all configurable knobs."
```

---

## Task 14: End-to-end test via DeerFlowClient

**Goal：** 用真实 agent 验证整个 PTC 流程走通。

**Files:**
- Create: `deer-flow/backend/tests/test_ptc_e2e.py`

- [ ] **Step 1: 写 e2e 测试**

Create `deer-flow/backend/tests/test_ptc_e2e.py`:

```python
"""End-to-end test: PTC with a real agent run.

This test is marked `live` because it requires API keys and a working model.
It is skipped by default; run with `pytest -m live`.
"""

import os
import pytest

pytestmark = pytest.mark.live


@pytest.fixture
def client():
    pytest.importorskip("anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("No API key configured for live test")
    from deerflow.client import DeerFlowClient
    return DeerFlowClient()


def test_code_execution_tool_is_registered(client):
    """After initialization, code_execution should be in the tool list."""
    tools = client.get_available_tools()
    tool_names = [t.name for t in tools]
    assert "code_execution" in tool_names, f"code_execution missing. Got: {tool_names}"


def test_code_execution_description_lists_sandbox_tools(client):
    """code_execution description should include bash, read_file, etc."""
    tools = client.get_available_tools()
    code_exec = next(t for t in tools if t.name == "code_execution")
    desc = code_exec.description
    for expected in ["bash", "read_file", "grep"]:
        assert expected in desc, f"{expected} missing from code_execution description"


def test_agent_uses_code_execution_for_batch_task(client):
    """An agent given a batch task should choose code_execution."""
    thread_id = "test-ptc-e2e"
    query = (
        "Use code_execution to: read every .py file under /mnt/user-data/workspace, "
        "count lines with 'def ', and print the total count. "
        "Don't call read_file or grep individually from the outer tool list."
    )
    result = client.chat(message=query, thread_id=thread_id)
    # We don't assert the exact count, just that the agent finished without error
    assert result is not None
    assert len(result) > 0
```

- [ ] **Step 2: 运行 e2e 测试**

```bash
cd deer-flow/backend
source ../.venv/bin/activate
export $(cat ../.env | grep -v '^#' | xargs)
PYTHONPATH=packages/harness uv run pytest tests/test_ptc_e2e.py -v -m live
```

Expected: 2 tests PASS (first two); third test (agent behavior) may or may not pick code_execution depending on model — log the outcome.

- [ ] **Step 3: 如果 agent 行为测试失败，用 trace inspector 定位**

```bash
python scripts/trace_inspector.py last
```

查看 LLM 实际输出，判断：
- agent 是否看到了 `code_execution` tool
- description 是否足够清晰让 LLM 选择它
- 是否需要在 agent system prompt 中加入 PTC 使用指引

If needed, document findings as a Phase 2 follow-up task.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_ptc_e2e.py
git commit -m "test(ptc): add end-to-end tests for code_execution tool registration

Verifies code_execution is registered alongside sandbox tools
when PTC is enabled. Marked 'live' since it requires API keys."
```

---

## Task 15: Update `deer-flow/backend/CLAUDE.md` documentation

**Files:**
- Modify: `deer-flow/backend/CLAUDE.md`

- [ ] **Step 1: 读取现有 CLAUDE.md 的 "Tool System" 章节**

Read `deer-flow/backend/CLAUDE.md` and find the "Tool System" section (around the line mentioning `get_available_tools(groups, include_mcp, model_name, subagent_enabled)`).

- [ ] **Step 2: 添加 PTC 子章节**

After the Tool System section's numbered list of tool sources, add:

```markdown
**Programmatic Tool Calling (PTC)** (new):
- When `config.ptc.enabled` is True, `get_available_tools()` also registers
  a `code_execution` tool that lets the LLM write Python code calling other
  tools programmatically.
- Tools opt in via `ptc_eligible: true` in config.yaml (sandbox tools do by default).
- Setting `config.ptc.include_mcp: true` exposes all MCP tools to the code
  execution environment and bypasses `tool_search` deferred loading for MCP.
- Implementation: `packages/harness/deerflow/sandbox/code_execution.py`
- MCP output schema handling follows the MCP 2025-06-18 spec
  (`outputSchema` / `structuredContent`).
- See `docs/superpowers/specs/2026-04-12-programmatic-tool-calling-design.md`
  for full design rationale.
```

- [ ] **Step 3: 在 "Configuration System" 章节的 `config.yaml` 部分加 `ptc` 条目**

Find the bullet list under `**`config.yaml`** key sections:` and add:

```markdown
- `ptc` - Programmatic Tool Calling (enabled, include_mcp, timeout_seconds, max_output_chars)
```

- [ ] **Step 4: Commit**

```bash
git add backend/CLAUDE.md
git commit -m "docs(ptc): document Programmatic Tool Calling in CLAUDE.md

Adds overview of code_execution tool, ptc_eligible config,
and the include_mcp opt-in for MCP tools."
```

---

## Task 16: Final verification

- [ ] **Step 1: 运行完整 PTC 测试套件**

```bash
cd deer-flow/backend
PYTHONPATH=packages/harness uv run pytest \
  tests/test_tool_config.py \
  tests/test_ptc_config.py \
  tests/test_code_execution_tool.py \
  tests/test_ptc_integration.py \
  -v
```

Expected: all PASS

- [ ] **Step 2: 运行 deer-flow 完整测试套件确认没有 regression**

```bash
cd deer-flow/backend && make test
```

Expected: all existing tests still PASS

- [ ] **Step 3: 运行 deer-agents e2e 脚本**

```bash
cd /Users/bytedance/Documents/aime/deer-agents
source .venv/bin/activate
export $(cat deer-flow/.env | grep -v '^#' | xargs)
python scripts/e2e_test.py
```

Expected: e2e script passes (no regressions to the full chain).

- [ ] **Step 4: 用 trace inspector 检查最近一次 agent run**

```bash
python scripts/trace_inspector.py recent
python scripts/trace_inspector.py last
```

确认 trace 里 `code_execution` tool 可见，如 agent 使用过它，确认行为符合预期。

- [ ] **Step 5: 最终 commit**

If any docs or small fixes remain:

```bash
git add <paths>
git commit -m "chore(ptc): final adjustments from verification"
```

---

## Summary Checklist

**Spec coverage:**
- [x] `ptc_eligible` field on ToolConfig (Task 2)
- [x] `PtcConfig` with `enabled` / `include_mcp` / `timeout_seconds` / `max_output_chars` (Task 3)
- [x] `_restricted_builtins()` and `_safe_modules()` (Task 4)
- [x] `_execute_code()` with timeout, exception handling, output truncation (Task 5)
- [x] `_extract_output_schema()` for MCP outputSchema (Task 6)
- [x] `_extract_structured_content()` for MCP structuredContent / JSON / dict / object (Task 6)
- [x] `_invoke_tool_with_runtime()` (Task 7)
- [x] `_build_function_docs()` with structured vs str return types and probe hint (Task 8)
- [x] `_build_tool_wrappers()` (Task 9)
- [x] `make_code_execution_tool()` factory (Task 10)
- [x] `get_available_tools()` integration (Task 11)
- [x] PTC overrides tool_search deferred for MCP (Task 12)
- [x] `config.example.yaml` markers and section (Task 13)
- [x] E2E test (Task 14)
- [x] Documentation (Task 15)

**Investigation tasks before implementation:**
- Task 0: LangChain tool invocation mechanism — informs Task 7
- Task 1: MCP outputSchema exposure in langchain-mcp-adapters — informs Task 6

**Implementation notes:**
- All helpers live in a single new file `sandbox/code_execution.py` (single-responsibility, keeps `sandbox/tools.py` from growing further)
- Tests live in `tests/test_code_execution_tool.py` (unit), `tests/test_ptc_integration.py` (integration), `tests/test_ptc_e2e.py` (live)
- Each task is self-contained and produces a green test suite before commit
- TDD discipline: write failing test, verify failure, implement minimal code, verify pass, commit
