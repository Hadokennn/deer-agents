# Purpose-Scoped Programmatic Tool Calling — Design Spec

> 全新的 purpose-scoped PTC 方案，替换原单一 `code_execution` tool 方案。
> 每个 PTC tool 在 config 中声明目的、可用 tools，大幅收窄 LLM 的决策空间。
> **不保留原方案的 code_execution，不使用 probe 路径，先跑起来再说。**

## Overview

在 DeerFlow 的 tool 层支持注册 **purpose-scoped PTC tools**。每个 PTC tool 对应一个明确的业务目的（如"分析告警关联性"、"回放故障时间线"），声明该目的下可调用的 tools（MCP 或内置）。LLM 生成 Python 代码在进程内执行，代码中可调用声明范围内的 tools。Tool 返回值留在代码作用域内处理，只有 `print()` 输出进入 model context。

## Motivation

### 原单一 `code_execution` 方案的问题

原方案将所有 `ptc_eligible` tools 汇入一个 `code_execution` tool，LLM 看到的 description 包含所有函数签名：

```python
# 原方案 code_execution description 中列出的函数
- bash(...) -> str
- read_file(...) -> str
- grep(...) -> str
- get_alert_metrics(...) -> dict
- query_log_cluster(...) -> str
- search_incident(...) -> dict
- query_operation_log(...) -> str
```

问题：
1. **意图空间过大**：LLM 需自己判断"该用哪几个 tool、怎么组合、输出什么结构"，组合空间爆炸
2. **幻觉风险高**：不同领域的 MCP tools（监控、日志、工单）被混在一起，LLM 可能生成无意义的组合
3. **无目的约束**：LLM 不知道"这次调用要达成什么目标"，容易跑偏

### 新方案

每个 PTC tool 是一个独立的、purpose-scoped 的 tool：

- **Config 声明目的**：`purpose` 字段明确告诉 LLM 这个 tool 要做什么
- **Config 约束 tool 范围**：`eligible_tools` 只列出与目的相关的 tools
- **先跑起来再说**：不强制 output schema，不使用 probe，让 LLM 自己处理返回值

## Design Decisions

| 决策 | 原方案 | 新方案 | 理由 |
|------|--------|--------|------|
| PTC tool 数量 | 1 个全局 `code_execution` | N 个 purpose-scoped tools | 收窄意图空间，降低幻觉 |
| Tool 选择 | `ptc_eligible` 标记，所有 eligible 汇入 | Config 中每个 PTC tool 声明自己的 `eligible_tools` | 目的相关性由人定义，不由 LLM 猜 |
| 目的约束 | 无 | `purpose` 字段 | LLM 有明确目标，代码生成更精准 |
| Output schema | 三路径（有 schema / 无 schema probe / 非 MCP） | 两路径（有 schema 用 / 无 schema 直接返回原始值） | 先跑起来再说，不强制 schema，不使用 probe |
| 与 deferred 的关系 | `ptc_eligible` 优先，自动取消 deferred | PTC tool 引用的 eligible tools 自动取消 deferred | 同理，LLM 需在 tool list 中看到完整 input schema |
| 执行环境 | In-process `exec()` | 不变 | 同原方案 |
| 安全模型 | 受限 namespace + 超时 + 截断 | 不变 | 同原方案 |

## Output Schema 优先级（宽松版）

**不强制 output schema，先跑起来再说：**

```
1. MCP tool 自身的 outputSchema  →  优先使用（forward-compatible,见下）
2. Config 中配置的 output_schema →  使用（如果有）
3. 都没有                        →  允许进入 PTC,返回原始值(LLM 自己处理)
```

所有 tool 都可以进 PTC，不管有没有 output schema。有 schema 的话 LLM 写代码更方便，没有的话 LLM 自己处理原始返回值。

> **当前状态 (2026-04-13):** `langchain-mcp-adapters 0.2.2` 不暴露 MCP tool 的
> `outputSchema`(见下方 "MCP Output Schema Handling" 章节的 adapter 兼容性小节)。
> 因此路径 1 目前永远走不通,只有路径 2 和路径 3 生效。路径 1 作为
> forward-compatible stub 保留,未来 adapter 版本支持后自动生效。

## Data Flow

```
┌──────────────────────────────────────────────────────────┐
│ Agent 初始化 — get_available_tools()                      │
│                                                          │
│ 1. 加载所有 tools (bash, grep, read_file, MCP...)        │
│                                                          │
│ 2. 遍历 config 中的 PTC tool 声明：                      │
│    for each ptc_config in config.ptc_tools:              │
│      a. 解析 eligible_tools，查找已加载的 tool 实例       │
│      b. 解析 output schema（可选）：                     │
│         - MCP tool 有 outputSchema → 使用                │
│         - Config 中有 output_schema → 使用               │
│         - 都没有 → 标记为无 schema，仍可使用             │
│      c. PTC tool 引用的 eligible tools 自动取消 deferred  │
│      d. 调用 make_ptc_tool(ptc_config, resolved_tools)   │
│      e. 加入最终 tool 列表                               │
│                                                          │
│ 3. 返回完整 tool 列表（含 PTC tools）                    │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 运行时 — LLM 选择并调用某个 PTC tool                      │
│                                                          │
│ AIMessage(tool_calls=[{                                  │
│   "name": "analyze_alert_correlation",                   │
│   "args": {"code": "...python code..."}                  │
│ }])                                                      │
│                                                          │
│ LLM 只看到与该目的相关的 tools 和 purpose 描述，          │
│ 不会混淆无关 tools                                       │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ PTC tool 执行                                            │
│                                                          │
│ 1. 构建 namespace:                                       │
│    - tool wrappers: {get_alert_metrics, query_log_...}   │
│    - safe modules: {json, re, math, ...}                 │
│    - restricted builtins                                 │
│                                                          │
│ 2. 注入 runtime 到 wrappers                              │
│                                                          │
│ 3. redirect_stdout → StringIO                            │
│                                                          │
│ 4. exec(code, namespace) with timeout                    │
│                                                          │
│ 5. 返回 stdout 内容（截断保护）                           │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ ToolMessage(content="处理后的摘要")                        │
│                                                          │
│ → 只有 print() 内容进入 model context                     │
│ → tool 原始返回值在代码中消费后丢弃                        │
└──────────────────────────────────────────────────────────┘
```

## File Changes

### 1. `deer-flow/backend/packages/harness/deerflow/config/tool_config.py`

**删除 `ToolConfig.ptc_eligible`**（因为不再用原方案），新增 `PTCToolConfig` 和 `PTCEligibleToolConfig`：

```python
from pydantic import BaseModel, Field, field_validator


class PTCEligibleToolConfig(BaseModel):
    name: str
    output_schema: dict | None = None  # 可选：当 MCP tool 未声明 outputSchema 时使用


class PTCToolConfig(BaseModel):
    name: str
    purpose: str                       # 目的描述，进入 tool description
    eligible_tools: list[PTCEligibleToolConfig] = Field(..., min_length=1)
    timeout_seconds: int | None = Field(
        default=None, ge=1,
        description="Max wall time for this PTC tool. None → use module default (30s).",
    )
    max_output_chars: int | None = Field(
        default=None, ge=100,
        description="Max stdout chars returned. None → use module default (20000).",
    )

    @field_validator("eligible_tools")
    @classmethod
    def _no_empty_eligible(cls, v):
        if not v:
            raise ValueError("eligible_tools must contain at least one tool")
        return v


class ToolConfig(BaseModel):
    name: str
    group: str
    use: str
    # ptc_eligible 字段已删除
```

### 2. `deer-flow/backend/packages/harness/deerflow/config/app_config.py`

新增 `ptc_tools` 字段到 `AppConfig`：

```python
from deerflow.config.tool_config import PTCToolConfig

class AppConfig(BaseModel):
    # ... 现有字段 ...
    ptc_tools: list[PTCToolConfig] = Field(default_factory=list, description="Purpose-scoped PTC tools")
```

### 3. `deer-flow/backend/packages/harness/deerflow/tools/tools.py`

`get_available_tools()` 改为遍历 `config.ptc_tools`：

```python
def get_available_tools(...) -> list[BaseTool]:
    tools = []
    tool_registry: dict[str, BaseTool] = {}

    for tool_config in config.tools:
        tool = resolve_variable(tool_config.use, BaseTool)
        tools.append(tool)
        tool_registry[tool_config.name] = tool

    # ... 其他 tools (built-in, MCP, ACP, subagent) ...
    # MCP tools 也加入 registry
    for mcp_tool in mcp_tools:
        tool_registry[mcp_tool.name] = mcp_tool

    # 校验并构建 PTC tools
    used_names = set(tool_registry.keys())  # 所有常规 tool 名字
    used_names.update(t.name for t in tools if t not in tool_registry.values())  # built-in 等

    for ptc_config in getattr(config, "ptc_tools", []):
        # Name collision check:PTC tool 名字不能与常规 tool 重复
        if ptc_config.name in used_names:
            raise ValueError(
                f"PTC tool name '{ptc_config.name}' collides with an existing tool. "
                f"Rename the PTC tool to something distinct (e.g. 'ptc_{ptc_config.name}')."
            )

        resolved = _resolve_ptc_eligible_tools(ptc_config, tool_registry)
        if resolved is None:
            continue  # eligible tool 找不到，跳过该 PTC tool 注册

        ptc_tool = make_ptc_tool(ptc_config, resolved)
        tools.append(ptc_tool)
        used_names.add(ptc_config.name)

    return tools


def _resolve_ptc_eligible_tools(
    ptc_config: PTCToolConfig,
    tool_registry: dict[str, BaseTool],
) -> list[tuple[BaseTool, dict | None]] | None:
    """解析 PTC tool 的 eligible_tools，返回 (tool, output_schema) 列表。

    Output schema 优先级：
    1. MCP tool 自身的 outputSchema
    2. Config 中配置的 output_schema
    3. 都没有 → output_schema = None（仍可使用）

    如果某个 eligible tool 找不到，返回 None（跳过整个 PTC tool）。
    """
    resolved = []

    for eligible in ptc_config.eligible_tools:
        tool = tool_registry.get(eligible.name)
        if tool is None:
            logger.warning(f"PTC tool '{ptc_config.name}': "
                           f"eligible tool '{eligible.name}' not found, skipping PTC tool")
            return None

        # 优先使用 MCP tool 自身的 outputSchema
        mcp_schema = _extract_output_schema(tool)
        if mcp_schema is not None:
            resolved.append((tool, mcp_schema))
        elif eligible.output_schema is not None:
            resolved.append((tool, eligible.output_schema))
        else:
            # 没有 schema，仍然可以用，output_schema = None
            resolved.append((tool, None))

        # PTC 引用的 tool 自动取消 deferred
        # (确保 LLM 在 tool list 中已看到其完整 input schema)

    return resolved
```

### 4. `deer-flow/backend/packages/harness/deerflow/sandbox/ptc.py`

新增模块，包含 `make_ptc_tool()` 工厂函数及辅助函数：

#### 4.1 工厂函数

```python
def make_ptc_tool(
    ptc_config: PTCToolConfig,
    resolved_tools: list[tuple[BaseTool, dict | None]],
) -> BaseTool:
    """根据 config 和已解析的 eligible tools 动态构建 purpose-scoped PTC tool。"""

    func_docs = _build_function_docs(resolved_tools)
    tool_wrappers = _build_tool_wrappers(resolved_tools)

    tool_name = ptc_config.name
    purpose = ptc_config.purpose

    # 每个 PTC tool 可覆盖默认 timeout / output 上限
    timeout = ptc_config.timeout_seconds or _DEFAULT_TIMEOUT
    max_output = ptc_config.max_output_chars or _DEFAULT_MAX_OUTPUT

    # 动态构建 docstring
    description = f"""{purpose}

Write Python code that calls the available functions and processes results.
Only print() output is returned to your context.

{func_docs}

Pre-imported: json, re, math, collections, itertools, functools, datetime
Use print() to output anything you want to see.

The `code` argument is the Python source to execute.
"""

    @tool(tool_name)
    def ptc_tool(runtime: ToolRuntime[ContextT, ThreadState], code: str) -> str:
        """Execute Python code programmatically."""
        return _execute_code(
            code,
            tool_wrappers,
            runtime,
            timeout=timeout,
            max_output_chars=max_output,
        )

    # 覆盖自动生成的 description
    ptc_tool.description = description
    return ptc_tool
```

#### 4.2 函数签名生成

```python
def _build_function_docs(resolved_tools: list[tuple[BaseTool, dict | None]]) -> str:
    """从 resolved tools 生成简洁的函数签名列表。"""
    if not resolved_tools:
        return "No tools are currently exposed to this PTC tool."

    import json as _json

    docs = []
    skip_params = {"runtime", "description"}

    for t, output_schema in resolved_tools:
        params = []
        if t.args_schema is not None:
            for name, field in t.args_schema.model_fields.items():
                if name in skip_params:
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

        sig = f"{t.name}({', '.join(params)}) -> {return_type}"
        desc = t.description.split("\n")[0]
        docs.append(f"- {sig}\n  {desc}{schema_hint}")

    return (
        "Available functions (full input schemas in the tool list above):\n\n"
        + "\n\n".join(docs)
    )
```

#### 4.3 Tool wrappers

```python
def _build_tool_wrappers(resolved_tools: list[tuple[BaseTool, dict | None]]) -> dict[str, callable]:
    """为每个 eligible tool 构建 Python callable wrapper。"""
    wrappers: dict[str, Any] = {}

    for t, _output_schema in resolved_tools:
        accepts_description = (
            t.args_schema is not None
            and "description" in t.args_schema.model_fields
        )

        def _make_wrapper(tool_ref=t, needs_desc=accepts_description):
            def wrapper(**kwargs):
                if needs_desc and "description" not in kwargs:
                    kwargs["description"] = "called from ptc tool"
                result = _invoke_tool_with_runtime(tool_ref, kwargs, wrapper._runtime)
                return _extract_structured_content(result)
            wrapper._runtime = None
            return wrapper

        wrappers[t.name] = _make_wrapper()

    return wrappers
```

#### 4.4 代码执行

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
    """在受限 namespace 中执行 LLM 生成的 Python 代码。"""

    # 注入 runtime 到每个 wrapper
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

    # SIGALRM 是 Unix-only，Windows 需要替代方案
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

#### 4.5 Tool 调用与结构化内容提取

```python
def _invoke_tool_with_runtime(tool_ref: BaseTool, kwargs: dict, runtime) -> object:
    """以正确的方式调用一个 LangChain @tool，将 runtime 透传进去。

    使用 `.func()` 绕过 pydantic ToolRuntime 验证，这样可以传 duck-typed runtime。
    这与 tests/test_sandbox_tools_security.py 中的模式一致。
    """
    return tool_ref.func(runtime=runtime, **kwargs)


def _extract_structured_content(tool_result: object) -> Any:
    """从 MCP tool 返回值中提取结构化内容。

    MCP tools via langchain-mcp-adapters 返回一个 `(content, artifact)` tuple。
    如果 artifact 有 `structured_content` key，返回它。
    否则返回原始结果。
    """
    if isinstance(tool_result, tuple) and len(tool_result) == 2:
        content, artifact = tool_result
        if isinstance(artifact, dict):
            structured = artifact.get("structured_content")
            if structured is not None:
                return structured
        return content
    return tool_result
```

#### 4.6 Output schema 提取

```python
def _extract_output_schema(tool: BaseTool) -> dict | None:
    """从 LangChain BaseTool 提取 MCP outputSchema。

    **当前状态 (2026-04-13):**
    langchain-mcp-adapters 0.2.2 不暴露 outputSchema(源码中零引用)。
    所以这个函数目前实际总是返回 None。

    保留这个函数是为了 forward compatibility:未来 adapter 版本
    如果支持 MCP 2025-06 的 outputSchema,只需修改这一个函数,
    其他代码无需改动。实现时如果在 tool 上发现了 outputSchema,
    可能在 metadata、response_format_schema 或其他字段,需要重新验证位置。
    """
    metadata = getattr(tool, "metadata", None) or {}
    return metadata.get("outputSchema")
```

#### 4.7 安全约束

```python
def _restricted_builtins() -> dict:
    """允许安全的内置函数，排除 __import__, eval, exec, open 等。"""
    import builtins
    allowed = {
        'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
        'sorted', 'reversed', 'min', 'max', 'sum', 'any', 'all',
        'abs', 'round', 'divmod', 'pow',
        'isinstance', 'issubclass', 'type', 'callable', 'hasattr', 'getattr',
        'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple', 'bytes',
        'frozenset', 'complex',
        'None', 'True', 'False',
        'repr', 'format', 'chr', 'ord',
        'StopIteration', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
        'AttributeError', 'RuntimeError', 'Exception',
    }
    return {k: getattr(builtins, k) for k in allowed if hasattr(builtins, k)}


def _safe_modules() -> dict:
    """预导入的安全标准库模块。"""
    import collections
    import datetime
    import functools
    import itertools
    import json
    import math
    import re

    return {
        'json': json,
        're': re,
        'math': math,
        'collections': collections,
        'itertools': itertools,
        'functools': functools,
        'datetime': datetime,
    }
```

### 5. `deer-flow/config.yaml`

**删除原 `ptc_eligible` 标记**，改为顶层 `ptc_tools` 声明：

```yaml
tools:
  - name: bash
    group: sandbox
    use: deerflow.sandbox.tools:bash_tool
    # ptc_eligible 已删除

  - name: read_file
    group: sandbox
    use: deerflow.sandbox.tools:read_file_tool

  - name: ask_clarification
    group: builtin
    use: deerflow.tools.builtin:ask_clarification_tool

  # MCP tools 正常注册
  # ...

# Purpose-scoped PTC tools
ptc_tools:
  - name: analyze_alert_correlation
    purpose: >
      Use this when you need to correlate multiple alerts across time
      and dimensions. Fetches metrics for N alerts in parallel, finds
      shared tags / time windows, and returns a structured correlation
      summary. Preferred over calling get_alert_metrics repeatedly.
    timeout_seconds: 60     # 覆盖默认 30s(可选)
    max_output_chars: 40000 # 覆盖默认 20000(可选)
    eligible_tools:
      - name: get_alert_metrics
        # MCP tool 自身有 outputSchema → 自动使用
      - name: query_log_cluster
        # 手动补充 schema(可选,当 MCP tool 未声明 outputSchema 时)
        output_schema:
          type: object
          properties:
            cluster_id: { type: string }
            log_count: { type: integer }
            sample_logs: { type: array, items: { type: string } }
          required: [cluster_id, log_count]

  - name: incident_timeline_replay
    purpose: >
      Use this when you need to reconstruct a chronological timeline
      of an incident from tickets, alerts, and operation logs.
      Returns a structured timeline sorted by event time.
    eligible_tools:
      - name: search_incident
      - name: get_alert_metrics
      - name: query_operation_log
```

### Writing a good `purpose` field

The `purpose` becomes the PTC tool's `description` field — LLM reads it to decide
whether to call this tool. So write it as a **trigger + scope** description, not
as a feature list.

**Good (trigger-oriented):**
```yaml
purpose: >
  Use this when you need to correlate multiple alerts across time.
  Fetches metrics in parallel and returns a correlation summary.
```

**Bad (feature-list):**
```yaml
purpose: >
  Analyzes alerts. Can query metrics, logs, and incidents.
```

The good version tells LLM **when to pick this tool** and **what it returns**.
The bad version lists features without telling LLM the triggering condition,
which leads to either over-triggering or never-triggering.

## MCP Output Schema Handling

### 协议背景

MCP 2025-06-18 规范中，tool 调用返回 `CallToolResult`：

```
CallToolResult {
    content: ContentItem[]       // 非结构化内容（text/image/audio/resource_link/resource）
    structuredContent?: object   // 结构化内容（仅当 tool 声明了 outputSchema 时存在）
    isError: boolean
}
```

- `content`：必选，向后兼容路径。即使有 `structuredContent` 也会在这里放一份 `TextContent`（JSON 序列化）
- `structuredContent`：可选，只有 tool 声明了 `outputSchema` 时才出现，是已解析的 JSON 对象
- `isError`：标记是否为错误

Tool 定义中的 `outputSchema` 是可选的 JSON Schema，声明了 output schema 的 tool **必须**在返回值中提供 `structuredContent`。

### 新方案的三种路径

| 场景 | Wrapper 行为 | LLM 负担 | Context 成本 |
|------|-------------|---------|-------------|
| MCP tool 有 `outputSchema` | 返回 `structuredContent`（parsed dict/list），signature 显示完整 schema | 零——代码里直接访问字段 | 初始化时 schema 进入 PTC tool description 一次 |
| MCP tool 无 `outputSchema`，config 中有 `output_schema` | 返回 `content` 中的 TextContent（`json.loads()` 解析），signature 显示 config 声明的 schema | 零——schema 由人补充 | 同上 |
| 都没有 | 返回原始值（str / dict / tuple），signature 显示 `-> Any` | LLM 自己写代码处理 | 无额外成本 |

### Adapter 兼容性

deer-flow 通过 `langchain-mcp-adapters` 的 `MultiServerMCPClient` 加载 MCP tools。
`_extract_output_schema()` 和 `_extract_structured_content()` 屏蔽差异。

**当前状态（2026-04-13,langchain-mcp-adapters 0.2.2）:**

- ❌ Adapter **不暴露** MCP `outputSchema`。源码中零引用。
  → `_extract_output_schema()` 目前总是返回 `None`
  → 实际路径只有两种生效：① config 中 `output_schema` 显式配置,或 ② 返回 `-> Any`
- ✅ Adapter 返回 `(content, artifact)` tuple(StructuredTool `response_format="content_and_artifact"`)
  → `_extract_structured_content()` 处理 tuple 提取 `artifact["structured_content"]`

**保留 `_extract_output_schema()` 的原因:** 当未来 adapter 版本支持 MCP 2025-06
的 `outputSchema` 时,只需修改 `_extract_output_schema()` 一个函数,其他代码无需改动。
这是 forward-compatible stub。

## Security Model

安全模型为 **受限 namespace + tool 自身边界 + 超时**，不做强沙箱隔离。

| 层 | 机制 | 说明 |
|---|------|------|
| Namespace | `_restricted_builtins()` | 排除 `__import__`, `eval`, `exec`, `open`, `os`, `subprocess` |
| Namespace | `_safe_modules()` | 只预导入数据处理类标准库 |
| Tool 边界 | Tool 内部校验 | 路径校验、权限检查等在 wrapper 调用时仍然生效 |
| 超时 | `signal.SIGALRM` | 默认 30 秒，防止死循环 |
| 输出截断 | `_MAX_OUTPUT_CHARS` | 默认 20000，与 `bash_tool` 对齐 |

**重要澄清：**

- **PTC 不在沙箱容器中执行**：`exec()` 在 DeerFlow 服务进程内运行，不是在沙箱容器中。
- 安全水位与当前 `LocalSandboxProvider + allow_host_bash: true` 一致。
- `SandboxAuditMiddleware` 不适用（它审计 bash 命令，不审计 exec 代码），如需审计可后续扩展。

## Constraints

- **名字不冲突**:PTC tool 的 `name` 不能与任何常规 tool(config.tools、built-in、MCP、ACP、subagent)重名。启动时校验,冲突直接 `ValueError` 快速失败(不静默覆盖)
- **eligible_tools 不可为空**:`PTCToolConfig` pydantic 验证阶段拒绝空列表(`min_length=1`),启动时失败而不是注册一个无用的 PTC tool
- **eligible tool 缺失时跳过整个 PTC tool**:如果 `eligible_tools` 中某个 tool 在 tool registry 中找不到(比如 MCP server 未启用),`_resolve_ptc_eligible_tools` 返回 `None`,整个 PTC tool 跳过注册,日志 warning。不影响其他 PTC tool
- **PTC tool 引用的 eligible tools 自动取消 deferred**,确保 LLM 在 tool list 中已看到完整 input schema
- **PTC tool 不可嵌套调用**:namespace 中不包含任何 PTC tool 自身或其他 PTC tools(wrapper 只包含 eligible_tools)
- **所有 tool 都可以进 PTC**:不强制 output schema。没有 schema 时 LLM 处理原始返回值(先跑起来再说)
- **超时 / 输出上限可按 PTC tool 配置**:`PTCToolConfig.timeout_seconds` 和 `PTCToolConfig.max_output_chars` 覆盖模块默认值(30s / 20000 chars)。未配置时使用默认
- **Unix-only 超时**:使用 `SIGALRM`。Windows 需要替代方案(threading timeout)
- **Tool wrapper 为同步调用**;如有 async-only 的 tool 需另行处理
- **Tool 调用方式**(`_invoke_tool_with_runtime`)使用 `tool.func(runtime=..., **kwargs)` 绕过 pydantic ToolRuntime 验证,传 duck-typed runtime(Task 0 已验证)
- **MCP outputSchema 当前不可得**:`langchain-mcp-adapters 0.2.2` 不暴露该字段,`_extract_output_schema()` 作为 forward-compatible stub 保留,未来 adapter 支持后只需改这一个函数

## Testing Strategy

1. **单元测试**
   - `_build_function_docs()`：生成包含完整 output schema 的函数签名（或 `-> Any`）
   - `_build_tool_wrappers()`：wrapper 返回结构化内容或原始值
   - `_resolve_ptc_eligible_tools()`：三种 output schema 路径（MCP 有 / config 有 / 都没有）
   - `_extract_output_schema()`：从 LangChain BaseTool 提取 MCP outputSchema
   - `_extract_structured_content()`：处理 dict/string/CallToolResult 三种输入
   - `_restricted_builtins()` / `_safe_modules()`：白名单正确
2. **集成测试**
   - `make_ptc_tool()` 构建 + 简单代码执行
   - MCP tool 有 outputSchema 时 wrapper 返回 structuredContent
   - MCP tool 无 outputSchema 时 wrapper 返回原始值
   - eligible tool 找不到时 PTC tool 跳过注册
3. **E2E 测试**
   - 完整 agent 对话，LLM 选择正确的 PTC tool 并成功批处理
   - 多个 PTC tools 场景下 LLM 根据 purpose 选择正确的 tool
   - 验证 token 消耗相比传统 tool calling 显著降低
4. **安全测试**：验证受限 namespace 拦截危险操作（`import os`, `open()`, `exec()` 等）
5. **边界测试**：超时、输出截断、异常处理、tool 找不到

## Migration from Original PTC — Clean Slate

**策略:完全清除原方案代码,从零重建。** 不做渐进重构,因为:

- 原方案的 `code_execution` 单一工具语义与新方案的 purpose-scoped PTC tools 根本不同
- 保留原代码会造成两套并存的困惑
- 底层 helper(如 `_execute_code`)的签名和依赖都会改变
- 新方案的测试覆盖点与原方案不同,原测试需要完全重写

### Step 1 — 删除原方案文件(git rm)

```bash
git rm deer-flow/backend/packages/harness/deerflow/sandbox/code_execution.py
git rm deer-flow/backend/packages/harness/deerflow/config/ptc_config.py
git rm deer-flow/backend/tests/test_code_execution_tool.py
git rm deer-flow/backend/tests/test_ptc_config.py
git rm deer-flow/backend/tests/test_ptc_integration.py
```

### Step 2 — 回退对现有文件的 PTC 相关修改

**`deer-flow/backend/packages/harness/deerflow/config/tool_config.py`**
- 删除 `ToolConfig.ptc_eligible` 字段

**`deer-flow/backend/packages/harness/deerflow/config/app_config.py`**
- 删除 `from deerflow.config.ptc_config import PtcConfig`
- 删除 `ptc: PtcConfig = Field(default_factory=PtcConfig)` 字段

**`deer-flow/backend/packages/harness/deerflow/tools/tools.py`**
- 删除 `_collect_ptc_eligible_tools()` 辅助函数
- 删除 `get_available_tools()` 末尾的 PTC 注册块
- 删除 tool_search 分支中的 `ptc_claims_mcp` 判断(恢复原版本)

**`deer-flow/backend/tests/test_tool_config.py`**
- 删除测试 `test_tool_config_defaults_ptc_eligible_to_false`
- 删除测试 `test_tool_config_accepts_ptc_eligible_true`
- 删除测试 `test_tool_config_rejects_invalid_ptc_eligible_type`
- 若这些是文件中唯一测试,删除整个文件

### Step 3 — 新增新方案文件

**新文件:**
- `deer-flow/backend/packages/harness/deerflow/sandbox/ptc.py`
  - `_restricted_builtins()` / `_safe_modules()`
  - `_execute_code()`(支持 timeout / max_output_chars 参数)
  - `_extract_output_schema()`(forward-compatible stub,永远返回 None)
  - `_extract_structured_content()`
  - `_invoke_tool_with_runtime()`
  - `_build_function_docs()`
  - `_build_tool_wrappers()`
  - `make_ptc_tool(ptc_config, resolved_tools)`

- `deer-flow/backend/tests/test_ptc.py`
  - 单元测试上述所有 helper
  - 集成测试 `make_ptc_tool()` 构建
  - 包含 Task 1 的 MCP adapter 发现作为注释

- `deer-flow/backend/tests/test_ptc_registration.py`
  - 测试 `get_available_tools()` 中的 PTC tool 注册
  - Name collision 校验测试
  - Missing eligible tool 跳过测试
  - Empty eligible_tools 拒绝测试(pydantic 层)

### Step 4 — 修改现有文件

**`deer-flow/backend/packages/harness/deerflow/config/tool_config.py`**
- 新增 `PTCEligibleToolConfig` 类
- 新增 `PTCToolConfig` 类(带 `min_length=1` 验证、`timeout_seconds` / `max_output_chars` 可选字段)

**`deer-flow/backend/packages/harness/deerflow/config/app_config.py`**
- 新增 `from deerflow.config.tool_config import PTCToolConfig`
- 新增 `ptc_tools: list[PTCToolConfig] = Field(default_factory=list)` 字段

**`deer-flow/backend/packages/harness/deerflow/tools/tools.py`**
- 新增 PTC tool 注册块(含 name collision 校验、_resolve_ptc_eligible_tools 调用)
- 在 MCP tools 加入 registry 后触发 PTC 构建
- PTC tools 引用的 eligible tools 自动取消 deferred(影响 tool_search 分支)

**`deer-flow/backend/CLAUDE.md`**
- 更新 Tool System 段落说明 purpose-scoped PTC

### 不变内容

- 执行环境:In-process `exec()`
- 安全模型:受限 namespace + 超时 + 截断
- Tool wrapper 隐藏 runtime 参数的模式
- Task 0 / Task 1 的 investigation commits 保留作为历史记录

### Git 状态现状

当前 branch `feat/purpose-scoped-ptc` 已有一个 unmerged 的 `deleted by us: deer-flow/backend/tests/test_ptc_integration.py`。
实施时需要先 `git add` 该删除以完成 merge,然后按 Step 1-4 进行清理和重建。
