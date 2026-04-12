# Programmatic Tool Calling (PTC) Design Spec

## Overview

在 DeerFlow 的 tool 层新增 `code_execution` tool，允许 LLM 生成 Python 代码在进程内执行，代码中可调用其他已注册 tools。Tool 返回值留在代码作用域内处理，只有 `print()` 输出进入 model context，从而大幅减少 token 消耗。

## Motivation

传统 tool calling 中，每次 tool 返回的原始数据全部进入 model context。当需要批量调用 tools（如遍历 N 个实体分别查询）时：
- 每次结果都膨胀 context window
- 多轮 round-trip 增加延迟
- LLM 被迫在大量噪声数据中提取关键信息

PTC 让 LLM 写一段代码完成批量调用 + 数据处理，只将最终摘要返回，token 可减少 80%+。

## Design Decisions

| 决策 | 选择 | 理由 |
|------|------|------|
| 实现层级 | deer-flow tool 层（方案 2） | 自然适配 tool 加载流程，deer-agents 通过 config 启用 |
| Tool 标记 | `ptc_eligible` 字段自声明 | 类似 Claude API 的 `allowed_callers`，每个 tool 声明自己是否可被代码调用 |
| 执行环境 | In-process `exec()` | 简单直接，tool wrapper 可直接调用，无需 IPC；安全水位与当前 `LocalSandboxProvider + allow_host_bash: true` 一致 |
| Input schema 传递 | 依赖 tool list 中已有的 schema | ptc_eligible tools 同时也是普通 tools，LLM 已在 tool list 中看到完整 input schema |
| Output schema 传递 | 遵循 MCP 2025-06-18 `outputSchema` / `structuredContent` | 声明了 output schema 的 tool 直接返回结构化对象，LLM 写代码无需猜字段；未声明的 tool LLM 需受控 probe 观察结构 |
| 与 deferred 的关系 | ptc_eligible 优先，自动取消 deferred | 两者互斥，系统自动解决冲突，不报错 |

## Data Flow

```
┌──────────────────────────────────────────────────────┐
│ Agent 初始化 — get_available_tools()                  │
│                                                      │
│ 1. 加载所有 tools (bash, grep, read_file, MCP...)    │
│ 2. ptc_eligible 的 tool 自动取消 deferred             │
│ 3. 筛选 ptc_eligible == true 的 tools                │
│ 4. 构建 code_execution_tool：                        │
│    - description: 函数名 + 一句话说明                 │
│    - 内部持有 eligible tools 的引用                   │
│ 5. 加入最终 tool 列表                                │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ 运行时 — LLM 调用 code_execution                     │
│                                                      │
│ AIMessage(tool_calls=[{                              │
│   "name": "code_execution",                          │
│   "args": {"code": "...python code..."}              │
│ }])                                                  │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ code_execution_tool 执行                              │
│                                                      │
│ 1. 构建 namespace:                                   │
│    - tool wrappers: {bash, read_file, grep, ...}     │
│    - safe modules: {json, re, math, ...}             │
│    - restricted builtins                             │
│                                                      │
│ 2. 注入 runtime 到 wrappers（sandbox 等上下文）       │
│                                                      │
│ 3. redirect_stdout → StringIO                        │
│                                                      │
│ 4. exec(code, namespace) with timeout                │
│                                                      │
│ 5. 返回 stdout 内容（截断保护）                       │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ ToolMessage(content="处理后的摘要")                    │
│                                                      │
│ → 只有 print() 内容进入 model context                 │
│ → tool 原始返回值在代码中消费后丢弃                    │
└──────────────────────────────────────────────────────┘
```

## File Changes

### 1. `deer-flow/backend/packages/harness/deerflow/config/tool_config.py`

`ToolConfig` 新增 `ptc_eligible` 字段：

```python
class ToolConfig(BaseModel):
    name: str
    group: str
    use: str
    ptc_eligible: bool = False  # 是否可在 code_execution 中被调用
```

### 2. `deer-flow/backend/packages/harness/deerflow/tools/tools.py`

`get_available_tools()` 末尾新增逻辑：

```python
def get_available_tools(...) -> list[BaseTool]:
    tools = []
    ptc_eligible_tools = []

    for tool_config in config.tools:
        # ptc_eligible 自动取消 deferred
        if tool_config.ptc_eligible:
            tool_config.deferred = False

        tool = resolve_variable(tool_config.use, BaseTool)
        tools.append(tool)

        if tool_config.ptc_eligible:
            ptc_eligible_tools.append(tool)

    # ... 其他 tools (built-in, MCP, ACP, subagent) ...

    # 有 eligible tools 时构建 code_execution
    if ptc_eligible_tools:
        from deerflow.sandbox.tools import make_code_execution_tool
        tools.append(make_code_execution_tool(ptc_eligible_tools))

    return tools
```

### 3. `deer-flow/backend/packages/harness/deerflow/sandbox/tools.py`

新增 `make_code_execution_tool()` 工厂函数及辅助函数：

#### 3.1 工厂函数

```python
def make_code_execution_tool(eligible_tools: list[BaseTool]) -> BaseTool:
    """根据 eligible tools 动态构建 code_execution tool。"""

    func_docs = _build_function_docs(eligible_tools)
    tool_wrappers = _build_tool_wrappers(eligible_tools)

    @tool("code_execution", parse_docstring=True)
    def code_execution_tool(runtime: ToolRuntime[ContextT, ThreadState], code: str) -> str:
        """Execute Python code with programmatic tool access.

        Write Python code that calls tools and processes results in-code.
        Only print() output is returned to your context.
        Use this when you need to batch-call tools and filter/aggregate
        large results to reduce context usage.

        {func_docs}

        Pre-imported: json, re, math, collections, itertools, functools
        """
        return _execute_code(code, tool_wrappers, runtime)

    return code_execution_tool
```

#### 3.2 函数签名生成

```python
def _build_function_docs(eligible_tools: list[BaseTool]) -> str:
    """从 eligible tools 生成简洁的函数签名列表。

    LLM 已在 tool list 中看到完整 input schema，此处只需列出
    函数名、参数名、和 output 处理提示。
    """
    docs = []
    skip_params = {"runtime", "description"}

    for t in eligible_tools:
        params = []
        for name, field in t.args_schema.model_fields.items():
            if name in skip_params:
                continue
            type_name = (field.annotation.__name__
                         if hasattr(field.annotation, '__name__')
                         else str(field.annotation))
            default = f" = {field.default!r}" if field.default is not None else ""
            params.append(f"{name}: {type_name}{default}")

        # 检查是否有 MCP output schema
        output_schema = _extract_output_schema(t)
        if output_schema is not None:
            # 有 schema：wrapper 返回 parsed dict/list
            return_type = "dict | list"
            schema_hint = f"\n  Returns structured data. Schema: {json.dumps(output_schema, ensure_ascii=False)}"
        else:
            # 无 schema：wrapper 返回 JSON string 或 plain string
            return_type = "str"
            schema_hint = "\n  Returns str (may be JSON — use json.loads() if needed)"

        sig = f"{t.name}({', '.join(params)}) -> {return_type}"
        desc = t.description.split('\n')[0]
        docs.append(f"- {sig}\n  {desc}{schema_hint}")

    probe_hint = (
        "\n\nFor tools returning str without a schema, you can first probe "
        "the structure by printing a single sample:\n"
        "  sample = json.loads(tool_name(...))\n"
        "  print(json.dumps(sample[0] if isinstance(sample, list) else sample, indent=2))\n"
        "Then use the observed structure in subsequent processing."
    )

    return (
        "Available functions (refer to tool schemas for full input details):\n\n"
        + "\n\n".join(docs)
        + probe_hint
    )


def _extract_output_schema(tool: BaseTool) -> dict | None:
    """从 LangChain BaseTool 提取 MCP outputSchema。

    langchain-mcp-adapters 把原始 MCP Tool metadata 暴露在
    tool.metadata 或 tool.response_format 等位置，实现阶段
    需根据当前 adapter 版本确认字段名。

    Returns:
        dict: JSON Schema of the tool's output structure
        None: tool 没有声明 outputSchema
    """
    # 实现占位——adapter 版本待验证
    # 候选路径：
    # - tool.metadata.get("outputSchema")
    # - tool.response_format
    # - tool._tool.outputSchema (mcp-adapters 内部字段)
    metadata = getattr(tool, "metadata", None) or {}
    return metadata.get("outputSchema")
```

#### 3.3 Tool wrappers

```python
def _build_tool_wrappers(eligible_tools: list[BaseTool]) -> dict[str, callable]:
    """为每个 eligible tool 构建 Python callable wrapper。

    隐藏 runtime 和 description 参数，
    只暴露业务参数给 LLM 生成的代码。

    返回值处理：
    - 有 outputSchema 的 MCP tool：返回 structuredContent（parsed dict/list）
    - 无 outputSchema 的 MCP tool：返回 content text（JSON string 或 plain string）
    - 非 MCP tool：返回 tool 原始返回值（通常是 str）
    """
    wrappers = {}

    for t in eligible_tools:
        accepts_description = "description" in t.args_schema.model_fields
        has_output_schema = _extract_output_schema(t) is not None

        def _make_wrapper(tool_ref, needs_desc=accepts_description, structured=has_output_schema):
            def wrapper(**kwargs):
                if needs_desc and "description" not in kwargs:
                    kwargs["description"] = "called from code_execution"
                result = _invoke_tool_with_runtime(tool_ref, kwargs, wrapper._runtime)
                # 有 output schema 时，返回 structured content
                if structured:
                    return _extract_structured_content(result)
                return result
            wrapper._runtime = None  # 执行时由 _execute_code 注入
            return wrapper

        wrappers[t.name] = _make_wrapper(t)

    return wrappers


def _invoke_tool_with_runtime(tool_ref: BaseTool, kwargs: dict, runtime) -> object:
    """以正确的方式调用一个 LangChain @tool，将 runtime 透传进去。

    实现注意事项：
    - LangChain `@tool` 装饰器包装了底层函数，直接 `_run(**kwargs)`
      在某些版本不能正确传递 runtime
    - 推荐通过 `tool_ref.invoke({**kwargs}, config={"configurable": {...}})`
      或直接调用 tool 内部的 `func`/`coroutine` 属性
    - MCP tools 是 langchain-mcp-adapters 包装的，需要调用 adapter 的
      调用接口（可能是 sync_wrapper 或 coroutine）
    - 实现阶段需在 deer-flow 当前 LangChain 版本下验证最稳定的调用方式
    """
    # 占位：实现时根据 LangChain 版本和 MCP adapter 确定最佳调用方式
    return tool_ref._run(**kwargs, runtime=runtime)


def _extract_structured_content(tool_result: object) -> dict | list | object:
    """从 MCP tool 返回值中提取 structuredContent。

    MCP 2025-06-18 规范：声明了 outputSchema 的 tool，返回值包含
    `structuredContent` 字段（类型化对象）。为了向后兼容，`content` 里
    也会有 TextContent 包含同样内容的 JSON 字符串。

    langchain-mcp-adapters 暴露方式待验证：
    - 可能直接返回 structuredContent 的 dict
    - 可能返回完整 CallToolResult，需要提取 `.structuredContent`
    - 可能返回字符串（向后兼容路径），需要 json.loads()

    这个函数屏蔽差异，总是返回结构化对象。
    """
    # 实现占位
    if isinstance(tool_result, (dict, list)):
        return tool_result
    if isinstance(tool_result, str):
        try:
            return json.loads(tool_result)
        except json.JSONDecodeError:
            return tool_result
    if hasattr(tool_result, "structuredContent"):
        return tool_result.structuredContent
    return tool_result
```

#### 3.4 代码执行

```python
import contextlib
import io
import signal

_DEFAULT_TIMEOUT = 30
_MAX_OUTPUT_CHARS = 20000

def _execute_code(
    code: str,
    tool_wrappers: dict[str, callable],
    runtime: ToolRuntime,
    timeout: int = _DEFAULT_TIMEOUT,
) -> str:
    """在受限 namespace 中执行 LLM 生成的 Python 代码。"""

    # 注入 runtime 到每个 wrapper
    for wrapper in tool_wrappers.values():
        wrapper._runtime = runtime

    namespace = {
        **tool_wrappers,
        **_safe_modules(),
        "__builtins__": _restricted_builtins(),
    }

    stdout = io.StringIO()

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution exceeded {timeout}s limit")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, namespace)
    except TimeoutError as e:
        return f"Error: {e}"
    except Exception as e:
        import traceback
        return f"Code execution error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    output = stdout.getvalue() or "(no output)"

    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(output)} total chars)"

    return output
```

#### 3.5 安全约束

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
    import json, re, math, collections, itertools, functools, datetime
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

### 4. `deer-flow/config.yaml`

各 tool 声明 `ptc_eligible`：

```yaml
tools:
  - name: bash
    group: sandbox
    use: deerflow.sandbox.tools:bash_tool
    ptc_eligible: true

  - name: read_file
    group: sandbox
    use: deerflow.sandbox.tools:read_file_tool
    ptc_eligible: true

  - name: write_file
    group: sandbox
    use: deerflow.sandbox.tools:write_file_tool
    ptc_eligible: true

  - name: grep
    group: sandbox
    use: deerflow.sandbox.tools:grep_tool
    ptc_eligible: true

  - name: glob
    group: sandbox
    use: deerflow.sandbox.tools:glob_tool
    ptc_eligible: true

  - name: ls
    group: sandbox
    use: deerflow.sandbox.tools:ls_tool
    ptc_eligible: true

  # 交互类 tool 不标记
  - name: ask_clarification
    group: builtin
    use: deerflow.tools.builtin:ask_clarification_tool

  - name: present_file
    group: builtin
    use: deerflow.tools.builtin:present_file_tool
```

## MCP Output Schema Handling

遵循 MCP 2025-06-18 协议规范（[spec 链接](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)）。

### 协议背景

MCP tool 定义可选声明 `outputSchema`：

```json
{
  "name": "get_weather_data",
  "inputSchema": { "type": "object", "properties": {...} },
  "outputSchema": {
    "type": "object",
    "properties": {
      "temperature": {"type": "number"},
      "conditions": {"type": "string"}
    },
    "required": ["temperature", "conditions"]
  }
}
```

当 tool 声明了 `outputSchema`，返回值 `CallToolResult` 必须包含 `structuredContent` 字段（类型化 JSON 对象），同时为向后兼容也会在 `content[]` 里放一份序列化的 `TextContent`：

```json
{
  "result": {
    "content": [{"type": "text", "text": "{\"temperature\": 22.5, ...}"}],
    "structuredContent": {"temperature": 22.5, "conditions": "Partly cloudy"}
  }
}
```

### PTC 的三种路径

| 场景 | Wrapper 行为 | LLM 负担 | Context 成本 |
|------|-------------|---------|-------------|
| MCP tool 有 `outputSchema` | 返回 `structuredContent`（parsed dict/list），signature 显示完整 schema | 零——代码里直接访问字段 | 初始化时 schema 进入 `code_execution` description 一次 |
| MCP tool 无 `outputSchema` | 返回 TextContent 原文（string） | 需 `json.loads()` 并**受控 probe** 观察结构 | Probe 一次 ~200 token |
| 非 MCP tool（sandbox/builtin） | 返回 tool 原始返回值 | 按 signature 处理 | 零 |

### 受控 Probe 指引

对无 `outputSchema` 的 tool，`code_execution` description 会给 LLM 明确指引：

```
For tools returning str without a schema, you can first probe
the structure by printing a single sample:
  sample = json.loads(tool_name(...))
  print(json.dumps(sample[0] if isinstance(sample, list) else sample, indent=2))
Then use the observed structure in subsequent processing.
```

关键约束：**只 print 一个样本的结构，不 print 全部数据**。200 token 成本换取后续无限次调用的正确性。

### Adapter 兼容性

deer-flow 通过 `langchain-mcp-adapters` 的 `MultiServerMCPClient` 加载 MCP tools。需要在实现阶段验证两个点：

1. **outputSchema 暴露位置**：adapter 把 MCP Tool 原始 metadata（包括 `outputSchema`）暴露在 LangChain BaseTool 的哪个属性上（候选：`metadata`、`response_format`、内部字段）
2. **structuredContent 返回位置**：tool 调用结果是直接返回 structuredContent，还是返回完整 CallToolResult 需要提取

`_extract_output_schema()` 和 `_extract_structured_content()` 屏蔽这些差异。如果当前 adapter 版本尚未支持 `outputSchema`，需要作为前置任务向上游 PR 或本地 patch 修复。

## Security Model

安全模型为 **受限 namespace + tool 自身边界 + 超时**，不做强沙箱隔离。

| 层 | 机制 | 说明 |
|---|------|------|
| Namespace | `_restricted_builtins()` | 排除 `__import__`, `eval`, `exec`, `open`, `os`, `subprocess` |
| Namespace | `_safe_modules()` | 只预导入数据处理类标准库 |
| Tool 边界 | Tool 内部校验 | 路径校验、权限检查等在 wrapper 调用时仍然生效 |
| 超时 | `signal.SIGALRM` | 默认 30 秒，防止死循环 |
| 输出截断 | `_MAX_OUTPUT_CHARS` | 默认 20000，与 `bash_tool` 对齐 |

与当前 `LocalSandboxProvider` + `allow_host_bash: true` 安全水位一致。`SandboxAuditMiddleware` 不适用（它审计 bash 命令，不审计 exec 代码），如需审计可后续扩展。

## Constraints

- `ptc_eligible` 与 `deferred` 互斥：标记 `ptc_eligible: true` 的 tool 自动取消 deferred，确保 LLM 在 tool list 中已看到其完整 input schema
- `code_execution` tool 本身不可嵌套调用（namespace 中不包含自身）
- 超时使用 `SIGALRM`，仅限 Unix 系统；Windows 需要替代方案（threading timeout）
- Tool wrapper 为同步调用；如有 async-only 的 tool 需另行处理
- Tool 调用方式（`_invoke_tool_with_runtime`）需在实现阶段根据 deer-flow 当前的 LangChain 版本确定最稳定的调用接口
- MCP tool 的 `outputSchema` 和 `structuredContent` 暴露方式依赖 `langchain-mcp-adapters` 当前版本，实现阶段需验证字段路径；如不支持需向上游 PR 或本地 patch

## Testing Strategy

1. **单元测试**
   - `_build_function_docs()`：有/无 outputSchema 的 tool 分别生成正确签名
   - `_build_tool_wrappers()`：wrapper 正确隐藏 runtime 参数
   - `_extract_output_schema()`：从 LangChain BaseTool 提取 MCP outputSchema
   - `_extract_structured_content()`：处理 dict/string/CallToolResult 三种输入
   - `_restricted_builtins()` / `_safe_modules()`：白名单正确
2. **集成测试**
   - `make_code_execution_tool()` 构建 + 简单代码执行
   - MCP tool 有 outputSchema 时 wrapper 返回 structuredContent
   - MCP tool 无 outputSchema 时 wrapper 返回原始 text
3. **E2E 测试**
   - 完整 agent 对话，LLM 选择 code_execution 并成功批处理
   - 验证 token 消耗相比传统 tool calling 显著降低
4. **安全测试**：验证受限 namespace 拦截危险操作（`import os`, `open()`, `exec()` 等）
5. **边界测试**：超时、输出截断、异常处理
