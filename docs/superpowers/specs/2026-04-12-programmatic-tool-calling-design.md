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
| Schema 传递 | 依赖 tool list 中已有的 schema | ptc_eligible tools 同时也是普通 tools，LLM 已在 tool list 中看到完整 schema，code_execution description 只需列函数名 |
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

    LLM 已在 tool list 中看到完整 schema，此处只需列出
    函数名和参数名作为代码编写参考。
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

        sig = f"{t.name}({', '.join(params)}) -> str"
        desc = t.description.split('\n')[0]
        docs.append(f"- {sig}\n  {desc}")

    return "Available functions (refer to tool schemas for full details):\n\n" + "\n\n".join(docs)
```

#### 3.3 Tool wrappers

```python
def _build_tool_wrappers(eligible_tools: list[BaseTool]) -> dict[str, callable]:
    """为每个 eligible tool 构建 Python callable wrapper。

    隐藏 runtime 和 description 参数，
    只暴露业务参数给 LLM 生成的代码。
    """
    wrappers = {}

    for t in eligible_tools:
        # 检查 tool 是否需要 description 参数
        accepts_description = "description" in t.args_schema.model_fields

        def _make_wrapper(tool_ref, needs_desc=accepts_description):
            def wrapper(**kwargs):
                if needs_desc and "description" not in kwargs:
                    kwargs["description"] = "called from code_execution"
                # 通过 LangChain 的 invoke 接口调用，
                # runtime 由 LangChain 框架按 ToolRuntime 注入机制传递
                return _invoke_tool_with_runtime(tool_ref, kwargs, wrapper._runtime)
            wrapper._runtime = None  # 执行时由 _execute_code 注入
            return wrapper

        wrappers[t.name] = _make_wrapper(t)

    return wrappers


def _invoke_tool_with_runtime(tool_ref: BaseTool, kwargs: dict, runtime) -> str:
    """以正确的方式调用一个 LangChain @tool，将 runtime 透传进去。

    实现注意事项：
    - LangChain `@tool` 装饰器包装了底层函数，直接 `_run(**kwargs)`
      在某些版本不能正确传递 runtime
    - 推荐通过 `tool_ref.invoke({**kwargs}, config={"configurable": {...}})`
      或直接调用 tool 内部的 `func`/`coroutine` 属性
    - 实现阶段需在 deer-flow 当前 LangChain 版本下验证最稳定的调用方式
    """
    # 占位：实现时根据 LangChain 版本确定最佳调用方式
    return tool_ref._run(**kwargs, runtime=runtime)
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

- `ptc_eligible` 与 `deferred` 互斥：标记 `ptc_eligible: true` 的 tool 自动取消 deferred，确保 LLM 在 tool list 中已看到其完整 schema
- `code_execution` tool 本身不可嵌套调用（namespace 中不包含自身）
- 超时使用 `SIGALRM`，仅限 Unix 系统；Windows 需要替代方案（threading timeout）
- Tool wrapper 为同步调用；如有 async-only 的 tool 需另行处理
- Tool 调用方式（`_invoke_tool_with_runtime`）需在实现阶段根据 deer-flow 当前的 LangChain 版本确定最稳定的调用接口

## Testing Strategy

1. **单元测试**：`_build_function_docs()`, `_build_tool_wrappers()`, `_restricted_builtins()`, `_safe_modules()`
2. **集成测试**：`make_code_execution_tool()` 构建 + 简单代码执行
3. **E2E 测试**：完整 agent 对话，验证 LLM 选择 code_execution 并成功执行代码
4. **安全测试**：验证受限 namespace 拦截危险操作（`import os`, `open()`, `exec()` 等）
5. **边界测试**：超时、输出截断、异常处理
