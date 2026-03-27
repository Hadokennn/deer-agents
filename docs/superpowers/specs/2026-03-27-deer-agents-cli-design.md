# Deer Agents CLI — Design Spec

## Overview

基于 DeerFlow harness 构建的 CLI agent 系统。统一终端入口，多 agent 动态切换，每个 agent 拥有独立的 config、skill、tool、MCP 和 middleware 配置。第一个 agent 为业务 oncall 答疑助手。

**目标用户**: 作者本人（SWE），通过构建和使用来学习 agent harness 最佳实践。

**非目标**: 不做 Web UI、不做多用户、不做部署平台。

---

## 1. 项目结构

```
deer-agents/
├── deer-flow/                      ← fork (git subtree), upstream remote 跟踪
│   └── backend/packages/harness/   ← 可直接修改的 harness 代码
│
├── agents/                         ← agent 定义层，每个 agent 一个目录
│   └── oncall/
│       ├── agent.yaml              ← agent 级配置（模型、tools、MCP servers）
│       ├── prompt.md               ← 系统提示词（Markdown，支持变量插值）
│       ├── skills/                 ← oncall 专属 skill（SKILL.md 格式）
│       │   └── runbook-lookup/
│       │       └── SKILL.md
│       └── knowledge/              ← 本地 runbook/FAQ 文档
│           ├── architecture.md
│           └── common-issues.md
│
├── cli/
│   ├── __main__.py                 ← `python -m cli` 或 `deer` 入口
│   ├── shell.py                    ← 交互式 REPL（prompt_toolkit）
│   ├── commands.py                 ← /switch, /status, /sessions, /resume, /help 等
│   └── renderer.py                 ← 流式输出渲染（rich, Markdown → 终端）
│
├── middlewares/                     ← 自定义 middleware（注入到 deer-flow chain）
│   ├── __init__.py
│   └── mcp_overflow.py             ← MCP response 过大时写入沙箱
│
├── config.yaml                     ← 全局配置（共享模型、默认 sandbox 等）
├── pyproject.toml
└── README.md
```

### 关键决策

- **deer-flow 用 git subtree**（非 submodule）— 可直接在 deer-agents 仓库里修改 deer-flow 代码并提交，同时 `git subtree pull` 同步上游。
- **middlewares/ 放根级别** — middleware 是 harness 层概念，多 agent 可复用，不放 agent 目录内。
- **agents/ 每个 agent 一个目录** — agent 的全部定义（config、prompt、skill、knowledge）自包含在一个目录里。

---

## 2. CLI 交互层

### 启动与使用

```bash
$ deer                          # 启动，加载默认 agent
$ deer --agent oncall           # 启动，直接进入 oncall agent

# REPL 内
🦌 > 帮我查下 redis 连接超时的 runbook
...（流式回答）

🦌 > /switch review             # 切换 agent
✓ Switched to review agent

🦌 review > ...

🦌 > /agents                    # 列出可用 agent
  oncall    ● active
  review    ○
  learning  ○

🦌 > /sessions                  # 列出历史会话
  oncall   #thread-abc  3h ago   "redis 连接超时排查"
  oncall   #thread-def  1d ago   "支付回调异常"

🦌 > /resume thread-abc         # 恢复历史对话
✓ Resumed oncall session: "redis 连接超时排查"

🦌 > /status                    # 当前 agent 状态
🦌 > /help                      # 帮助
🦌 > /exit                      # 退出
```

### 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| REPL | `prompt_toolkit` | 自动补全、多行输入、历史记录 |
| 渲染 | `rich` | Markdown → 终端，代码高亮、表格 |
| 流式 | `DeerFlowClient.stream()` | 监听 `StreamEvent`，实时输出 |

### Agent 生命周期

```
deer 启动
  → 加载 config.yaml（全局配置）
  → 加载 agents/{name}/agent.yaml（merge 到全局配置）
  → 创建 DeerFlowClient(config=merged, agent_name=name, checkpointer=sqlite)
  → 进入 REPL 循环

/switch review
  → 创建新 DeerFlowClient(config=review_config)
  → 替换当前 client（旧 agent 的 thread 不销毁）

/resume thread-abc
  → 从 sessions 元数据查到 agent_name + thread_id
  → 创建 DeerFlowClient(agent_name=..., checkpointer=sqlite)
  → stream(thread_id="thread-abc") 自动恢复历史上下文
```

**关键点**: `/switch` 不是热切换，而是创建新的 `DeerFlowClient` 实例。每个 agent 有独立的 thread、sandbox、memory。简单可靠。

---

## 3. Session Resume

### 持久化方案

- **Checkpointer**: LangGraph 内置 `SqliteSaver`，存储在 `~/.deer-agents/checkpoints.db`
- **Session 元数据**: `~/.deer-agents/sessions/` 目录，每个 session 一个 JSON 文件

```json
{
  "thread_id": "thread-abc",
  "agent_name": "oncall",
  "title": "redis 连接超时排查",
  "created_at": "2026-03-27T10:30:00Z",
  "last_active_at": "2026-03-27T11:15:00Z"
}
```

- **thread_id 是 resume 的 key** — 同一个 thread_id + checkpointer 自动带上完整消息历史
- **Title 自动生成** — 复用 DeerFlow 的 `TitleMiddleware`

---

## 4. Agent 配置与合并

### 全局 config.yaml

```yaml
default_agent: oncall

models:
  - name: doubao-seed-1.8
    use: deerflow.models.patched_deepseek:PatchedChatDeepSeek
    model: doubao-seed-1-8-251228
    api_base: https://ark.cn-beijing.volces.com/api/v3
    api_key: $VOLCENGINE_API_KEY
    supports_thinking: true

sandbox:
  type: local

checkpointer:
  type: sqlite
  path: ~/.deer-agents/checkpoints.db

sessions:
  dir: ~/.deer-agents/sessions/
```

### Agent 级 agent.yaml（增量覆盖）

```yaml
name: oncall
display_name: "Oncall 答疑助手"
description: "业务 oncall 答疑，连接告警平台和 runbook"

model: doubao-seed-1.8
thinking_enabled: true

tool_groups:
  - web
  - file
  - bash

mcp_servers:
  - name: alert-platform
    command: npx
    args: ["alert-mcp-server", "--env=prod"]
  - name: log-search
    command: npx
    args: ["log-mcp-server"]

extra_middlewares:
  - use: middlewares.mcp_overflow:McpOverflowMiddleware
    config:
      max_response_size: 8192
      sandbox_path: /tmp/mcp_responses/

code_repos:
  - name: payment-service
    path: /Users/bytedance/code/payment-service
  - name: gateway
    path: /Users/bytedance/code/gateway

knowledge_dirs:
  - ./agents/oncall/knowledge/

skills_dir: ./agents/oncall/skills/

prompt: ./agents/oncall/prompt.md
```

### 合并规则

| 字段 | 策略 | 理由 |
|------|------|------|
| models, sandbox, checkpointer | agent 覆盖全局 | 基础设施通常共享 |
| mcp_servers | agent 独占，不继承 | 每个 agent 连不同的外部系统 |
| extra_middlewares | 追加到默认 chain | 不破坏 deer-flow 原有 middleware 顺序 |
| tool_groups | agent 独占 | 不同 agent 需要不同 tool 集 |
| code_repos, knowledge_dirs, skills_dir, prompt | agent 独占 | agent 特有资源 |

---

## 5. MCP Overflow Middleware

### 问题

MCP server 返回的 response 可能巨大（如日志查询返回 50KB+），直接塞入 messages 导致 context rot 和 token 浪费。

### 方案

```
MCP tool 返回 response
    → McpOverflowMiddleware 拦截
    → size > max_response_size (default 8KB)?
        No  → 原样通过
        Yes → 写入沙箱 /tmp/mcp_responses/{tool_call_id}.txt
            → 替换 ToolMessage 内容为:
              "Response too large (52KB), saved to /tmp/mcp_responses/xxx.txt
               Use read_file to inspect specific sections."
```

### Middleware Chain 位置

```
[SummarizationMiddleware]   order=500
[McpOverflowMiddleware]     order=550  ← 插入点
[TitleMiddleware]           order=600
```

在 Summarization 之后（summarization 压缩历史，McpOverflow 处理当前轮），在 Title 之前（title 生成不需要完整 MCP response）。

### 实现骨架

```python
from deerflow.agents.middlewares.base import BaseMiddleware

class McpOverflowMiddleware(BaseMiddleware):
    """Intercept oversized MCP tool responses, write to sandbox."""

    order = 550

    def __init__(self, max_response_size: int = 8192, sandbox_path: str = "/tmp/mcp_responses/"):
        self.max_response_size = max_response_size
        self.sandbox_path = sandbox_path

    async def __call__(self, state, config):
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None

        if not self._is_tool_message(last_msg):
            return state

        content = last_msg.content
        if len(content) <= self.max_response_size:
            return state

        file_path = f"{self.sandbox_path}{last_msg.tool_call_id}.txt"
        self._write_to_sandbox(state, file_path, content)

        size_kb = len(content) // 1024
        last_msg.content = (
            f"Response too large ({size_kb}KB), saved to {file_path}\n"
            f"Use read_file tool to inspect specific sections."
        )
        return state
```

### 后续演化方向（不在 MVP 范围）

- 智能摘要：用 LLM 生成 response 的结构化摘要替代简单提示
- 分块索引：大 response 切分后建索引，agent 按 section 查询
- 阈值自适应：根据 context 剩余空间动态调整

---

## 6. Oncall Agent Prompt & Skills

### prompt.md

```markdown
# Oncall 答疑助手

你是一个业务 oncall 答疑 agent，帮助工程师快速定位和解决线上问题。

## 你的能力
- 查询告警平台获取实时告警详情（通过 alert-platform MCP）
- 搜索日志定位异常（通过 log-search MCP）
- 查阅本地 runbook 和架构文档（knowledge/ 目录）
- 阅读业务代码定位根因（code_repos 配置的仓库）

## 工作流程
1. **先理解问题** — 确认告警名称、服务名、时间范围
2. **查数据** — 拉告警详情、搜相关日志
3. **查文档** — 看 runbook 里有没有已知处理方案
4. **查代码** — 如果需要定位根因，读相关代码
5. **给结论** — 明确的处理建议，附带证据

## 约束
- 不确定时说"不确定"，不要编造处理方案
- 涉及数据变更操作必须给出命令但不自动执行
- 每次回答附带信息来源（哪个告警、哪段日志、哪个文件）
```

### Skill 示例: runbook-lookup

```markdown
---
name: runbook-lookup
description: 在本地 knowledge 目录中查找与当前问题相关的 runbook
---

## Use when
- 用户描述了一个线上问题或告警
- 需要查找已有的处理方案

## Don't use when
- 用户在问架构设计问题
- 问题明显不在已有 runbook 覆盖范围内

## Steps
1. 从用户描述中提取关键词（服务名、错误类型）
2. 在 knowledge/ 目录下 grep 相关文件
3. 读取匹配的 runbook 内容
4. 总结处理步骤并呈现给用户
```

### 设计要点

- **prompt.md 不写死工具名** — 只描述能力，换 MCP server 不改 prompt
- **每个 Skill 必须有 Don't use when** — 遵循 agent-skills-design 规则
- **knowledge/ 是静态文档** — 手动维护的 runbook，agent 用 file tools 读取，不需要 RAG

---

## 7. 端到端数据流

```
用户: "线上 payment-service 出现大量 redis 连接超时告警，帮我查一下"
    │
    ▼
CLI Shell (prompt_toolkit)
  → 解析输入 → 非 / 命令 → 发送给当前 agent client
    │
    ▼
DeerFlowClient.stream(msg, thread_id="thread-xyz")
  → checkpointer: SQLite 自动恢复历史
  → config: 全局 + oncall agent.yaml merged
    │
    ▼
Middleware Chain (11层 + McpOverflow)
    │
    ▼
Lead Agent (LLM 推理，多轮 tool 调用)
  │
  ├─ 第1轮: alert-platform MCP → get_alerts() → 3KB → 正常通过
  ├─ 第2轮: log-search MCP → search_logs() → 60KB → ★ McpOverflow 写入沙箱
  ├─ 第3轮: read_file("knowledge/common-issues.md") → 找到 runbook
  ├─ 第4轮: bash/read_file → 读业务代码定位连接池配置
  └─ 第5轮: 综合输出结论
    │
    ▼
CLI Renderer (rich)
  → Markdown 渲染到终端
  → 结构化输出: 根因 + 处理建议 + 证据链
```

---

## 8. 代码检索（Phase 2，此处仅记录决策）

**MVP 方案**: 本地路径直读。`agent.yaml` 配置 `code_repos` 路径列表，agent 通过 `read_file` / `bash(grep)` 直接访问本地已有仓库。

**后续优化方向**（单独设计）:
- 预 clone 到统一目录 + 软链
- 代码索引加速检索
- 智能代码定位（基于告警信息自动缩小搜索范围）

---

## 9. 技术依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.12+ | 与 deer-flow 对齐 |
| deer-flow (fork) | latest | harness 核心 |
| prompt_toolkit | latest | 交互式 REPL |
| rich | latest | 终端 Markdown 渲染 |
| LangGraph (via deer-flow) | — | agent 编排、checkpointer |
| LangChain (via deer-flow) | — | LLM 抽象层 |

---

## 10. Scope 边界

### In Scope (MVP)
- 项目骨架 + deer-flow fork 集成
- CLI 统一入口 + REPL + /switch + /resume
- Oncall agent（config + prompt + skill + MCP）
- McpOverflow middleware
- SQLite checkpointer + session 管理
- 本地代码路径直读

### Out of Scope
- Web UI
- 其他 agent（review, guided learning）— 后续迭代
- 代码检索优化 — 单独设计
- Docker/K8s sandbox — MVP 用 local
- 多用户 / 权限
- MCP response 智能摘要
