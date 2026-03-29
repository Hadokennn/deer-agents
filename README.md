# Deer Agents

Agent harness 学习与实践项目。基于 [DeerFlow](https://github.com/bytedance/deer-flow) harness 构建，通过动手搭建 CLI agent 系统来理解 agent harness 的每一层：config 合并、middleware chain、checkpoint 持久化、tool 编排、反馈机制。

**这个项目的目标不是做一个好用的 agent，而是理解和实践搭建 agent 的每个细节。**

## 已实践的 Harness 概念

### Config 分层

```
config.yaml              ← deer-agents 自身配置（checkpointer、sessions）
deer-flow/config.yaml    ← harness 配置（models、tools、memory、sandbox）
agents/oncall/agent.yaml ← agent 级增量覆盖（MCP、middlewares、prompt）
```

合并规则：MCP/tools/prompt 是 agent 独占（不继承全局），models/sandbox 继承全局。`cli/app.py` 实现。

### Middleware 注入

在 deer-flow 的 11 层 middleware chain 中注入自定义 middleware，不破坏原有顺序：

- `middlewares/mcp_overflow.py` — MCP 响应 > 8KB 时写入沙箱，防 context rot
- 通过 `extra_middlewares` 参数注入（deer-flow 补丁：`_build_middlewares()` + `DeerFlowClient._get_runnable_config()`）

### Checkpoint Time-Travel

LangGraph 每个 step 都创建 checkpoint，支持从任意 step 恢复执行：

```bash
# 查看 step 历史
python scripts/trace_replay.py steps <thread_id>

# 从 step 10 重放（不重跑之前的 tool）
python scripts/trace_replay.py replay <thread_id> --from-step 10
```

### 反馈机制

三层反馈，从 CLI 到 LangSmith 到 checkpoint：

| 层 | 工具 | 看什么 |
|----|------|--------|
| CLI 渲染 | `renderer.py` | tool calls、results、errors、token usage 实时展示 |
| LangSmith trace | `scripts/trace_inspector.py` | 完整 step 链、每步 input/output、耗时 |
| Checkpoint replay | `scripts/trace_replay.py` | 定位异常 step，修改代码后从该 step 重放验证 |

### 开发 Agent 的 CLAUDE.md

`CLAUDE.md` 编码了项目的开发流程 — 让 Claude（开发 agent）知道改完代码后要跑 e2e、行为异常时查 trace、写解析代码前先 dump 真实格式。这是"项目级 harness"的一部分。

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e deer-flow/backend/packages/harness/
pip install -e .

# API keys 在 deer-flow/.env 中配置
python -m cli
```

## 项目结构

```
deer-agents/
├── deer-flow/                ← DeerFlow fork (git subtree)
├── agents/
│   └── oncall/
│       ├── agent.yaml        ← agent 配置
│       ├── prompt.md         ← 系统提示词
│       ├── skills/           ← agent 专属 skill
│       └── knowledge/        ← 本地知识库
├── cli/
│   ├── __main__.py           ← 入口
│   ├── shell.py              ← 交互式 REPL
│   ├── renderer.py           ← 流式渲染（tool calls + usage 可见）
│   ├── commands.py           ← /switch /replay /diagnose /trace 等
│   ├── sessions.py           ← 会话元数据
│   ├── app.py                ← config 合并
│   └── bootstrap.py          ← 共享初始化（env、checkpointer 路径）
├── middlewares/
│   └── mcp_overflow.py       ← 自定义 middleware 示例
├── scripts/
│   ├── e2e_test.py           ← 端到端验证
│   ├── trace_inspector.py    ← LangSmith trace 查看
│   ├── trace_replay.py       ← checkpoint 诊断 + 重放
│   └── dump_events.py        ← StreamEvent 格式观察
├── config.yaml               ← deer-agents 配置
├── CLAUDE.md                 ← 开发 agent 指引
└── .deer-flow/               ← 运行时数据（checkpoints、sessions）
```

## REPL 命令

| 命令 | 说明 |
|------|------|
| `/agents` | 列出可用 agent |
| `/switch <name>` | 切换 agent |
| `/sessions` | 查看历史会话 |
| `/resume <id>` | 恢复历史对话 |
| `/status` | 当前 agent 状态 |
| `/trace` | 查看 LangSmith traces |
| `/replay` | 查看当前 thread 的 step 历史 |
| `/replay <N>` | 从 step N 重放 |
| `/diagnose` | 自动检测异常 step |
| `/help` | 帮助 |
| `/exit` | 退出 |

## DeerFlow 同步

```bash
git fetch deer-flow-upstream
git subtree pull --prefix=deer-flow deer-flow-upstream main --squash
```

## 踩过的坑（学到的）

| 坑 | 根因 | 教训 |
|----|------|------|
| renderer 不展示消息 | StreamEvent.data 是 flat dict 不是 tuple | 先 dump 真实格式再写代码 |
| config 加载报 sandbox missing | `get_app_config()` 读了根目录 config 而不是 deer-flow 的 | 用 `DEER_FLOW_CONFIG_PATH` 锁定 |
| model use 路径报 ModuleNotFoundError | `src.models.*` 应为 `deerflow.models.*` | fork 后检查所有 import 路径 |
| SqliteSaver.from_conn_string 报错 | 返回 context manager，需要 `__enter__()` | 读 API 签名，不猜 |
| 之前以为 checkpoint 不支持 step 重放 | explore agent 结论错误，未自己验证 | 不信二手结论，自己跑代码验证 |

## License

MIT
