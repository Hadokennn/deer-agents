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

- `middlewares/mcp_overflow.py` — Programmatic Tool Processing（见下文）
- 通过 `extra_middlewares` 参数注入（deer-flow 补丁：`_build_middlewares()` + `DeerFlowClient._get_runnable_config()`）

### Programmatic Tool Calling

**核心思想：LLM 是决策者，不是数据搬运工。** 中间数据在执行环境内 tool-to-tool 流转处理，LLM 只看最终的处理结果，大幅降低 token 消耗。

```
传统模式:  LLM → search → 6KB 回 LLM → LLM → fetch → 50KB 回 LLM → LLM 总结
                                        累计 56KB 进 context

Programmatic: LLM → search → 执行环境提取 top5 摘要 → 500B 回 LLM
                                        节省 90%+ token
```

通过 middleware 的 `wrap_tool_call` 拦截 tool 返回，在执行环境内处理：

| tool 类型 | 处理方式 | 效果 |
|-----------|---------|------|
| web_search | 提取 title + snippet + URL（top 5） | 6KB → ~500B |
| log_search | 提取 ERROR/WARN 行 + 统计 | 60KB → ~1KB |
| 未知大响应 | head + tail 截断，完整数据存沙箱 | 兜底方案 |

实现在 `middlewares/mcp_overflow.py`（`ToolResponseProcessorMiddleware`），可通过 `agent.yaml` 的 `extra_middlewares` 配置阈值和沙箱路径。完整数据始终保存在沙箱中，LLM 需要时可用 `read_file` 深入查看。

**后续方向：**
- 更多 tool 类型的专用 extractor（MCP server 返回的业务数据）
- tool chain：一个 tool 的输出自动触发下一个 tool，中间不经过 LLM
- 条件性 tool 注入：基于 state 自动执行前置 tool（如 oncall 问题自动先查 runbook）

### Checkpoint Time-Travel

LangGraph 每个 step 都创建 checkpoint，支持从任意 step 恢复执行：

- steps：从 checkpoint 的本地缓存的图历史还原步骤列表
- diagnose：在步骤上打标记，标出可疑问题，并建议从哪一步 replay
- replay：从指定步骤序号对应的 checkpoint 接着跑 agent（不是从头）

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
├── pipelines/                ← Mode 1: 声明式 Pipeline DSL
│   ├── parser.py             ← YAML → Pipeline dataclass
│   ├── resolver.py           ← ${var} 模板引擎
│   ├── registry.py           ← name → tool lookup
│   ├── executor.py           ← step-by-step 执行
│   ├── tool_factory.py       ← Pipeline → StructuredTool
│   └── loader.py             ← agent 集成 helper
├── codeact/                  ← Mode 2: 生成式 CodeAct Executor
│   ├── sandbox.py            ← 受限 Python 沙箱
│   ├── namespace.py          ← BaseTool → Python callable
│   ├── code_act_tool.py      ← 包装为 LangChain tool
│   └── prompt.py             ← LLM 描述模板
├── evals/
│   ├── framework/            ← 评测框架（agent 无关）
│   │   ├── types.py          ← EvalCase / EvalResult / EvalReport
│   │   ├── runner.py         ← 加载 case、调度 scorer、聚合结果
│   │   └── report.py         ← 控制台 + JSON 报告输出
│   └── oncall/               ← oncall agent 评测（首个实现）
│       ├── fixtures.py       ← 共享 mock 数据
│       ├── tool_cases.json   ← Layer 1 mock 回归 5 case
│       ├── tool_cases_live.json ← Layer 1 live 真实 MCP 3 case
│       ├── tool_eval.py      ← Layer 1 scorer (mock + live)
│       ├── e2e_cases.json    ← Layer 3 端到端 case
│       ├── e2e_eval.py       ← Layer 3 scorer (跑真实 agent)
│       └── process_eval.py   ← 启发式规则（被 e2e_eval 复用）
├── scripts/
│   ├── run_eval.py           ← 评测 CLI 入口
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

## 评测体系

两层评测，mock 和 live 分开：

| 命令 | 模式 | 评什么 | 速度 |
|------|------|--------|------|
| `run_eval.py tool` | Mock | 工具逻辑有没有改坏（回归） | 秒级 |
| `run_eval.py tool --live` | Live | 真实 MCP 返回是否正确 | 分钟级 |
| `run_eval.py e2e` | Live | 跑完整 agent → 评过程 + 结论 | 分钟级 |

Layer 1 (Tool) 两种模式：mock 测逻辑，live 测真实 API。Layer 3 (E2E) 天然 live — 跑真实 agent pipeline，同时评过程（启发式规则：调了哪些 tool、token 用量、步骤数）和结论（输出是否包含关键信息）。

没有独立的 Layer 2 — 过程评分并入 E2E，因为评旧 transcript 测不出代码改动是否有效。

### 运行评测

```bash
# Layer 1 mock（每次改代码必跑，秒级）
python scripts/run_eval.py tool

# Layer 1 live（打真实 MCP，验证 API 集成）
python scripts/run_eval.py tool --live

# Layer 3 E2E（跑完整 agent，评过程 + 结论）
python scripts/run_eval.py e2e

# 过滤 & 保存
python scripts/run_eval.py tool --tag fallback
python scripts/run_eval.py tool --case happy_path_field_found
python scripts/run_eval.py tool --save       # JSON 报告 → .deer-flow/eval-reports/
python scripts/run_eval.py all               # tool + e2e 全跑
```

### Layer 1 Mock Case（5 个冷启动）

| Case | 场景 | 覆盖 |
|------|------|------|
| `happy_path_field_found` | 三级类目直接命中，字段精确匹配 | 正常流程 |
| `category_fallback` | 叶子节点无模板，父级回退 | 容错路径 |
| `ambiguous_template` | 同类目多个模板，需用户确认 | 歧义处理 |
| `field_not_found` | 字段不在 schema 中 | 空结果处理 |
| `cross_concern_with_code` | schema + 组件代码联合定位 | 跨关注点 |

### Layer 1 Live Case（3 个）

| Case | 场景 | 断言类型 |
|------|------|---------|
| `live_overview` | 真实类目概览 | 结构性（status=overview, field_count>0） |
| `live_field_found` | 真实字段定位 | 结构性（status=found, 有 field_key） |
| `live_not_found` | 虚构类目 | 结构性（status=not_found） |

### Layer 3 E2E Case（2 个冷启动）

| Case | 场景 | 评分 |
|------|------|------|
| `e2e_field_diagnosis` | 字段 schema 诊断 | 过程：调了 locate_field_schema + 结论：有内容 |
| `e2e_overview_request` | 模板字段概览 | 过程：调了 tool + 结论：有响应 |

### 添加新 Agent 评测

```bash
mkdir -p evals/review
# 创建: evals/review/tool_cases.json + evals/review/tool_eval.py
python scripts/run_eval.py tool --agent review
```

### Regression 策略

每次修 bug → 加一个 case 到 `*_cases.json`。改代码后跑 `python scripts/run_eval.py tool`，新旧 case 都 PASS 才没改坏。

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
