# CLAUDE.md — deer-agents 开发指引

## 项目概述

基于 DeerFlow harness 的 CLI agent 系统。deer-flow 作为 git subtree 集成，可直接修改 harness 代码。

## 开发调试工具

### e2e 测试（改代码后必跑）

```bash
source .venv/bin/activate
export $(cat deer-flow/.env | grep -v '^#' | xargs)
python scripts/e2e_test.py
```

验证完整链路：config → model → client → chat → stream → renderer。

### LangSmith Trace Inspector（agent 行为异常时必用）

```bash
python scripts/trace_inspector.py recent      # 最近的 agent runs
python scripts/trace_inspector.py last        # 最近一次 run 的完整 step 链
python scripts/trace_inspector.py detail <id> # 特定 run 的详情
```

**When to use:**
- renderer 不展示内容 → 查 trace 看 event 实际格式
- agent 没调 tool 或调错 tool → 查 trace 看 LLM 实际输出
- 响应异常（太慢、内容错） → 查 trace 看 token 用量和每步耗时
- 修改 prompt/middleware 后 → 跑 e2e + 查 trace 确认行为变化

### Event 格式观察（写解析代码前必用）

```bash
python scripts/dump_events.py
```

先看真实数据结构，再写代码。不要基于文档或猜测编写解析逻辑。

## 关键路径

| 组件 | 路径 |
|------|------|
| CLI 入口 | `cli/__main__.py` |
| REPL Shell | `cli/shell.py` |
| 流式渲染 | `cli/renderer.py` |
| Agent 配置 | `agents/<name>/agent.yaml` |
| 全局配置 | `config.yaml`（deer-agents 自身）|
| Harness 配置 | `deer-flow/config.yaml`（模型/tools/memory）|
| 自定义 Middleware | `middlewares/` |
| deer-flow 补丁 | `deer-flow/backend/packages/harness/deerflow/agents/lead_agent/agent.py`（extra_middlewares 注入）|
| deer-flow 补丁 | `deer-flow/backend/packages/harness/deerflow/client.py`（extra_middlewares 透传）|

## 环境

- API keys 在 `deer-flow/.env`
- `DEER_FLOW_CONFIG_PATH` 由 shell.py 自动设为 `deer-flow/config.yaml`
- LangSmith 已开启，project = "deer-agents"

## 注意事项

- deer-flow/config.yaml 中 model 的 `use` 字段必须是 `deerflow.*` 前缀（不是 `src.*`）
- `SqliteSaver.from_conn_string()` 返回 context manager，需要 `__enter__()` 获取实例
- deer-flow 的 `.gitignore` 忽略 `config.yaml`，需要 `git add -f` 强制跟踪
