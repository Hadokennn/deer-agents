# Deer Agents

基于 [DeerFlow](https://github.com/bytedance/deer-flow) harness 构建的 CLI agent 系统。统一终端入口，多 agent 动态切换，每个 agent 拥有独立的 config、prompt、skill、tool 和 MCP 配置。

```
$ deer
🦌 > 线上 payment-service redis 连接超时，帮我查一下
...

🦌 > /switch review
✓ Switched to review agent

🦌 review > 帮我 review 这个 PR
```

## Quick Start

```bash
# 环境要求: Python 3.12+
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e deer-flow/backend/packages/harness/
pip install -e .

# 配置 API Key
export VOLCENGINE_API_KEY=your-key-here

# 启动
python -m cli
# 或
deer
```

## 项目结构

```
deer-agents/
├── deer-flow/              ← DeerFlow fork (git subtree)
├── agents/
│   └── oncall/
│       ├── agent.yaml      ← agent 配置 (模型/tools/MCP)
│       ├── prompt.md       ← 系统提示词
│       ├── skills/         ← agent 专属 skill
│       └── knowledge/      ← 本地知识库 (runbook)
├── cli/
│   ├── __main__.py         ← 入口
│   ├── shell.py            ← 交互式 REPL
│   ├── commands.py         ← /switch /resume /agents 等命令
│   └── renderer.py         ← 流式 Markdown 渲染
├── middlewares/
│   └── mcp_overflow.py     ← MCP 大响应写沙箱防 context rot
└── config.yaml             ← 全局配置
```

## REPL 命令

| 命令 | 说明 |
|------|------|
| `/agents` | 列出可用 agent |
| `/switch <name>` | 切换 agent |
| `/sessions` | 查看历史会话 |
| `/resume <id>` | 恢复历史对话 |
| `/status` | 当前 agent 状态 |
| `/help` | 帮助 |
| `/exit` | 退出 |

## 创建新 Agent

在 `agents/` 下创建目录：

```
agents/my-agent/
├── agent.yaml      ← 必须
├── prompt.md       ← 系统提示词
├── skills/         ← 可选
└── knowledge/      ← 可选
```

`agent.yaml` 示例：

```yaml
name: my-agent
display_name: "My Agent"
model: doubao-seed-1.8
tool_groups: [web, file, bash]
mcp_servers: []
prompt: ./agents/my-agent/prompt.md
```

## DeerFlow 同步

```bash
# 拉取上游更新
git subtree pull --prefix=deer-flow deer-flow-upstream main --squash
```

## License

MIT
