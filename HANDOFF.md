# HANDOFF

## Objective
基于 DeerFlow harness 构建 CLI agent 系统，通过动手实践学习 agent harness 的每个细节（config、middleware、checkpoint、反馈机制、programmatic tool calling、知识库检索）。

## Current State
- Branch: `main`
- Last commit: `cbd0f40 — feat: CodeIndexMiddleware — enrich code search with symbol index`
- Uncommitted: `tools/schema_locator.py` + `tests/test_schema_locator.py` + config updates
- Tests: 47 pass, 0 fail

### 已实现的 Harness 能力
1. **CLI Shell** — prompt_toolkit REPL，多 agent 切换（/switch），session resume（/resume）
2. **Config 分层** — `config.yaml`（deer-agents）+ `deer-flow/config.yaml`（harness）+ `agents/oncall/agent.yaml`（agent 级）
3. **Middleware 注入** — deer-flow 补丁支持 `extra_middlewares`，两个自定义 middleware：
   - `ToolResponseProcessorMiddleware` — programmatic tool processing（search result 提取、log ERROR 提取、大响应截断）
   - `CodeIndexMiddleware` — bash grep 拦截，prepend tree-sitter 符号索引结果
4. **Checkpoint Time-Travel** — `trace_replay.py` 支持 steps/diagnose/replay from checkpoint
5. **反馈机制** — renderer 展示 tool calls/results/usage，LangSmith trace inspector，`--verbose` 模式
6. **符号索引** — tree-sitter 扫描 10K ts/tsx 文件 → 23K 符号，搜索瞬间返回

### deer-flow 补丁（2 个文件）
- `deer-flow/backend/packages/harness/deerflow/agents/lead_agent/agent.py` — `_build_middlewares()` + `make_lead_agent()` 加 `extra_middlewares` 参数
- `deer-flow/backend/packages/harness/deerflow/client.py` — `_get_runnable_config()` 透传 `extra_middlewares`
- `deer-flow/config.yaml` — `use` 路径从 `src.*` 改为 `deerflow.*`（git add -f 强制跟踪，因为 deer-flow/.gitignore 排除了 config.yaml）

## What Worked
- **先观察再实现** — dump_events.py 先看 StreamEvent 实际格式，再写 renderer，避免了基于文档假设的 bug
- **e2e 测试脚本** — `scripts/e2e_test.py` 分步验证（config → model → client → chat → stream → renderer），每次改代码后跑一遍
- **薄壳架构** — CLI 是薄壳，核心逻辑在 DeerFlowClient + middleware，保持简单
- **git subtree** — deer-flow 代码可直接改 + `git subtree pull` 同步上游
- **bootstrap.py** — 统一的 env/checkpointer 初始化，所有 scripts 复用

## Dead Ends (Don't Retry)
- **Explore agent 说"LangGraph checkpoint 不支持 step 级重放"** — 错误结论。实际上 `get_state_history()` 返回每个 step 的 checkpoint，`agent.stream(None, config_with_checkpoint_id)` 可以从任意 step 恢复。**永远自己验证，不信二手结论。**
- **根目录 config.yaml 同时当 deer-flow config** — 会导致 `get_app_config()` 误读（缺少 sandbox 等必须字段）。解决方案：分离为两个 config，用 `DEER_FLOW_CONFIG_PATH` 环境变量锁定。
- **SqliteSaver.from_conn_string() 直接当实例用** — 它返回 context manager，需要 `__enter__()` 拿实例。

## What's Left (Ordered)

### 下一步：Text-to-SQL 数据访问层
用户有底层业务数据表，现有 MCP API 太死板（严格参数、场景固定）。讨论中的方案：
- 给 agent 一个 `query_db` tool，让 LLM 根据 schema 生成 SQL
- schema 注入用"数据字典"（类似符号索引的思路），只把相关表 schema 注入 context
- 混合模式：简单查询用 Text-to-SQL，复杂业务操作用现有 MCP API
- **需要先了解**：用户的数据库类型（MySQL/PG/内部平台）、表数量、安全约束

### Oncall 诊断系统完善（基于执行层 trade-off 分析）

> 四种执行机制的选择标准：路径是否已知 × 中间数据是否大
>
> | 机制 | 驱动者 | Context 影响 | 适合场景 |
> |------|--------|-------------|---------|
> | CLI Tool | LLM 逐步 | 每步进 parent | 探索性、一次性操作 |
> | Programmatic Tool | 代码一气呵成 | 只有最终结果 | 固定流程、已知路径 |
> | Skill SOP | LLM 按步骤走 | 每步进 parent | 已知流程但需 LLM 判断 |
> | Sub-agent | 独立 LLM | 隔离，只回传摘要 | 复杂分析、中间数据大 |
>
> 嵌套组合：Skill 是编排层，Programmatic Tool 和 CLI Tool 是执行层，Sub-agent 是隔离层。

- [ ] **Phase 2 schema 定位** — `locate_field_schema` Programmatic Tool（✅ 已实现，待接入真实 MCP）
  - 路径已知（MCP1→MCP2→搜索），中间数据大（schema 存文件不进 context）
  - 类目树向上回溯（叶子→父→祖父）纯代码驱动
  - 待办：配置 `bytedance-mcp-ace_ai` MCP server，用真实数据验证

- [ ] **Phase 3 组件代码定位** — Programmatic Tool
  - 从字段 schema 的 `x-component` 提取组件名 → grep 代码仓库
  - 复用 `CodeIndexMiddleware` 的符号索引能力
  - 可和 Phase 2 串成一个更大的 Programmatic Tool，或独立为 `locate_component_code`

- [ ] **Phase 1-4 完整诊断流程** — Skill SOP
  - Phase 1（解析意图）需要 LLM 判断 → Skill 步骤
  - Phase 2 调用 `locate_field_schema` Programmatic Tool
  - Phase 3 调用组件定位 Programmatic Tool 或 CLI grep
  - Phase 4（综合分析）需要 LLM 推理 → Skill 步骤
  - 如果 Phase 4 的 schema + 代码上下文过大 → 委派 Sub-agent 隔离分析

- [ ] **BudgetGuardMiddleware** — before_model 预算熔断
  - 超过 N 轮 tool call 或 token 预算 → 注入 SystemMessage 强制收敛
  - 防止 agent 在错误方向上无限探索

- [ ] **safe_bash** — 执行边界
  - oncall agent 只需只读能力，白名单命令（grep/cat/find/git log/git blame）
  - 工具 description 里声明限制 + 执行层硬拦截，两层都做

- [ ] **ContextCompactionMiddleware** — before_model 上下文压缩
  - 滑动窗口：最近 K 轮完整保留
  - 窗口外 ToolMessage：已被 ToolResponseProcessor 处理的直接丢弃，未处理的做摘要
  - 窗口外 AIMessage：保留 tool_calls，删推理文本
  - 注意和 ToolResponseProcessorMiddleware 的两层压缩协调

- [ ] **Sub-agent 程序化路由** — middleware 自动判断是否委派
  - before_model 检测 tool response 总大小，超阈值注入指令走 sub-agent
  - 确定性逻辑不走 LLM，不放 prompt

### 后续 backlog
1. **代码结构图**（Code Graph） — FileEntry.imports 已提取，下一步构建 `A imports B` 依赖图，支持调用链追溯
2. **Tool Chaining** — wrap_tool_call 里一个 tool 输出自动触发下一个 tool（如 search_alerts → get_detail → search_logs），省 LLM 轮次
3. **条件性 tool 注入** — before_model 检测 oncall 问题 → 自动先查 runbook（programmatic tool calling 的延伸）
4. **更多 agent** — review agent、guided learning agent（复用同样的 config/middleware/CLI 架构）

## Key Decisions
- **Fork 方式选 git subtree** 而非 submodule — 因为要直接改 harness 代码，subtree 提交流程更顺畅
- **DeerFlowClient 作为核心** — 不自己造 agent loop，复用 deer-flow 的 middleware chain + LangGraph 编排
- **项目目标是学习而非产品** — README 记录已实践的概念和踩过的坑，代码是实验场
- **Programmatic tool calling > LLM 决定一切** — 中间数据在执行环境流转，LLM 只看处理后的结果，降低 token 消耗
- **符号索引 > Embedding RAG** — 代码检索是搜索+导航问题，tree-sitter 精确匹配比 embedding 相似度更可靠
