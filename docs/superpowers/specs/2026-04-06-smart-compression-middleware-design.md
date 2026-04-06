# SmartCompressionMiddleware 设计

## 问题

DeerFlow harness 使用 LangChain 的 SummarizationMiddleware，机制是 `RemoveMessage(REMOVE_ALL_MESSAGES)` + 一条 summary HumanMessage。原始消息在 checkpoint 中被永久替换，无法回退。且压缩策略是一刀切 — 所有旧消息都被 LLM 压成一段摘要，丢失了 tool 调用决策链、错误上下文等结构化信息。

## 目标

1. **安全** — 归档原始消息，压缩失败可回退
2. **精细** — 按消息类型分策略压缩，不一刀切
3. **兼容** — Drop-in 替换现有 SummarizationMiddleware，复用触发配置

## 核心机制

```
触发（沿用 harness SummarizationConfig: token/消息数/比例）
  ↓
归档原始消息 → .deer-flow/threads/{thread_id}/archive.jsonl (追加)
  ↓ (归档失败 → 跳过压缩，原始不动)
更新 ThreadState.compressed_messages 定位信息
  ↓
按类型策略压缩，same-ID 替换写回 state
```

## 消息分区

```
[--- 压缩区 (older) ---][--- 保留区 (recent, keep config) ---]
```

保留区由 `keep` 配置决定（如最近 20 条消息）。只处理压缩区。

## 压缩策略

### 规则式（零成本）

| 消息类型 | 条件 | 处理 |
|---------|------|------|
| SystemMessage | — | 不压缩 |
| HumanMessage | — | 不压缩 |
| AIMessage | 有 `tool_calls` | 保留 tool_calls 字段，清空 content |
| AIMessage | content 是 block 列表 | 删 `type: "thinking"` blocks，保留 `type: "text"` |
| ToolMessage | < threshold (默认 2KB) | 不压缩 |
| ToolMessage | >= threshold 且非 error | 截断：head + `[archived, {N}B]` + tail |
| ToolMessage | 包含 error | 不压缩 |

### LLM 摘要（仅必要时）

| 消息类型 | 条件 | 处理 |
|---------|------|------|
| AIMessage | content 是纯文本字符串 | LLM 摘要："保留架构决策和结论，删除推理过程" |

### 配对约束

AIMessage(tool_calls) + 对应 ToolMessage 始终作为一组处理，不拆分。LangGraph 要求 tool_call 和 ToolMessage 配对，拆了会报错。

### 衰减

保留区不压缩。压缩区内按上述策略统一处理，不做进一步分级。

## 归档

### 存储格式

JSONL 文件，路径: `.deer-flow/threads/{thread_id}/archive.jsonl`

每行一条 LangChain message 的完整序列化 JSON：

```jsonl
{"id": "msg-uuid-1", "type": "ai", "content": "完整原文...", "tool_calls": [...], ...}
{"id": "msg-uuid-2", "type": "tool", "content": "{50KB MCP response}", "name": "locate_field_schema", ...}
```

追加写入，不覆盖。

### 定位信息

ThreadState 新增字段 `compressed_messages`，记录被压缩消息的 ID 和在 JSONL 中的行号：

```python
compressed_messages: {
    "msg-uuid-1": {"line": 0},
    "msg-uuid-2": {"line": 1},
}
```

该字段随 checkpoint 持久化。Map 只存 ID + 行号，体积可忽略。

### 安全保证

1. **Archive-first** — 先追加 JSONL，成功后才替换 state
2. **归档失败 = 不压缩** — `before_model()` 返回 None，原始消息不动
3. **回滚** — 从 state 取 compressed_messages → 读 JSONL 对应行 → 反序列化 → same-ID 替换回 state

## 集成

### 位置

替换 middleware chain 第 8 位的 SummarizationMiddleware。

### 接口

继承 `AgentMiddleware`，实现 `before_model()` / `abefore_model()`。

### 配置

复用 `SummarizationConfig` 的 trigger/keep，新增 `use_smart_compression` 开关和 `compression` 段：

```yaml
summarization:
  enabled: true
  use_smart_compression: false    # 默认 false = harness 原生 SummarizationMiddleware
                                  # true = SmartCompressionMiddleware
  trigger:
    - token_fraction: 0.6
  keep:
    messages: 20
  compression:                    # 仅 use_smart_compression: true 时生效
    tool_message_threshold: 2048  # bytes，小于不压缩
    tool_truncate_head: 500       # bytes
    tool_truncate_tail: 200       # bytes
    ai_summary_model: null        # null = 用 summarization.model_name
```

**开关逻辑：** `_create_summarization_middleware()` 根据 `use_smart_compression` 决定实例化哪个类。默认行为不变。

## 文件结构

```
summarization/
    smart_compression.py      # SmartCompressionMiddleware 主体
    compression_strategies.py # 各类型压缩策略（纯函数）
    compression_archive.py    # JSONL 归档读写
```

## 不做的事

- 不改触发机制 — 沿用 harness 的 SummarizationConfig
- 不改 LangGraph 的 add_messages reducer — 用原生 same-ID 替换
- 不在 state 里存原始消息体 — 只存 ID + 行号指针
