# Oncall Knowledge Base — 从 DeerFlow Memory 机制演化的诊断经验沉淀系统

> 核心思路：DeerFlow 的 memory 机制设计精巧，但目标是"认识用户"（长期助手）。Oncall agent 每次面对新问题，需要的不是用户画像，而是**可泛化的诊断经验**。复用 memory 的工程骨架，重新设计 Extract 和 Store 层。

---

## 1. 为什么 DeerFlow Memory 对任务型 Agent 是负优化

### 现象

现有 oncall memory 存储的 facts（来自 `agents/oncall/memory.json`）：

```
"outId field has x-disabled: false setting"          → 下个 case 没用
"Uses OutIdChecker validator rule (rule_type: 2)"    → 下个 case 没用
"Platform has 40 total configured fields"            → 下个 case 没用
```

26 条 facts 全是**单个 case 的细节**，对下一个完全不同的 oncall 问题不仅无用，还会污染 context。

### 根因

DeerFlow Memory 的设计假设是**连续对话、同一用户、长期关系**：

| 假设 | DeerFlow（助手） | Oncall Agent（任务） |
|------|----------------|-------------------|
| 对话连续性 | 多轮跨 session | 每个 case 独立 |
| 信息生命周期 | 长期有效 | case 结束即过期 |
| 注入策略 | 每次都注入 top N | 仅相似问题时注入 |

**结论：DeerFlow memory 的工程机制（Capture → Extract → Store → Inject）值得复用，但 schema 和策略需要为 oncall 场景重新设计。**

---

## 2. DeerFlow Memory 机制拆解

### 四阶段管道

```
Capture ──→ Extract ──→ Store ──→ Inject
(Middleware)   (LLM)     (JSON)   (Prompt)
```

### 关键设计点

| 设计 | 文件 | 机制 |
|------|------|------|
| 异步去抖 | `memory/queue.py` | 30s debounce，同 thread 去重，不阻塞 agent 执行 |
| 信号检测 | `middlewares/memory_middleware.py:132-171` | 正则检测 correction（"不对"/"你理解错了"）和 reinforcement（"完全正确"），调整 fact confidence |
| 消息过滤 | `middlewares/memory_middleware.py:71-129` | 只保留 human + final AI response，过滤 tool calls 和 upload blocks |
| LLM 提取 | `memory/updater.py:269-360` | 结构化 prompt → JSON 输出 → 去重 → 原子写入 |
| Token 预算注入 | `memory/prompt.py:201-317` | 按 confidence 排序，逐条计算 token，塞满预算为止 |
| State 外存储 | `memory/storage.py` | 不进 checkpoint，prompt 时动态加载，跨 session 持久化 |
| 原子写入 | `memory/storage.py:142-146` | temp file + rename，防写入中途崩溃 |

### Memory Schema（用户画像导向）

```json
{
  "user": {
    "workContext": { "summary": "..." },
    "personalContext": { "summary": "..." },
    "topOfMind": { "summary": "..." }
  },
  "history": {
    "recentMonths": { "summary": "..." },
    "earlierContext": { "summary": "..." }
  },
  "facts": [
    { "id": "fact_xxx", "content": "...", "category": "preference|knowledge|context|behavior|goal|correction", "confidence": 0.95 }
  ]
}
```

---

## 3. Oncall Knowledge Base 设计

### 3.1 新的 Store Schema（诊断模式导向）

```json
{
  "version": "1.0",
  "lastUpdated": "2026-04-07T09:32:10Z",
  "patterns": [
    {
      "id": "pattern_a1b2c3d4",
      "symptom": "商家反馈字段不显示/不能编辑",
      "symptom_keywords": ["字段", "不显示", "不能编辑", "看不到"],
      "misdiagnosis_trap": "容易误判为 schema 配置问题（x_hidden / reaction_rules）",
      "actual_root_cause": "runtime 业务逻辑控制（use-model.ts 中的条件渲染）",
      "root_cause_type": "runtime_business_logic",
      "diagnostic_shortcut": "先查 use-model.ts 条件渲染 → 再查 schema x_hidden → 最后查 reaction_rules",
      "key_files": ["use-model.ts"],
      "resolution": "引导商家开启对应账户配置（如 enableThirdProduct）",
      "confidence": 0.95,
      "source_cases": ["outId字段不显示-团购水果模板"],
      "times_matched": 0,
      "createdAt": "2026-04-07T09:32:10Z"
    }
  ]
}
```

### 各字段设计意图

| 字段 | 为什么需要 | 面试怎么说 |
|------|-----------|-----------|
| `symptom_keywords` | 新问题进来时做关键词匹配 | "这是最简单的检索方案，不需要向量数据库，keyword overlap 在 oncall 场景够用" |
| `misdiagnosis_trap` | **最有价值的字段** — 防止重蹈覆辙 | "Agent 第一次走了弯路，第二次不应该再走。trap 把弯路显式化" |
| `diagnostic_shortcut` | 节省 step/token 的核心 | "没有 shortcut 要 8 步 65K token，有 shortcut 可能 3 步 15K token" |
| `root_cause_type` | 问题分类，支持统计和聚类 | "类型化后可以发现：60% 的问题是 runtime_business_logic 类，说明这是平台的系统性弱点" |
| `times_matched` | 命中越多说明越通用 | "高频 pattern 应该被固化为 PTC 或 Skill，而不是每次靠 LLM 匹配" |
| `source_cases` | 溯源 + 支持 pattern 合并 | "多个 case 归到同一个 pattern 时，confidence 应该上升" |

### 3.2 新的 Extract Prompt

```
你是一个诊断经验总结系统。分析刚完成的 oncall 诊断过程，提取可泛化的诊断模式。

诊断记录：
<transcript>
{transcript}
</transcript>

提取规则：
1. 走了哪些弯路？提取为 misdiagnosis_trap
2. 最终根因属于哪类？（schema配置 / runtime逻辑 / 数据问题 / 权限问题 / 依赖变更）
3. 如果下次遇到类似症状，最短排查路径是什么？提取为 diagnostic_shortcut
4. 哪些文件/工具是定位关键？

不要记录的（case 细节，不可泛化）：
- 具体字段名（outId / price 等）
- 具体模板 ID / category_id
- 具体商家信息

要记录的（跨 case 可复用的经验）：
- 问题模式（"字段不显示"而非"outId 不显示"）
- 排查路径（"先查 runtime 逻辑再查 schema"）
- 易犯错误（"容易误判为 schema 配置问题"）

输出 JSON：
{output_schema}
```

### 3.3 Inject 策略 — 从无条件到条件注入

**DeerFlow 方式（不适合 oncall）：**
```python
# 每次都注入 top 15 facts
def inject():
    return top_facts_by_confidence(limit=15)
```

**Oncall Knowledge 方式：**
```python
def inject_knowledge(user_question: str, knowledge_store) -> str:
    # 1. 提取症状关键词
    keywords = extract_symptom_keywords(user_question)
    
    # 2. 匹配 patterns（keyword overlap scoring）
    matched = []
    for pattern in knowledge_store["patterns"]:
        score = keyword_overlap(keywords, pattern["symptom_keywords"])
        if score > threshold:
            matched.append((score, pattern))
    
    # 3. 取 top 3，注入 shortcut
    matched.sort(reverse=True)
    if not matched:
        return ""  # 没匹配就不注入，保持 context 干净
    
    # 4. 格式化为诊断提示
    hints = []
    for _, pattern in matched[:3]:
        hints.append(f"""
相似问题经验：
- 症状：{pattern["symptom"]}
- 常见误判：{pattern["misdiagnosis_trap"]}
- 推荐路径：{pattern["diagnostic_shortcut"]}
- 关键文件：{pattern["key_files"]}
""")
    
    # 5. 更新 times_matched
    for _, pattern in matched[:3]:
        pattern["times_matched"] += 1
    
    return "\n".join(hints)
```

**核心区别：匹配才注入，不匹配就保持 context 干净。**

---

## 4. 可复用 vs 需重写

| DeerFlow 组件 | 复用？ | 说明 |
|--------------|--------|------|
| `FileMemoryStorage` | ✅ 直接复用 | 原子写入、mtime 缓存、per-agent 隔离 |
| `MemoryMiddleware` (capture) | ⚠️ 改触发条件 | 不是 after_agent，而是 case 完成时 |
| `queue.py` (debounce) | ❌ 不需要 | oncall 是一次性任务，不需要去抖 |
| `updater.py` (LLM extract) | ⚠️ 改 prompt | 从"认识用户"改成"提取诊断模式" |
| `prompt.py` (inject) | ⚠️ 改注入策略 | 从无条件 top N 改成条件匹配 |
| correction 检测 | ✅ 复用思路 | 用户说"不对"时更新 pattern 的 trap 字段 |

---

## 5. 演进路线

### Phase 1: 手动沉淀（验证 schema 设计）

- 手动跑 3-5 个 oncall case
- 每个 case 结束后，手动填写 pattern JSON
- 验证：pattern 的 symptom_keywords 能否匹配新问题

### Phase 2: 自动提取（复用 DeerFlow Extract 管道）

- 实现 OncallKnowledgeExtractor（改写 MemoryUpdater 的 prompt）
- case 完成后自动调用 LLM 提取 pattern
- 与手动填写的 pattern 做 diff，校准 LLM 提取质量

### Phase 3: 自动注入 + 效果验证

- 实现条件注入逻辑（keyword matching）
- A/B 对比：有 knowledge 注入 vs 无注入的 step 数、token 数、准确率
- 收集 times_matched 数据，高频 pattern 考虑固化为 PTC

---

## 6. 面试话术

### 当被问到"memory 怎么用"时

> "DeerFlow 的 memory 机制设计很精巧——异步去抖、信号检测、原子写入、token 预算注入。但它的目标是'认识用户'，对任务型 agent 是负优化。我把它的工程骨架复用过来，重新设计了 Extract 和 Store 层——不存 case 细节，存**可泛化的诊断模式**：症状关键词、误诊陷阱、排查捷径。注入策略也从'每次都注入 top N'改成了'症状匹配时才注入'，保持新 case 的 context 干净。"

### 当被问到"这个设计的 trade-off"时

> "Keyword matching 精度不如向量检索，但 oncall 场景的症状描述相对固定（'字段不显示'、'价格不对'），keyword overlap 够用。上向量数据库是过度工程化——先跑起来，patterns 超过 50 个再考虑。"

### 当被问到"怎么验证效果"时

> "对比同类问题：有 knowledge shortcut 注入时 3 步 15K token，没有时 8 步 65K token。核心指标是 step 数和 token 数的下降，不是准确率——因为 shortcut 改变的是路径效率，不是最终答案。"
