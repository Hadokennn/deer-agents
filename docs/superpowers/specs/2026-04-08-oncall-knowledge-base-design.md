# Oncall Knowledge Base — Spec

> 复用 DeerFlow memory 的工程骨架（middleware capture → LLM extract → atomic store → prompt inject），重新设计 schema 和策略，为 oncall agent 沉淀可泛化的诊断经验。

## 1. 问题

DeerFlow memory 存的是用户画像 facts，oncall agent 每次面对新 case，需要的不是 "outId field has x-disabled: false"（下个 case 没用），而是 "字段不显示时，先查 runtime 条件渲染再查 schema"（跨 case 可复用）。

**目标：** 高评分的诊断 case 自动沉淀为 pattern，下次遇到相似症状时注入 diagnostic shortcut，减少 step 数和 token 消耗。

## 2. 数据流

```
[写入路径 — after_agent]
Agent 给结论 → 问评分 → 用户 "8" → Agent "感谢"
                                         │
                              after_agent() 触发
                                         │
                    检测最近 messages 中的评分 (regex)
                                         │
                              score ≥ 7 ?
                              ├─ No → return None
                              └─ Yes ↓
                         过滤 messages (human + final AI)
                                         │
                              trim to 8K token budget
                                         │
                              LLM extract pattern (JSON)
                                         │
                    去重 (root_cause_type + symptom overlap)
                              ├─ 重复 → 合并
                              └─ 新增 → append
                                         │
                         原子写入 knowledge.json

[读取路径 — before_model]
用户 "字段不显示" → before_model() 触发
                          │
         keyword overlap 匹配 patterns
                          │
                   top 3, score > 0
                   ├─ 无匹配 → 不注入，保持 context 干净
                   └─ 有匹配 → 注入 <diagnostic_knowledge> 到 system prompt
```

## 3. Pattern Schema

```json
{
  "version": "1.0",
  "lastUpdated": "2026-04-08T09:00:00Z",
  "patterns": [
    {
      "id": "pattern_a1b2c3d4",
      "symptom": "商家反馈字段不显示/不能编辑",
      "symptom_keywords": ["字段", "不显示", "不能编辑", "看不到", "隐藏"],
      "misdiagnosis_trap": "容易误判为 schema 配置问题（x_hidden / reaction_rules）",
      "actual_root_cause": "runtime 业务逻辑控制（use-model.ts 中的条件渲染）",
      "root_cause_type": "runtime_business_logic",
      "diagnostic_shortcut": "先查 use-model.ts 条件渲染 → 再查 schema x_hidden → 最后查 reaction_rules",
      "key_files": ["use-model.ts"],
      "resolution": "引导商家开启对应账户配置（如 enableThirdProduct）",
      "confidence": 0.95,
      "source_cases": ["字段不显示-团购水果模板"],
      "times_matched": 0,
      "createdAt": "2026-04-08T09:00:00Z"
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | `pattern_{uuid[:8]}`，唯一标识 |
| `symptom` | string | 泛化的症状描述（不含具体字段名） |
| `symptom_keywords` | string[] | 匹配用关键词，含同义词 |
| `misdiagnosis_trap` | string | 容易走的弯路——最有价值的字段 |
| `actual_root_cause` | string | 真正的根因 |
| `root_cause_type` | enum | `schema_config` \| `runtime_business_logic` \| `data_issue` \| `permission` \| `dependency_change` |
| `diagnostic_shortcut` | string | 最短排查路径，格式 "先查 X → 再查 Y → 最后查 Z" |
| `key_files` | string[] | 关键文件名（不含行号） |
| `resolution` | string | 解决建议 |
| `confidence` | float | 0.7-1.0，LLM 提取时给出，合并时取 max |
| `source_cases` | string[] | 来源 case 摘要，支持溯源和合并 |
| `times_matched` | int | 被匹配命中次数，高频 pattern 考虑固化为 Skill |
| `createdAt` | string | ISO 8601 UTC |

### 去重规则

两个 pattern 满足以下条件时视为重复，合并而非新增：
- `root_cause_type` 相同
- `symptom` 与新 pattern 的 `symptom_keywords` 的 keyword overlap > 50%

合并策略：
- `source_cases`: 追加
- `confidence`: 取 max
- `symptom_keywords`: union 去重

## 4. 组件设计

### 4.1 KnowledgeMiddleware

**文件：** `middlewares/knowledge.py`

**职责：** `after_agent` 做写入（检测评分 → 提取 → 存储），`before_model` 做读取（匹配 → 注入）。

```python
class KnowledgeMiddleware(AgentMiddleware):
    def __init__(self,
                 knowledge_path: str,
                 score_threshold: int = 7,
                 max_inject_patterns: int = 3,
                 extract_model: str | None = None,
                 max_extract_tokens: int = 8000):
        ...

    def after_agent(self, state, runtime) -> dict | None:
        """评分 ≥ threshold 时提取 pattern。"""

    def before_model(self, state, runtime) -> dict | None:
        """匹配 knowledge，注入 shortcut 到 system prompt。"""
```

**评分检测逻辑：**
- 扫最后 4 条 human messages
- 正则：`r'^\s*(\d{1,2})\s*[,，分]?\s*(.*)'`
- 提取 score (int) 和 comment (str)
- 仅在 agent 最近一条 AI message 包含 "评分" 关键词时才激活检测（防止误匹配普通数字）

**消息过滤逻辑（复用 MemoryMiddleware 思路）：**
- 保留 human messages + 不含 tool_calls 的 AI messages
- 去掉 `<uploaded_files>` 块
- trim 到 `max_extract_tokens` 预算

**Inject 逻辑：**
- 判断条件：messages 中只有 1 条 human message（即用户的首次描述）时才注入
- 后续轮次（messages 含多条 human message）不再注入，避免重复污染 context
- 注入位置：system message 尾部追加 `<diagnostic_knowledge>` 块

### 4.2 KnowledgeStore

**文件：** `knowledge/store.py`

**职责：** 读写 knowledge.json + keyword overlap 检索。

```python
class KnowledgeStore:
    def load(self) -> dict
    def add_pattern(self, pattern: dict) -> None       # 去重 + 合并 + 原子写入
    def match(self, text: str, top_k: int) -> list[dict]  # keyword overlap
```

**匹配算法：**
- `score = 命中关键词数 / 总关键词数`
- 排序：score 降序，同分按 confidence 降序
- 命中的 pattern 自动 `times_matched += 1`

**原子写入：** temp file + `os.replace()`（复用 FileMemoryStorage 模式）。

**mtime 缓存：** 文件未变时直接返回内存缓存，避免频繁 IO。

### 4.3 Extractor

**文件：** `knowledge/extractor.py`

**职责：** 调 LLM 提取 pattern。

```python
def extract_pattern(
    messages: list,
    score: int,
    comment: str | None,
    model_name: str | None = None,
) -> dict | None
```

- 格式化 messages 为 transcript 文本
- 填充 prompt 模板（score、comment、transcript）
- 调 LLM，解析 JSON 输出
- 返回 `None` 如果 LLM 输出 null 或解析失败
- 补充元数据：`id`、`source_cases`、`times_matched`、`createdAt`

### 4.4 Extract Prompt

**文件：** `knowledge/prompts.py`

**输入变量：** `{score}`, `{comment}`, `{transcript}`

**提取规则：**
1. 症状泛化——用通用描述替代具体字段名
2. 弯路提取——agent 最初尝试了什么、为什么不对
3. 捷径提炼——最短排查路径，"先查 X → 再查 Y"
4. 关键文件——保留文件名，去掉行号

**过滤规则：** 不记录具体字段名、模板 ID、category_id、商家信息。

**输出：** 严格 JSON，匹配 Pattern Schema。无完整诊断过程时输出 `null`。

**`root_cause_type` 枚举：**

| 值 | 含义 |
|----|------|
| `schema_config` | schema 配置问题（x_hidden、reaction_rules、validator） |
| `runtime_business_logic` | 运行时业务逻辑（条件渲染、权限判断） |
| `data_issue` | 数据问题（值错误、缺失、格式不对） |
| `permission` | 权限问题（角色、账户配置） |
| `dependency_change` | 依赖变更（上游接口改动、组件升级） |

### 4.5 Inject 格式

```xml
<diagnostic_knowledge>
以下是与当前问题相似的历史诊断经验，供参考：

【经验 1】(confidence: 0.95, 命中 3 次)
- 症状：商家反馈字段不显示/不能编辑
- 常见误判：容易误判为 schema 配置问题（x_hidden / reaction_rules）
- 推荐路径：先查 use-model.ts 条件渲染 → 再查 schema x_hidden → 最后查 reaction_rules
- 关键文件：use-model.ts
- 历史根因：runtime 业务逻辑控制（条件渲染）

注意：以上为历史经验，当前问题可能不同。如果排查路径与经验不符，按实际情况判断。
</diagnostic_knowledge>
```

## 5. SKILL.md 修改

在 `agents/oncall/skills/schema-diagnosis/SKILL.md` 末尾新增 Step 5：

```markdown
## Step 5: 评分收集

诊断结论呈现给用户后，**必须**主动请求评分：

> "本次诊断到此结束。请对诊断过程评分（1-10 分），可以附带评语。
> 评分标准：定位是否准确、过程是否高效、建议是否可操作。"

收到评分后回复确认即可，不需要额外操作（middleware 会自动处理后续提取）。
```

## 6. 配置

```yaml
# agents/oncall/agent.yaml — extra_middlewares 新增
- use: middlewares.knowledge:KnowledgeMiddleware
  config:
    knowledge_path: agents/oncall/knowledge.json
    score_threshold: 7
    max_inject_patterns: 3
    extract_model: null
    max_extract_tokens: 8000
```

## 7. 文件结构

```
middlewares/
    knowledge.py              # KnowledgeMiddleware
knowledge/
    __init__.py
    store.py                  # KnowledgeStore
    extractor.py              # extract_pattern()
    prompts.py                # EXTRACT_PATTERN_PROMPT
agents/oncall/
    knowledge.json            # 初始：{"version":"1.0","lastUpdated":"","patterns":[]}
    agent.yaml                # 新增 middleware 配置
    skills/schema-diagnosis/
        SKILL.md              # 新增 Step 5
```

## 8. 验证计划

1. **单元测试**：KnowledgeStore 的 add/match/merge/dedup
2. **单元测试**：评分检测 regex（正例 + 反例）
3. **单元测试**：extract prompt → mock LLM → 验证 JSON 解析
4. **集成测试**：middleware after_agent 完整流程（mock state with messages）
5. **集成测试**：middleware before_model inject（mock store with patterns）
6. **E2E**：跑一个真实 oncall case → 评分 8 → 检查 knowledge.json 是否写入
7. **E2E**：新 case 相似症状 → 检查 inject 是否命中
