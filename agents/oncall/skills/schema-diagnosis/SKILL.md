---
name: schema-diagnosis
description: 定位商品模板 schema 配置问题。引导澄清五元组，调用 locate_field_schema 定位字段，决定直接分析或委派 sub-agent。
allowed-tools:
  - locate_field_schema
  - task
  - bash
  - read_file
---

# 商品模板 Schema 诊断

用户反馈表单字段问题（不显示、缺选项、校验异常）时，定位到具体的 schema 配置。

## Use when

- 用户反馈表单字段不显示、选项缺失、校验报错、组件行为异常
- 用户提到"模板"、"配置"、"schema"、"字段"、"组件"等关键词
- 需要查看某个类目/商品类型下字段的配置详情

## Don't use when

- 用户反馈的是字段**值**错误（数据问题，不是配置问题）
- 用户在问代码逻辑而非模板配置
- 问题跟商品模板无关（如告警、日志、部署）

## Step 1: 澄清五元组

从用户描述中提取以下信息。缺少的**必须向用户确认**，不要猜测。

**必须拿到：**
- `category_full_name` — 类目路径，格式 `一级>二级>三级`（如 `购物>果蔬生鲜>水果`）。能精确到叶子节点最好，至少要到二级。

**尽量拿到：**
- 商品类型 — 用户可能说"团购"、"代金券"、"配送"等中文名称

**匹配规则：** 读取 `./template_properties_map.json` 的枚举，做以下匹配：
1. 先尝试匹配 `product_sub_type`（更精确），如"权益卡"→ `101`
2. 匹配不上 `product_sub_type` 再匹配 `product_type`，如"团购"→ `1`
3. 都匹配不上就只传 `category_full_name`，让 `locate_template` 返回所有匹配

**可选：**
- `field_name` — 用户关心的具体字段名称（标题、key、组件名均可）

## Step 2: 调用 locate_field_schema

根据 Step 1 的结果，调用 `locate_field_schema` 工具：

```
locate_field_schema(
  category_full_name="购物>果蔬生鲜>水果",
  product_type="1",          # 如果匹配到了才传
  field_name="商家名称"       # 如果用户指定了才传
)
```

**关键：不传的参数不要传空字符串，直接不传。**

处理返回结果：
- `status: "found"` → 进入 Step 3
- `status: "ambiguous"` → 把 candidates 展示给用户，让用户选择，然后重新调用
- `status: "not_found"` → 告知用户未找到，建议检查类目名称
- `status: "field_not_found"` → 展示 `available_fields` 列表，让用户确认字段名
- `status: "schema_error"` → 报告错误，检查 MCP 连接

## Step 3: 分析结果 — 直接回答 or 委派 sub-agent

**判断规则：**

### 直接分析（返回字段 ≤ 3 个）
如果 `locate_field_schema` 返回了 `status: "found"` 且字段信息清晰：
- 直接根据返回的 `reaction_rules`、`validator_rules`、`x_component_props` 等分析问题原因
- 结合用户描述给出结论

### 委派 sub-agent（以下任一条件成立）
- 用户问题涉及**多个字段的联动关系**
- 返回的 schema 片段包含复杂的 `reaction_rules` 需要交叉分析
- 需要对比全量 schema 中多个字段（超过 3 个）的配置
- 不确定问题根因，需要更深入分析

委派方式：直接将 `locate_field_schema` 返回的 `next_action` 字段内容作为 task prompt。`next_action` 已包含完整的 sub-agent prompt（源码读取命令、schema 数据、分析要点）。

## Step 4: 评估 sub-agent 结果并决策

Sub-agent 负责执行（读代码、分析 schema），lead agent 负责**感知和决策**。

### 评估 sub-agent 输出

收到 sub-agent 结果后，检查以下三点：

1. **是否读了源码？** — 结果中是否包含具体的代码片段引用（函数名、行号、逻辑描述），而非"可能"、"推测"等措辞
2. **根因是否明确？** — 是否指出了具体原因（如"组件 X 在 app.tsx:25 行判断了 platform === 'pc'"），而非泛泛的"可能是平台差异"
3. **证据链是否完整？** — schema 配置 + 代码逻辑是否对得上，能否解释用户看到的现象

### 决策

| 评估结果 | 行动 |
|----------|------|
| 三点都满足 | **直接呈现**：整理 sub-agent 的分析，格式化输出给用户 |
| 缺源码分析（sub-agent 没读到代码或只是推测）| **针对性补充**：自己用 bash 读 sub-agent 没读到的那几个文件，补充代码分析 |
| 根因不明确 | **追问 sub-agent**：再派一次 task，缩小范围，指定要分析的具体文件和问题 |
| schema 数据不够 | **针对性补充**：read_file 读全量 schema 中缺失的关联字段，补充后自己给结论 |

**关键原则：不要全盘重做。** sub-agent 已经做过的工作（已读的文件、已分析的配置）不要重复。只补充缺失的部分。

### 输出格式
1. 问题根因（1-2 句话说清楚）
2. 关键证据（schema 配置 + 代码逻辑，引用具体的字段配置和代码位置）
3. 解决建议（具体可操作的步骤）

## Step 5: 评分收集

诊断结论呈现给用户后，**必须**主动请求评分：

> "本次诊断到此结束。请对诊断过程评分（1-10 分），可以附带评语。
> 评分标准：定位是否准确、过程是否高效、建议是否可操作。"

收到评分后回复确认即可，不需要额外操作（middleware 会自动处理后续提取）。

## 重要注意事项

- 五元组信息不确定时**必须问用户**，不要假设
- `locate_field_schema` 已经把全量 schema 存到了 `schema_path`，sub-agent 可以直接 `read_file` 读取
- 全量 schema 通常 100KB+，**绝对不要**把全量 schema 放进对话上下文
- sub-agent 已经做过的工作不要重复，只针对性补充缺失的部分
