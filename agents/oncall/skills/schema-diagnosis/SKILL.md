---
name: schema-diagnosis
description: 定位商品模板 schema 配置问题。引导澄清五元组，调用 locate_template_schema 定位字段，决定直接分析或委派 sub-agent。Use when：用户反馈表单字段不显示、选项缺失、校验报错、组件行为异常，用户提到"模板"、"配置"、"schema"、"字段"、"组件"等关键词，需要查看某个类目/商品类型下字段的配置详情。
allowed-tools:
  - locate_template_schema
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

**匹配规则：** 读取 `./referrences/template_properties_map.json` 的枚举，做以下匹配：
1. 先尝试匹配 `product_sub_type`（更精确），如"权益卡"→ `101`
2. 匹配不上 `product_sub_type` 再匹配 `product_type`，如"团购"→ `1`
3. 都匹配不上就只传 `category_full_name`，让 `locate_template` 返回所有匹配

**尽量拿到：**
- `field_name` — 用户关心的具体字段名称（标题、key、组件名均可）

## Step 2: 调用 locate_template_schema

## Step 3: 委派 sub-agent 分析

**强制规则：当 `locate_template_schema` 返回包含 `next_action` 字段时，必须使用 `task` 工具委派 sub-agent。不要自己分析。**

> 这不是并行分解，而是 **context 隔离**——源码分析会产生大量输出，必须隔离到 sub-agent 中，避免污染 lead agent 的上下文。即使只有 1 个任务，也必须委派。

**委派方式：**
1. 使用 `task` 工具，**subagent_type 指定为 `"code-analyst"`**
2. 将 `next_action` 字段的完整内容作为 task 的 prompt
3. 不要修改、截断、或省略 `next_action` 的内容

**不需要委派的情况（仅当 `next_action` 不存在时）：**
- `locate_template_schema` 返回的字段信息已足够清晰（无组件源码需要分析）
- 直接根据 `reaction_rules`、`validator_rules` 等给出结论

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
- `locate_template_schema` 已经把全量 schema 存到了 `schema_path`，sub-agent 可以直接 `read_file` 读取
- 全量 schema 通常 100KB+，**绝对不要**把全量 schema 放进对话上下文
- sub-agent 已经做过的工作不要重复，只针对性补充缺失的部分
