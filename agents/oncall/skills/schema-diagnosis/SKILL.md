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

委派方式：
```
task(
  description="分析商品模板 schema 配置问题",
  prompt="用户问题：{用户原始问题}\n\n已定位到模板 schema 文件：{schema_path}\n\n请读取该文件，分析以下字段的配置：{相关字段列表}\n\n重点关注：reaction_rules（显隐条件）、validator_rules（校验规则）、x-component-props（组件属性）",
  subagent_type="general-purpose"
)
```

**sub-agent prompt 要包含：**
1. 用户原始问题
2. schema 文件路径（`schema_path`）
3. 需要关注的字段列表
4. 分析重点（显隐/校验/组件行为）

## 重要注意事项

- 五元组信息不确定时**必须问用户**，不要假设
- `locate_field_schema` 已经把全量 schema 存到了 `schema_path`，sub-agent 可以直接 `read_file` 读取
- 全量 schema 通常 100KB+，**绝对不要**把全量 schema 放进对话上下文
