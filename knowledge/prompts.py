"""Prompt templates for diagnostic pattern extraction."""

EXTRACT_PATTERN_PROMPT = """\
你是一个诊断经验总结系统。分析刚完成的 oncall 诊断对话，提取可泛化的诊断模式。

用户评分：{score}/10
用户评语：{comment}

对话记录：
<transcript>
{transcript}
</transcript>

## 提取规则

1. **症状泛化**：用通用描述替代具体字段名
   - 正确："商家反馈字段不显示/不能编辑"
   - 错误："outId 字段不显示"

2. **弯路提取**：agent 最初尝试了什么方向、为什么不对、最终转向了哪里
   - 这是最有价值的信息，认真分析对话中的排查过程

3. **捷径提炼**：如果下次遇到类似症状，最短路径是什么
   - 格式："先查 X → 再查 Y → 最后查 Z"

4. **关键文件**：哪些文件/函数是定位问题的关键（保留文件名，去掉具体行号）

## 不要记录的（case 细节，不可泛化）
- 具体字段名（outId / price / accountName）
- 具体模板 ID / category_id / 商家信息
- 具体的 schema 配置值

## 输出 JSON（严格遵循，不要增减字段）

```json
{{
  "symptom": "一句话描述症状模式",
  "symptom_keywords": ["关键词1", "关键词2"],
  "misdiagnosis_trap": "容易走的弯路",
  "actual_root_cause": "真正的根因",
  "root_cause_type": "<schema_config | runtime_business_logic | data_issue | permission | dependency_change>",
  "diagnostic_shortcut": "先查 X → 再查 Y → 最后查 Z",
  "key_files": ["file1.ts"],
  "resolution": "解决建议",
  "confidence": 0.85
}}
```

只输出 JSON，不要其他文字。如果对话中没有完整的诊断过程（比如只是闲聊），输出 null。\
"""
