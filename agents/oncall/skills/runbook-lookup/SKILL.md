---
name: runbook-lookup
description: 在本地 knowledge 目录中查找与当前问题相关的 runbook
---

## Use when
- 用户描述了一个线上问题或告警
- 需要查找已有的处理方案

## Don't use when
- 用户在问架构设计问题
- 问题明显不在已有 runbook 覆盖范围内

## Steps
1. 从用户描述中提取关键词（服务名、错误类型）
2. 在 knowledge/ 目录下 grep 相关文件
3. 读取匹配的 runbook 内容
4. 总结处理步骤并呈现给用户
