"""Code analysis subagent — reads source code and analyzes behavior.

Restricted to file system tools only (no MCP, no web, no subagent nesting).
Used by schema diagnosis to analyze component source code against schema config.
"""

from deerflow.subagents.config import SubagentConfig

CODE_ANALYST_CONFIG = SubagentConfig(
    name="code-analyst",
    description="""Source code analysis specialist for reading and analyzing code files.

Use this subagent when:
- You need to read source code files and analyze their behavior
- You need to correlate schema/config with runtime code logic
- You need to trace component rendering logic or business rules
- Analysis requires reading multiple files across a codebase

Do NOT use when:
- You need to call external APIs or MCP tools
- You need to search the web or fetch URLs
- You need to run build/test/deploy commands
- A simple single file read suffices (use read_file directly)""",
    system_prompt="""你是一个源码分析专家。你的任务是读取代码文件，分析其行为逻辑，结合提供的配置数据给出诊断结论。

<guidelines>
- 严格使用 bash 和 read_file 读取代码，不要调用其他工具
- 读取文件时优先使用 bash + sed 读取关键片段，避免读取整个大文件
- 分析代码逻辑时要引用具体的文件名、函数名、行号
- 不要猜测——如果代码中没有找到证据，明确说明"未找到相关逻辑"
- 完成分析后直接给出结论，不要发起额外的探索
</guidelines>

<output_format>
分析结果必须包含：
1. 问题根因（1-2 句话）
2. 关键证据（代码片段引用，格式：文件名:行号）
3. 解决建议（具体可操作的步骤）
</output_format>
""",
    tools=["bash", "ls", "read_file"],
    disallowed_tools=["task", "ask_clarification", "present_files"],
    model="inherit",
    max_turns=30,
)
