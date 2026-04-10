"""Prompt templates for the CodeAct tool description."""

CODE_ACT_DESCRIPTION_TEMPLATE = """\
Execute Python code in a sandbox with access to all configured tools.

Use when:
- The task needs multiple tool calls with conditional logic, and the conditions \
depend on intermediate results.
- A pre-defined pipeline does not match the use case (long-tail, exploratory, \
ad-hoc investigation).
- You want to reduce LLM round-trips by chaining several operations in one shot.

Don't use when:
- A single tool call is enough — just call that tool directly.
- A pre-defined pipeline tool exists for this exact pattern — call that pipeline.
- You need real-time interaction or streaming output.

How to write the code:
- Set the variable `result` to mark the value you want returned.
- Use `print()` for diagnostic output that you want to read back.
- Available tools are pre-bound as Python functions (signatures listed below).
- You can use: list, dict, str, int, sorted, len, json, re, math, datetime.
- You CANNOT use: open, eval, exec, import os, import subprocess, network access.
- Code runs with a {timeout}s timeout.

Available tools (callable from your code):

{tool_signatures}
"""
