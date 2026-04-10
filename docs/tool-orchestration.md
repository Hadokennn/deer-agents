# Tool Orchestration

deer-agents has **two complementary tool orchestration mechanisms** that work
side by side. The LLM picks between them based on the problem.

**Spec:** `docs/superpowers/specs/2026-04-09-pipeline-engine-design.md`

## TL;DR

| Use case | Mechanism |
|---|---|
| Single tool call | Just call the tool directly (no orchestration needed) |
| Known multi-step path, high frequency | **Mode 1: Pipeline DSL** (YAML) |
| Unknown path, exploratory | **Mode 2: CodeAct** (LLM writes Python) |

The two modes are not replacements for each other — they cover different
points on the "is the path known?" spectrum.

## Mode 1: Declarative Pipeline DSL

Write a YAML file under `agents/<your_agent>/pipelines/`. The pipeline becomes
a regular tool the LLM can call. Inside the pipeline, code interprets the
YAML — no LLM in the loop.

### When to use it
- Path is fixed: you know the steps before runtime.
- Steps chain existing tools (MCP, Python, even other pipelines).
- The logic between steps is just variable passing + conditional branches.

### When NOT to use it
- The flow needs LLM judgment between steps — use Skill SOP instead.
- The flow needs loops or parallelism — use Mode 2 (CodeAct) or write a Python tool.
- A single tool call would do — just call the tool directly.

### Example

```yaml
# agents/oncall/pipelines/my_pipeline.yaml
name: my_pipeline
description: |
  One sentence about what this does.
  Use when: <specific trigger>.
  Don't use when: <specific anti-trigger>.

input:
  query:
    type: str
    description: What this parameter is for

steps:
  - id: step_one
    tool: existing_tool_name
    input:
      param: ${input.query}

  - id: step_two
    tool: another_tool
    input:
      data: ${step_one.result}
    when: ${step_one.success}

  - id: optional_step
    tool: maybe_fails
    input: {}
    optional: true

output:
  result: ${step_one.result | step_two.result}
  source: ${step_one.source | "default"}
```

### Variable language

| Syntax | Meaning |
|---|---|
| `${input.field}` | Pipeline input parameter |
| `${step_id.field}` | A previous step's output field |
| `${step_id.nested.field}` | Nested dict access |
| `${a \| b \| c}` | First non-None wins |
| `"text ${var} text"` | String interpolation (returns str) |
| `${var}` (whole string) | Returns the raw value (preserves type) |
| `"literal"` | Literal string in fallback chains |

### Step modifiers

| Field | Effect |
|---|---|
| `when: <expr>` | Run only if expression is truthy |
| `unless: <expr>` | Run only if expression is falsy |
| `optional: true` | Tool failure or skip sets context to None instead of raising |

### Wiring it up

```python
from pathlib import Path
from pipelines import load_pipelines_for_agent

agent_dir = Path("agents/oncall")
all_tools = load_pipelines_for_agent(agent_dir, base_tools)
```

The loader scans `<agent_dir>/pipelines/*.yaml` and returns
`base_tools + pipeline_tools`. Pipeline tools are also registered in the
internal registry, so a pipeline can call another pipeline.

### Example file

`agents/oncall/pipelines/example_lookup.yaml` shows fallback resolution +
conditional steps.

---

## Mode 2: Generative CodeAct Executor

The LLM submits a chunk of Python code that runs in a restricted sandbox with
all configured tools bound as Python functions. The LLM gets back stdout +
return value, never sees intermediate tool calls.

### When to use it
- Path is unknown: you don't know the steps before seeing the data.
- The flow has conditionals that depend on inspecting intermediate results.
- You want to reduce LLM round-trips by chaining several operations in one shot.

### When NOT to use it
- A single tool call is enough — call that tool directly.
- A pre-defined pipeline matches the pattern — use Mode 1.
- You need real-time streaming or multi-turn interaction.

### Wiring it up

```python
from codeact import make_code_act_tool

code_executor = make_code_act_tool(available_tools=all_tools)
agent_tools = all_tools + [code_executor]
```

`available_tools` can include pipeline tools from Mode 1. They'll appear in
the LLM's code as regular Python functions.

### Sandbox boundaries

Default sandbox configuration:
- **Allowed imports**: `json, re, math, datetime, collections, itertools, functools, operator, typing, string`
- **Blocked builtins**: `open, eval, exec, compile, __import__` (only the safe `__import__` proxy is exposed via the import whitelist)
- **Timeout**: 10 seconds (configurable)
- **Stdout limit**: 50 KB (truncated past this)
- **No network, no subprocess, no file system access** (use `file_read` / `file_write` tools)

### Threat model

The sandbox protects against **accidental misuse** (LLM writing buggy code,
infinite loops, accidental imports), **not adversarial code**. Real isolation
needs gVisor / Firecracker, deferred to v3.

### Example file

`tests/test_codeact_e2e.py` shows realistic LLM-style code calling a
pipeline tool from within the sandbox.

---

## How they fit together

| Aspect | Mode 1 (Pipeline DSL) | Mode 2 (CodeAct) |
|---|---|---|
| Who decides flow | Programmer (write YAML) | LLM (write code at runtime) |
| Decision time | Write time | Runtime |
| Flexibility | Limited (DSL) | Maximum (Python) |
| Predictability | High | Low |
| New scenario cost | Write YAML file | Zero |
| Best for | High-frequency known paths | Long-tail / exploratory |

**LLM holds both as tools and picks based on the problem.** The platform
evolution is: start with everything in CodeAct, observe what high-frequency
patterns emerge, codify those into Mode 1 pipelines.

## Boundaries (v1)

Not supported in either mode:
- Streaming output
- Multi-turn interaction within one orchestration call
- True process isolation (Mode 2 sandbox is in-process)

Mode 1 specifically does not support: parallelism, loops, dynamic step generation.
Mode 2 specifically does not support: async code, persistent state across calls,
network access, automatic retry on code failure.

If you need any of these, write a Python tool instead (see `tools/schema_locator.py`
for the "complex Python tool" pattern).
