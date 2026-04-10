"""End-to-end: CodeAct executes code that uses a Mode 1 pipeline tool.

This is the integration test that proves Mode 1 + Mode 2 compose:
- Phase 1 produced a pipeline tool (example_lookup) wrapping kv_lookup
- Phase 2 produces a code_execute tool that can call the pipeline as a function
- The "LLM-written" code in this test (which is just hand-written by the test author)
  uses both the raw kv_lookup tool and the example_lookup pipeline tool
"""

from pathlib import Path

from langchain_core.tools import StructuredTool

from codeact import make_code_act_tool
from pipelines import load_pipelines_for_agent

ONCALL_AGENT_DIR = Path(__file__).parent.parent / "agents" / "oncall"


_STORE: dict[str, dict[str, str]] = {
    "service-a": {"runbook": "Restart service-a via systemd"},
    "default": {"runbook": "Generic runbook: check logs first"},
}


def _kv_lookup_tool() -> StructuredTool:
    def _run(namespace: str, key: str) -> dict:
        ns = _STORE.get(namespace, {})
        if key in ns:
            return {"value": ns[key], "source": namespace, "missing": False}
        return {"value": None, "source": None, "missing": True}

    return StructuredTool.from_function(
        func=_run,
        name="kv_lookup",
        description="Look up a key in an in-memory store",
    )


def _build_full_toolset() -> list:
    """Replicates what an agent setup would do: base tools + pipelines + codeact."""
    base = [_kv_lookup_tool()]
    with_pipelines = load_pipelines_for_agent(ONCALL_AGENT_DIR, base)
    code_act = make_code_act_tool(available_tools=with_pipelines)
    return with_pipelines + [code_act]


def _get_code_executor(tools):
    return next(t for t in tools if t.name == "code_execute")


def test_codeact_can_call_raw_kv_lookup_tool():
    tools = _build_full_toolset()
    code_executor = _get_code_executor(tools)

    code = """
result = kv_lookup(namespace="service-a", key="runbook")
"""
    out = code_executor.invoke({"code": code})

    assert out["success"] is True
    assert out["return_value"]["value"] == "Restart service-a via systemd"
    assert out["return_value"]["source"] == "service-a"


def test_codeact_can_call_pipeline_tool_as_function():
    tools = _build_full_toolset()
    code_executor = _get_code_executor(tools)

    code = """
result = example_lookup(key="runbook", namespace="service-a")
"""
    out = code_executor.invoke({"code": code})

    assert out["success"] is True
    assert out["return_value"]["value"] == "Restart service-a via systemd"
    assert out["return_value"]["source"] == "service-a"


def test_codeact_writes_multistep_logic_with_conditionals():
    """The realistic case: LLM writes multi-step exploration code."""
    tools = _build_full_toolset()
    code_executor = _get_code_executor(tools)

    code = """
# Try service-a first; if missing, try via pipeline (which has its own fallback)
direct = kv_lookup(namespace="service-a", key="missing_key")

if direct["missing"]:
    via_pipeline = example_lookup(key="runbook", namespace="service-a")
    print(f"direct missing, pipeline says: {via_pipeline['source']}")
    result = {
        "found_via": "pipeline",
        "value": via_pipeline["value"],
        "source": via_pipeline["source"],
    }
else:
    result = {
        "found_via": "direct",
        "value": direct["value"],
        "source": direct["source"],
    }
"""
    out = code_executor.invoke({"code": code})

    assert out["success"] is True
    assert out["return_value"]["found_via"] == "pipeline"
    assert out["return_value"]["value"] == "Restart service-a via systemd"
    assert "direct missing" in out["stdout"]


def test_codeact_handles_tool_exception_gracefully():
    """If LLM-written code calls a tool that errors, the exception is captured."""
    tools = _build_full_toolset()
    code_executor = _get_code_executor(tools)

    code = """
try:
    result = kv_lookup(namespace=42, key="runbook")
except Exception as e:
    result = {"error": type(e).__name__}
"""
    out = code_executor.invoke({"code": code})

    assert out["success"] is True  # the code itself didn't crash
    assert "error" in out["return_value"]
