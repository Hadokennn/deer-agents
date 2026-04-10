"""End-to-end test: load example_lookup.yaml, register a real tool, invoke it."""

from pathlib import Path

from langchain_core.tools import StructuredTool

from pipelines import load_pipelines_for_agent

ONCALL_AGENT_DIR = Path(__file__).parent.parent / "agents" / "oncall"


_STORE: dict[str, dict[str, str]] = {
    "service-a": {"runbook": "Restart service-a via systemd"},
    "default": {
        "runbook": "Generic runbook: check logs first",
        "ping": "Generic ping playbook",
    },
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


def test_example_lookup_pipeline_loads_and_invokes():
    base_tools = [_kv_lookup_tool()]
    all_tools = load_pipelines_for_agent(ONCALL_AGENT_DIR, base_tools)

    pipeline_tool = next((t for t in all_tools if t.name == "example_lookup"), None)
    assert pipeline_tool is not None, "example_lookup pipeline tool not loaded"
    assert "example pipeline for oncall" in pipeline_tool.description.lower()


def test_example_lookup_finds_in_primary_namespace():
    all_tools = load_pipelines_for_agent(ONCALL_AGENT_DIR, [_kv_lookup_tool()])
    pipeline_tool = next(t for t in all_tools if t.name == "example_lookup")

    result = pipeline_tool.invoke({"key": "runbook", "namespace": "service-a"})
    assert result["value"] == "Restart service-a via systemd"
    assert result["source"] == "service-a"


def test_example_lookup_falls_back_to_default():
    all_tools = load_pipelines_for_agent(ONCALL_AGENT_DIR, [_kv_lookup_tool()])
    pipeline_tool = next(t for t in all_tools if t.name == "example_lookup")

    result = pipeline_tool.invoke({"key": "ping", "namespace": "service-a"})
    assert result["value"] == "Generic ping playbook"
    assert result["source"] == "default"


def test_example_lookup_returns_not_found():
    all_tools = load_pipelines_for_agent(ONCALL_AGENT_DIR, [_kv_lookup_tool()])
    pipeline_tool = next(t for t in all_tools if t.name == "example_lookup")

    result = pipeline_tool.invoke({"key": "nope", "namespace": "service-a"})
    assert result["value"] is None
    assert result["source"] == "not_found"
