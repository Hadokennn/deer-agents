"""Tests for pipelines/loader.py — agent integration helper."""

from pathlib import Path

from langchain_core.tools import StructuredTool

from pipelines.loader import load_pipelines_for_agent


def _echo_tool() -> StructuredTool:
    def _run(value: str = "") -> dict:
        return {"echoed": value}
    return StructuredTool.from_function(func=_run, name="echo", description="Echo")


def _write_pipeline_yaml(dir_path: Path, name: str, body: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{name}.yaml").write_text(body, encoding="utf-8")


def test_load_no_pipeline_dir_returns_base_tools(tmp_path):
    base = [_echo_tool()]
    result = load_pipelines_for_agent(tmp_path, base)
    assert result == base


def test_load_empty_pipeline_dir_returns_base_tools(tmp_path):
    (tmp_path / "pipelines").mkdir()
    base = [_echo_tool()]
    result = load_pipelines_for_agent(tmp_path, base)
    assert result == base


def test_load_one_pipeline_appends_pipeline_tool(tmp_path):
    _write_pipeline_yaml(tmp_path / "pipelines", "wrap_echo", """
name: wrap_echo
description: Echoes via the echo tool
input:
  text:
    type: str
    description: Text to echo
steps:
  - id: e
    tool: echo
    input:
      value: ${input.text}
output:
  echoed: ${e.echoed}
""")
    base = [_echo_tool()]
    result = load_pipelines_for_agent(tmp_path, base)

    assert len(result) == 2
    pipeline_tool = next(t for t in result if t.name == "wrap_echo")
    invoked = pipeline_tool.invoke({"text": "hi"})
    assert invoked == {"echoed": "hi"}


def test_load_pipeline_can_call_another_pipeline(tmp_path):
    _write_pipeline_yaml(tmp_path / "pipelines", "inner", """
name: inner
description: Inner pipeline
input:
  v:
    type: str
    description: value
steps:
  - id: e
    tool: echo
    input:
      value: ${input.v}
output:
  echoed: ${e.echoed}
""")
    _write_pipeline_yaml(tmp_path / "pipelines", "outer", """
name: outer
description: Outer pipeline that calls inner
input:
  text:
    type: str
    description: text
steps:
  - id: call_inner
    tool: inner
    input:
      v: ${input.text}
output:
  result: ${call_inner.echoed}
""")
    base = [_echo_tool()]
    result = load_pipelines_for_agent(tmp_path, base)

    outer_tool = next(t for t in result if t.name == "outer")
    assert outer_tool.invoke({"text": "nested"}) == {"result": "nested"}


def test_load_preserves_base_tool_order(tmp_path):
    base_a = StructuredTool.from_function(
        func=lambda: {"x": 1}, name="a", description="A"
    )
    base_b = StructuredTool.from_function(
        func=lambda: {"x": 2}, name="b", description="B"
    )
    base = [base_a, base_b]

    result = load_pipelines_for_agent(tmp_path, base)
    assert result[0] is base_a
    assert result[1] is base_b
