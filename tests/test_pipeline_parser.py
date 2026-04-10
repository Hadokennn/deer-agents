"""Tests for pipelines/parser.py — YAML to Pipeline dataclass parsing."""

import pytest

from pipelines.errors import PipelineDefinitionError
from pipelines.parser import Pipeline, PipelineParser, PipelineStep


def _write_yaml(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_parse_minimal_pipeline(tmp_path):
    yaml_path = _write_yaml(tmp_path, "minimal.yaml", """
name: minimal
description: A minimal pipeline
input:
  query:
    type: str
    description: A query string
steps:
  - id: search
    tool: web_search
    input:
      q: ${input.query}
output:
  results: ${search.items}
""")
    pipeline = PipelineParser.parse_file(yaml_path)

    assert isinstance(pipeline, Pipeline)
    assert pipeline.name == "minimal"
    assert pipeline.description == "A minimal pipeline"
    assert pipeline.input_schema == {"query": {"type": "str", "description": "A query string"}}
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0].id == "search"
    assert pipeline.steps[0].tool == "web_search"
    assert pipeline.steps[0].input == {"q": "${input.query}"}
    assert pipeline.steps[0].when is None
    assert pipeline.steps[0].unless is None
    assert pipeline.steps[0].optional is False
    assert pipeline.output_template == {"results": "${search.items}"}


def test_parse_pipeline_with_conditions(tmp_path):
    yaml_path = _write_yaml(tmp_path, "conditions.yaml", """
name: with_conditions
description: Pipeline with when/unless/optional
input:
  service:
    type: str
    description: Service name
steps:
  - id: a
    tool: tool_a
    input: {}
  - id: b
    tool: tool_b
    input: {}
    when: ${a.found}
  - id: c
    tool: tool_c
    input: {}
    unless: ${a.found}
    optional: true
output:
  result: ${b.value | c.value}
""")
    pipeline = PipelineParser.parse_file(yaml_path)

    assert len(pipeline.steps) == 3
    assert pipeline.steps[1].when == "${a.found}"
    assert pipeline.steps[1].unless is None
    assert pipeline.steps[2].when is None
    assert pipeline.steps[2].unless == "${a.found}"
    assert pipeline.steps[2].optional is True


def test_parse_dir_loads_all_yaml(tmp_path):
    _write_yaml(tmp_path, "p1.yaml", """
name: p1
description: First
input: {}
steps:
  - id: s
    tool: t
    input: {}
output: {}
""")
    _write_yaml(tmp_path, "p2.yaml", """
name: p2
description: Second
input: {}
steps:
  - id: s
    tool: t
    input: {}
output: {}
""")
    (tmp_path / "readme.txt").write_text("not a pipeline")

    pipelines = PipelineParser.parse_dir(tmp_path)
    names = sorted(p.name for p in pipelines)
    assert names == ["p1", "p2"]


def test_parse_dir_missing_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert PipelineParser.parse_dir(missing) == []


def test_parse_missing_name_raises(tmp_path):
    yaml_path = _write_yaml(tmp_path, "bad.yaml", """
description: Missing name
input: {}
steps:
  - id: s
    tool: t
    input: {}
output: {}
""")
    with pytest.raises(PipelineDefinitionError, match="missing required field 'name'"):
        PipelineParser.parse_file(yaml_path)


def test_parse_missing_steps_raises(tmp_path):
    yaml_path = _write_yaml(tmp_path, "bad.yaml", """
name: bad
description: No steps
input: {}
output: {}
""")
    with pytest.raises(PipelineDefinitionError, match="missing required field 'steps'"):
        PipelineParser.parse_file(yaml_path)


def test_parse_step_missing_id_raises(tmp_path):
    yaml_path = _write_yaml(tmp_path, "bad.yaml", """
name: bad
description: Step missing id
input: {}
steps:
  - tool: t
    input: {}
output: {}
""")
    with pytest.raises(PipelineDefinitionError, match="step 0 missing 'id'"):
        PipelineParser.parse_file(yaml_path)


def test_parse_duplicate_step_ids_raises(tmp_path):
    yaml_path = _write_yaml(tmp_path, "bad.yaml", """
name: bad
description: Duplicate step ids
input: {}
steps:
  - id: a
    tool: t1
    input: {}
  - id: a
    tool: t2
    input: {}
output: {}
""")
    with pytest.raises(PipelineDefinitionError, match="duplicate step id 'a'"):
        PipelineParser.parse_file(yaml_path)
