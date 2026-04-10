"""YAML → Pipeline dataclass parsing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pipelines.errors import PipelineDefinitionError


@dataclass
class PipelineStep:
    """One step in a pipeline."""

    id: str
    tool: str
    input: dict[str, Any]
    when: str | None = None
    unless: str | None = None
    optional: bool = False


@dataclass
class Pipeline:
    """A complete pipeline definition loaded from YAML."""

    name: str
    description: str
    input_schema: dict[str, dict[str, Any]]
    steps: list[PipelineStep]
    output_template: dict[str, Any] = field(default_factory=dict)


_REQUIRED_TOP_FIELDS = ("name", "description", "input", "steps")
_REQUIRED_STEP_FIELDS = ("id", "tool", "input")


class PipelineParser:
    """Loads and validates pipeline YAML files."""

    @staticmethod
    def parse_file(yaml_path: Path) -> Pipeline:
        try:
            data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise PipelineDefinitionError(f"{yaml_path}: invalid YAML: {e}") from e

        if not isinstance(data, dict):
            raise PipelineDefinitionError(f"{yaml_path}: top level must be a mapping")

        for field_name in _REQUIRED_TOP_FIELDS:
            if field_name not in data:
                raise PipelineDefinitionError(
                    f"{yaml_path}: missing required field '{field_name}'"
                )

        steps = PipelineParser._parse_steps(data["steps"], yaml_path)

        return Pipeline(
            name=str(data["name"]),
            description=str(data["description"]),
            input_schema=dict(data.get("input") or {}),
            steps=steps,
            output_template=dict(data.get("output") or {}),
        )

    @staticmethod
    def parse_dir(dir_path: Path) -> list[Pipeline]:
        path = Path(dir_path)
        if not path.exists():
            return []
        return [PipelineParser.parse_file(f) for f in sorted(path.glob("*.yaml"))]

    @staticmethod
    def _parse_steps(steps_data: Any, yaml_path: Path) -> list[PipelineStep]:
        if not isinstance(steps_data, list):
            raise PipelineDefinitionError(f"{yaml_path}: 'steps' must be a list")

        steps: list[PipelineStep] = []
        seen_ids: set[str] = set()
        for idx, step_data in enumerate(steps_data):
            if not isinstance(step_data, dict):
                raise PipelineDefinitionError(
                    f"{yaml_path}: step {idx} must be a mapping"
                )
            for field_name in _REQUIRED_STEP_FIELDS:
                if field_name not in step_data:
                    raise PipelineDefinitionError(
                        f"{yaml_path}: step {idx} missing '{field_name}'"
                    )

            step_id = str(step_data["id"])
            if step_id in seen_ids:
                raise PipelineDefinitionError(
                    f"{yaml_path}: duplicate step id '{step_id}'"
                )
            seen_ids.add(step_id)

            steps.append(
                PipelineStep(
                    id=step_id,
                    tool=str(step_data["tool"]),
                    input=dict(step_data["input"] or {}),
                    when=step_data.get("when"),
                    unless=step_data.get("unless"),
                    optional=bool(step_data.get("optional", False)),
                )
            )
        return steps
