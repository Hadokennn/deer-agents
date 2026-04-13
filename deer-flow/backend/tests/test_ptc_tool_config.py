"""Tests for PTCToolConfig and PTCEligibleToolConfig pydantic models."""

import pydantic
import pytest

from deerflow.config.tool_config import PTCEligibleToolConfig, PTCToolConfig


def test_eligible_tool_config_minimal():
    cfg = PTCEligibleToolConfig(name="bash")
    assert cfg.name == "bash"
    assert cfg.output_schema is None


def test_eligible_tool_config_with_output_schema():
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    cfg = PTCEligibleToolConfig(name="bash", output_schema=schema)
    assert cfg.output_schema == schema


def test_ptc_tool_config_minimal():
    cfg = PTCToolConfig(
        name="analyze_alerts",
        purpose="Use this when you need to analyze alerts",
        eligible_tools=[PTCEligibleToolConfig(name="bash")],
    )
    assert cfg.name == "analyze_alerts"
    assert cfg.timeout_seconds is None
    assert cfg.max_output_chars is None
    assert len(cfg.eligible_tools) == 1


def test_ptc_tool_config_with_overrides():
    cfg = PTCToolConfig(
        name="analyze_alerts",
        purpose="...",
        eligible_tools=[PTCEligibleToolConfig(name="bash")],
        timeout_seconds=60,
        max_output_chars=40000,
    )
    assert cfg.timeout_seconds == 60
    assert cfg.max_output_chars == 40000


def test_ptc_tool_config_rejects_empty_eligible_tools():
    with pytest.raises(pydantic.ValidationError):
        PTCToolConfig(
            name="analyze_alerts",
            purpose="...",
            eligible_tools=[],
        )


def test_ptc_tool_config_rejects_negative_timeout():
    with pytest.raises(pydantic.ValidationError):
        PTCToolConfig(
            name="analyze_alerts",
            purpose="...",
            eligible_tools=[PTCEligibleToolConfig(name="bash")],
            timeout_seconds=0,
        )


def test_ptc_tool_config_rejects_small_max_output():
    with pytest.raises(pydantic.ValidationError):
        PTCToolConfig(
            name="analyze_alerts",
            purpose="...",
            eligible_tools=[PTCEligibleToolConfig(name="bash")],
            max_output_chars=50,
        )


def test_ptc_tool_config_accepts_multiple_eligible_tools():
    cfg = PTCToolConfig(
        name="timeline_replay",
        purpose="...",
        eligible_tools=[
            PTCEligibleToolConfig(name="search_incident"),
            PTCEligibleToolConfig(name="get_alert_metrics"),
            PTCEligibleToolConfig(name="query_operation_log"),
        ],
    )
    assert len(cfg.eligible_tools) == 3
    assert cfg.eligible_tools[0].name == "search_incident"


def test_app_config_has_empty_ptc_tools_by_default():
    """AppConfig.ptc_tools defaults to empty list when not specified."""
    from deerflow.config.app_config import AppConfig
    from deerflow.config.sandbox_config import SandboxConfig

    cfg = AppConfig(
        sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"),
    )
    assert cfg.ptc_tools == []


def test_app_config_accepts_ptc_tools():
    """AppConfig.ptc_tools accepts a list of PTCToolConfig."""
    from deerflow.config.app_config import AppConfig
    from deerflow.config.sandbox_config import SandboxConfig

    cfg = AppConfig(
        sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"),
        ptc_tools=[
            {
                "name": "analyze_alerts",
                "purpose": "Use this to analyze alerts",
                "eligible_tools": [{"name": "bash"}],
            }
        ],
    )
    assert len(cfg.ptc_tools) == 1
    assert cfg.ptc_tools[0].name == "analyze_alerts"
    assert cfg.ptc_tools[0].eligible_tools[0].name == "bash"


def test_app_config_ptc_tools_validates_min_length():
    """Pydantic validation fires when eligible_tools is empty."""
    from deerflow.config.app_config import AppConfig
    from deerflow.config.sandbox_config import SandboxConfig

    with pytest.raises(pydantic.ValidationError):
        AppConfig(
            sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"),
            ptc_tools=[
                {
                    "name": "broken",
                    "purpose": "...",
                    "eligible_tools": [],
                }
            ],
        )
