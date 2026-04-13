"""Integration tests for PTC tool registration in get_available_tools()."""

from unittest.mock import MagicMock

import pytest

from deerflow.tools.tools import get_available_tools


def _make_fake_config(*, tools=None, ptc_tools=None, tool_search_enabled=False):
    """Build a MagicMock AppConfig with sensible defaults."""
    cfg = MagicMock()
    cfg.tools = tools or []
    cfg.models = []
    cfg.tool_search = MagicMock(enabled=tool_search_enabled)
    cfg.skill_evolution = None
    cfg.ptc_tools = ptc_tools or []
    cfg.get_model_config = MagicMock(return_value=None)
    return cfg


def test_ptc_tool_registered_when_eligible_tools_exist(monkeypatch):
    """A PTC tool whose eligible_tools all resolve should be registered."""
    from deerflow.config.tool_config import (
        PTCEligibleToolConfig,
        PTCToolConfig,
        ToolConfig,
    )

    fake_config = _make_fake_config(
        tools=[
            ToolConfig(
                name="bash",
                group="sandbox",
                use="deerflow.sandbox.tools:bash_tool",
            )
        ],
        ptc_tools=[
            PTCToolConfig(
                name="analyze_bash",
                purpose="Batch bash analysis",
                eligible_tools=[PTCEligibleToolConfig(name="bash")],
            )
        ],
    )
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: fake_config)
    monkeypatch.setattr("deerflow.tools.tools.is_host_bash_allowed", lambda c: True)

    tools = get_available_tools(include_mcp=False)
    tool_names = [t.name for t in tools]
    assert "bash" in tool_names
    assert "analyze_bash" in tool_names


def test_ptc_tool_skipped_when_eligible_tool_missing(monkeypatch):
    """If an eligible tool can't be found, the whole PTC tool is skipped."""
    from deerflow.config.tool_config import (
        PTCEligibleToolConfig,
        PTCToolConfig,
        ToolConfig,
    )

    fake_config = _make_fake_config(
        tools=[
            ToolConfig(
                name="bash",
                group="sandbox",
                use="deerflow.sandbox.tools:bash_tool",
            )
        ],
        ptc_tools=[
            PTCToolConfig(
                name="uses_missing",
                purpose="References a tool that doesn't exist",
                eligible_tools=[
                    PTCEligibleToolConfig(name="bash"),
                    PTCEligibleToolConfig(name="nonexistent_tool"),
                ],
            )
        ],
    )
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: fake_config)
    monkeypatch.setattr("deerflow.tools.tools.is_host_bash_allowed", lambda c: True)

    tools = get_available_tools(include_mcp=False)
    tool_names = [t.name for t in tools]
    assert "bash" in tool_names
    assert "uses_missing" not in tool_names  # PTC tool skipped, not an error


def test_ptc_tool_name_collision_raises(monkeypatch):
    """PTC tool name matching an existing tool triggers ValueError."""
    from deerflow.config.tool_config import (
        PTCEligibleToolConfig,
        PTCToolConfig,
        ToolConfig,
    )

    fake_config = _make_fake_config(
        tools=[
            ToolConfig(
                name="bash",
                group="sandbox",
                use="deerflow.sandbox.tools:bash_tool",
            )
        ],
        ptc_tools=[
            PTCToolConfig(
                name="bash",  # collides with existing tool
                purpose="Conflicts",
                eligible_tools=[PTCEligibleToolConfig(name="bash")],
            )
        ],
    )
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: fake_config)
    monkeypatch.setattr("deerflow.tools.tools.is_host_bash_allowed", lambda c: True)

    with pytest.raises(ValueError, match="collides"):
        get_available_tools(include_mcp=False)


def test_multiple_ptc_tools_registered(monkeypatch):
    """Multiple PTC tools can coexist with different eligible_tools."""
    from deerflow.config.tool_config import (
        PTCEligibleToolConfig,
        PTCToolConfig,
        ToolConfig,
    )

    fake_config = _make_fake_config(
        tools=[
            ToolConfig(name="bash", group="sandbox", use="deerflow.sandbox.tools:bash_tool"),
            ToolConfig(name="read_file", group="sandbox", use="deerflow.sandbox.tools:read_file_tool"),
        ],
        ptc_tools=[
            PTCToolConfig(
                name="batch_bash",
                purpose="Batch shell analysis",
                eligible_tools=[PTCEligibleToolConfig(name="bash")],
            ),
            PTCToolConfig(
                name="batch_read",
                purpose="Batch file reading",
                eligible_tools=[PTCEligibleToolConfig(name="read_file")],
            ),
        ],
    )
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: fake_config)
    monkeypatch.setattr("deerflow.tools.tools.is_host_bash_allowed", lambda c: True)

    tools = get_available_tools(include_mcp=False)
    tool_names = [t.name for t in tools]
    assert "batch_bash" in tool_names
    assert "batch_read" in tool_names


def test_no_ptc_tools_when_config_is_empty(monkeypatch):
    """Empty ptc_tools → no PTC tools registered, normal tools unaffected."""
    from deerflow.config.tool_config import ToolConfig

    fake_config = _make_fake_config(
        tools=[
            ToolConfig(
                name="bash",
                group="sandbox",
                use="deerflow.sandbox.tools:bash_tool",
            )
        ],
    )
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: fake_config)
    monkeypatch.setattr("deerflow.tools.tools.is_host_bash_allowed", lambda c: True)

    tools = get_available_tools(include_mcp=False)
    tool_names = [t.name for t in tools]
    assert "bash" in tool_names
    # No PTC tools should be added
    ptc_names = [n for n in tool_names if n.startswith("ptc_") or n.endswith("_ptc")]
    assert len(ptc_names) == 0
