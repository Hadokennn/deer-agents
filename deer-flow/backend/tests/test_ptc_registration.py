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


# ---------- Tool-search deferred interaction ----------
#
# Architecture note: get_available_tools() always returns ALL tools (including
# deferred ones) so that ToolNode can execute them. The DeferredToolFilterMiddleware
# reads the deferred registry at model-bind time and removes deferred tools from
# the LLM's visible schema. Therefore these tests check the deferred REGISTRY
# state, not the get_available_tools() return list.


def _get_deferred_names() -> set[str]:
    """Return the set of tool names currently in the deferred registry."""
    from deerflow.tools.builtins.tool_search import get_deferred_registry
    registry = get_deferred_registry()
    if registry is None:
        return set()
    return {e.name for e in registry.entries}


def _make_fake_mcp_tool(name: str):
    class _FakeMcpTool:
        def __init__(self, n):
            self.name = n
            self.metadata = {}
            self.args_schema = None
            self.description = f"fake mcp tool {n}"
            def _func(**kwargs):
                return "ok"
            self.func = _func
    return _FakeMcpTool(name)


def test_mcp_tools_referenced_by_ptc_are_not_deferred(monkeypatch):
    """MCP tools named in a PTC tool's eligible_tools must NOT be in the
    deferred registry, so the LLM can see their input schema and write
    correct code in the PTC wrapper."""
    from deerflow.config.tool_config import (
        PTCEligibleToolConfig,
        PTCToolConfig,
    )

    fake_config = _make_fake_config(
        tools=[],
        ptc_tools=[
            PTCToolConfig(
                name="query_metrics_batch",
                purpose="Batch metric queries",
                eligible_tools=[PTCEligibleToolConfig(name="get_alert_metrics")],
            )
        ],
        tool_search_enabled=True,  # deferred loading is ON
    )

    referenced = _make_fake_mcp_tool("get_alert_metrics")
    not_referenced = _make_fake_mcp_tool("something_else")
    fake_mcp_tools = [referenced, not_referenced]

    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: fake_config)
    monkeypatch.setattr("deerflow.tools.tools.is_host_bash_allowed", lambda c: True)
    monkeypatch.setattr("deerflow.mcp.cache.get_cached_mcp_tools", lambda: fake_mcp_tools)

    fake_ext = MagicMock()
    fake_ext.get_enabled_mcp_servers = MagicMock(return_value={"srv": object()})
    monkeypatch.setattr(
        "deerflow.config.extensions_config.ExtensionsConfig.from_file",
        lambda: fake_ext,
    )

    get_available_tools(include_mcp=True)
    deferred = _get_deferred_names()

    # The referenced MCP tool must NOT be in the deferred registry
    assert "get_alert_metrics" not in deferred

    # The non-referenced MCP tool SHOULD be in the deferred registry
    assert "something_else" in deferred


def test_all_mcp_tools_visible_when_all_referenced_by_ptc(monkeypatch):
    """If every MCP tool is referenced by PTC, the deferred registry is
    empty (nothing deferred)."""
    from deerflow.config.tool_config import (
        PTCEligibleToolConfig,
        PTCToolConfig,
    )

    fake_mcp_tools = [_make_fake_mcp_tool("mcp_one"), _make_fake_mcp_tool("mcp_two")]

    fake_config = _make_fake_config(
        tools=[],
        ptc_tools=[
            PTCToolConfig(
                name="batch_workflow",
                purpose="Uses all MCP tools",
                eligible_tools=[
                    PTCEligibleToolConfig(name="mcp_one"),
                    PTCEligibleToolConfig(name="mcp_two"),
                ],
            )
        ],
        tool_search_enabled=True,
    )

    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: fake_config)
    monkeypatch.setattr("deerflow.tools.tools.is_host_bash_allowed", lambda c: True)
    monkeypatch.setattr("deerflow.mcp.cache.get_cached_mcp_tools", lambda: fake_mcp_tools)
    fake_ext = MagicMock()
    fake_ext.get_enabled_mcp_servers = MagicMock(return_value={"srv": object()})
    monkeypatch.setattr(
        "deerflow.config.extensions_config.ExtensionsConfig.from_file",
        lambda: fake_ext,
    )

    get_available_tools(include_mcp=True)
    deferred = _get_deferred_names()

    # All MCP tools referenced by PTC — none should be deferred
    assert "mcp_one" not in deferred
    assert "mcp_two" not in deferred


def test_tool_search_unchanged_when_no_ptc_tools(monkeypatch):
    """Sanity: with PTC absent and tool_search ON, all MCP tools go to
    the deferred registry (original behavior unchanged)."""
    fake_mcp_tools = [_make_fake_mcp_tool("alpha"), _make_fake_mcp_tool("beta")]

    fake_config = _make_fake_config(
        tools=[],
        ptc_tools=[],
        tool_search_enabled=True,
    )
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: fake_config)
    monkeypatch.setattr("deerflow.tools.tools.is_host_bash_allowed", lambda c: True)
    monkeypatch.setattr("deerflow.mcp.cache.get_cached_mcp_tools", lambda: fake_mcp_tools)
    fake_ext = MagicMock()
    fake_ext.get_enabled_mcp_servers = MagicMock(return_value={"srv": object()})
    monkeypatch.setattr(
        "deerflow.config.extensions_config.ExtensionsConfig.from_file",
        lambda: fake_ext,
    )

    tools = get_available_tools(include_mcp=True)
    tool_names = [getattr(t, "name", str(t)) for t in tools]
    deferred = _get_deferred_names()

    # Deferred path: both MCP tools in the deferred registry, tool_search in the tool list
    assert "alpha" in deferred
    assert "beta" in deferred
    assert "tool_search" in tool_names
