# tests/test_config_merge.py
import tempfile
from pathlib import Path

import yaml


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data))


def test_merge_agent_overrides_model():
    """Agent-level model field overrides global default."""
    from cli.app import merge_agent_config

    global_cfg = {"default_agent": "oncall", "models": [{"name": "model-a"}]}
    agent_cfg = {"name": "oncall", "model": "model-b"}
    merged = merge_agent_config(global_cfg, agent_cfg)
    assert merged["model"] == "model-b"


def test_merge_mcp_servers_not_inherited():
    """MCP servers come only from agent config, not global."""
    from cli.app import merge_agent_config

    global_cfg = {"mcp_servers": [{"name": "global-mcp"}]}
    agent_cfg = {"name": "oncall", "mcp_servers": [{"name": "oncall-mcp"}]}
    merged = merge_agent_config(global_cfg, agent_cfg)
    assert len(merged["mcp_servers"]) == 1
    assert merged["mcp_servers"][0]["name"] == "oncall-mcp"


def test_merge_mcp_servers_empty_when_agent_has_none():
    """If agent defines no MCP servers, result has empty list."""
    from cli.app import merge_agent_config

    global_cfg = {"mcp_servers": [{"name": "global-mcp"}]}
    agent_cfg = {"name": "oncall"}
    merged = merge_agent_config(global_cfg, agent_cfg)
    assert merged["mcp_servers"] == []


def test_merge_models_inherited():
    """Models from global config are inherited when agent doesn't override."""
    from cli.app import merge_agent_config

    global_cfg = {"models": [{"name": "model-a"}]}
    agent_cfg = {"name": "oncall"}
    merged = merge_agent_config(global_cfg, agent_cfg)
    assert merged["models"] == [{"name": "model-a"}]


def test_load_agent_config_from_dir(tmp_path):
    """load_agent_config reads agent.yaml from agents/{name}/ dir."""
    from cli.app import load_agent_config

    agent_dir = tmp_path / "agents" / "oncall"
    agent_dir.mkdir(parents=True)
    _write_yaml(agent_dir / "agent.yaml", {"name": "oncall", "model": "test-model"})

    cfg = load_agent_config("oncall", project_root=tmp_path)
    assert cfg["name"] == "oncall"
    assert cfg["model"] == "test-model"
