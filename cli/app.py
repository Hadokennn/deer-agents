# cli/app.py
"""DeerApp — config loading, agent client lifecycle, session management."""

from pathlib import Path
from typing import Any

import yaml

# Fields that are agent-exclusive (not inherited from global config)
_AGENT_EXCLUSIVE_FIELDS = {"mcp_servers", "tool_groups", "extra_middlewares",
                           "code_repos", "knowledge_dirs", "skills_dir", "prompt"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_global_config(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load global config.yaml from project root."""
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Global config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def load_agent_config(agent_name: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load agent.yaml from agents/{name}/agent.yaml."""
    agent_path = project_root / "agents" / agent_name / "agent.yaml"
    if not agent_path.exists():
        raise FileNotFoundError(f"Agent config not found: {agent_path}")
    with open(agent_path) as f:
        return yaml.safe_load(f) or {}


def merge_agent_config(global_cfg: dict[str, Any], agent_cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge global and agent configs.

    Rules:
    - Agent-exclusive fields (mcp_servers, tool_groups, etc.) come only from agent config.
    - Other fields: agent overrides global (shallow merge).
    """
    merged = {**global_cfg, **agent_cfg}

    # Agent-exclusive fields: use agent's value or empty default
    for field in _AGENT_EXCLUSIVE_FIELDS:
        if field in agent_cfg:
            merged[field] = agent_cfg[field]
        else:
            # Default to empty list for list-like fields, None for others
            merged[field] = [] if field in {"mcp_servers", "tool_groups",
                                            "extra_middlewares", "code_repos",
                                            "knowledge_dirs"} else None

    return merged


def list_available_agents(project_root: Path = PROJECT_ROOT) -> list[str]:
    """List agent names by scanning agents/ directory."""
    agents_dir = project_root / "agents"
    if not agents_dir.exists():
        return []
    return sorted(
        d.name for d in agents_dir.iterdir()
        if d.is_dir() and (d / "agent.yaml").exists()
    )


def resolve_agent_name(requested: str | None, global_cfg: dict[str, Any],
                       project_root: Path = PROJECT_ROOT) -> str:
    """Resolve agent name: explicit arg > config default > first available."""
    if requested:
        return requested
    default = global_cfg.get("default_agent")
    if default:
        return default
    available = list_available_agents(project_root)
    if available:
        return available[0]
    raise RuntimeError("No agents found in agents/ directory")
