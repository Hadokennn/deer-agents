# Deer Agents CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI agent system on top of DeerFlow harness, with a unified REPL shell, agent switching, session resume, and an oncall Q&A agent as the first agent.

**Architecture:** Thin CLI shell (prompt_toolkit + rich) wrapping DeerFlowClient. Each agent is a directory of config/prompt/skills/knowledge. Custom middlewares inject into deer-flow's middleware chain via a small patch to `_build_middlewares()`. SQLite checkpointer enables session resume.

**Tech Stack:** Python 3.12+, DeerFlow harness (fork), prompt_toolkit, rich, LangGraph SqliteSaver, PyYAML

---

## File Structure

```
deer-agents/
├── deer-flow/                          ← git subtree from bytedance/deer-flow
├── agents/
│   └── oncall/
│       ├── agent.yaml                  ← agent config (model, tools, MCP, middlewares)
│       ├── prompt.md                   ← system prompt
│       ├── skills/
│       │   └── runbook-lookup/
│       │       └── SKILL.md
│       └── knowledge/
│           └── common-issues.md        ← placeholder runbook
├── cli/
│   ├── __init__.py
│   ├── __main__.py                     ← entry point
│   ├── app.py                          ← DeerApp: config loading, client lifecycle
│   ├── shell.py                        ← REPL loop (prompt_toolkit)
│   ├── commands.py                     ← /switch, /agents, /sessions, /resume, /help, /exit
│   ├── renderer.py                     ← stream event → rich terminal output
│   └── sessions.py                     ← session metadata CRUD (~/.deer-agents/sessions/)
├── middlewares/
│   ├── __init__.py
│   └── mcp_overflow.py                 ← MCP response overflow → sandbox file
├── tests/
│   ├── __init__.py
│   ├── test_config_merge.py
│   ├── test_commands.py
│   ├── test_sessions.py
│   └── test_mcp_overflow.py
├── config.yaml                         ← global config
├── pyproject.toml
└── README.md
```

---

### Task 1: Project Scaffold + deer-flow Subtree

**Files:**
- Create: `pyproject.toml`
- Create: `cli/__init__.py`
- Create: `middlewares/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/bytedance/Documents/aime/deer-agents
git init
```

- [ ] **Step 2: Create .gitignore**

```gitignore
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/
.env
~/.deer-agents/
```

- [ ] **Step 3: Add deer-flow as git subtree**

```bash
cd /Users/bytedance/Documents/aime/deer-agents
git add .gitignore
git commit -m "chore: initial commit with gitignore"
git remote add deer-flow-upstream git@github.com:bytedance/deer-flow.git
git fetch deer-flow-upstream
git subtree add --prefix=deer-flow deer-flow-upstream main --squash
```

This brings the entire deer-flow repo under `deer-flow/`. Future syncs:
```bash
git subtree pull --prefix=deer-flow deer-flow-upstream main --squash
```

- [ ] **Step 4: Create pyproject.toml**

```toml
[project]
name = "deer-agents"
version = "0.1.0"
description = "CLI agent system built on DeerFlow harness"
requires-python = ">=3.12"
dependencies = [
    "prompt_toolkit>=3.0.0",
    "rich>=13.0.0",
    "pyyaml>=6.0",
]

[project.scripts]
deer = "cli.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 5: Create empty __init__.py files**

Create these empty files:
- `cli/__init__.py`
- `middlewares/__init__.py`
- `tests/__init__.py`

- [ ] **Step 6: Set up Python environment and install dependencies**

```bash
cd /Users/bytedance/Documents/aime/deer-agents
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e deer-flow/backend/packages/harness/
pip install -e .
```

- [ ] **Step 7: Verify deerflow is importable**

```bash
source .venv/bin/activate
python -c "from deerflow.client import DeerFlowClient; print('OK')"
```

Expected: `OK`

- [ ] **Step 8: Commit scaffold**

```bash
git add pyproject.toml cli/ middlewares/ tests/
git commit -m "chore: project scaffold with deer-flow subtree"
```

---

### Task 2: Config Merging Layer

**Files:**
- Create: `cli/app.py`
- Create: `config.yaml`
- Create: `agents/oncall/agent.yaml`
- Test: `tests/test_config_merge.py`

- [ ] **Step 1: Write the failing test for config merging**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/bytedance/Documents/aime/deer-agents
source .venv/bin/activate
python -m pytest tests/test_config_merge.py -v
```

Expected: FAIL — `cli.app` does not exist yet.

- [ ] **Step 3: Implement config merging in cli/app.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_config_merge.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Create global config.yaml**

```yaml
# config.yaml — Global configuration for deer-agents
default_agent: oncall

models:
  - name: doubao-seed-1.8
    display_name: Doubao-Seed-1.8
    use: deerflow.models.patched_deepseek:PatchedChatDeepSeek
    model: doubao-seed-1-8-251228
    api_base: https://ark.cn-beijing.volces.com/api/v3
    api_key: $VOLCENGINE_API_KEY
    supports_thinking: true
    supports_vision: true

sandbox:
  type: local

checkpointer:
  type: sqlite
  path: ~/.deer-agents/checkpoints.db

sessions:
  dir: ~/.deer-agents/sessions/
```

- [ ] **Step 6: Create oncall agent.yaml**

```yaml
# agents/oncall/agent.yaml
name: oncall
display_name: "Oncall 答疑助手"
description: "业务 oncall 答疑，连接告警平台和 runbook"

model: doubao-seed-1.8
thinking_enabled: true

tool_groups:
  - web
  - file
  - bash

mcp_servers: []
# Example:
#  - name: alert-platform
#    command: npx
#    args: ["alert-mcp-server", "--env=prod"]

extra_middlewares:
  - use: middlewares.mcp_overflow:McpOverflowMiddleware
    config:
      max_response_size: 8192
      sandbox_path: /tmp/mcp_responses/

code_repos: []
# Example:
#  - name: payment-service
#    path: /Users/bytedance/code/payment-service

knowledge_dirs:
  - ./agents/oncall/knowledge/

skills_dir: ./agents/oncall/skills/

prompt: ./agents/oncall/prompt.md
```

- [ ] **Step 7: Commit**

```bash
git add cli/app.py config.yaml agents/oncall/agent.yaml tests/test_config_merge.py
git commit -m "feat: config merging layer with global + agent override"
```

---

### Task 3: Session Metadata Manager

**Files:**
- Create: `cli/sessions.py`
- Test: `tests/test_sessions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sessions.py
import json
from pathlib import Path


def test_create_and_list_sessions(tmp_path):
    """Create a session and verify it appears in list."""
    from cli.sessions import SessionManager

    mgr = SessionManager(sessions_dir=tmp_path)
    mgr.create("thread-abc", agent_name="oncall")

    sessions = mgr.list_all()
    assert len(sessions) == 1
    assert sessions[0]["thread_id"] == "thread-abc"
    assert sessions[0]["agent_name"] == "oncall"
    assert sessions[0]["title"] is None


def test_update_session_title(tmp_path):
    """Updating title persists to disk."""
    from cli.sessions import SessionManager

    mgr = SessionManager(sessions_dir=tmp_path)
    mgr.create("thread-1", agent_name="oncall")
    mgr.update("thread-1", title="redis 排查")

    reloaded = mgr.get("thread-1")
    assert reloaded["title"] == "redis 排查"


def test_list_sessions_sorted_by_last_active(tmp_path):
    """Sessions are sorted most-recent-first."""
    from cli.sessions import SessionManager
    import time

    mgr = SessionManager(sessions_dir=tmp_path)
    mgr.create("old", agent_name="oncall")
    time.sleep(0.01)
    mgr.create("new", agent_name="review")

    sessions = mgr.list_all()
    assert sessions[0]["thread_id"] == "new"
    assert sessions[1]["thread_id"] == "old"


def test_get_nonexistent_session(tmp_path):
    """Getting a nonexistent session returns None."""
    from cli.sessions import SessionManager

    mgr = SessionManager(sessions_dir=tmp_path)
    assert mgr.get("does-not-exist") is None


def test_touch_updates_last_active(tmp_path):
    """Touching a session updates its last_active_at."""
    from cli.sessions import SessionManager

    mgr = SessionManager(sessions_dir=tmp_path)
    mgr.create("thread-1", agent_name="oncall")
    original = mgr.get("thread-1")["last_active_at"]

    import time
    time.sleep(0.01)
    mgr.touch("thread-1")
    updated = mgr.get("thread-1")["last_active_at"]
    assert updated > original
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_sessions.py -v
```

Expected: FAIL — `cli.sessions` does not exist.

- [ ] **Step 3: Implement SessionManager**

```python
# cli/sessions.py
"""Session metadata CRUD — thin JSON files in ~/.deer-agents/sessions/."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SessionManager:
    """Manage session metadata files.

    Each session is stored as {thread_id}.json in the sessions directory.
    The actual conversation state lives in the LangGraph checkpointer (SQLite);
    these files only hold lightweight metadata (title, agent_name, timestamps).
    """

    def __init__(self, sessions_dir: Path | str):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, thread_id: str) -> Path:
        return self.sessions_dir / f"{thread_id}.json"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(self, thread_id: str, agent_name: str) -> dict[str, Any]:
        """Create a new session metadata file."""
        now = self._now()
        data = {
            "thread_id": thread_id,
            "agent_name": agent_name,
            "title": None,
            "created_at": now,
            "last_active_at": now,
        }
        self._path(thread_id).write_text(json.dumps(data, indent=2))
        return data

    def get(self, thread_id: str) -> dict[str, Any] | None:
        """Get session metadata by thread_id. Returns None if not found."""
        path = self._path(thread_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def update(self, thread_id: str, **fields) -> dict[str, Any] | None:
        """Update specific fields of a session."""
        data = self.get(thread_id)
        if data is None:
            return None
        data.update(fields)
        data["last_active_at"] = self._now()
        self._path(thread_id).write_text(json.dumps(data, indent=2))
        return data

    def touch(self, thread_id: str) -> None:
        """Update last_active_at timestamp."""
        self.update(thread_id)

    def list_all(self) -> list[dict[str, Any]]:
        """List all sessions, sorted by last_active_at descending."""
        sessions = []
        for path in self.sessions_dir.glob("*.json"):
            try:
                sessions.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        sessions.sort(key=lambda s: s.get("last_active_at", ""), reverse=True)
        return sessions
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_sessions.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/sessions.py tests/test_sessions.py
git commit -m "feat: session metadata manager with CRUD operations"
```

---

### Task 4: Stream Renderer

**Files:**
- Create: `cli/renderer.py`

- [ ] **Step 1: Implement the stream renderer**

This component renders `StreamEvent` objects to the terminal using `rich`. It's hard to unit test (terminal output), so we test it manually in Task 8.

```python
# cli/renderer.py
"""Render DeerFlowClient StreamEvents to the terminal using rich."""

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

console = Console()


def render_stream(stream_events) -> str | None:
    """Consume a stream of StreamEvents, rendering AI text incrementally.

    Args:
        stream_events: Generator of StreamEvent from DeerFlowClient.stream()

    Returns:
        The final title if one was generated, else None.
    """
    collected_text = ""
    title = None

    with Live("", console=console, refresh_per_second=8, vertical_overflow="visible") as live:
        for event in stream_events:
            if event.type == "values":
                # Full state snapshot — extract title
                title = event.data.get("title") or title

            elif event.type == "messages-tuple":
                # Per-message update: (message_dict, metadata_dict)
                msg_data = event.data
                if isinstance(msg_data, (list, tuple)) and len(msg_data) >= 1:
                    msg = msg_data[0]
                    # AI text content
                    content = ""
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                    else:
                        content = getattr(msg, "content", "")

                    if isinstance(content, str) and content:
                        collected_text = content
                        live.update(Markdown(collected_text))

            elif event.type == "end":
                break

    # Final render with full markdown
    if collected_text:
        console.print()  # newline after live display

    return title
```

- [ ] **Step 2: Commit**

```bash
git add cli/renderer.py
git commit -m "feat: rich stream renderer for terminal output"
```

---

### Task 5: CLI Commands

**Files:**
- Create: `cli/commands.py`
- Test: `tests/test_commands.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_commands.py


def test_parse_command_switch():
    from cli.commands import parse_command
    cmd = parse_command("/switch review")
    assert cmd.name == "switch"
    assert cmd.args == "review"


def test_parse_command_no_args():
    from cli.commands import parse_command
    cmd = parse_command("/agents")
    assert cmd.name == "agents"
    assert cmd.args == ""


def test_parse_command_resume():
    from cli.commands import parse_command
    cmd = parse_command("/resume thread-abc")
    assert cmd.name == "resume"
    assert cmd.args == "thread-abc"


def test_parse_non_command_returns_none():
    from cli.commands import parse_command
    assert parse_command("hello world") is None
    assert parse_command("") is None


def test_parse_command_exit():
    from cli.commands import parse_command
    cmd = parse_command("/exit")
    assert cmd.name == "exit"


def test_parse_command_help():
    from cli.commands import parse_command
    cmd = parse_command("/help")
    assert cmd.name == "help"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_commands.py -v
```

Expected: FAIL — `cli.commands` does not exist.

- [ ] **Step 3: Implement commands**

```python
# cli/commands.py
"""CLI commands — /switch, /agents, /sessions, /resume, /help, /exit, /status."""

from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class ParsedCommand:
    name: str
    args: str


def parse_command(user_input: str) -> ParsedCommand | None:
    """Parse a /command from user input. Returns None if not a command."""
    text = user_input.strip()
    if not text.startswith("/"):
        return None
    parts = text[1:].split(maxsplit=1)
    name = parts[0] if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    return ParsedCommand(name=name, args=args)


def handle_help() -> None:
    """Print available commands."""
    console.print("\n[bold]Available commands:[/bold]")
    console.print("  /switch <agent>   Switch to a different agent")
    console.print("  /agents           List available agents")
    console.print("  /sessions         List previous sessions")
    console.print("  /resume <id>      Resume a previous session")
    console.print("  /status           Show current agent status")
    console.print("  /help             Show this help")
    console.print("  /exit             Exit deer")
    console.print()


def handle_agents(available: list[str], current: str) -> None:
    """Display available agents with active marker."""
    console.print()
    for name in available:
        marker = "[green]● active[/green]" if name == current else "○"
        console.print(f"  {name:16s} {marker}")
    console.print()


def handle_sessions(sessions: list[dict[str, Any]]) -> None:
    """Display session history."""
    if not sessions:
        console.print("\n  No previous sessions.\n")
        return
    console.print()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Agent", style="cyan")
    table.add_column("Thread ID")
    table.add_column("Last Active")
    table.add_column("Title")
    for s in sessions[:20]:  # Show last 20
        table.add_row(
            s.get("agent_name", "?"),
            s.get("thread_id", "?"),
            s.get("last_active_at", "?")[:16],
            s.get("title") or "(untitled)",
        )
    console.print(table)
    console.print()


def handle_status(agent_name: str, thread_id: str, config: dict[str, Any]) -> None:
    """Display current agent status."""
    console.print(f"\n  Agent:     [cyan]{agent_name}[/cyan]")
    console.print(f"  Thread:    {thread_id}")
    console.print(f"  Model:     {config.get('model', 'default')}")
    mcp_count = len(config.get("mcp_servers", []))
    console.print(f"  MCP:       {mcp_count} server(s)")
    console.print()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_commands.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/commands.py tests/test_commands.py
git commit -m "feat: CLI command parser and display handlers"
```

---

### Task 6: REPL Shell + App Entry Point

**Files:**
- Create: `cli/shell.py`
- Create: `cli/__main__.py`

- [ ] **Step 1: Implement the REPL shell**

```python
# cli/shell.py
"""Interactive REPL shell using prompt_toolkit."""

import uuid

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console

from cli.app import (
    PROJECT_ROOT,
    list_available_agents,
    load_agent_config,
    load_global_config,
    merge_agent_config,
    resolve_agent_name,
)
from cli.commands import (
    handle_agents,
    handle_help,
    handle_sessions,
    handle_status,
    parse_command,
)
from cli.renderer import render_stream
from cli.sessions import SessionManager

console = Console()


class DeerShell:
    """Interactive agent shell."""

    def __init__(self, agent_name: str | None = None):
        self.global_cfg = load_global_config()
        self.agent_name = resolve_agent_name(agent_name, self.global_cfg)
        self.agent_cfg = self._load_merged_config(self.agent_name)
        self.thread_id = str(uuid.uuid4())[:8]
        self.client = None  # Lazy init

        # Session management
        sessions_dir = self.global_cfg.get("sessions", {}).get("dir", "~/.deer-agents/sessions/")
        from pathlib import Path
        self.session_mgr = SessionManager(Path(sessions_dir).expanduser())

        # Prompt history
        history_path = Path("~/.deer-agents/history").expanduser()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_session = PromptSession(history=FileHistory(str(history_path)))

    def _load_merged_config(self, agent_name: str) -> dict:
        agent_cfg = load_agent_config(agent_name)
        return merge_agent_config(self.global_cfg, agent_cfg)

    def _ensure_client(self):
        """Lazy-initialize or re-create DeerFlowClient."""
        if self.client is not None:
            return

        from deerflow.client import DeerFlowClient
        from langgraph.checkpoint.sqlite import SqliteSaver
        from pathlib import Path

        # Set up checkpointer
        cp_cfg = self.global_cfg.get("checkpointer", {})
        cp_path = Path(cp_cfg.get("path", "~/.deer-agents/checkpoints.db")).expanduser()
        cp_path.parent.mkdir(parents=True, exist_ok=True)

        self._checkpointer = SqliteSaver.from_conn_string(str(cp_path))
        self._checkpointer.setup()

        # Point DeerFlowClient at our project's config.yaml
        config_path = str(PROJECT_ROOT / "config.yaml")

        self.client = DeerFlowClient(
            config_path=config_path,
            checkpointer=self._checkpointer,
            model_name=self.agent_cfg.get("model"),
            thinking_enabled=self.agent_cfg.get("thinking_enabled", True),
            agent_name=self.agent_name,
        )

    def _switch_agent(self, new_agent: str) -> bool:
        """Switch to a different agent. Returns True on success."""
        available = list_available_agents()
        if new_agent not in available:
            console.print(f"  [red]Unknown agent: {new_agent}[/red]")
            console.print(f"  Available: {', '.join(available)}")
            return False

        self.agent_name = new_agent
        self.agent_cfg = self._load_merged_config(new_agent)
        self.thread_id = str(uuid.uuid4())[:8]
        self.client = None  # Force re-create on next message
        console.print(f"  [green]✓ Switched to {new_agent} agent[/green]")
        return True

    def _resume_session(self, thread_id: str) -> bool:
        """Resume a previous session. Returns True on success."""
        session = self.session_mgr.get(thread_id)
        if session is None:
            console.print(f"  [red]Session not found: {thread_id}[/red]")
            return False

        agent_name = session["agent_name"]
        self.agent_name = agent_name
        self.agent_cfg = self._load_merged_config(agent_name)
        self.thread_id = thread_id
        self.client = None  # Force re-create
        title = session.get("title") or "(untitled)"
        console.print(f"  [green]✓ Resumed {agent_name} session: \"{title}\"[/green]")
        return True

    def _handle_command(self, cmd) -> bool:
        """Handle a parsed command. Returns False if shell should exit."""
        if cmd.name == "exit":
            return False
        elif cmd.name == "help":
            handle_help()
        elif cmd.name == "agents":
            handle_agents(list_available_agents(), self.agent_name)
        elif cmd.name == "switch":
            if not cmd.args:
                console.print("  Usage: /switch <agent_name>")
            else:
                self._switch_agent(cmd.args.strip())
        elif cmd.name == "sessions":
            handle_sessions(self.session_mgr.list_all())
        elif cmd.name == "resume":
            if not cmd.args:
                console.print("  Usage: /resume <thread_id>")
            else:
                self._resume_session(cmd.args.strip())
        elif cmd.name == "status":
            handle_status(self.agent_name, self.thread_id, self.agent_cfg)
        else:
            console.print(f"  Unknown command: /{cmd.name}. Type /help for available commands.")
        return True

    def _send_message(self, text: str) -> None:
        """Send a message to the current agent and render the response."""
        self._ensure_client()

        # Create session if this is the first message on this thread
        if self.session_mgr.get(self.thread_id) is None:
            self.session_mgr.create(self.thread_id, agent_name=self.agent_name)

        try:
            events = self.client.stream(text, thread_id=self.thread_id)
            title = render_stream(events)

            # Update session metadata
            if title:
                self.session_mgr.update(self.thread_id, title=title)
            else:
                self.session_mgr.touch(self.thread_id)

        except KeyboardInterrupt:
            console.print("\n  [yellow]Interrupted[/yellow]")
        except Exception as e:
            console.print(f"\n  [red]Error: {e}[/red]")

    def run(self) -> None:
        """Main REPL loop."""
        console.print(f"\n  [bold]🦌 Deer Agents[/bold] — {self.agent_name} agent ready")
        console.print(f"  Type /help for commands, /exit to quit\n")

        while True:
            try:
                # Dynamic prompt showing current agent
                prompt_text = f"🦌 {self.agent_name} > " if self.agent_name != self.global_cfg.get("default_agent") else "🦌 > "
                user_input = self.prompt_session.prompt(prompt_text)

                if not user_input.strip():
                    continue

                # Check for commands
                cmd = parse_command(user_input)
                if cmd is not None:
                    if not self._handle_command(cmd):
                        break
                    continue

                # Regular message — send to agent
                self._send_message(user_input)

            except KeyboardInterrupt:
                continue  # Ctrl+C just cancels current input
            except EOFError:
                break  # Ctrl+D exits

        console.print("\n  [dim]Bye 👋[/dim]\n")
```

- [ ] **Step 2: Implement the entry point**

```python
# cli/__main__.py
"""Entry point for `deer` CLI or `python -m cli`."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="deer",
        description="CLI agent system built on DeerFlow harness",
    )
    parser.add_argument(
        "--agent", "-a",
        type=str,
        default=None,
        help="Agent to start with (default: from config.yaml)",
    )
    args = parser.parse_args()

    from cli.shell import DeerShell
    shell = DeerShell(agent_name=args.agent)
    shell.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke test — verify CLI starts**

```bash
cd /Users/bytedance/Documents/aime/deer-agents
source .venv/bin/activate
python -m cli --help
```

Expected: Shows help text with `--agent` flag.

- [ ] **Step 4: Commit**

```bash
git add cli/shell.py cli/__main__.py
git commit -m "feat: interactive REPL shell with agent switching and session resume"
```

---

### Task 7: Oncall Agent Definition (Prompt + Skill + Knowledge)

**Files:**
- Create: `agents/oncall/prompt.md`
- Create: `agents/oncall/skills/runbook-lookup/SKILL.md`
- Create: `agents/oncall/knowledge/common-issues.md`

- [ ] **Step 1: Create oncall prompt**

```markdown
# agents/oncall/prompt.md

# Oncall 答疑助手

你是一个业务 oncall 答疑 agent，帮助工程师快速定位和解决线上问题。

## 你的能力
- 查询告警平台获取实时告警详情（通过 MCP 工具）
- 搜索日志定位异常（通过 MCP 工具）
- 查阅本地 runbook 和架构文档（knowledge/ 目录）
- 阅读业务代码定位根因（code_repos 配置的本地仓库）

## 工作流程
1. **先理解问题** — 确认告警名称、服务名、时间范围
2. **查数据** — 拉告警详情、搜相关日志
3. **查文档** — 看 runbook 里有没有已知处理方案
4. **查代码** — 如果需要定位根因，读相关代码
5. **给结论** — 明确的处理建议，附带证据

## 约束
- 不确定时说"不确定"，不要编造处理方案
- 涉及数据变更操作必须给出命令但不自动执行
- 每次回答附带信息来源（哪个告警、哪段日志、哪个文件）
```

- [ ] **Step 2: Create runbook-lookup skill**

```markdown
# agents/oncall/skills/runbook-lookup/SKILL.md
---
name: runbook-lookup
description: 在本地 knowledge 目录中查找与当前问题相关的 runbook
---

## Use when
- 用户描述了一个线上问题或告警
- 需要查找已有的处理方案

## Don't use when
- 用户在问架构设计问题
- 问题明显不在已有 runbook 覆盖范围内

## Steps
1. 从用户描述中提取关键词（服务名、错误类型）
2. 在 knowledge/ 目录下 grep 相关文件
3. 读取匹配的 runbook 内容
4. 总结处理步骤并呈现给用户
```

- [ ] **Step 3: Create placeholder knowledge doc**

```markdown
# agents/oncall/knowledge/common-issues.md

# 常见问题 Runbook

## Redis 连接超时

**症状**: `redis.exceptions.ConnectionError: Connection timed out`

**常见原因**:
1. 连接池 `max_connections` 不足 — 检查 QPS 与连接池大小的比例
2. Redis 实例内存满 — 检查 `used_memory` 和 `maxmemory`
3. 网络分区 — 检查 pod 和 redis 实例之间的网络连通性

**处理步骤**:
1. 确认告警时间范围和影响服务
2. 查看 redis 监控面板（连接数、内存、QPS）
3. 检查服务端连接池配置
4. 如果是连接池不足：临时调大 `REDIS_MAX_CONN` 环境变量并滚动重启

---

## MySQL 慢查询

**症状**: 接口 P99 延迟飙升，数据库 CPU 告警

**常见原因**:
1. 缺少索引 — `EXPLAIN` 查看执行计划
2. 全表扫描 — 检查 WHERE 条件
3. 锁等待 — 检查 `SHOW PROCESSLIST`

**处理步骤**:
1. 从慢查询日志找到 TOP SQL
2. EXPLAIN 分析执行计划
3. 临时方案：KILL 阻塞查询
4. 长期方案：添加索引或优化 SQL
```

- [ ] **Step 4: Commit**

```bash
git add agents/oncall/prompt.md agents/oncall/skills/ agents/oncall/knowledge/
git commit -m "feat: oncall agent definition with prompt, skill, and knowledge"
```

---

### Task 8: MCP Overflow Middleware

**Files:**
- Create: `middlewares/mcp_overflow.py`
- Modify: `deer-flow/backend/packages/harness/deerflow/agents/lead_agent/agent.py:208-265`
- Test: `tests/test_mcp_overflow.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcp_overflow.py
from unittest.mock import MagicMock


def test_small_response_passes_through():
    """Responses under threshold are not modified."""
    from middlewares.mcp_overflow import McpOverflowMiddleware

    mw = McpOverflowMiddleware(max_response_size=100)
    # Simulate a small tool response
    result = mw.process_tool_response(content="short response", tool_call_id="call-1")
    assert result == "short response"


def test_large_response_gets_replaced(tmp_path):
    """Responses over threshold are written to file and replaced with pointer."""
    from middlewares.mcp_overflow import McpOverflowMiddleware

    sandbox_path = str(tmp_path) + "/"
    mw = McpOverflowMiddleware(max_response_size=50, sandbox_path=sandbox_path)

    big_content = "x" * 200
    result = mw.process_tool_response(content=big_content, tool_call_id="call-abc")

    # Result should be a pointer, not the original content
    assert "call-abc" in result
    assert "read_file" in result
    assert len(result) < len(big_content)

    # File should exist on disk
    from pathlib import Path
    written_file = Path(sandbox_path) / "call-abc.txt"
    assert written_file.exists()
    assert written_file.read_text() == big_content


def test_non_string_content_passes_through():
    """Non-string content (e.g., list) is not processed."""
    from middlewares.mcp_overflow import McpOverflowMiddleware

    mw = McpOverflowMiddleware(max_response_size=10)
    result = mw.process_tool_response(content=["not", "a", "string"], tool_call_id="call-1")
    assert result == ["not", "a", "string"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_mcp_overflow.py -v
```

Expected: FAIL — `middlewares.mcp_overflow` does not exist.

- [ ] **Step 3: Implement MCP overflow middleware**

```python
# middlewares/mcp_overflow.py
"""MCP Overflow Middleware — write oversized tool responses to sandbox files."""

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


class McpOverflowMiddleware(AgentMiddleware[AgentState]):
    """Intercept oversized tool responses and write them to sandbox files.

    When a tool response exceeds max_response_size bytes, the content is
    written to a file in sandbox_path and the ToolMessage content is replaced
    with a pointer instructing the agent to use read_file.
    """

    def __init__(self, max_response_size: int = 8192, sandbox_path: str = "/tmp/mcp_responses/"):
        self.max_response_size = max_response_size
        self.sandbox_path = sandbox_path
        Path(self.sandbox_path).mkdir(parents=True, exist_ok=True)

    def process_tool_response(self, content: Any, tool_call_id: str) -> Any:
        """Core logic: check size and replace if needed. Returns new content."""
        if not isinstance(content, str):
            return content
        if len(content) <= self.max_response_size:
            return content

        # Write to file
        file_path = Path(self.sandbox_path) / f"{tool_call_id}.txt"
        file_path.write_text(content)

        size_kb = len(content) // 1024
        logger.info("MCP response overflow: %dKB written to %s", size_kb, file_path)

        return (
            f"Response too large ({size_kb}KB), saved to {file_path}\n"
            f"Use read_file tool to inspect specific sections."
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = handler(request)
        if isinstance(result, ToolMessage):
            result.content = self.process_tool_response(
                result.content,
                tool_call_id=str(request.tool_call.get("id", "unknown")),
            )
        return result

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)
        if isinstance(result, ToolMessage):
            result.content = self.process_tool_response(
                result.content,
                tool_call_id=str(request.tool_call.get("id", "unknown")),
            )
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_mcp_overflow.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Patch deer-flow to support extra_middlewares injection**

Modify `deer-flow/backend/packages/harness/deerflow/agents/lead_agent/agent.py` — add `extra_middlewares` parameter to `_build_middlewares()` and insert them before `LoopDetectionMiddleware`:

In function `_build_middlewares` (line 208), change the signature:

```python
def _build_middlewares(config: RunnableConfig, model_name: str | None, agent_name: str | None = None, extra_middlewares: list | None = None):
```

Before the `LoopDetectionMiddleware` line (line 260), add:

```python
    # Inject extra middlewares (from deer-agents or custom config)
    if extra_middlewares:
        middlewares.extend(extra_middlewares)
```

So the end of the function looks like:

```python
    ...
    if subagent_enabled:
        max_concurrent_subagents = config.get("configurable", {}).get("max_concurrent_subagents", 3)
        middlewares.append(SubagentLimitMiddleware(max_concurrent=max_concurrent_subagents))

    # Inject extra middlewares (from deer-agents or custom config)
    if extra_middlewares:
        middlewares.extend(extra_middlewares)

    # LoopDetectionMiddleware — detect and break repetitive tool call loops
    middlewares.append(LoopDetectionMiddleware())

    # ClarificationMiddleware should always be last
    middlewares.append(ClarificationMiddleware())
    return middlewares
```

Also update both calls to `_build_middlewares` in `make_lead_agent` (lines 331, 340) to pass through extra_middlewares from config:

```python
    extra_mws = config.get("configurable", {}).get("extra_middlewares", None)
```

Add this line before the `if is_bootstrap:` check (line 326), then update the two calls:

```python
    # Line 331
    middleware=_build_middlewares(config, model_name=model_name, extra_middlewares=extra_mws),
    # Line 340
    middleware=_build_middlewares(config, model_name=model_name, agent_name=agent_name, extra_middlewares=extra_mws),
```

- [ ] **Step 6: Commit**

```bash
git add middlewares/mcp_overflow.py tests/test_mcp_overflow.py deer-flow/backend/packages/harness/deerflow/agents/lead_agent/agent.py
git commit -m "feat: MCP overflow middleware + deer-flow extra_middlewares injection"
```

---

### Task 9: Wire Middleware into DeerShell

**Files:**
- Modify: `cli/shell.py`

- [ ] **Step 1: Update _ensure_client to pass extra_middlewares**

In `cli/shell.py`, update the `_ensure_client` method to load and pass extra middlewares from agent config:

Replace the `self.client = DeerFlowClient(...)` block with:

```python
    def _ensure_client(self):
        """Lazy-initialize or re-create DeerFlowClient."""
        if self.client is not None:
            return

        from deerflow.client import DeerFlowClient
        from langgraph.checkpoint.sqlite import SqliteSaver
        from pathlib import Path

        # Set up checkpointer
        cp_cfg = self.global_cfg.get("checkpointer", {})
        cp_path = Path(cp_cfg.get("path", "~/.deer-agents/checkpoints.db")).expanduser()
        cp_path.parent.mkdir(parents=True, exist_ok=True)

        self._checkpointer = SqliteSaver.from_conn_string(str(cp_path))
        self._checkpointer.setup()

        # Load extra middlewares from agent config
        extra_mws = self._load_extra_middlewares()

        # Point DeerFlowClient at our project's config.yaml
        config_path = str(PROJECT_ROOT / "config.yaml")

        self.client = DeerFlowClient(
            config_path=config_path,
            checkpointer=self._checkpointer,
            model_name=self.agent_cfg.get("model"),
            thinking_enabled=self.agent_cfg.get("thinking_enabled", True),
            agent_name=self.agent_name,
        )

        # Stash extra middlewares to pass via configurable
        self._extra_middlewares = extra_mws

    def _load_extra_middlewares(self):
        """Instantiate extra middleware from agent config."""
        from importlib import import_module

        extra_cfg = self.agent_cfg.get("extra_middlewares", [])
        middlewares = []
        for mw_cfg in extra_cfg:
            use = mw_cfg["use"]  # e.g., "middlewares.mcp_overflow:McpOverflowMiddleware"
            module_path, class_name = use.rsplit(":", 1)
            mod = import_module(module_path)
            cls = getattr(mod, class_name)
            kwargs = mw_cfg.get("config", {})
            middlewares.append(cls(**kwargs))
        return middlewares
```

Then update `_send_message` to pass extra_middlewares through the stream call:

```python
    def _send_message(self, text: str) -> None:
        """Send a message to the current agent and render the response."""
        self._ensure_client()

        if self.session_mgr.get(self.thread_id) is None:
            self.session_mgr.create(self.thread_id, agent_name=self.agent_name)

        try:
            events = self.client.stream(
                text,
                thread_id=self.thread_id,
                extra_middlewares=self._extra_middlewares if self._extra_middlewares else None,
            )
            title = render_stream(events)

            if title:
                self.session_mgr.update(self.thread_id, title=title)
            else:
                self.session_mgr.touch(self.thread_id)

        except KeyboardInterrupt:
            console.print("\n  [yellow]Interrupted[/yellow]")
        except Exception as e:
            console.print(f"\n  [red]Error: {e}[/red]")
```

> **Note:** This depends on `DeerFlowClient.stream()` passing `extra_middlewares` through to the configurable dict. You'll need to also patch `DeerFlowClient.stream()` to accept and forward `**kwargs` to `configurable`. The existing `stream()` method already accepts `**kwargs` and passes them to configurable, so `extra_middlewares` will flow through to `make_lead_agent` → `_build_middlewares`.

- [ ] **Step 2: Verify the stream kwargs flow**

Check that `DeerFlowClient.stream()` passes unknown kwargs into configurable. Read the stream method to confirm, and if needed, add `extra_middlewares` to the configurable dict construction.

- [ ] **Step 3: Commit**

```bash
git add cli/shell.py
git commit -m "feat: wire extra middlewares from agent config into DeerFlowClient"
```

---

### Task 10: End-to-End Smoke Test

**Files:** No new files — manual verification.

- [ ] **Step 1: Run all unit tests**

```bash
cd /Users/bytedance/Documents/aime/deer-agents
source .venv/bin/activate
python -m pytest tests/ -v
```

Expected: All tests pass (config merge, sessions, commands, mcp overflow).

- [ ] **Step 2: Start the CLI in dry-run mode**

```bash
python -m cli --help
```

Expected: Shows help with `--agent` flag.

- [ ] **Step 3: Start the REPL (will need valid API key)**

```bash
export VOLCENGINE_API_KEY=your-key-here
python -m cli
```

Expected: Sees `🦌 Deer Agents — oncall agent ready`.

Test commands:
- `/help` → shows command list
- `/agents` → shows oncall with active marker
- `/status` → shows agent info
- `/exit` → exits cleanly

- [ ] **Step 4: Test a simple conversation (requires API key)**

```
🦌 > 你好，你是谁？
```

Expected: Agent responds based on oncall prompt, stream renders in terminal.

- [ ] **Step 5: Test session resume**

```
🦌 > /sessions
```

Expected: Shows the session from Step 4 with auto-generated title.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore: end-to-end verification complete"
```

---

## Summary

| Task | What it builds | Tests |
|------|---------------|-------|
| 1 | Project scaffold + deer-flow subtree | Import check |
| 2 | Config merging (global + agent) | 5 unit tests |
| 3 | Session metadata manager | 5 unit tests |
| 4 | Stream renderer (rich) | Manual |
| 5 | CLI commands (/switch, etc.) | 6 unit tests |
| 6 | REPL shell + entry point | Smoke test |
| 7 | Oncall agent (prompt/skill/knowledge) | N/A (content) |
| 8 | MCP overflow middleware + deer-flow patch | 3 unit tests |
| 9 | Wire middleware into shell | Integration |
| 10 | End-to-end smoke test | Manual |
