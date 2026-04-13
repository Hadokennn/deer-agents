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
    handle_diagnose,
    handle_help,
    handle_replay,
    handle_sessions,
    handle_status,
    handle_trace,
    parse_command,
)
from cli.renderer import render_stream
from cli.sessions import SessionManager

console = Console()


class DeerShell:
    """Interactive agent shell."""

    def __init__(self, agent_name: str | None = None, verbose: bool = False):
        self.global_cfg = load_global_config()
        self.agent_name = resolve_agent_name(agent_name, self.global_cfg)
        self.agent_cfg = self._load_merged_config(self.agent_name)
        self.thread_id = str(uuid.uuid4())[:8]
        self.client = None  # Lazy init
        self._checkpointer = None
        self._extra_middlewares = []
        self._verbose = verbose

        # Session management (path relative to PROJECT_ROOT)
        from pathlib import Path
        sessions_raw = self.global_cfg.get("sessions", {}).get("dir", ".deer-flow/sessions/")
        sessions_path = Path(sessions_raw).expanduser()
        if not sessions_path.is_absolute():
            sessions_path = PROJECT_ROOT / sessions_path
        self.session_mgr = SessionManager(sessions_path)

        # Prompt history
        history_path = PROJECT_ROOT / ".deer-flow" / "history"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_session = PromptSession(history=FileHistory(str(history_path)))

    def _load_merged_config(self, agent_name: str) -> dict:
        agent_cfg = load_agent_config(agent_name)
        return merge_agent_config(self.global_cfg, agent_cfg)

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

    def _ensure_client(self):
        """Lazy-initialize or re-create DeerFlowClient."""
        if self.client is not None:
            return

        from deerflow.client import DeerFlowClient
        from cli.bootstrap import setup_env, create_checkpointer

        setup_env()
        self._checkpointer, self._cp_context = create_checkpointer()

        # Load extra middlewares from agent config
        self._extra_middlewares = self._load_extra_middlewares()

        config_path = str(PROJECT_ROOT / "deer-flow" / "config.yaml")

        # Per-agent skill whitelist (None = all skills)
        agent_skills = self.agent_cfg.get("skills")
        available_skills = set(agent_skills) if agent_skills else None

        self.client = DeerFlowClient(
            config_path=config_path,
            checkpointer=self._checkpointer,
            model_name=self.agent_cfg.get("model"),
            thinking_enabled=self.agent_cfg.get("thinking_enabled", False),
            subagent_enabled=self.agent_cfg.get("subagent_enabled", False),
            agent_name=self.agent_name,
            middlewares=self._extra_middlewares or None,
            available_skills=available_skills,
        )

        ptc_tools_raw = self.agent_cfg.get("ptc_tools")
        if ptc_tools_raw:
            from deerflow.config.tool_config import PTCToolConfig
            from deerflow.config.app_config import set_app_config

            ptc_tools = [PTCToolConfig(**c) for c in ptc_tools_raw]
            patched = self.client._app_config.model_copy(update={"ptc_tools": ptc_tools})
            set_app_config(patched)
            self.client._app_config = patched

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
        elif cmd.name == "trace":
            handle_trace(cmd.args)
        elif cmd.name == "replay":
            handle_replay(cmd.args, self.thread_id)
        elif cmd.name == "diagnose":
            handle_diagnose(self.thread_id)
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
            memory_cfg = self.agent_cfg.get("memory", {})
            events = self.client.stream(
                text,
                thread_id=self.thread_id,
                memory_enabled=memory_cfg.get("enabled", True),
            )
            result = render_stream(events, verbose=self._verbose)

            # Update session metadata
            if result.title:
                self.session_mgr.update(self.thread_id, title=result.title)
            else:
                self.session_mgr.touch(self.thread_id)

        except KeyboardInterrupt:
            console.print("\n  [yellow]Interrupted[/yellow]")
        except ConnectionError as e:
            console.print(f"\n  [red]Connection error: Cannot reach the model API.[/red]")
            console.print(f"  [dim]{e}[/dim]")
        except TimeoutError as e:
            console.print(f"\n  [red]Request timed out. The model may be overloaded.[/red]")
            console.print(f"  [dim]{e}[/dim]")
        except Exception as e:
            console.print(f"\n  [red]Error ({type(e).__name__}): {e}[/red]")
            if self._verbose:
                import traceback
                console.print(f"  [dim]{traceback.format_exc()}[/dim]")
            else:
                console.print(f"  [dim]Run with --verbose for full traceback.[/dim]")

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
