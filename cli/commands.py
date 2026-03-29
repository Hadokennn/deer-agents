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
    console.print("  /trace            Show recent LangSmith traces")
    console.print("  /trace <id>       Show detail of a specific trace")
    console.print("  /replay           Show step history for current thread")
    console.print("  /replay <N>       Replay from step N")
    console.print("  /diagnose         Auto-detect anomalies in current thread")
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


def handle_replay(args: str, thread_id: str) -> None:
    """Show step history or replay from a step."""
    try:
        import sys
        sys.path.insert(0, ".")
        from scripts.trace_replay import cmd_steps, cmd_replay
        if not args:
            cmd_steps(thread_id)
        else:
            step_num = int(args.strip())
            cmd_replay(thread_id, step_num)
    except ValueError:
        console.print("  Usage: /replay <step_number>")
    except Exception as e:
        console.print(f"  [red]Replay error: {e}[/red]")


def handle_diagnose(thread_id: str) -> None:
    """Auto-diagnose current thread."""
    try:
        import sys
        sys.path.insert(0, ".")
        from scripts.trace_replay import cmd_diagnose
        cmd_diagnose(thread_id)
    except Exception as e:
        console.print(f"  [red]Diagnose error: {e}[/red]")


def handle_trace(args: str) -> None:
    """Show LangSmith traces."""
    try:
        import sys
        sys.path.insert(0, ".")
        from scripts.trace_inspector import cmd_recent, cmd_detail
        if not args:
            cmd_recent(limit=5)
        else:
            cmd_detail(run_id=args.strip())
    except ImportError:
        console.print("  [red]LangSmith not configured. Check deer-flow/.env[/red]")
    except Exception as e:
        console.print(f"  [red]Trace error: {e}[/red]")


def handle_status(agent_name: str, thread_id: str, config: dict[str, Any]) -> None:
    """Display current agent status."""
    console.print(f"\n  Agent:     [cyan]{agent_name}[/cyan]")
    console.print(f"  Thread:    {thread_id}")
    console.print(f"  Model:     {config.get('model', 'default')}")
    mcp_count = len(config.get("mcp_servers", []))
    console.print(f"  MCP:       {mcp_count} server(s)")
    console.print()
