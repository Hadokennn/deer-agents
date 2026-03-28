# cli/renderer.py
"""Render DeerFlowClient StreamEvents to the terminal using rich."""

import json
from dataclasses import dataclass, field

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

console = Console()


@dataclass
class RenderResult:
    title: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls_count: int = 0
    error_count: int = 0


def _summarize_args(args: dict, max_len: int = 80) -> str:
    """One-line summary of tool call arguments."""
    try:
        s = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(args)
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


def render_stream(stream_events, verbose: bool = False) -> RenderResult:
    """Consume a stream of StreamEvents, rendering to terminal.

    Shows: AI text (streaming), tool calls, tool results, errors, token usage.

    Args:
        stream_events: Generator of StreamEvent from DeerFlowClient.stream()
        verbose: Show extra detail (full tool results, event types).

    Returns:
        RenderResult with title, usage, and counts.
    """
    result = RenderResult()
    collected_text = ""
    live = None

    def _stop_live():
        nonlocal live
        if live is not None:
            live.stop()
            live = None

    def _start_live():
        nonlocal live
        if live is None:
            live = Live("", console=console, refresh_per_second=8, vertical_overflow="visible")
            live.start()

    try:
        for event in stream_events:
            if verbose:
                keys = list(event.data.keys()) if isinstance(event.data, dict) else "..."
                console.print(f"  [dim]EVENT: type={event.type} keys={keys}[/dim]")

            if event.type == "values":
                result.title = event.data.get("title") or result.title

            elif event.type == "messages-tuple":
                msg = event.data
                if not isinstance(msg, dict):
                    continue

                msg_type = msg.get("type")

                # AI text content — stream with Live
                if msg_type == "ai" and msg.get("content"):
                    _start_live()
                    collected_text = msg["content"]
                    live.update(Markdown(collected_text))

                # AI tool calls — print static lines
                elif msg_type == "ai" and "tool_calls" in msg:
                    _stop_live()
                    for tc in msg["tool_calls"]:
                        name = tc.get("name", "?")
                        args = tc.get("args", {})
                        result.tool_calls_count += 1
                        console.print(f"  [cyan]⚙ Tool: {name}[/cyan] [dim]{_summarize_args(args)}[/dim]")

                # Tool results
                elif msg_type == "tool":
                    _stop_live()
                    name = msg.get("name", "?")
                    content = msg.get("content", "")
                    is_error = isinstance(content, str) and content.startswith("Error:")

                    if is_error:
                        result.error_count += 1
                        console.print(f"  [red]  ↳ Error ({name}): {content[:200]}[/red]")
                    elif verbose:
                        console.print(f"  [dim]  ↳ Result ({name}): {content[:500]}[/dim]")
                    else:
                        preview = content[:100].replace("\n", " ")
                        console.print(f"  [dim]  ↳ Result ({name}): {preview}{'...' if len(content) > 100 else ''}[/dim]")

            elif event.type == "end":
                _stop_live()
                result.usage = event.data.get("usage", {})
                break

    finally:
        _stop_live()

    # Final newline after AI text
    if collected_text:
        console.print()

    # Token usage summary
    if result.usage.get("total_tokens"):
        inp = result.usage.get("input_tokens", 0)
        out = result.usage.get("output_tokens", 0)
        console.print(f"  [dim]Tokens: {inp}in / {out}out[/dim]")

    return result
