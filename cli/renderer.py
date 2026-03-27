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
