"""Per-type message compression strategies.

All functions are pure — no IO, no LLM, independently testable.
compress_message() returns:
  - None: no compression needed
  - "needs_llm": plain text AIMessage, caller should use LLM summary
  - BaseMessage: compressed copy (same ID, reduced content)
"""

from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


@dataclass
class CompressionConfig:
    tool_message_threshold: int = 2048  # bytes
    tool_truncate_head: int = 500
    tool_truncate_tail: int = 200
    ai_min_text_length: int = 200  # Don't LLM-summarize short texts


def compress_message(
    msg: BaseMessage,
    config: CompressionConfig = CompressionConfig(),
) -> BaseMessage | str | None:
    """Compress a single message.

    Returns:
        None — no compression needed
        "needs_llm" — plain text AI, caller should LLM-summarize
        BaseMessage — compressed copy with same ID
    """
    if isinstance(msg, (HumanMessage, SystemMessage)):
        return None

    if isinstance(msg, AIMessage):
        return _compress_ai(msg, config)

    if isinstance(msg, ToolMessage):
        return _compress_tool(msg, config)

    return None


def _compress_ai(msg: AIMessage, config: CompressionConfig) -> AIMessage | str | None:
    """Compress AIMessage based on content structure."""
    # Case 1: Has tool_calls — keep calls, clear content
    if msg.tool_calls:
        if not msg.content:
            return None  # Already minimal
        return AIMessage(
            content="",
            tool_calls=msg.tool_calls,
            id=msg.id,
        )

    # Case 2: Content is block list — strip thinking blocks
    if isinstance(msg.content, list):
        has_thinking = any(
            isinstance(block, dict) and block.get("type") == "thinking"
            for block in msg.content
        )
        if not has_thinking:
            return None
        kept = [
            block for block in msg.content
            if not (isinstance(block, dict) and block.get("type") == "thinking")
        ]
        return AIMessage(content=kept, id=msg.id)

    # Case 3: Plain text string
    if isinstance(msg.content, str):
        if len(msg.content) < config.ai_min_text_length:
            return None  # Too short to bother
        return "needs_llm"

    return None


def _compress_tool(msg: ToolMessage, config: CompressionConfig) -> ToolMessage | None:
    """Compress ToolMessage if over threshold and not an error."""
    if not isinstance(msg.content, str):
        return None

    if _is_tool_error(msg.content):
        return None

    content_bytes = len(msg.content.encode("utf-8"))
    if content_bytes <= config.tool_message_threshold:
        return None

    head = msg.content[:config.tool_truncate_head]
    tail = msg.content[-config.tool_truncate_tail:]
    truncated = f"{head}\n\n[archived, {content_bytes}B total]\n\n{tail}"

    return ToolMessage(
        content=truncated,
        tool_call_id=msg.tool_call_id,
        name=msg.name,
        id=msg.id,
    )


def _is_tool_error(content: str) -> bool:
    """Check if tool content is an error message."""
    return content.startswith("Error:")
