"""Build rule-based "ruled out" hints from loop regions.

Pure-function module; no LangGraph runtime dependency. Used by
LoopDetectionMiddleware.wrap_model_call to construct the HumanMessage
that replaces a loop region in the LLM-visible message view.
"""


def _extract_text(content) -> str:
    """Extract concatenated text from AIMessage.content.

    Handles three shapes:
      - str: returned as-is
      - None: returned as empty string
      - list (Anthropic block format): concatenates {"type": "text", ...} blocks,
        ignores other block types (e.g., thinking, tool_use)
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


_SALIENT_FIELDS = ("path", "url", "query", "command", "pattern", "glob", "cmd")

_INTENT_MIN_CHARS = 20
_INTENT_MAX_CHARS = 120


def _salient_args(args: dict) -> str:
    """Format args dict for hint display, keeping only whitelisted fields.

    Whitelist matches stable_tool_key in loop_hash to avoid drift.
    """
    items = [f"{k}={args[k]!r}" for k in _SALIENT_FIELDS if args.get(k) is not None]
    if items:
        return ", ".join(items)
    if not args:
        return "{}"
    return str(args)[:40]


def _extract_intent(ai_message) -> str | None:
    """Extract original intent from an AIMessage.content as a single short string.

    Returns None if content is shorter than _INTENT_MIN_CHARS (likely just filler).
    Otherwise returns text truncated to _INTENT_MAX_CHARS.
    """
    text = _extract_text(ai_message.content).strip()
    if len(text) < _INTENT_MIN_CHARS:
        return None
    if len(text) > _INTENT_MAX_CHARS:
        return text[:_INTENT_MAX_CHARS].rstrip() + "..."
    return text
