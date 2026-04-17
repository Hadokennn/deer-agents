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
