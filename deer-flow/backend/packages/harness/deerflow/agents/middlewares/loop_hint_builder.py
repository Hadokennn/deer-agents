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


from langchain_core.messages import AIMessage, ToolMessage

_FALLBACK_HINT = (
    "[LOOP RECOVERY] Repeated tool calls detected. Stop calling tools and produce "
    "your final answer using whatever information you have so far."
)


def build_rule_hint(messages: list, start: int, end: int) -> str:
    """Build a human-readable rule-based hint for a loop region.

    Args:
        messages: full message list (the patched view's source)
        start: inclusive start index of the merged loop region
        end: inclusive end index (last message belonging to the region,
             including paired ToolMessages)

    Returns:
        Hint string. Falls back to _FALLBACK_HINT when no AIMessage/ToolMessage
        pairs can be extracted from the region.
    """
    region = messages[start : end + 1]

    # Walk region, pair each AIMessage.tool_call with its ToolMessage by id
    pending: dict[str, tuple[str, str]] = {}   # tool_call_id -> (name, salient_args_str)
    attempts: list[tuple[str, str, str, bool]] = []   # (name, args, result_preview, is_error)

    for msg in region:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", None) or []:
                tc_id = tc.get("id")
                if tc_id is None:
                    continue
                name = tc.get("name", "?")
                args = tc.get("args", {}) if isinstance(tc.get("args"), dict) else {}
                pending[tc_id] = (name, _salient_args(args))
        elif isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", None)
            if tc_id and tc_id in pending:
                name, args = pending.pop(tc_id)
                content = msg.content if msg.content is not None else ""
                preview = (str(content)[:120]).strip() if content else "(empty)"
                is_error = isinstance(content, str) and content.startswith("Error:")
                attempts.append((name, args, preview, is_error))

    if not attempts:
        return _FALLBACK_HINT

    # Dedupe by (name, args)
    seen: dict[tuple[str, str], tuple[str, bool]] = {}
    for name, args, preview, is_err in attempts:
        seen.setdefault((name, args), (preview, is_err))

    errors = [(k, v) for k, v in seen.items() if v[1]]
    unhelpful = [(k, v) for k, v in seen.items() if not v[1]]

    lines: list[str] = []
    intent = _extract_intent(messages[start]) if isinstance(messages[start], AIMessage) else None
    if intent:
        lines.append("[LOOP RECOVERY] Original intent at the start of this loop region:")
        lines.append(f'  "{intent}"')
        lines.append("")
        lines.append("These tool-call paths have been ruled out:")
    else:
        lines.append("[LOOP RECOVERY] These tool-call paths have been ruled out:")

    if errors:
        lines.append("")
        lines.append("Failed with errors:")
        for (name, args), (preview, _) in errors:
            lines.append(f"  ✗ {name}({args}) → {preview[:80]}")

    if unhelpful:
        lines.append("")
        lines.append("Returned unhelpful results:")
        for (name, args), (preview, _) in unhelpful:
            lines.append(f"  ○ {name}({args}) → {preview[:80]}")

    lines.append("")
    lines.append("Do NOT retry the ruled-out paths. Reassess your approach toward the original")
    lines.append("intent. Choose:")
    lines.append("  (a) a different tool")
    lines.append("  (b) different arguments")
    lines.append("  (c) produce a final answer using partial information.")

    return "\n".join(lines)


_FILLER_PHRASES = (
    "let me",
    "i'll",
    "i will",
    "let's",
    "now i'll",
    "now let me",
    "i need to",
    "going to",
    "next i'll",
)

_MEANINGFUL_MIN_CHARS = 80


def _has_meaningful_text(content) -> bool:
    """Return True if content contains substantive reasoning/conclusion text.

    Signal definition:
      - length >= _MEANINGFUL_MIN_CHARS (filter out filler like "ok", "trying...")
      - NOT all sentences start with filler phrases (filter mechanical narration)
    """
    text = _extract_text(content).strip()
    if len(text) < _MEANINGFUL_MIN_CHARS:
        return False
    sentences = [s.strip().lower() for s in text.split(".") if s.strip()]
    if not sentences:
        return False
    all_filler = all(
        any(s.startswith(p) for p in _FILLER_PHRASES) for s in sentences
    )
    if all_filler:
        return False
    return True


def build_no_progress_hint(messages: list, start: int, end: int) -> str:
    """Build hint for V2.1 no-progress detection.

    Counts tool-call-bearing AIMessages in the region; produces an actionable
    three-choice prompt guiding the model out of exploratory paralysis.
    """
    tool_call_turns = sum(
        1
        for m in messages[start : end + 1]
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
    )

    return (
        f"[NO PROGRESS] You've made {tool_call_turns} tool calls in recent turns "
        "without forming a hypothesis, conclusion, or progress statement.\n\n"
        "Stop exploring. Choose:\n"
        "  (a) commit to ONE specific approach and pursue it\n"
        "  (b) summarize what you know and produce a final answer\n"
        "  (c) explicitly state why the task may not be solvable with available tools"
    )
