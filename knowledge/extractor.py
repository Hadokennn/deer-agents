"""LLM-based diagnostic pattern extraction."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from knowledge.prompts import EXTRACT_PATTERN_PROMPT

try:
    from deerflow.models import create_chat_model
except ImportError:  # allow tests to import without deerflow on sys.path
    create_chat_model = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _format_messages(messages: list[Any]) -> str:
    """Format messages as 'role: content' lines for the extract prompt."""
    lines = []
    for msg in messages:
        msg_type = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
            content = " ".join(parts)
        role = "用户" if msg_type == "human" else "助手"
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def _parse_llm_json(raw: str) -> dict | None:
    """Parse LLM response as JSON, stripping markdown code blocks."""
    text = raw.strip()

    # Strip ```json ... ``` wrapper
    m = _CODE_BLOCK_RE.search(text)
    if m:
        text = m.group(1).strip()

    if text.lower() == "null":
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON: %s", text[:200])
        return None


def _extract_case_summary(messages: list[Any]) -> str:
    """Build a short case summary from the first human message."""
    for msg in messages:
        if getattr(msg, "type", None) == "human":
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content[:50]
    return "unknown"


def extract_pattern(
    messages: list[Any],
    score: int,
    comment: str | None = None,
    model_name: str | None = None,
) -> dict | None:
    """Call LLM to extract a diagnostic pattern from conversation messages.

    Returns a pattern dict ready for KnowledgeStore.add_pattern(), or None.
    """
    transcript = _format_messages(messages)
    prompt = EXTRACT_PATTERN_PROMPT.format(
        score=score,
        comment=comment or "无",
        transcript=transcript,
    )

    model = create_chat_model(name=model_name, thinking_enabled=False)
    response = model.invoke(prompt)

    raw = response.content
    if isinstance(raw, list):
        raw = " ".join(
            p if isinstance(p, str) else p.get("text", "")
            for p in raw
        )

    parsed = _parse_llm_json(raw)
    if parsed is None:
        return None

    # Add metadata
    parsed["id"] = f"pattern_{uuid4().hex[:8]}"
    parsed["source_cases"] = [_extract_case_summary(messages)]
    parsed["times_matched"] = 0
    parsed["createdAt"] = datetime.now(timezone.utc).isoformat()

    return parsed
