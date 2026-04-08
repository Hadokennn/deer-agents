"""Knowledge extraction middleware for oncall diagnostic patterns.

after_agent: detect user rating >= threshold, extract pattern via LLM, save to store.
before_model: match user's first message against known patterns, inject shortcuts.
"""

from __future__ import annotations

import logging
import re
from copy import copy
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)

_UPLOAD_BLOCK_RE = re.compile(
    r"<uploaded_files>[\s\S]*?</uploaded_files>\n*", re.IGNORECASE
)
_RATING_RE = re.compile(r"^\s*(\d{1,2})\s*[,，分]?\s*(.*)", re.DOTALL)
_RATING_PROMPT_KEYWORDS = ("评分", "打分", "rate", "rating")


class KnowledgeMiddleware(AgentMiddleware[AgentState]):
    """Extract diagnostic patterns from high-rated oncall cases."""

    def __init__(
        self,
        knowledge_path: str = "knowledge.json",
        score_threshold: int = 7,
        max_inject_patterns: int = 3,
        extract_model: str | None = None,
        max_extract_tokens: int = 8000,
    ):
        super().__init__()
        self.store = KnowledgeStore(knowledge_path)
        self.score_threshold = score_threshold
        self.max_inject_patterns = max_inject_patterns
        self.extract_model = extract_model
        self.max_extract_tokens = max_extract_tokens

    # ------------------------------------------------------------------
    # after_agent -- extract patterns from rated conversations
    # ------------------------------------------------------------------

    @override
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        score, comment = self._detect_rating(messages)
        if score is None or score < self.score_threshold:
            return None

        logger.info("Knowledge extraction triggered: score=%d comment=%s", score, comment)

        filtered = self._filter_messages(messages)
        if not filtered:
            return None

        trimmed = self._trim_to_budget(filtered)

        try:
            from knowledge.extractor import extract_pattern

            pattern = extract_pattern(
                trimmed,
                score=score,
                comment=comment,
                model_name=self.extract_model,
            )
            if pattern:
                self.store.add_pattern(pattern)
                logger.info(
                    "Pattern extracted and saved: %s (%s)",
                    pattern.get("symptom", ""),
                    pattern.get("id", ""),
                )
        except Exception:
            logger.exception("Knowledge extraction failed")

        return None

    # ------------------------------------------------------------------
    # before_model -- inject matched patterns
    # ------------------------------------------------------------------

    @override
    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        messages = state.get("messages", [])
        # Only inject on first user turn
        human_msgs = [m for m in messages if getattr(m, "type", None) == "human"]
        if len(human_msgs) != 1:
            return None

        user_text = self._extract_text(human_msgs[0])
        if not user_text:
            return None

        matched = self.store.match(user_text, top_k=self.max_inject_patterns)
        if not matched:
            return None

        hint_block = self._format_inject(matched)
        return self._inject_into_system(state, hint_block)

    # ------------------------------------------------------------------
    # Rating detection
    # ------------------------------------------------------------------

    def _detect_rating(self, messages: list[Any]) -> tuple[int | None, str | None]:
        """Detect numeric rating in recent messages, only if agent asked for it."""
        # Find last AI message that mentions rating
        ai_asked = False
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai":
                text = self._extract_text(msg)
                if any(kw in text for kw in _RATING_PROMPT_KEYWORDS):
                    ai_asked = True
                break  # only check the last AI message

        if not ai_asked:
            return None, None

        # Find last human message with a number
        for msg in reversed(messages):
            if getattr(msg, "type", None) != "human":
                continue
            text = self._extract_text(msg).strip()
            m = _RATING_RE.match(text)
            if m:
                score = int(m.group(1))
                if 1 <= score <= 10:
                    comment = m.group(2).strip()
                    return score, comment
            break  # only check the last human message

        return None, None

    # ------------------------------------------------------------------
    # Message filtering (mirrors MemoryMiddleware approach)
    # ------------------------------------------------------------------

    def _filter_messages(self, messages: list[Any]) -> list[Any]:
        """Keep human + final AI messages, strip upload blocks."""
        filtered = []
        skip_next_ai = False
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            if msg_type == "human":
                text = self._extract_text(msg)
                if "<uploaded_files>" in text:
                    stripped = _UPLOAD_BLOCK_RE.sub("", text).strip()
                    if not stripped:
                        skip_next_ai = True
                        continue
                    clean = copy(msg)
                    clean.content = stripped
                    filtered.append(clean)
                    skip_next_ai = False
                else:
                    filtered.append(msg)
                    skip_next_ai = False
            elif msg_type == "ai":
                if not getattr(msg, "tool_calls", None):
                    if skip_next_ai:
                        skip_next_ai = False
                        continue
                    filtered.append(msg)
        return filtered

    def _trim_to_budget(self, messages: list[Any]) -> list[Any]:
        """Trim messages to fit token budget (rough char estimate)."""
        # ~4 chars per token
        char_budget = self.max_extract_tokens * 4
        total = 0
        result = []
        # Keep from end (most recent messages are most valuable)
        for msg in reversed(messages):
            text = self._extract_text(msg)
            total += len(text)
            if total > char_budget:
                break
            result.append(msg)
        result.reverse()
        return result

    # ------------------------------------------------------------------
    # Inject formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_inject(patterns: list[dict]) -> str:
        """Format matched patterns as XML block for system prompt."""
        lines = [
            "<diagnostic_knowledge>",
            "以下是与当前问题相似的历史诊断经验，供参考：\n",
        ]
        for i, p in enumerate(patterns, 1):
            conf = p.get("confidence", 0)
            matched = p.get("times_matched", 0)
            lines.append(f"【经验 {i}】(confidence: {conf}, 命中 {matched} 次)")
            lines.append(f"- 症状：{p.get('symptom', '')}")
            lines.append(f"- 常见误判：{p.get('misdiagnosis_trap', '')}")
            lines.append(f"- 推荐路径：{p.get('diagnostic_shortcut', '')}")
            kf = ", ".join(p.get("key_files", []))
            lines.append(f"- 关键文件：{kf}")
            lines.append(f"- 历史根因：{p.get('actual_root_cause', '')}")
            lines.append("")

        lines.append("注意：以上为历史经验，当前问题可能不同。如果排查路径与经验不符，按实际情况判断。")
        lines.append("</diagnostic_knowledge>")
        return "\n".join(lines)

    @staticmethod
    def _inject_into_system(state: AgentState, hint_block: str) -> dict | None:
        """Append hint_block to the system message content."""
        messages = list(state.get("messages", []))
        for i, msg in enumerate(messages):
            if getattr(msg, "type", None) == "system":
                updated = copy(msg)
                updated.content = f"{updated.content}\n\n{hint_block}"
                messages[i] = updated
                return {"messages": messages}
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(msg: Any) -> str:
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
            return " ".join(parts)
        return str(content)
