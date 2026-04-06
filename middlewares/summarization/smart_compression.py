"""SmartCompressionMiddleware — archive-first context compression.

Hooks into before_model() to:
1. Check trigger conditions (message count / token count)
2. Partition messages into compress zone and keep zone
3. Archive originals before modifying
4. Compress per strategy (compress_message)
5. Return state update with same-ID replacement
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from middlewares.summarization.compression_archive import archive_messages
from middlewares.summarization.compression_strategies import CompressionConfig, compress_message

logger = logging.getLogger(__name__)

_DEFAULT_ARCHIVE_DIR = Path(".deer-flow/threads")


class SmartCompressionMiddleware(AgentMiddleware):  # type: ignore[type-arg]
    """Compresses older messages by type, archives originals to JSONL.

    Use when: conversation grows long and needs context reduction before LLM call.
    Don't use when: conversation is short or you need full history fidelity.
    """

    def __init__(
        self,
        trigger: list[tuple[str, int | float]],
        keep: tuple[str, int | float] = ("messages", 20),
        compression_config: CompressionConfig | None = None,
        model: Any = None,
        archive_base_dir: Path | str | None = None,
    ):
        self._trigger = trigger
        self._keep = keep
        self._config = compression_config or CompressionConfig()
        self._model = model
        self._archive_base_dir = Path(archive_base_dir) if archive_base_dir else _DEFAULT_ARCHIVE_DIR

    def _should_compress(self, messages: Sequence[BaseMessage]) -> bool:
        for ttype, tvalue in self._trigger:
            if ttype == "messages" and len(messages) > int(tvalue):
                return True
            if ttype == "tokens":
                total_chars = sum(
                    len(m.content) if isinstance(m.content, str) else 0
                    for m in messages
                )
                if total_chars // 4 > int(tvalue):
                    return True
        return False

    def _determine_cutoff(self, messages: Sequence[BaseMessage]) -> int:
        """messages[:cutoff] = compress zone, [cutoff:] = keep zone."""
        ktype, kvalue = self._keep

        if ktype == "messages":
            cutoff = max(0, len(messages) - int(kvalue))
        elif ktype == "tokens":
            char_budget = int(kvalue) * 4
            chars_seen = 0
            cutoff = len(messages)
            for i in range(len(messages) - 1, -1, -1):
                content = messages[i].content
                chars_seen += len(content) if isinstance(content, str) else 0
                if chars_seen >= char_budget:
                    cutoff = i + 1
                    break
                cutoff = i
        else:
            cutoff = 0

        # Don't split AI+Tool pairs
        if 0 < cutoff < len(messages) and isinstance(messages[cutoff], ToolMessage):
            cutoff = max(0, cutoff - 1)

        return cutoff

    def _get_thread_id(self, state: Any, runtime: Any) -> str:
        """Extract thread_id from runtime or generate fallback."""
        # Try runtime.context dict
        ctx = getattr(runtime, "context", None)
        if isinstance(ctx, dict):
            tid = ctx.get("thread_id")
            if tid:
                return str(tid)

        # Try runtime.config.configurable
        config = getattr(runtime, "config", None)
        if isinstance(config, dict):
            tid = config.get("configurable", {}).get("thread_id")
            if tid:
                return str(tid)

        return f"unknown-{uuid.uuid4().hex[:8]}"

    def _compress_zone(
        self, messages: Sequence[BaseMessage], thread_id: str
    ) -> tuple[list[BaseMessage], dict[str, dict]]:
        """Compress messages, archive originals. Returns (compressed_list, pointers)."""
        to_archive: list[BaseMessage] = []
        compressed: list[BaseMessage] = []

        for msg in messages:
            result = compress_message(msg, self._config)

            if result is None:
                compressed.append(msg)
            elif result == "needs_llm":
                if self._model is not None and isinstance(msg, AIMessage):
                    summary = self._llm_summarize(msg)
                    to_archive.append(msg)
                    compressed.append(summary)
                else:
                    compressed.append(msg)
            elif isinstance(result, BaseMessage):
                to_archive.append(msg)
                compressed.append(result)
            else:
                compressed.append(msg)

        pointers: dict[str, dict] = {}
        if to_archive:
            try:
                pointers = archive_messages(thread_id, to_archive, base_dir=self._archive_base_dir)
            except Exception:
                logger.exception("Archive failed, skipping compression")
                return list(messages), {}

        return compressed, pointers

    def _llm_summarize(self, msg: AIMessage) -> AIMessage:
        prompt = (
            "请压缩以下 AI 回复，保留架构决策和结论，删除推理过程。"
            "只输出压缩后的内容，不要解释。\n\n"
            f"{msg.content}"
        )
        try:
            response = self._model.invoke([HumanMessage(content=prompt)])
            return AIMessage(content=response.content, id=msg.id)
        except Exception:
            logger.warning("LLM summarization failed for msg %s, keeping original", msg.id)
            return msg

    def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:  # type: ignore[override]
        messages = list(state.get("messages", []))

        if not self._should_compress(messages):
            return None

        cutoff = self._determine_cutoff(messages)
        if cutoff <= 0:
            return None

        thread_id = self._get_thread_id(state, runtime)
        compress_zone = messages[:cutoff]
        keep_zone = messages[cutoff:]

        compressed, pointers = self._compress_zone(compress_zone, thread_id)

        if not pointers:
            return None  # Nothing was actually compressed

        logger.info(
            "Compressed %d/%d messages (thread=%s, archived=%d)",
            len(pointers), len(compress_zone), thread_id, len(pointers),
        )

        return {
            "messages": compressed + list(keep_zone),
            "compressed_messages": pointers,
        }

    async def abefore_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:  # type: ignore[override]
        return self.before_model(state, runtime)
