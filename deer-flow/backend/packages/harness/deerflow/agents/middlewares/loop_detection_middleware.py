"""Middleware to detect and break repetitive tool call loops.

P0 safety: prevents the agent from calling the same tool with the same
arguments indefinitely until the recursion limit kills the run.

Detection strategy:
  1. View-layer patching (wrap_model_call): detect loops in the message
     history and replace the looping region with a [LOOP RECOVERY] hint
     before the model sees it.
  2. Hard-stop safety net (after_model): if the same tool-call hash appears
     >= hard_limit times in the sliding window, strip all tool_calls from
     the response so the agent is forced to produce a final text answer.
"""

import logging
import threading
from collections import OrderedDict, defaultdict
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.loop_hash import (
    hash_tool_calls as _hash_tool_calls,
    normalize_tool_call_args as _normalize_tool_call_args,
    stable_tool_key as _stable_tool_key,
)
from deerflow.agents.middlewares.loop_hint_builder import (
    _has_meaningful_text,
    build_rule_hint,
)

logger = logging.getLogger(__name__)

# Defaults — can be overridden via constructor
_DEFAULT_REWIND_THRESHOLD = 3  # rewind threshold for _detect_all_loops
_DEFAULT_HARD_LIMIT = 5  # force-stop after 5 identical calls
_DEFAULT_WINDOW_SIZE = 20  # track last N tool calls
_DEFAULT_MAX_TRACKED_THREADS = 100  # LRU eviction limit


_HARD_STOP_MSG = "[FORCED STOP] Repeated tool calls exceeded the safety limit. Producing final answer with results collected so far."


class LoopDetectionMiddleware(AgentMiddleware[AgentState]):
    """Detects and breaks repetitive tool call loops.

    Args:
        rewind_threshold: Number of identical tool call sets before
            view-layer patching collapses the loop region into a hint.
            Default: 3.
        hard_limit: Number of identical tool call sets before stripping
            tool_calls entirely. Default: 5.
        window_size: Size of the sliding window for tracking calls.
            Default: 20.
        max_tracked_threads: Maximum number of threads to track before
            evicting the least recently used. Default: 100.
    """

    def __init__(
        self,
        rewind_threshold: int = _DEFAULT_REWIND_THRESHOLD,
        hard_limit: int = _DEFAULT_HARD_LIMIT,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        max_tracked_threads: int = _DEFAULT_MAX_TRACKED_THREADS,
    ):
        super().__init__()
        self.rewind_threshold = rewind_threshold
        self.hard_limit = hard_limit
        self.window_size = window_size
        self.max_tracked_threads = max_tracked_threads
        self._lock = threading.Lock()
        # Per-thread tracking using OrderedDict for LRU eviction
        self._history: OrderedDict[str, list[str]] = OrderedDict()
        # Observation-only dedup cache: tracks (thread_id, loop_hash) reported once.
        # Pure optimization — clearing it only causes some loops to be re-reported,
        # never affects patching behavior.
        self._reported_loops: dict[str, set[str]] = defaultdict(set)

    def _detect_all_loops(self, messages: list) -> list[tuple[str, int, int]]:
        """Find all tool-call hashes that meet rewind_threshold in messages.

        Returns list of (hash, first_ai_idx, last_ai_idx) sorted by first_ai_idx.
        Indices reference AIMessage positions; ToolMessage absorption happens later.
        """
        from langchain_core.messages import AIMessage as _AIMessage

        hash_counts: dict[str, int] = {}
        hash_first_idx: dict[str, int] = {}
        hash_last_idx: dict[str, int] = {}

        for i, msg in enumerate(messages):
            if not isinstance(msg, _AIMessage):
                continue
            tcs = getattr(msg, "tool_calls", None)
            if not tcs:
                continue
            try:
                h = _hash_tool_calls(tcs)
            except Exception:
                logger.warning("Failed to hash tool_calls at index %d, skipping", i)
                continue
            hash_counts[h] = hash_counts.get(h, 0) + 1
            hash_first_idx.setdefault(h, i)
            hash_last_idx[h] = i

        loops = [
            (h, hash_first_idx[h], hash_last_idx[h])
            for h, count in hash_counts.items()
            if count >= self.rewind_threshold
        ]
        return sorted(loops, key=lambda t: t[1])

    def _expand_for_tool_messages(
        self,
        messages: list,
        tool_call_ids: set[str],
        region_end: int,
    ) -> int:
        """Walk forward from region_end absorbing ToolMessages whose tool_call_id
        is in tool_call_ids. Stops at the first non-matching message.

        Returns the new (inclusive) end index.
        """
        from langchain_core.messages import ToolMessage as _ToolMessage

        i = region_end + 1
        while i < len(messages):
            msg = messages[i]
            if isinstance(msg, _ToolMessage) and getattr(msg, "tool_call_id", None) in tool_call_ids:
                i += 1
                continue
            break
        return i - 1

    def _merge_overlapping(
        self, regions: list[tuple[str, int, int]]
    ) -> list[tuple[set[str], int, int]]:
        """Merge overlapping or adjacent regions, aggregating their hash sets.

        Two regions [a, b] and [c, d] (a <= c) merge if c <= b + 1.
        """
        if not regions:
            return []
        merged: list[tuple[set[str], int, int]] = []
        for h, start, end in regions:
            if merged and start <= merged[-1][2] + 1:
                hashes, m_start, m_end = merged[-1]
                merged[-1] = (hashes | {h}, m_start, max(m_end, end))
            else:
                merged.append(({h}, start, end))
        return merged

    def _collect_tool_call_ids_in_range(
        self, messages: list, start: int, end: int
    ) -> set[str]:
        """Collect tool_call_id values from AIMessages in [start, end]."""
        ids: set[str] = set()
        for i in range(start, end + 1):
            msg = messages[i]
            if isinstance(msg, AIMessage):
                for tc in getattr(msg, "tool_calls", None) or []:
                    tc_id = tc.get("id")
                    if tc_id:
                        ids.add(tc_id)
        return ids

    def _apply_patches_to_merged(
        self, messages: list, merged: list[tuple[set[str], int, int]]
    ) -> list:
        """Apply patches for pre-merged loop regions from end to start.

        Returns a new list (does not mutate input).
        """
        patched = list(messages)
        for hashes, start, end in sorted(merged, key=lambda r: -r[1]):
            tc_ids = self._collect_tool_call_ids_in_range(patched, start, end)
            expanded_end = self._expand_for_tool_messages(patched, tc_ids, end)
            hint = build_rule_hint(patched, start, expanded_end)
            patched = patched[:start] + [HumanMessage(content=hint)] + patched[expanded_end + 1 :]
        return patched

    def _apply_all_patches(self, messages: list) -> list:
        """Detect all loops, merge overlapping, apply patches from end to start.

        Returns a new list (does not mutate input).
        """
        loops = self._detect_all_loops(messages)
        if not loops:
            return list(messages)

        merged = self._merge_overlapping(loops)
        return self._apply_patches_to_merged(messages, merged)

    def _detect_no_progress(
        self,
        messages: list,
        window: int = 15,
        min_window_to_trigger: int = 10,
        no_progress_ratio: float = 0.85,
    ) -> tuple[int, int] | None:
        """V2.1 detector: detects "lots of tool calls with no meaningful thinking".

        Returns (patch_start_idx, patch_end_idx) of the AIMessage range to patch,
        or None if not triggered.
        """
        recent_ai_with_idx: list[tuple[int, AIMessage]] = []
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                recent_ai_with_idx.append((i, messages[i]))
                if len(recent_ai_with_idx) >= window:
                    break
        if len(recent_ai_with_idx) < min_window_to_trigger:
            return None

        recent_ai_with_idx.reverse()

        no_progress = sum(
            1 for _, ai in recent_ai_with_idx
            if getattr(ai, "tool_calls", None)
            and not _has_meaningful_text(ai.content)
        )
        if no_progress / len(recent_ai_with_idx) < no_progress_ratio:
            return None

        patch_start = recent_ai_with_idx[0][0]
        patch_end = recent_ai_with_idx[-1][0]
        return patch_start, patch_end

    def _get_thread_id(self, runtime: Runtime) -> str:
        """Extract thread_id from runtime context for per-thread tracking."""
        thread_id = runtime.context.get("thread_id") if runtime.context else None
        if thread_id:
            return thread_id
        return "default"

    def _get_thread_id_from_request(self, request: ModelRequest) -> str:
        """Extract thread_id from request.runtime.context, fallback to 'default'."""
        runtime = getattr(request, "runtime", None)
        if runtime is None:
            return "default"
        ctx = getattr(runtime, "context", None)
        if not ctx:
            return "default"
        return ctx.get("thread_id", "default")

    def _log_first_detected_loops(
        self,
        thread_id: str,
        merged: list[tuple[set[str], int, int]],
    ) -> None:
        """Emit `loop.rewind.first_detected` log once per (thread_id, loop_hash)."""
        reported = self._reported_loops[thread_id]
        for hashes, start, end in merged:
            for h in hashes:
                if h not in reported:
                    reported.add(h)
                    logger.warning(
                        "loop.rewind.first_detected",
                        extra={
                            "thread_id": thread_id,
                            "loop_hash": h,
                            "loop_start_idx": start,
                            "loop_end_idx": end,
                        },
                    )

    def _evict_if_needed(self) -> None:
        """Evict least recently used threads if over the limit.

        Must be called while holding self._lock.
        """
        while len(self._history) > self.max_tracked_threads:
            evicted_id, _ = self._history.popitem(last=False)
            logger.debug("Evicted loop tracking for thread %s (LRU)", evicted_id)

    def _track_and_check(self, state: AgentState, runtime: Runtime) -> tuple[str | None, bool]:
        """Track tool calls and return (hard_stop_msg_or_none, should_hard_stop).

        Note: warn tier removed in favor of wrap_model_call view patching.
        Only the hard_stop safety net remains in after_model.
        """
        messages = state.get("messages", [])
        if not messages:
            return None, False

        last_msg = messages[-1]
        if getattr(last_msg, "type", None) != "ai":
            return None, False

        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return None, False

        thread_id = self._get_thread_id(runtime)
        try:
            call_hash = _hash_tool_calls(tool_calls)
        except Exception:
            return None, False

        with self._lock:
            if thread_id in self._history:
                self._history.move_to_end(thread_id)
            else:
                self._history[thread_id] = []
                self._evict_if_needed()

            history = self._history[thread_id]
            history.append(call_hash)
            if len(history) > self.window_size:
                history[:] = history[-self.window_size :]

            count = history.count(call_hash)
            tool_names = [tc.get("name", "?") for tc in tool_calls]

            if count >= self.hard_limit:
                logger.error(
                    "Loop hard limit reached - forcing stop",
                    extra={
                        "thread_id": thread_id,
                        "call_hash": call_hash,
                        "count": count,
                        "tools": tool_names,
                    },
                )
                return _HARD_STOP_MSG, True

        return None, False

    @staticmethod
    def _append_text(content: str | list | None, text: str) -> str | list:
        """Append *text* to AIMessage content, handling str, list, and None.

        When content is a list of content blocks (e.g. Anthropic thinking mode),
        we append a new ``{"type": "text", ...}`` block instead of concatenating
        a string to a list, which would raise ``TypeError``.
        """
        if content is None:
            return text
        if isinstance(content, list):
            return [*content, {"type": "text", "text": f"\n\n{text}"}]
        if isinstance(content, str):
            return content + f"\n\n{text}"
        # Fallback: coerce unexpected types to str to avoid TypeError
        return str(content) + f"\n\n{text}"

    def _apply(self, state: AgentState, runtime: Runtime) -> dict | None:
        _, hard_stop = self._track_and_check(state, runtime)

        if hard_stop:
            messages = state.get("messages", [])
            last_msg = messages[-1]
            stripped_msg = last_msg.model_copy(
                update={
                    "tool_calls": [],
                    "content": self._append_text(last_msg.content, _HARD_STOP_MSG),
                }
            )
            return {"messages": [stripped_msg]}

        return None

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        loops = self._detect_all_loops(request.messages)
        if not loops:
            return handler(request)

        merged = self._merge_overlapping(loops)
        thread_id = self._get_thread_id_from_request(request)
        self._log_first_detected_loops(thread_id, merged)

        patched = self._apply_patches_to_merged(request.messages, merged)
        request = request.override(messages=patched)
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        loops = self._detect_all_loops(request.messages)
        if not loops:
            return await handler(request)

        merged = self._merge_overlapping(loops)
        thread_id = self._get_thread_id_from_request(request)
        self._log_first_detected_loops(thread_id, merged)

        patched = self._apply_patches_to_merged(request.messages, merged)
        request = request.override(messages=patched)
        return await handler(request)

    def reset(self, thread_id: str | None = None) -> None:
        """Clear tracking state. If thread_id given, clear only that thread."""
        with self._lock:
            if thread_id:
                self._history.pop(thread_id, None)
                self._reported_loops.pop(thread_id, None)
            else:
                self._history.clear()
                self._reported_loops.clear()
