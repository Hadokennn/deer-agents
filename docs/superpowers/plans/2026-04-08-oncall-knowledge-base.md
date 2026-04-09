# Oncall Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a diagnostic pattern knowledge base that automatically extracts reusable diagnosis experience from high-rated oncall cases and injects relevant shortcuts into new cases.

**Architecture:** `KnowledgeMiddleware` (in `middlewares/knowledge.py`) uses `after_agent` to detect user ratings and trigger LLM pattern extraction, and `before_model` to inject matched patterns into the system prompt. `KnowledgeStore` (in `knowledge/store.py`) handles atomic persistence and keyword-overlap matching. `knowledge/extractor.py` wraps the LLM extraction call. `knowledge/prompts.py` holds the extraction prompt template.

**Tech Stack:** Python 3.12+, LangChain AgentMiddleware, DeerFlow model factory (`create_chat_model`), JSON file storage with atomic writes.

**Spec:** `docs/superpowers/specs/2026-04-08-oncall-knowledge-base-design.md`

---

## File Structure

```
knowledge/                        # NEW module
    __init__.py                   # Empty
    store.py                      # KnowledgeStore — load, add_pattern, match, atomic save
    extractor.py                  # extract_pattern() — format messages, call LLM, parse JSON
    prompts.py                    # EXTRACT_PATTERN_PROMPT template

middlewares/
    knowledge.py                  # NEW — KnowledgeMiddleware (after_agent + before_model)

agents/oncall/
    knowledge.json                # NEW — empty pattern store
    agent.yaml                    # MODIFY — add KnowledgeMiddleware to extra_middlewares
    skills/schema-diagnosis/
        SKILL.md                  # MODIFY — add Step 5 (rating collection)

tests/
    test_knowledge_store.py       # NEW — KnowledgeStore unit tests
    test_knowledge_extractor.py   # NEW — extractor unit tests
    test_knowledge_middleware.py   # NEW — middleware integration tests
```

---

### Task 1: KnowledgeStore — Storage + Matching

**Files:**
- Create: `knowledge/__init__.py`
- Create: `knowledge/store.py`
- Create: `tests/test_knowledge_store.py`

- [ ] **Step 1: Write test for KnowledgeStore.load() and empty init**

```python
# tests/test_knowledge_store.py
import json
from pathlib import Path

from knowledge.store import KnowledgeStore


def test_load_nonexistent_creates_empty(tmp_path):
    store = KnowledgeStore(str(tmp_path / "knowledge.json"))
    data = store.load()
    assert data["version"] == "1.0"
    assert data["patterns"] == []


def test_load_existing_file(tmp_path):
    f = tmp_path / "knowledge.json"
    f.write_text(json.dumps({
        "version": "1.0",
        "lastUpdated": "",
        "patterns": [{"id": "p1", "symptom": "test"}],
    }))
    store = KnowledgeStore(str(f))
    data = store.load()
    assert len(data["patterns"]) == 1
    assert data["patterns"][0]["id"] == "p1"


def test_load_caches_by_mtime(tmp_path):
    f = tmp_path / "knowledge.json"
    f.write_text(json.dumps({"version": "1.0", "lastUpdated": "", "patterns": []}))
    store = KnowledgeStore(str(f))
    d1 = store.load()
    d2 = store.load()
    assert d1 is d2  # same object = cache hit
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bytedance/Documents/aime/deer-agents && python -m pytest tests/test_knowledge_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'knowledge'`

- [ ] **Step 3: Write KnowledgeStore — load + cache + atomic save**

```python
# knowledge/__init__.py
# (empty)

# knowledge/store.py
"""Diagnostic pattern storage with keyword-overlap matching."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_EMPTY_STORE = {"version": "1.0", "lastUpdated": "", "patterns": []}


class KnowledgeStore:
    """Read/write knowledge.json with mtime caching and atomic saves."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._cache: dict | None = None
        self._mtime: float = 0

    def load(self) -> dict:
        """Load data, returning cached copy if file unchanged."""
        if not self._path.exists():
            if self._cache is None:
                self._cache = json.loads(json.dumps(_EMPTY_STORE))
            return self._cache

        mtime = self._path.stat().st_mtime
        if self._cache is not None and mtime == self._mtime:
            return self._cache

        try:
            self._cache = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load %s: %s", self._path, e)
            self._cache = json.loads(json.dumps(_EMPTY_STORE))
        self._mtime = mtime
        return self._cache

    def _atomic_save(self, data: dict) -> None:
        """Write via temp file + atomic rename."""
        data["lastUpdated"] = datetime.now(timezone.utc).isoformat()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self._path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, str(self._path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._cache = data
        self._mtime = self._path.stat().st_mtime

    def add_pattern(self, pattern: dict) -> None:
        """Add pattern with dedup + merge. Atomic save."""
        data = self.load()
        existing = self._find_duplicate(data["patterns"], pattern)
        if existing:
            self._merge_pattern(existing, pattern)
        else:
            data["patterns"].append(pattern)
        self._atomic_save(data)

    def match(self, text: str, top_k: int = 3) -> list[dict]:
        """Return top_k patterns by keyword overlap score (> 0)."""
        data = self.load()
        scored = []
        for p in data["patterns"]:
            score = self._keyword_overlap(text, p.get("symptom_keywords", []))
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda x: (-x[0], -x[1].get("confidence", 0)))

        results = []
        dirty = False
        for _, p in scored[:top_k]:
            p["times_matched"] = p.get("times_matched", 0) + 1
            results.append(p)
            dirty = True
        if dirty:
            self._atomic_save(data)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_overlap(text: str, keywords: list[str]) -> float:
        """Fraction of keywords found as substrings in text."""
        if not keywords:
            return 0
        hits = sum(1 for kw in keywords if kw in text)
        return hits / len(keywords)

    @staticmethod
    def _find_duplicate(patterns: list[dict], new: dict) -> dict | None:
        """Match by root_cause_type + symptom keyword overlap > 50%."""
        new_type = new.get("root_cause_type", "")
        new_kws = new.get("symptom_keywords", [])
        for p in patterns:
            if p.get("root_cause_type") != new_type:
                continue
            score = KnowledgeStore._keyword_overlap(
                p.get("symptom", ""), new_kws
            )
            if score > 0.5:
                return p
        return None

    @staticmethod
    def _merge_pattern(existing: dict, new: dict) -> None:
        """Merge new into existing: append source_cases, max confidence, union keywords."""
        existing["source_cases"] = list(
            set(existing.get("source_cases", []) + new.get("source_cases", []))
        )
        existing["confidence"] = max(
            existing.get("confidence", 0), new.get("confidence", 0)
        )
        existing["symptom_keywords"] = list(
            set(existing.get("symptom_keywords", []) + new.get("symptom_keywords", []))
        )
```

- [ ] **Step 4: Run tests to verify load/cache pass**

Run: `cd /Users/bytedance/Documents/aime/deer-agents && python -m pytest tests/test_knowledge_store.py -v`
Expected: 3 PASS

- [ ] **Step 5: Write tests for add_pattern + dedup + merge**

```python
# Append to tests/test_knowledge_store.py

def _make_pattern(**overrides):
    base = {
        "id": "pattern_test1",
        "symptom": "字段不显示",
        "symptom_keywords": ["字段", "不显示", "看不到"],
        "misdiagnosis_trap": "容易误判为 schema 配置",
        "actual_root_cause": "runtime 逻辑",
        "root_cause_type": "runtime_business_logic",
        "diagnostic_shortcut": "先查 use-model.ts",
        "key_files": ["use-model.ts"],
        "resolution": "检查条件渲染",
        "confidence": 0.9,
        "source_cases": ["case1"],
        "times_matched": 0,
        "createdAt": "2026-04-08T00:00:00Z",
    }
    base.update(overrides)
    return base


def test_add_pattern_new(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k.json"))
    store.add_pattern(_make_pattern())
    data = store.load()
    assert len(data["patterns"]) == 1
    assert data["patterns"][0]["id"] == "pattern_test1"
    assert data["lastUpdated"] != ""


def test_add_pattern_dedup_merges(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k.json"))
    store.add_pattern(_make_pattern(confidence=0.8, source_cases=["case1"]))
    store.add_pattern(_make_pattern(
        id="pattern_test2",
        confidence=0.95,
        source_cases=["case2"],
        symptom_keywords=["字段", "不显示", "隐藏"],
    ))
    data = store.load()
    assert len(data["patterns"]) == 1  # merged, not two
    assert data["patterns"][0]["confidence"] == 0.95
    assert "case2" in data["patterns"][0]["source_cases"]
    assert "隐藏" in data["patterns"][0]["symptom_keywords"]


def test_add_pattern_different_type_no_merge(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k.json"))
    store.add_pattern(_make_pattern(root_cause_type="runtime_business_logic"))
    store.add_pattern(_make_pattern(
        id="pattern_test2",
        root_cause_type="schema_config",
    ))
    data = store.load()
    assert len(data["patterns"]) == 2
```

- [ ] **Step 6: Run tests**

Run: `cd /Users/bytedance/Documents/aime/deer-agents && python -m pytest tests/test_knowledge_store.py -v`
Expected: 6 PASS

- [ ] **Step 7: Write tests for match()**

```python
# Append to tests/test_knowledge_store.py

def test_match_keyword_overlap(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k.json"))
    store.add_pattern(_make_pattern())
    results = store.match("商家反馈字段不显示了")
    assert len(results) == 1
    assert results[0]["id"] == "pattern_test1"
    assert results[0]["times_matched"] == 1


def test_match_no_overlap_returns_empty(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k.json"))
    store.add_pattern(_make_pattern())
    results = store.match("价格计算错误")
    assert results == []


def test_match_respects_top_k(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k.json"))
    for i in range(5):
        store.add_pattern(_make_pattern(
            id=f"p{i}",
            root_cause_type=f"type_{i}",  # distinct types so no merge
            confidence=0.5 + i * 0.1,
        ))
    results = store.match("字段不显示", top_k=2)
    assert len(results) == 2
```

- [ ] **Step 8: Run tests**

Run: `cd /Users/bytedance/Documents/aime/deer-agents && python -m pytest tests/test_knowledge_store.py -v`
Expected: 9 PASS

- [ ] **Step 9: Commit**

```bash
git add knowledge/__init__.py knowledge/store.py tests/test_knowledge_store.py
git commit -m "feat(knowledge): KnowledgeStore with atomic save, dedup/merge, keyword matching"
```

---

### Task 2: Extract Prompt + Extractor

**Files:**
- Create: `knowledge/prompts.py`
- Create: `knowledge/extractor.py`
- Create: `tests/test_knowledge_extractor.py`

- [ ] **Step 1: Write the extract prompt**

```python
# knowledge/prompts.py
"""Prompt templates for diagnostic pattern extraction."""

EXTRACT_PATTERN_PROMPT = """\
你是一个诊断经验总结系统。分析刚完成的 oncall 诊断对话，提取可泛化的诊断模式。

用户评分：{score}/10
用户评语：{comment}

对话记录：
<transcript>
{transcript}
</transcript>

## 提取规则

1. **症状泛化**：用通用描述替代具体字段名
   - 正确："商家反馈字段不显示/不能编辑"
   - 错误："outId 字段不显示"

2. **弯路提取**：agent 最初尝试了什么方向、为什么不对、最终转向了哪里
   - 这是最有价值的信息，认真分析对话中的排查过程

3. **捷径提炼**：如果下次遇到类似症状，最短路径是什么
   - 格式："先查 X → 再查 Y → 最后查 Z"

4. **关键文件**：哪些文件/函数是定位问题的关键（保留文件名，去掉具体行号）

## 不要记录的（case 细节，不可泛化）
- 具体字段名（outId / price / accountName）
- 具体模板 ID / category_id / 商家信息
- 具体的 schema 配置值

## 输出 JSON（严格遵循，不要增减字段）

```json
{{
  "symptom": "一句话描述症状模式",
  "symptom_keywords": ["关键词1", "关键词2"],
  "misdiagnosis_trap": "容易走的弯路",
  "actual_root_cause": "真正的根因",
  "root_cause_type": "<schema_config | runtime_business_logic | data_issue | permission | dependency_change>",
  "diagnostic_shortcut": "先查 X → 再查 Y → 最后查 Z",
  "key_files": ["file1.ts"],
  "resolution": "解决建议",
  "confidence": 0.85
}}
```

只输出 JSON，不要其他文字。如果对话中没有完整的诊断过程（比如只是闲聊），输出 null。\
"""
```

- [ ] **Step 2: Write tests for extract_pattern()**

```python
# tests/test_knowledge_extractor.py
import json
from unittest.mock import MagicMock, patch

from knowledge.extractor import extract_pattern, _format_messages, _parse_llm_json


def test_format_messages():
    """Messages are formatted as 'role: content' lines."""
    msgs = [
        MagicMock(type="human", content="字段不显示"),
        MagicMock(type="ai", content="我来帮你看看", tool_calls=None),
    ]
    text = _format_messages(msgs)
    assert "用户: 字段不显示" in text
    assert "助手: 我来帮你看看" in text


def test_parse_llm_json_valid():
    raw = '```json\n{"symptom": "test", "confidence": 0.9}\n```'
    result = _parse_llm_json(raw)
    assert result["symptom"] == "test"


def test_parse_llm_json_null():
    assert _parse_llm_json("null") is None


def test_parse_llm_json_invalid():
    assert _parse_llm_json("not json at all") is None


def test_extract_pattern_happy_path():
    llm_response = json.dumps({
        "symptom": "字段不显示",
        "symptom_keywords": ["字段", "不显示"],
        "misdiagnosis_trap": "误判为 schema",
        "actual_root_cause": "runtime 逻辑",
        "root_cause_type": "runtime_business_logic",
        "diagnostic_shortcut": "先查 use-model.ts",
        "key_files": ["use-model.ts"],
        "resolution": "检查条件渲染",
        "confidence": 0.9,
    })
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content=llm_response)

    with patch("knowledge.extractor.create_chat_model", return_value=mock_model):
        msgs = [MagicMock(type="human", content="字段不显示")]
        result = extract_pattern(msgs, score=8, comment="不错")

    assert result is not None
    assert result["symptom"] == "字段不显示"
    assert result["id"].startswith("pattern_")
    assert result["times_matched"] == 0
    assert result["source_cases"] == ["字段不显示"]


def test_extract_pattern_null_response():
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="null")

    with patch("knowledge.extractor.create_chat_model", return_value=mock_model):
        msgs = [MagicMock(type="human", content="hello")]
        result = extract_pattern(msgs, score=8)

    assert result is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/bytedance/Documents/aime/deer-agents && python -m pytest tests/test_knowledge_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'knowledge.extractor'`

- [ ] **Step 4: Write extractor.py**

```python
# knowledge/extractor.py
"""LLM-based diagnostic pattern extraction."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from knowledge.prompts import EXTRACT_PATTERN_PROMPT

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
    from deerflow.models import create_chat_model

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
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/bytedance/Documents/aime/deer-agents && python -m pytest tests/test_knowledge_extractor.py -v`
Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
git add knowledge/prompts.py knowledge/extractor.py tests/test_knowledge_extractor.py
git commit -m "feat(knowledge): LLM extractor with prompt template and JSON parsing"
```

---

### Task 3: KnowledgeMiddleware

**Files:**
- Create: `middlewares/knowledge.py`
- Create: `tests/test_knowledge_middleware.py`

- [ ] **Step 1: Write tests for rating detection**

```python
# tests/test_knowledge_middleware.py
from unittest.mock import MagicMock

from middlewares.knowledge import KnowledgeMiddleware


def _msg(type: str, content: str, tool_calls=None):
    m = MagicMock()
    m.type = type
    m.content = content
    m.tool_calls = tool_calls
    return m


class TestRatingDetection:
    def _mw(self, tmp_path):
        return KnowledgeMiddleware(knowledge_path=str(tmp_path / "k.json"))

    def test_detect_score_after_rating_prompt(self, tmp_path):
        mw = self._mw(tmp_path)
        messages = [
            _msg("human", "字段不显示"),
            _msg("ai", "根因是 runtime 逻辑。请对本次诊断评分（1-10 分）"),
            _msg("human", "8"),
        ]
        score, comment = mw._detect_rating(messages)
        assert score == 8
        assert comment == ""

    def test_detect_score_with_comment(self, tmp_path):
        mw = self._mw(tmp_path)
        messages = [
            _msg("ai", "请评分"),
            _msg("human", "9 这次定位很快"),
        ]
        score, comment = mw._detect_rating(messages)
        assert score == 9
        assert comment == "这次定位很快"

    def test_no_rating_prompt_returns_none(self, tmp_path):
        mw = self._mw(tmp_path)
        messages = [
            _msg("ai", "根因是 runtime 逻辑"),
            _msg("human", "8"),  # number but no rating prompt
        ]
        score, comment = mw._detect_rating(messages)
        assert score is None

    def test_score_out_of_range_returns_none(self, tmp_path):
        mw = self._mw(tmp_path)
        messages = [
            _msg("ai", "请评分"),
            _msg("human", "15"),
        ]
        score, comment = mw._detect_rating(messages)
        assert score is None

    def test_chinese_fen_suffix(self, tmp_path):
        mw = self._mw(tmp_path)
        messages = [
            _msg("ai", "请评分"),
            _msg("human", "8分"),
        ]
        score, comment = mw._detect_rating(messages)
        assert score == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bytedance/Documents/aime/deer-agents && python -m pytest tests/test_knowledge_middleware.py::TestRatingDetection -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write KnowledgeMiddleware**

```python
# middlewares/knowledge.py
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
    # after_agent — extract patterns from rated conversations
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
    # before_model — inject matched patterns
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
    def _inject_into_system(state: dict, hint_block: str) -> dict | None:
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
```

- [ ] **Step 4: Run rating detection tests**

Run: `cd /Users/bytedance/Documents/aime/deer-agents && python -m pytest tests/test_knowledge_middleware.py::TestRatingDetection -v`
Expected: 5 PASS

- [ ] **Step 5: Write tests for after_agent and before_model**

```python
# Append to tests/test_knowledge_middleware.py
import json
from pathlib import Path
from unittest.mock import patch


class TestAfterAgent:
    def test_extracts_on_high_score(self, tmp_path):
        kpath = str(tmp_path / "k.json")
        mw = KnowledgeMiddleware(knowledge_path=kpath, score_threshold=7)

        state = {
            "messages": [
                _msg("human", "字段不显示怎么办"),
                _msg("ai", "根因是 runtime 逻辑。请评分"),
                _msg("human", "8 很准确"),
            ]
        }

        mock_pattern = {
            "symptom": "字段不显示",
            "symptom_keywords": ["字段", "不显示"],
            "root_cause_type": "runtime_business_logic",
            "confidence": 0.9,
        }

        with patch("knowledge.extractor.extract_pattern", return_value=mock_pattern) as mock_extract:
            mw.after_agent(state, runtime=MagicMock())
            mock_extract.assert_called_once()

        data = json.loads(Path(kpath).read_text())
        assert len(data["patterns"]) == 1

    def test_skips_low_score(self, tmp_path):
        mw = KnowledgeMiddleware(
            knowledge_path=str(tmp_path / "k.json"), score_threshold=7
        )
        state = {
            "messages": [
                _msg("ai", "请评分"),
                _msg("human", "3"),
            ]
        }
        with patch("knowledge.extractor.extract_pattern") as mock_extract:
            mw.after_agent(state, runtime=MagicMock())
            mock_extract.assert_not_called()


class TestBeforeModel:
    def test_injects_on_first_human_message(self, tmp_path):
        kpath = tmp_path / "k.json"
        kpath.write_text(json.dumps({
            "version": "1.0",
            "lastUpdated": "",
            "patterns": [{
                "id": "p1",
                "symptom": "字段不显示",
                "symptom_keywords": ["字段", "不显示"],
                "misdiagnosis_trap": "误判为 schema",
                "actual_root_cause": "runtime",
                "root_cause_type": "runtime_business_logic",
                "diagnostic_shortcut": "先查 use-model.ts",
                "key_files": ["use-model.ts"],
                "resolution": "检查条件渲染",
                "confidence": 0.95,
                "source_cases": ["case1"],
                "times_matched": 0,
                "createdAt": "2026-04-08T00:00:00Z",
            }],
        }))

        mw = KnowledgeMiddleware(knowledge_path=str(kpath))
        sys_msg = _msg("system", "你是 oncall 助手")
        state = {
            "messages": [
                sys_msg,
                _msg("human", "商家反馈字段不显示"),
            ]
        }
        result = mw.before_model(state, runtime=MagicMock())
        assert result is not None
        sys_content = result["messages"][0].content
        assert "<diagnostic_knowledge>" in sys_content
        assert "use-model.ts" in sys_content

    def test_skips_after_first_turn(self, tmp_path):
        kpath = tmp_path / "k.json"
        kpath.write_text(json.dumps({
            "version": "1.0", "lastUpdated": "",
            "patterns": [{"id": "p1", "symptom_keywords": ["字段"]}],
        }))
        mw = KnowledgeMiddleware(knowledge_path=str(kpath))
        state = {
            "messages": [
                _msg("system", "你是助手"),
                _msg("human", "字段不显示"),
                _msg("ai", "我来查"),
                _msg("human", "是水果类目的"),  # second human msg
            ]
        }
        result = mw.before_model(state, runtime=MagicMock())
        assert result is None
```

- [ ] **Step 6: Run all middleware tests**

Run: `cd /Users/bytedance/Documents/aime/deer-agents && python -m pytest tests/test_knowledge_middleware.py -v`
Expected: 12 PASS

- [ ] **Step 7: Commit**

```bash
git add middlewares/knowledge.py tests/test_knowledge_middleware.py
git commit -m "feat(knowledge): KnowledgeMiddleware with rating detection, extraction, and injection"
```

---

### Task 4: Config + SKILL.md + Initial Data

**Files:**
- Create: `agents/oncall/knowledge.json`
- Modify: `agents/oncall/agent.yaml`
- Modify: `agents/oncall/skills/schema-diagnosis/SKILL.md`

- [ ] **Step 1: Create empty knowledge store**

```json
{
  "version": "1.0",
  "lastUpdated": "",
  "patterns": []
}
```

Write this to `agents/oncall/knowledge.json`.

- [ ] **Step 2: Add KnowledgeMiddleware to agent.yaml**

Read `agents/oncall/agent.yaml`, then append to `extra_middlewares`:

```yaml
  - use: middlewares.knowledge:KnowledgeMiddleware
    config:
      knowledge_path: agents/oncall/knowledge.json
      score_threshold: 7
      max_inject_patterns: 3
      extract_model: null
      max_extract_tokens: 8000
```

- [ ] **Step 3: Add Step 5 to SKILL.md**

Read `agents/oncall/skills/schema-diagnosis/SKILL.md`, then append before the "重要注意事项" section:

```markdown
## Step 5: 评分收集

诊断结论呈现给用户后，**必须**主动请求评分：

> "本次诊断到此结束。请对诊断过程评分（1-10 分），可以附带评语。
> 评分标准：定位是否准确、过程是否高效、建议是否可操作。"

收到评分后回复确认即可，不需要额外操作（middleware 会自动处理后续提取）。
```

- [ ] **Step 4: Commit**

```bash
git add agents/oncall/knowledge.json agents/oncall/agent.yaml agents/oncall/skills/schema-diagnosis/SKILL.md
git commit -m "feat(knowledge): wire up KnowledgeMiddleware in oncall agent config"
```

---

### Task 5: Full Pipeline Verification

- [ ] **Step 1: Run all knowledge tests**

Run: `cd /Users/bytedance/Documents/aime/deer-agents && python -m pytest tests/test_knowledge_store.py tests/test_knowledge_extractor.py tests/test_knowledge_middleware.py -v`
Expected: All PASS

- [ ] **Step 2: Run all existing tests to check for regressions**

Run: `cd /Users/bytedance/Documents/aime/deer-agents && python -m pytest tests/ -v`
Expected: All existing tests still PASS

- [ ] **Step 3: Verify config loading**

Run:
```bash
cd /Users/bytedance/Documents/aime/deer-agents && python -c "
from cli.shell import OncallShell
shell = OncallShell.__new__(OncallShell)
import yaml
cfg = yaml.safe_load(open('agents/oncall/agent.yaml'))
shell.agent_cfg = cfg
mws = shell._load_extra_middlewares()
print(f'{len(mws)} middlewares loaded')
for mw in mws:
    print(f'  {type(mw).__name__}')
"
```
Expected: 3 middlewares — ToolResponseProcessorMiddleware, CodeIndexMiddleware, KnowledgeMiddleware

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(knowledge): oncall diagnostic knowledge base — complete implementation"
```

---

## Verification Summary

After implementation, verify:

1. **9 KnowledgeStore tests pass** — load, cache, add, dedup, merge, match
2. **6 Extractor tests pass** — format, parse, happy path, null response
3. **12 Middleware tests pass** — rating detection (5), after_agent (2), before_model (2+)
4. **No regression** — all existing tests pass
5. **Config loads** — KnowledgeMiddleware appears in middleware chain
