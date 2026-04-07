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
            assert self._cache is not None
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
        assert self._cache is not None
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
