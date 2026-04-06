"""JSONL archive for original messages before compression.

Append-only. Each line is one LangChain message serialized via dumpd().
"""

import json
import logging
from pathlib import Path

from langchain_core.load import dumpd, load
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

ARCHIVE_FILENAME = "archive.jsonl"


def _archive_path(thread_id: str, base_dir: Path) -> Path:
    return base_dir / thread_id / ARCHIVE_FILENAME


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def archive_messages(
    thread_id: str,
    messages: list[BaseMessage],
    base_dir: Path = Path(".deer-flow/threads"),
) -> dict[str, dict]:
    """Archive messages to JSONL. Returns {msg_id: {"line": N}} map.

    Appends to existing file. Returns empty dict if messages is empty.
    """
    if not messages:
        return {}

    path = _archive_path(thread_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    start_line = _count_lines(path)
    pointers: dict[str, dict] = {}

    with path.open("a", encoding="utf-8") as f:
        for i, msg in enumerate(messages):
            line = json.dumps(dumpd(msg), ensure_ascii=False)
            f.write(line + "\n")
            if msg.id is not None:
                pointers[msg.id] = {"line": start_line + i}

    logger.info(
        "Archived %d messages to %s (lines %d-%d)",
        len(messages),
        path,
        start_line,
        start_line + len(messages) - 1,
    )
    return pointers


def read_archived_messages(
    thread_id: str,
    pointers: dict[str, dict],
    base_dir: Path = Path(".deer-flow/threads"),
) -> list[BaseMessage]:
    """Read archived messages by ID from JSONL. Returns list in pointer order."""
    if not pointers:
        return []

    path = _archive_path(thread_id, base_dir)
    if not path.exists():
        logger.warning("Archive not found: %s", path)
        return []

    needed_lines = {v["line"] for v in pointers.values()}
    line_to_data: dict[int, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in needed_lines:
                line_to_data[i] = line.strip()
            if i > max(needed_lines):
                break

    result = []
    for msg_id, meta in pointers.items():
        raw = line_to_data.get(meta["line"])
        if raw:
            msg = load(json.loads(raw))
            result.append(msg)
        else:
            logger.warning("Line %d not found in archive for msg %s", meta["line"], msg_id)

    return result
