"""Session metadata CRUD — thin JSON files in ~/.deer-agents/sessions/."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SessionManager:
    """Manage session metadata files.

    Each session is stored as {thread_id}.json in the sessions directory.
    The actual conversation state lives in the LangGraph checkpointer (SQLite);
    these files only hold lightweight metadata (title, agent_name, timestamps).
    """

    def __init__(self, sessions_dir: Path | str):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, thread_id: str) -> Path:
        return self.sessions_dir / f"{thread_id}.json"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(self, thread_id: str, agent_name: str) -> dict[str, Any]:
        """Create a new session metadata file."""
        now = self._now()
        data = {
            "thread_id": thread_id,
            "agent_name": agent_name,
            "title": None,
            "created_at": now,
            "last_active_at": now,
        }
        self._path(thread_id).write_text(json.dumps(data, indent=2))
        return data

    def get(self, thread_id: str) -> dict[str, Any] | None:
        """Get session metadata by thread_id. Returns None if not found."""
        path = self._path(thread_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def update(self, thread_id: str, **fields) -> dict[str, Any] | None:
        """Update specific fields of a session."""
        data = self.get(thread_id)
        if data is None:
            return None
        data.update(fields)
        data["last_active_at"] = self._now()
        self._path(thread_id).write_text(json.dumps(data, indent=2))
        return data

    def touch(self, thread_id: str) -> None:
        """Update last_active_at timestamp."""
        self.update(thread_id)

    def list_all(self) -> list[dict[str, Any]]:
        """List all sessions, sorted by last_active_at descending."""
        sessions = []
        for path in self.sessions_dir.glob("*.json"):
            try:
                sessions.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        sessions.sort(key=lambda s: s.get("last_active_at", ""), reverse=True)
        return sessions
