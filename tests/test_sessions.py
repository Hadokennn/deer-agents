import json
from pathlib import Path


def test_create_and_list_sessions(tmp_path):
    """Create a session and verify it appears in list."""
    from cli.sessions import SessionManager

    mgr = SessionManager(sessions_dir=tmp_path)
    mgr.create("thread-abc", agent_name="oncall")

    sessions = mgr.list_all()
    assert len(sessions) == 1
    assert sessions[0]["thread_id"] == "thread-abc"
    assert sessions[0]["agent_name"] == "oncall"
    assert sessions[0]["title"] is None


def test_update_session_title(tmp_path):
    """Updating title persists to disk."""
    from cli.sessions import SessionManager

    mgr = SessionManager(sessions_dir=tmp_path)
    mgr.create("thread-1", agent_name="oncall")
    mgr.update("thread-1", title="redis 排查")

    reloaded = mgr.get("thread-1")
    assert reloaded["title"] == "redis 排查"


def test_list_sessions_sorted_by_last_active(tmp_path):
    """Sessions are sorted most-recent-first."""
    from cli.sessions import SessionManager
    import time

    mgr = SessionManager(sessions_dir=tmp_path)
    mgr.create("old", agent_name="oncall")
    time.sleep(0.01)
    mgr.create("new", agent_name="review")

    sessions = mgr.list_all()
    assert sessions[0]["thread_id"] == "new"
    assert sessions[1]["thread_id"] == "old"


def test_get_nonexistent_session(tmp_path):
    """Getting a nonexistent session returns None."""
    from cli.sessions import SessionManager

    mgr = SessionManager(sessions_dir=tmp_path)
    assert mgr.get("does-not-exist") is None


def test_touch_updates_last_active(tmp_path):
    """Touching a session updates its last_active_at."""
    from cli.sessions import SessionManager

    mgr = SessionManager(sessions_dir=tmp_path)
    mgr.create("thread-1", agent_name="oncall")
    original = mgr.get("thread-1")["last_active_at"]

    import time
    time.sleep(0.01)
    mgr.touch("thread-1")
    updated = mgr.get("thread-1")["last_active_at"]
    assert updated > original
