# cli/bootstrap.py
"""Shared bootstrap — load config, resolve paths, setup checkpointer."""

import os
from pathlib import Path

from cli.app import PROJECT_ROOT, load_global_config


def setup_env():
    """Load deer-flow .env and pin DEER_FLOW_CONFIG_PATH."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / "deer-flow" / ".env")
    os.environ.setdefault("DEER_FLOW_CONFIG_PATH", str(PROJECT_ROOT / "deer-flow" / "config.yaml"))


def get_checkpointer_path() -> Path:
    """Resolve checkpointer path from config.yaml, relative to PROJECT_ROOT."""
    cfg = load_global_config()
    raw = cfg.get("checkpointer", {}).get("path", ".deer-flow/checkpoints.db")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def create_checkpointer():
    """Create and setup SqliteSaver from config. Returns (saver, context_manager)."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    cp_path = get_checkpointer_path()
    ctx = SqliteSaver.from_conn_string(str(cp_path))
    saver = ctx.__enter__()
    saver.setup()
    return saver, ctx
