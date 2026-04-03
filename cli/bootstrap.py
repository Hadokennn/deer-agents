# cli/bootstrap.py
"""Shared bootstrap — load config, resolve paths, setup checkpointer."""

import logging
import os
from pathlib import Path

from cli.app import PROJECT_ROOT, load_global_config

LOG_PATH = PROJECT_ROOT / ".deer-flow" / "cli.log"


def setup_env():
    """Load deer-flow .env and pin DEER_FLOW_CONFIG_PATH."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / "deer-flow" / ".env")
    os.environ.setdefault("DEER_FLOW_CONFIG_PATH", str(PROJECT_ROOT / "deer-flow" / "config.yaml"))


def setup_logging(verbose: bool = False):
    """Configure logging to file + optional stderr.

    Always writes to .deer-flow/cli.log (DEBUG level).
    With verbose=True, also logs WARNING+ to stderr.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # File handler — always on, captures everything
    fh = logging.FileHandler(str(LOG_PATH), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(fh)

    # Stderr handler — only with verbose
    if verbose:
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        sh.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(sh)


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
