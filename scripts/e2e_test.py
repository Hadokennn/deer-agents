"""End-to-end test: bypass REPL, call DeerFlowClient directly."""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from cli.bootstrap import setup_env, create_checkpointer
from cli.app import PROJECT_ROOT

setup_env()
DEERFLOW_CONFIG = str(PROJECT_ROOT / "deer-flow" / "config.yaml")


def test_config_loads():
    """Step 1: Can deer-flow config load without error?"""
    print("=" * 60)
    print("Step 1: Loading deer-flow config...")
    from deerflow.config.app_config import reload_app_config
    config = reload_app_config(DEERFLOW_CONFIG)
    print(f"  Models: {[m.name for m in config.models]}")
    print(f"  Tools:  {[t.name for t in config.tools]}")
    print(f"  Sandbox: {config.sandbox}")
    print("  ✓ Config loaded OK")
    return config


def test_model_resolves(config):
    """Step 2: Can the model class be instantiated?"""
    print("=" * 60)
    print("Step 2: Resolving model...")
    model_cfg = config.models[0]
    print(f"  Model: {model_cfg.name}")
    print(f"  Use:   {model_cfg.use}")

    from deerflow.models import create_chat_model
    model = create_chat_model(name=model_cfg.name, thinking_enabled=False)
    print(f"  Class: {type(model).__name__}")
    print("  ✓ Model resolved OK")
    return model


def test_client_creates():
    """Step 3: Can DeerFlowClient be created?"""
    print("=" * 60)
    print("Step 3: Creating DeerFlowClient...")
    from deerflow.client import DeerFlowClient

    checkpointer, cp_ctx = create_checkpointer()

    client = DeerFlowClient(
        config_path=DEERFLOW_CONFIG,
        checkpointer=checkpointer,
        thinking_enabled=False,
    )
    print(f"  ✓ Client created OK")
    return client, cp_ctx


def test_chat(client):
    """Step 4: Can we send a message and get a response?"""
    print("=" * 60)
    print("Step 4: Sending '你好' via client.chat()...")
    response = client.chat("你好", thread_id="e2e-test-1")
    print(f"  Response: {response[:200]}")
    print("  ✓ Chat OK")


def test_stream(client):
    """Step 5: Can we stream a response?"""
    print("=" * 60)
    print("Step 5: Streaming '你好' via client.stream()...")
    for event in client.stream("你好", thread_id="e2e-test-2"):
        print(f"  Event: type={event.type}, data_keys={list(event.data.keys()) if isinstance(event.data, dict) else type(event.data).__name__}")
        if event.type == "end":
            break
    print("  ✓ Stream OK")


def test_renderer(client):
    """Step 6: Does renderer extract AI content correctly?"""
    print("=" * 60)
    print("Step 6: Testing renderer with real stream...")
    from cli.renderer import render_stream
    from io import StringIO
    from rich.console import Console

    # Capture output to verify renderer produces content
    buf = StringIO()
    import cli.renderer as r
    original_console = r.console
    r.console = Console(file=buf, force_terminal=True)

    try:
        title = render_stream(client.stream("说一个字", thread_id="e2e-test-3"))
        output = buf.getvalue()
        print(f"  Title: {title}")
        print(f"  Output length: {len(output)} chars")
        print(f"  Output preview: {repr(output[:100])}")
        assert len(output) > 0, "Renderer produced no output!"
        print("  ✓ Renderer OK")
    finally:
        r.console = original_console


if __name__ == "__main__":
    print("\n🦌 Deer Agents E2E Test\n")

    try:
        config = test_config_loads()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        sys.exit(1)

    try:
        model = test_model_resolves(config)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    try:
        client, cp_ctx = test_client_creates()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    try:
        test_chat(client)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    try:
        test_stream(client)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    try:
        test_renderer(client)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 All steps passed!")
