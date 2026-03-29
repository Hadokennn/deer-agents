"""Dump raw StreamEvent format to understand actual data structure."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.bootstrap import setup_env, create_checkpointer
from cli.app import PROJECT_ROOT

setup_env()

from deerflow.client import DeerFlowClient

checkpointer, cp_ctx = create_checkpointer()

client = DeerFlowClient(
    config_path=str(PROJECT_ROOT / "deer-flow" / "config.yaml"),
    checkpointer=checkpointer,
    thinking_enabled=False,
)

print("=== Dumping raw events for '你好' ===\n")
for event in client.stream("你好", thread_id="dump-test"):
    print(f"TYPE: {event.type}")
    print(f"DATA type: {type(event.data).__name__}")
    if isinstance(event.data, dict):
        for k, v in event.data.items():
            v_repr = repr(v)[:200]
            print(f"  {k}: {v_repr}")
    else:
        print(f"  {repr(event.data)[:300]}")
    print()
    if event.type == "end":
        break
