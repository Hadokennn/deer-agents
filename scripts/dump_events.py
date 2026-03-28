"""Dump raw StreamEvent format to understand actual data structure."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["DEER_FLOW_CONFIG_PATH"] = str(PROJECT_ROOT / "deer-flow" / "config.yaml")

from deerflow.client import DeerFlowClient
from langgraph.checkpoint.sqlite import SqliteSaver

cp_path = Path("~/.deer-agents/checkpoints.db").expanduser()
cp_path.parent.mkdir(parents=True, exist_ok=True)
cp_ctx = SqliteSaver.from_conn_string(str(cp_path))
checkpointer = cp_ctx.__enter__()
checkpointer.setup()

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
