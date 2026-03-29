"""Trace Replay — diagnose, browse steps, and replay from checkpoints.

Usage:
    python scripts/trace_replay.py steps <thread_id>                # Show step history
    python scripts/trace_replay.py diagnose <thread_id>             # Auto-detect anomalies
    python scripts/trace_replay.py replay <thread_id> --from-step N # Resume from step N
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "deer-flow" / ".env")
os.environ.setdefault("DEER_FLOW_CONFIG_PATH", str(Path(__file__).resolve().parent.parent / "deer-flow" / "config.yaml"))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig


# ---------------------------------------------------------------------------
# Step data model
# ---------------------------------------------------------------------------

@dataclass
class Step:
    index: int
    checkpoint_id: str
    next_node: str
    messages: list
    is_key_step: bool  # model, tools, __start__, END — skip middleware internals
    anomalies: list  # list of (severity, message) tuples


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver
    cp_path = Path("~/.deer-agents/checkpoints.db").expanduser()
    ctx = SqliteSaver.from_conn_string(str(cp_path))
    ck = ctx.__enter__()
    ck.setup()
    return ck, ctx


def _get_agent(checkpointer):
    from deerflow.client import DeerFlowClient
    config_path = str(Path(__file__).resolve().parent.parent / "deer-flow" / "config.yaml")
    client = DeerFlowClient(config_path=config_path, checkpointer=checkpointer, thinking_enabled=False)
    dummy_config = RunnableConfig(configurable={"thread_id": "_dummy", "thinking_enabled": False}, recursion_limit=100)
    client._ensure_agent(dummy_config)
    return client._agent


def _build_steps(agent, thread_id: str) -> list[Step]:
    """Build chronological step list from state history."""
    config = RunnableConfig(configurable={"thread_id": thread_id, "thinking_enabled": False}, recursion_limit=100)
    history = list(agent.get_state_history(config))
    history.reverse()  # chronological order

    key_nodes = {"__start__", "model", "tools"}

    steps = []
    for i, state in enumerate(history):
        next_nodes = state.next
        next_node = next_nodes[0] if next_nodes else "END"
        cp_id = state.config["configurable"]["checkpoint_id"]
        msgs = state.values.get("messages", [])

        is_key = next_node in key_nodes or next_node == "END"
        steps.append(Step(
            index=i,
            checkpoint_id=cp_id,
            next_node=next_node,
            messages=msgs,
            is_key_step=is_key,
            anomalies=[],
        ))

    return steps


def _msg_summary(m) -> str:
    """One-line summary of a message."""
    if isinstance(m, HumanMessage):
        return f'Human: "{str(m.content)[:50]}"'
    elif isinstance(m, AIMessage):
        if getattr(m, "tool_calls", None):
            tc_names = [tc["name"] for tc in m.tool_calls]
            tc_args = ""
            if m.tool_calls:
                import json
                try:
                    tc_args = json.dumps(m.tool_calls[0].get("args", {}), ensure_ascii=False)[:60]
                except (TypeError, ValueError):
                    tc_args = str(m.tool_calls[0].get("args", {}))[:60]
            return f"AI: tool_call {', '.join(tc_names)}({tc_args})"
        elif m.content:
            return f'AI: "{str(m.content)[:60]}"'
        else:
            return "AI: (empty)"
    elif isinstance(m, ToolMessage):
        name = getattr(m, "name", "?")
        content = str(m.content)[:50] if m.content else "(empty)"
        return f"Tool({name}): {content}"
    return f"{type(m).__name__}: ..."


# ---------------------------------------------------------------------------
# diagnose — auto-detect anomalies
# ---------------------------------------------------------------------------

def _diagnose_steps(steps: list[Step]) -> list[Step]:
    """Annotate steps with detected anomalies."""
    tool_call_streak = {}  # tool_name -> count

    for i, step in enumerate(steps):
        msgs = step.messages

        # Check last message for issues
        if msgs:
            last = msgs[-1]

            # Tool error
            if isinstance(last, ToolMessage) and isinstance(last.content, str) and last.content.startswith("Error:"):
                step.anomalies.append(("error", f"Tool error: {last.content[:80]}"))

            # Tool loop detection
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                for tc in last.tool_calls:
                    name = tc.get("name", "?")
                    tool_call_streak[name] = tool_call_streak.get(name, 0) + 1
                    if tool_call_streak[name] >= 3:
                        step.anomalies.append(("warning", f"Tool '{name}' called {tool_call_streak[name]}x in a row (loop?)"))
            else:
                tool_call_streak.clear()

            # Empty AI response at END
            if step.next_node == "END" and isinstance(last, AIMessage) and not last.content:
                step.anomalies.append(("warning", "Final AI response is empty"))

        # Token spike (compare consecutive model steps)
        if step.next_node in ("model", "END") and i > 0:
            prev_key_models = [s for s in steps[:i] if "model" in s.next_node and s.is_key_step]
            if prev_key_models:
                # Token comparison would need usage_metadata - skip for now
                pass

    return steps


# ---------------------------------------------------------------------------
# Command: steps
# ---------------------------------------------------------------------------

def cmd_steps(thread_id: str, show_all: bool = False):
    """Show step history for a thread."""
    ck, ctx = _get_checkpointer()
    agent = _get_agent(ck)
    steps = _build_steps(agent, thread_id)

    if not steps:
        print(f"No steps found for thread: {thread_id}")
        return

    print(f"\nThread: {thread_id}  ({len(steps)} total steps)\n")

    for step in steps:
        if not show_all and not step.is_key_step:
            continue

        # What changed at this step (last message)
        change = ""
        if step.messages:
            change = _msg_summary(step.messages[-1])

        # Anomaly markers
        marker = "  "
        if any(a[0] == "error" for a in step.anomalies):
            marker = "✗ "
        elif any(a[0] == "warning" for a in step.anomalies):
            marker = "⚠ "
        else:
            marker = "  "

        cp_short = step.checkpoint_id[:12]
        print(f"  {marker}#{step.index:<3} [{cp_short}] next={step.next_node:20s}  {change}")

        for severity, msg in step.anomalies:
            color = "31" if severity == "error" else "33"  # red / yellow
            print(f"       \033[{color}m↳ {msg}\033[0m")

    print()


# ---------------------------------------------------------------------------
# Command: diagnose
# ---------------------------------------------------------------------------

def cmd_diagnose(thread_id: str):
    """Auto-diagnose a thread's execution."""
    ck, ctx = _get_checkpointer()
    agent = _get_agent(ck)
    steps = _build_steps(agent, thread_id)
    steps = _diagnose_steps(steps)

    anomaly_steps = [s for s in steps if s.anomalies]

    print(f"\nThread: {thread_id}  ({len(steps)} steps, {len(anomaly_steps)} with issues)\n")

    if not anomaly_steps:
        print("  ✓ No anomalies detected.\n")
        # Still show key steps summary
        for step in steps:
            if step.is_key_step:
                change = _msg_summary(step.messages[-1]) if step.messages else ""
                print(f"    #{step.index:<3} [{step.next_node:20s}] {change}")
        print()
        return

    print("  Issues found:\n")
    for step in anomaly_steps:
        change = _msg_summary(step.messages[-1]) if step.messages else ""
        print(f"    #{step.index:<3} [{step.next_node:20s}] {change}")
        for severity, msg in step.anomalies:
            icon = "✗" if severity == "error" else "⚠"
            color = "31" if severity == "error" else "33"
            print(f"         \033[{color}m{icon} {msg}\033[0m")
        print()

    # Suggest replay point
    first_issue = anomaly_steps[0]
    replay_from = max(0, first_issue.index - 1)
    print(f"  Suggested: replay from step #{replay_from}")
    print(f"    python scripts/trace_replay.py replay {thread_id} --from-step {replay_from}")
    print()


# ---------------------------------------------------------------------------
# Command: replay
# ---------------------------------------------------------------------------

def cmd_replay(thread_id: str, from_step: int):
    """Resume agent execution from a specific step's checkpoint."""
    ck, ctx = _get_checkpointer()
    agent = _get_agent(ck)
    steps = _build_steps(agent, thread_id)

    if from_step < 0 or from_step >= len(steps):
        print(f"Invalid step {from_step}. Thread has {len(steps)} steps (0-{len(steps)-1}).")
        return

    target = steps[from_step]
    print(f"\nReplay from step #{from_step}")
    print(f"  Checkpoint: {target.checkpoint_id[:16]}")
    print(f"  Next node:  {target.next_node}")
    if target.messages:
        print(f"  Last msg:   {_msg_summary(target.messages[-1])}")
    print(f"\n  Resuming...\n")

    config = RunnableConfig(
        configurable={
            "thread_id": thread_id,
            "checkpoint_id": target.checkpoint_id,
            "thinking_enabled": False,
        },
        recursion_limit=100,
    )

    # Resume: pass None to continue from checkpoint
    for chunk in agent.stream(None, config=config, stream_mode="values"):
        msgs = chunk.get("messages", [])
        if msgs:
            last = msgs[-1]
            if isinstance(last, AIMessage):
                if getattr(last, "tool_calls", None):
                    for tc in last.tool_calls:
                        print(f"  ⚙ Tool: {tc['name']}({str(tc.get('args', {}))[:80]})")
                elif last.content:
                    print(f"  AI: {str(last.content)[:200]}")
            elif isinstance(last, ToolMessage):
                name = getattr(last, "name", "?")
                content = str(last.content)[:100] if last.content else "(empty)"
                is_err = isinstance(last.content, str) and last.content.startswith("Error:")
                prefix = "  ✗" if is_err else "  ↳"
                print(f"{prefix} Tool({name}): {content}")

    print(f"\n  ✓ Replay complete.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "steps":
        if len(sys.argv) < 3:
            print("Usage: trace_replay.py steps <thread_id> [--all]")
            return
        show_all = "--all" in sys.argv
        cmd_steps(sys.argv[2], show_all=show_all)

    elif cmd == "diagnose":
        if len(sys.argv) < 3:
            print("Usage: trace_replay.py diagnose <thread_id>")
            return
        cmd_diagnose(sys.argv[2])

    elif cmd == "replay":
        if len(sys.argv) < 3:
            print("Usage: trace_replay.py replay <thread_id> --from-step N")
            return
        thread_id = sys.argv[2]
        from_step = None
        for i, arg in enumerate(sys.argv):
            if arg == "--from-step" and i + 1 < len(sys.argv):
                from_step = int(sys.argv[i + 1])
        if from_step is None:
            print("Missing --from-step N")
            return
        cmd_replay(thread_id, from_step)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
