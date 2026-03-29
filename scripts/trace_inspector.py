"""Trace Inspector — query LangSmith traces for deer-agents runs.

Usage:
    python scripts/trace_inspector.py recent         # Last 10 runs
    python scripts/trace_inspector.py last            # Detail of most recent run
    python scripts/trace_inspector.py detail <run_id> # Detail of specific run
"""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.bootstrap import setup_env
setup_env()


def _get_client():
    from langsmith import Client
    return Client(
        api_key=os.environ.get("LANGSMITH_API_KEY"),
        api_url=os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    )


def _get_project():
    return os.environ.get("LANGSMITH_PROJECT", "deer-agents").strip('"')


def _time_ago(dt):
    if dt is None:
        return "?"
    now = datetime.now(timezone.utc)
    diff = now - dt
    secs = int(diff.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    elif secs < 3600:
        return f"{secs // 60}m ago"
    elif secs < 86400:
        return f"{secs // 3600}h ago"
    else:
        return f"{secs // 86400}d ago"


def _extract_user_input(run):
    """Extract the user message from run inputs."""
    if not run.inputs or "messages" not in run.inputs:
        return ""
    for m in run.inputs["messages"]:
        if isinstance(m, dict):
            if m.get("type") == "human":
                return m.get("content", "")[:60]
            # LangChain serialized format: [{"id": [..., "HumanMessage"], "kwargs": {"content": ...}}]
            if isinstance(m.get("id"), list) and any("HumanMessage" in str(x) for x in m.get("id", [])):
                return m.get("kwargs", {}).get("content", "")[:60]
        elif isinstance(m, list) and len(m) > 0:
            # Nested list format
            for item in m:
                if isinstance(item, dict):
                    if isinstance(item.get("id"), list) and any("HumanMessage" in str(x) for x in item.get("id", [])):
                        return item.get("kwargs", {}).get("content", "")[:60]
    return str(run.inputs)[:60]


def _extract_ai_output(run):
    """Extract the final AI response from run outputs."""
    if not run.outputs:
        return ""
    # outputs may have 'messages' key with list of messages
    if "messages" in run.outputs:
        for m in reversed(run.outputs["messages"]):
            if isinstance(m, dict) and m.get("type") == "ai" and m.get("content"):
                return m["content"][:120]
    return ""


def cmd_recent(limit=10):
    """List recent root runs."""
    client = _get_client()
    project = _get_project()

    runs = list(client.list_runs(
        project_name=project,
        is_root=True,
        run_type="chain",  # Filter to agent runs, not memory updater LLM calls
        limit=limit,
    ))

    if not runs:
        print("No runs found.")
        return

    print(f"\nRecent runs ({project}):\n")
    for r in runs:
        status = "✓" if r.status == "success" else "✗"
        elapsed = ""
        if r.start_time and r.end_time:
            elapsed = f"{(r.end_time - r.start_time).total_seconds():.1f}s"
        tokens = r.total_tokens or 0
        user_input = _extract_user_input(r)
        time_ago = _time_ago(r.start_time)

        print(f"  {status} {str(r.id)[:8]}  {time_ago:>8s}  {elapsed:>6s}  tok={tokens:>6}  \"{user_input}\"")
        if r.error:
            print(f"    [ERROR] {r.error[:100]}")
    print()


def cmd_detail(run_id=None):
    """Show detailed trace for a specific run."""
    client = _get_client()
    project = _get_project()

    if run_id is None:
        # Get most recent agent run
        runs = list(client.list_runs(
            project_name=project,
            is_root=True,
            run_type="chain",
            limit=1,
        ))
        if not runs:
            print("No runs found.")
            return
        run = runs[0]
    else:
        # Find run by prefix match
        runs = list(client.list_runs(
            project_name=project,
            is_root=True,
            run_type="chain",
            limit=50,
        ))
        matches = [r for r in runs if str(r.id).startswith(run_id)]
        if not matches:
            print(f"Run not found: {run_id}")
            return
        run = matches[0]

    # Header
    user_input = _extract_user_input(run)
    ai_output = _extract_ai_output(run)
    status = "success" if run.status == "success" else f"error: {run.error[:80]}" if run.error else run.status
    elapsed = ""
    if run.start_time and run.end_time:
        elapsed = f"{(run.end_time - run.start_time).total_seconds():.1f}s"

    print(f"\n{'=' * 60}")
    print(f"Run {str(run.id)[:8]} — \"{user_input}\"")
    print(f"  Status:   {status}")
    print(f"  Duration: {elapsed}")
    print(f"  Tokens:   {run.prompt_tokens or '?'}in / {run.completion_tokens or '?'}out (total: {run.total_tokens or '?'})")
    if run.start_time:
        print(f"  Time:     {run.start_time.strftime('%Y-%m-%d %H:%M:%S')} ({_time_ago(run.start_time)})")

    # Trace URL
    print(f"  Trace:    https://smith.langchain.com/public/{run.id}")

    # Child runs (steps)
    print(f"\n  Steps:")
    children = list(client.list_runs(
        project_name=project,
        trace_id=run.trace_id,
        limit=50,
    ))

    # Sort by start_time
    children.sort(key=lambda c: c.start_time or datetime.min.replace(tzinfo=timezone.utc))

    step_num = 0
    for c in children:
        if c.id == run.id:
            continue  # Skip root

        step_num += 1
        c_elapsed = ""
        if c.start_time and c.end_time:
            c_elapsed = f"{(c.end_time - c.start_time).total_seconds():.1f}s"

        c_status = "✓" if c.status == "success" else "✗"
        tok = f"tok={c.total_tokens}" if c.total_tokens else ""

        # Summarize based on run_type
        if c.run_type == "llm":
            # LLM call — show what it produced
            output_summary = ""
            if c.outputs:
                gens = c.outputs.get("generations", [[]])
                if gens and gens[0]:
                    gen = gens[0][0] if isinstance(gens[0], list) else gens[0]
                    if isinstance(gen, dict):
                        text = gen.get("text", "")
                        msg = gen.get("message", {})
                        tool_calls = msg.get("kwargs", {}).get("tool_calls") if isinstance(msg, dict) else None
                        if tool_calls:
                            tc_names = [tc.get("name", "?") for tc in tool_calls]
                            output_summary = f"→ tool_call: {', '.join(tc_names)}"
                        elif text:
                            output_summary = f"→ \"{text[:60]}{'...' if len(text) > 60 else ''}\""
            print(f"  {step_num:>3}. {c_status} [LLM] {c.name[:30]:30s} {c_elapsed:>6s} {tok:>10s}  {output_summary}")

        elif c.run_type == "tool":
            # Tool execution
            output_summary = ""
            if c.outputs:
                content = c.outputs.get("content") or c.outputs.get("output", "")
                if isinstance(content, str):
                    output_summary = f"→ \"{content[:60]}{'...' if len(content) > 60 else ''}\""
            print(f"  {step_num:>3}. {c_status} [Tool] {c.name[:30]:30s} {c_elapsed:>6s} {tok:>10s}  {output_summary}")

        else:
            # Middleware or other
            print(f"  {step_num:>3}. {c_status} [{c.run_type:>5}] {c.name[:30]:30s} {c_elapsed:>6s} {tok:>10s}")

    if ai_output:
        print(f"\n  Final response:")
        print(f"    {ai_output}")

    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "recent":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        cmd_recent(limit)
    elif cmd == "last":
        cmd_detail(run_id=None)
    elif cmd == "detail":
        if len(sys.argv) < 3:
            print("Usage: trace_inspector.py detail <run_id>")
            return
        cmd_detail(run_id=sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
