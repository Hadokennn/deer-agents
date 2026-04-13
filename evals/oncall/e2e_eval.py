"""Layer 3: End-to-end eval — run full agent pipeline, score process + output.

Runs the oncall agent with a real query via DeerFlowClient, captures the stream
events, then scores:
  - Process: heuristic rules on tool calls (imported from process_eval)
  - Output: structural checks on the final AI response

Scorer contract: evaluate(case, **kwargs) -> EvalResult
"""

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from evals.framework.types import EvalCase, EvalResult
from evals.oncall.process_eval import apply_heuristic_rules


# ---------------------------------------------------------------------------
# Event capture
# ---------------------------------------------------------------------------


@dataclass
class CapturedRun:
    """Captured data from a full agent run."""

    final_response: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list[str] = field(default_factory=list)


def capture_run(client, query: str, thread_id: str) -> CapturedRun:
    """Run agent and capture all stream events."""
    run = CapturedRun()

    for event in client.stream(query, thread_id=thread_id):
        if event.type == "messages-tuple":
            msg = event.data
            if not isinstance(msg, dict):
                continue

            msg_type = msg.get("type")

            if msg_type == "ai" and msg.get("content"):
                run.final_response = msg["content"]

            elif msg_type == "ai" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    run.tool_calls.append({
                        "name": tc.get("name", "?"),
                        "args": tc.get("args", {}),
                    })

            elif msg_type == "tool":
                name = msg.get("name", "?")
                content = msg.get("content", "")
                # Attach result to most recent matching tool call
                for tc in reversed(run.tool_calls):
                    if tc["name"] == name and "result" not in tc:
                        tc["result"] = content[:500]
                        break
                if isinstance(content, str) and content.startswith("Error:"):
                    run.errors.append(f"{name}: {content[:200]}")

        elif event.type == "end":
            usage = event.data.get("usage", {})
            run.total_tokens = usage.get("total_tokens", 0)
            run.input_tokens = usage.get("input_tokens", 0)
            run.output_tokens = usage.get("output_tokens", 0)
            break

    return run


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------


def _create_client():
    """Create DeerFlowClient with oncall agent config.

    Returns (client, cp_ctx). Caller must keep cp_ctx alive — the SQLite
    connection closes when cp_ctx is garbage collected.
    """
    from cli.app import PROJECT_ROOT, load_agent_config, load_global_config, merge_agent_config
    from cli.bootstrap import create_checkpointer, setup_env

    setup_env()
    global_cfg = load_global_config()
    agent_cfg = merge_agent_config(global_cfg, load_agent_config("oncall"))

    checkpointer, cp_ctx = create_checkpointer()
    config_path = str(PROJECT_ROOT / "deer-flow" / "config.yaml")

    from deerflow.client import DeerFlowClient  # type: ignore[import-not-found]

    client = DeerFlowClient(
        config_path=config_path,
        checkpointer=checkpointer,
        model_name=agent_cfg.get("model"),
        thinking_enabled=agent_cfg.get("thinking_enabled", False),
        subagent_enabled=agent_cfg.get("subagent_enabled", False),
        agent_name="oncall",
    )

    ptc_tools_raw = agent_cfg.get("ptc_tools")
    if ptc_tools_raw:
        from deerflow.config.tool_config import PTCToolConfig
        from deerflow.config.app_config import set_app_config

        ptc_tools = [PTCToolConfig(**c) for c in ptc_tools_raw]
        patched = client._app_config.model_copy(update={"ptc_tools": ptc_tools})
        set_app_config(patched)
        client._app_config = patched

    return client, cp_ctx


# ---------------------------------------------------------------------------
# Output checks
# ---------------------------------------------------------------------------


def _check_output(response: str, output_checks: dict) -> dict[str, bool]:
    """Score final AI response against expected output properties."""
    checks: dict[str, bool] = {}

    if "must_contain" in output_checks:
        for term in output_checks["must_contain"]:
            checks[f"contains_{term}"] = term in response

    if "must_not_contain" in output_checks:
        for term in output_checks["must_not_contain"]:
            checks[f"not_contains_{term}"] = term not in response

    if "min_length" in output_checks:
        checks["min_length"] = len(response) >= output_checks["min_length"]

    if "has_response" in output_checks:
        checks["has_response"] = bool(response.strip()) == output_checks["has_response"]

    return checks


# ---------------------------------------------------------------------------
# Transcript conversion (stream events → process_eval format)
# ---------------------------------------------------------------------------


def _build_transcript(run: CapturedRun) -> list[dict]:
    """Convert CapturedRun to transcript format for heuristic rules."""
    transcript = []
    for tc in run.tool_calls:
        transcript.append({
            "run_type": "tool",
            "name": tc["name"],
            "inputs": tc.get("args", {}),
            "outputs": tc.get("result", ""),
            "total_tokens": 0,
        })
    # Add a synthetic LLM step with the token usage
    if run.total_tokens:
        transcript.append({
            "run_type": "llm",
            "name": "model",
            "inputs": {},
            "outputs": {},
            "total_tokens": run.total_tokens,
        })
    return transcript


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def evaluate(case: EvalCase, **_kwargs) -> EvalResult:
    """Run full agent pipeline and score process + output."""
    start = time.monotonic()
    cp_ctx = None
    try:
        client, cp_ctx = _create_client()
        thread_id = f"eval-e2e-{uuid.uuid4().hex[:8]}"
        query = case.input["query"]

        run = capture_run(client, query, thread_id)

    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return EvalResult(
            case_id=case.id, passed=False, score=0.0,
            details={}, actual={}, elapsed_ms=elapsed, error=str(e),
        )

    elapsed = (time.monotonic() - start) * 1000

    # Score process (heuristic rules)
    process_rules = case.expected.get("process_rules", [])
    transcript = _build_transcript(run)
    process_checks = apply_heuristic_rules(transcript, process_rules)

    # Score output
    output_checks_spec = case.expected.get("output_checks", {})
    output_checks = _check_output(run.final_response, output_checks_spec)

    # Combine all checks
    checks = {**process_checks, **output_checks}
    passed = all(checks.values()) if checks else False
    score = sum(checks.values()) / len(checks) if checks else 0.0

    actual = {
        "response_preview": run.final_response[:300],
        "tool_calls": [tc["name"] for tc in run.tool_calls],
        "total_tokens": run.total_tokens,
        "errors": run.errors,
    }

    return EvalResult(
        case_id=case.id, passed=passed, score=score,
        details=checks, actual=actual, elapsed_ms=elapsed,
    )
