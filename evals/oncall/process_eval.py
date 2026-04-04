"""Layer 2: Process-level eval — heuristic rules on agent transcripts."""

import json
import time
from pathlib import Path

from evals.framework.types import EvalCase, EvalResult


def _load_transcript(case: EvalCase) -> list[dict]:
    """Load transcript from case input.

    Supports:
    - input.transcript: inline list of step dicts
    - input.transcript_path: path to saved JSON file
    """
    if "transcript" in case.input:
        return case.input["transcript"]

    path = case.input.get("transcript_path")
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    msg = f"Case {case.id}: no transcript or transcript_path in input"
    raise ValueError(msg)


def apply_heuristic_rules(
    transcript: list[dict], rules: list[dict]
) -> dict[str, bool]:
    """Apply heuristic rules to transcript. Returns per-rule pass/fail."""
    tool_steps = [s for s in transcript if s.get("run_type") == "tool"]
    total_tokens = sum(s.get("total_tokens", 0) for s in transcript)

    checks: dict[str, bool] = {}
    for rule in rules:
        rule_type = rule["rule"]

        if rule_type == "called_tool":
            tool_name = rule["tool_name"]
            min_times = rule.get("min_times", 1)
            count = sum(1 for s in tool_steps if s.get("name") == tool_name)
            checks["called_tool"] = count >= min_times

        elif rule_type == "no_redundant_calls":
            tool_name = rule["tool_name"]
            max_times = rule.get("max_times", 2)
            count = sum(1 for s in tool_steps if s.get("name") == tool_name)
            checks["no_redundant_calls"] = count <= max_times

        elif rule_type == "token_budget":
            max_tokens = rule["max_tokens"]
            checks["token_budget"] = total_tokens <= max_tokens

        elif rule_type == "step_count":
            max_steps = rule["max_steps"]
            key_steps = [
                s for s in transcript if s.get("run_type") in ("llm", "tool")
            ]
            checks["step_count"] = len(key_steps) <= max_steps

    return checks


def evaluate(case: EvalCase, **kwargs) -> EvalResult:
    """Evaluate a process-level case."""
    start = time.monotonic()
    try:
        transcript = _load_transcript(case)
        rules = case.expected.get("heuristic_rules", [])
        checks = apply_heuristic_rules(transcript, rules)
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return EvalResult(
            case_id=case.id,
            passed=False,
            score=0.0,
            details={},
            actual={},
            elapsed_ms=elapsed,
            error=str(e),
        )

    elapsed = (time.monotonic() - start) * 1000
    passed = all(checks.values())
    score = sum(checks.values()) / len(checks) if checks else 0.0

    return EvalResult(
        case_id=case.id,
        passed=passed,
        score=score,
        details=checks,
        actual={
            "total_tokens": sum(s.get("total_tokens", 0) for s in transcript),
            "step_count": len(transcript),
        },
        elapsed_ms=elapsed,
    )
