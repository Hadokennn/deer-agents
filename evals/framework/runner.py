"""Eval runner: load cases, dispatch to scorer, collect results."""

import importlib
import json
from pathlib import Path

from evals.framework.types import EvalCase, EvalReport, EvalResult

EVALS_ROOT = Path(__file__).resolve().parent.parent  # evals/


def load_cases(
    agent: str,
    layer: str,
    *,
    case_ids: list[str] | None = None,
    tags: list[str] | None = None,
    live: bool = False,
) -> list[EvalCase]:
    """Load eval cases from JSON file, optionally filtered.

    For tool layer with live=True, loads tool_cases_live.json instead.
    """
    if live and layer == "tool":
        case_file = EVALS_ROOT / agent / "tool_cases_live.json"
    else:
        case_file = EVALS_ROOT / agent / f"{layer}_cases.json"
    if not case_file.exists():
        raise FileNotFoundError(f"No cases found: {case_file}")

    raw = json.loads(case_file.read_text(encoding="utf-8"))
    cases = [EvalCase(**c) for c in raw]

    if case_ids:
        cases = [c for c in cases if c.id in case_ids]
    if tags:
        cases = [c for c in cases if any(t in c.tags for t in tags)]

    return cases


def run_eval(
    layer: str,
    *,
    agent: str = "oncall",
    case_ids: list[str] | None = None,
    tags: list[str] | None = None,
    live: bool = False,
    **kwargs,
) -> EvalReport:
    """Run eval suite for a given layer and agent."""
    cases = load_cases(agent, layer, case_ids=case_ids, tags=tags, live=live)
    kwargs["live"] = live

    # Import scorer module: evals.{agent}.{layer}_eval
    mod = importlib.import_module(f"evals.{agent}.{layer}_eval")
    evaluate_fn = mod.evaluate

    results: list[EvalResult] = []
    for case in cases:
        result = evaluate_fn(case, **kwargs)
        results.append(result)

    report = EvalReport.build(agent=agent, layer=layer, results=results)

    # Enrich summary with tag breakdown
    tag_stats: dict[str, dict] = {}
    for case, result in zip(cases, results):
        for tag in case.tags:
            if tag not in tag_stats:
                tag_stats[tag] = {"total": 0, "passed": 0}
            tag_stats[tag]["total"] += 1
            if result.passed:
                tag_stats[tag]["passed"] += 1
    report.summary["by_tag"] = tag_stats

    return report
