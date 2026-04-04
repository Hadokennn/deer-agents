"""Eval report: console output + JSON persistence."""

import json
from datetime import datetime
from pathlib import Path

from evals.framework.types import EvalReport

REPORT_DIR = Path(".deer-flow/eval-reports")


def print_report(report: EvalReport) -> None:
    """Print report to console."""
    print(f"\nEval Report: {report.agent} / {report.layer}")
    print(f"  {report.timestamp}\n")

    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"  {r.case_id:40s} {status:4s}  {r.score:.2f}  ({r.elapsed_ms:.0f}ms)"
        )
        if not r.passed:
            for check, ok in r.details.items():
                if not ok:
                    actual_val = r.actual.get(check, "N/A")
                    print(f"    - {check}: got {actual_val!r}")
        if r.error:
            print(f"    ERROR: {r.error}")

    s = report.summary
    print(
        f"\n  Summary: {s['passed']}/{s['total']} passed ({s['pass_rate']:.1%})"
        f"  avg_score={s['avg_score']:.2f}  total={s['elapsed_ms']:.0f}ms"
    )

    if "by_tag" in s:
        print("  By tag:")
        for tag, stats in s["by_tag"].items():
            rate = stats["passed"] / stats["total"] if stats["total"] else 0
            print(f"    {tag:20s} {stats['passed']}/{stats['total']} ({rate:.0%})")
    print()


def save_report(report: EvalReport) -> Path:
    """Save report as JSON. Returns path."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.fromisoformat(report.timestamp).strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"{report.agent}_{report.layer}_{ts}.json"

    data = {
        "agent": report.agent,
        "layer": report.layer,
        "timestamp": report.timestamp,
        "summary": report.summary,
        "cases": [
            {
                "case_id": r.case_id,
                "passed": r.passed,
                "score": r.score,
                "details": r.details,
                "elapsed_ms": r.elapsed_ms,
                "error": r.error,
            }
            for r in report.results
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
