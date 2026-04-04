"""Tests for evals/framework/types.py."""

from evals.framework.types import EvalCase, EvalReport, EvalResult


def test_eval_case_from_dict():
    d = {
        "id": "test1",
        "layer": "tool",
        "input": {"x": 1},
        "expected": {"status": "found"},
        "tags": ["cold-start"],
    }
    case = EvalCase(**d)
    assert case.id == "test1"
    assert case.tags == ["cold-start"]


def test_eval_case_default_tags():
    case = EvalCase(id="t", layer="tool", input={}, expected={})
    assert case.tags == []


def test_eval_result():
    r = EvalResult(
        case_id="t1",
        passed=True,
        score=1.0,
        details={"status": True},
        actual={"status": "found"},
        elapsed_ms=12.5,
    )
    assert r.passed
    assert r.error is None


def test_eval_report_summary():
    results = [
        EvalResult(case_id="t1", passed=True, score=1.0, details={}, actual={}, elapsed_ms=10),
        EvalResult(case_id="t2", passed=False, score=0.5, details={}, actual={}, elapsed_ms=20),
    ]
    report = EvalReport.build(agent="oncall", layer="tool", results=results)
    assert report.summary["total"] == 2
    assert report.summary["passed"] == 1
    assert report.summary["pass_rate"] == 0.5
