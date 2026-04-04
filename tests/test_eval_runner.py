"""Tests for evals/framework/runner.py."""

from evals.framework.runner import load_cases, run_eval
from evals.framework.types import EvalReport


def test_load_cases():
    cases = load_cases("oncall", "tool")
    assert len(cases) == 5
    ids = [c.id for c in cases]
    assert "happy_path_field_found" in ids


def test_load_cases_filter_by_tag():
    cases = load_cases("oncall", "tool", tags=["fallback"])
    assert len(cases) == 1
    assert cases[0].id == "category_fallback"


def test_load_cases_filter_by_id():
    cases = load_cases("oncall", "tool", case_ids=["ambiguous_template"])
    assert len(cases) == 1
    assert cases[0].id == "ambiguous_template"


def test_load_live_cases():
    cases = load_cases("oncall", "tool", live=True)
    assert len(cases) == 3
    ids = [c.id for c in cases]
    assert "live_overview" in ids


def test_run_eval_tool_layer(tmp_path):
    report = run_eval("tool", agent="oncall", tmp_dir=tmp_path)
    assert isinstance(report, EvalReport)
    assert report.summary["total"] == 5
    assert report.summary["pass_rate"] > 0
    assert "by_tag" in report.summary
    assert "cold-start" in report.summary["by_tag"]
