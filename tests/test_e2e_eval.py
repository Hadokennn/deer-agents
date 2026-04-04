"""Tests for evals/oncall/e2e_eval.py — unit tests for capture/scoring logic.

Does NOT run the real agent — tests the scoring functions with mock data.
"""

from evals.oncall.e2e_eval import CapturedRun, _build_transcript, _check_output
from evals.oncall.process_eval import apply_heuristic_rules


def test_build_transcript():
    run = CapturedRun(
        final_response="诊断结果...",
        tool_calls=[
            {"name": "locate_field_schema", "args": {"field_name": "价格"}, "result": "found"},
            {"name": "task", "args": {"prompt": "分析..."}, "result": "ok"},
        ],
        total_tokens=25000,
    )
    transcript = _build_transcript(run)
    assert len(transcript) == 3  # 2 tool + 1 LLM
    assert transcript[0]["run_type"] == "tool"
    assert transcript[0]["name"] == "locate_field_schema"
    assert transcript[2]["run_type"] == "llm"
    assert transcript[2]["total_tokens"] == 25000


def test_build_transcript_applies_heuristics():
    run = CapturedRun(
        tool_calls=[
            {"name": "locate_field_schema", "args": {}},
        ],
        total_tokens=10000,
    )
    transcript = _build_transcript(run)
    rules = [
        {"rule": "called_tool", "tool_name": "locate_field_schema", "min_times": 1},
        {"rule": "token_budget", "max_tokens": 50000},
    ]
    checks = apply_heuristic_rules(transcript, rules)
    assert all(checks.values())


def test_check_output_must_contain():
    response = "该字段的 schema 配置中有 reaction_rules 控制显隐"
    checks = _check_output(response, {
        "must_contain": ["schema", "reaction"],
        "must_not_contain": ["不确定"],
        "min_length": 10,
    })
    assert checks["contains_schema"] is True
    assert checks["contains_reaction"] is True
    assert checks["not_contains_不确定"] is True
    assert checks["min_length"] is True


def test_check_output_failures():
    response = "不确定"
    checks = _check_output(response, {
        "must_contain": ["schema"],
        "must_not_contain": ["不确定"],
        "min_length": 50,
    })
    assert checks["contains_schema"] is False
    assert checks["not_contains_不确定"] is False
    assert checks["min_length"] is False


def test_check_output_has_response():
    assert _check_output("有内容", {"has_response": True})["has_response"] is True
    assert _check_output("", {"has_response": True})["has_response"] is False
    assert _check_output("  ", {"has_response": True})["has_response"] is False
