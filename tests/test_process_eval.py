"""Tests for evals/oncall/process_eval.py — Layer 2 scorer."""

from evals.framework.types import EvalCase
from evals.oncall.process_eval import apply_heuristic_rules, evaluate

GOOD_TRANSCRIPT = [
    {
        "run_type": "tool",
        "name": "locate_field_schema",
        "inputs": {"field_name": "价格"},
        "outputs": {"status": "found"},
        "total_tokens": 0,
    },
    {
        "run_type": "llm",
        "name": "model",
        "inputs": {},
        "outputs": {},
        "total_tokens": 8000,
    },
    {
        "run_type": "tool",
        "name": "task",
        "inputs": {"prompt": "分析..."},
        "outputs": {},
        "total_tokens": 0,
    },
    {
        "run_type": "llm",
        "name": "model",
        "inputs": {},
        "outputs": {},
        "total_tokens": 15000,
    },
]

BAD_TRANSCRIPT = [
    {
        "run_type": "tool",
        "name": "locate_field_schema",
        "inputs": {"field_name": "价格"},
        "outputs": {},
        "total_tokens": 0,
    },
    {
        "run_type": "tool",
        "name": "locate_field_schema",
        "inputs": {"field_name": "价格"},
        "outputs": {},
        "total_tokens": 0,
    },
    {
        "run_type": "tool",
        "name": "locate_field_schema",
        "inputs": {"field_name": "价格"},
        "outputs": {},
        "total_tokens": 0,
    },
    {
        "run_type": "llm",
        "name": "model",
        "inputs": {},
        "outputs": {},
        "total_tokens": 60000,
    },
]


def test_heuristic_good_transcript():
    rules = [
        {"rule": "called_tool", "tool_name": "locate_field_schema", "min_times": 1},
        {
            "rule": "no_redundant_calls",
            "tool_name": "locate_field_schema",
            "max_times": 2,
        },
        {"rule": "token_budget", "max_tokens": 50000},
        {"rule": "step_count", "max_steps": 15},
    ]
    checks = apply_heuristic_rules(GOOD_TRANSCRIPT, rules)
    assert all(checks.values())


def test_heuristic_bad_transcript():
    rules = [
        {
            "rule": "no_redundant_calls",
            "tool_name": "locate_field_schema",
            "max_times": 2,
        },
        {"rule": "token_budget", "max_tokens": 50000},
    ]
    checks = apply_heuristic_rules(BAD_TRANSCRIPT, rules)
    assert checks["no_redundant_calls"] is False
    assert checks["token_budget"] is False


def test_evaluate_with_inline_transcript():
    case = EvalCase(
        id="efficient_flow",
        layer="process",
        input={"transcript": GOOD_TRANSCRIPT},
        expected={
            "heuristic_rules": [
                {
                    "rule": "called_tool",
                    "tool_name": "locate_field_schema",
                    "min_times": 1,
                },
                {"rule": "token_budget", "max_tokens": 50000},
            ]
        },
    )
    result = evaluate(case)
    assert result.passed
    assert result.score == 1.0


def test_evaluate_bad_transcript():
    case = EvalCase(
        id="wasteful_flow",
        layer="process",
        input={"transcript": BAD_TRANSCRIPT},
        expected={
            "heuristic_rules": [
                {
                    "rule": "no_redundant_calls",
                    "tool_name": "locate_field_schema",
                    "max_times": 2,
                },
            ]
        },
    )
    result = evaluate(case)
    assert not result.passed
    assert result.details["no_redundant_calls"] is False
