"""Tests for evals/oncall/tool_eval.py — Layer 1 scorer."""

from evals.framework.types import EvalCase
from evals.oncall.tool_eval import evaluate


def test_evaluate_happy_path(tmp_path):
    case = EvalCase(
        id="happy_path_field_found",
        layer="tool",
        input={
            "category_full_name": "购物>果蔬生鲜>水果",
            "product_type": "1",
            "field_name": "商家名称",
            "mock": "single_locate",
        },
        expected={
            "status": "found",
            "field_key": "accountName",
            "group": "merchantInfo",
            "x_component": "AccountName",
            "has_reaction_rules": True,
        },
    )
    result = evaluate(case, tmp_dir=tmp_path)
    assert result.passed
    assert result.score == 1.0
    assert result.details["status"] is True
    assert result.details["field_key"] is True


def test_evaluate_field_not_found(tmp_path):
    case = EvalCase(
        id="field_not_found",
        layer="tool",
        input={
            "category_full_name": "购物>果蔬生鲜>水果",
            "product_type": "1",
            "field_name": "不存在的字段",
            "mock": "single_locate",
        },
        expected={
            "status": "field_not_found",
            "has_available_fields": True,
        },
    )
    result = evaluate(case, tmp_dir=tmp_path)
    assert result.passed
    assert result.score == 1.0


def test_evaluate_ambiguous(tmp_path):
    case = EvalCase(
        id="ambiguous_template",
        layer="tool",
        input={
            "category_full_name": "购物>果蔬生鲜>水果",
            "field_name": "any",
            "mock": "ambiguous_locate",
        },
        expected={
            "status": "ambiguous",
            "candidates_count": 2,
        },
    )
    result = evaluate(case, tmp_dir=tmp_path)
    assert result.passed
    assert result.score == 1.0


def test_evaluate_category_fallback(tmp_path):
    case = EvalCase(
        id="category_fallback",
        layer="tool",
        input={
            "category_full_name": "购物>果蔬生鲜>水果",
            "product_type": "1",
            "field_name": "商家名称",
            "mock": "fallback_locate",
        },
        expected={
            "status": "found",
            "category": "购物>果蔬生鲜",
            "field_key": "accountName",
            "locate_call_count": 2,
        },
    )
    result = evaluate(case, tmp_dir=tmp_path)
    assert result.passed
    assert result.score == 1.0


def test_evaluate_cross_concern(tmp_path):
    case = EvalCase(
        id="cross_concern_with_code",
        layer="tool",
        input={
            "category_full_name": "购物>果蔬生鲜>水果",
            "product_type": "1",
            "field_name": "商家名称",
            "mock": "single_locate",
            "mock_component_code": True,
        },
        expected={
            "status": "found",
            "has_component_sources": True,
            "has_next_action": True,
        },
    )
    result = evaluate(case, tmp_dir=tmp_path)
    assert result.passed
    assert result.score == 1.0


def test_structural_checks(tmp_path):
    """Structural checks used by live mode (has_field_key, has_schema_path)."""
    case = EvalCase(
        id="structural",
        layer="tool",
        input={
            "category_full_name": "购物>果蔬生鲜>水果",
            "product_type": "1",
            "field_name": "商家名称",
            "mock": "single_locate",
        },
        expected={
            "status": "found",
            "has_field_key": True,
            "has_schema_path": True,
        },
    )
    result = evaluate(case, tmp_dir=tmp_path)
    assert result.passed
    assert result.details["has_field_key"] is True
    assert result.details["has_schema_path"] is True
