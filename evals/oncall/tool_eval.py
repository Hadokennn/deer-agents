"""Layer 1: Tool-level eval for SchemaLocatorTool.

Two modes:
- Mock (default): Uses fake MCP responses. Tests tool logic. Seconds.
- Live (--live):   Uses real MCP server. Tests real API integration. Minutes.

Scorer contract: evaluate(case, **kwargs) -> EvalResult
"""

import json
import time
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import MagicMock

from evals.framework.types import EvalCase, EvalResult
from evals.oncall.fixtures import (
    PARENT_DIM,
    SAMPLE_DETAIL,
    SAMPLE_DIM,
)
from tools.schema_locator import SchemaLocatorTool

TOOL_PARAM_KEYS = ("category_full_name", "field_name", "product_type", "product_sub_type")


# ---------------------------------------------------------------------------
# Mock MCP builders (for offline regression)
# ---------------------------------------------------------------------------


def _build_mcp_tools(mock_type: str) -> tuple[dict, MagicMock]:
    """Build mock MCP tools based on case mock type."""
    locate_tool = MagicMock()
    detail_tool = MagicMock()
    detail_tool.invoke.return_value = [
        {"type": "text", "text": json.dumps(SAMPLE_DETAIL)}
    ]

    if mock_type == "single_locate":
        locate_tool.invoke.return_value = [
            {"type": "text", "text": json.dumps([{"config_dimension": SAMPLE_DIM}])}
        ]
    elif mock_type == "fallback_locate":
        locate_tool.invoke.side_effect = [
            [{"type": "text", "text": "[]"}],
            [{"type": "text", "text": json.dumps([{"config_dimension": PARENT_DIM}])}],
        ]
    elif mock_type == "ambiguous_locate":
        dims = [
            {"config_dimension": {**SAMPLE_DIM, "product_type": "1"}},
            {"config_dimension": {**SAMPLE_DIM, "product_type": "11"}},
        ]
        locate_tool.invoke.return_value = [
            {"type": "text", "text": json.dumps(dims)}
        ]
    elif mock_type == "empty_locate":
        locate_tool.invoke.return_value = [
            {"type": "text", "text": "[]"}
        ]
    else:
        msg = f"Unknown mock type: {mock_type}"
        raise ValueError(msg)

    mcp_tools = {
        "bytedance-mcp-ace_ai_ace_ai_locate_template": locate_tool,
        "bytedance-mcp-ace_ai_ace_ai_get_last_template_detail": detail_tool,
    }
    return mcp_tools, locate_tool


# ---------------------------------------------------------------------------
# Assertion checkers
# ---------------------------------------------------------------------------


def _check_expected(
    actual: dict, expected: dict, locate_tool: MagicMock | None
) -> dict:
    """Compare actual tool output against expected fields."""
    checks: dict[str, bool] = {}
    for key, exp_val in expected.items():
        # Exact match checks
        if key == "status":
            checks["status"] = actual.get("status") == exp_val
        elif key == "field_key":
            checks["field_key"] = actual.get("field_key") == exp_val
        elif key == "group":
            checks["group"] = actual.get("group") == exp_val
        elif key == "x_component":
            checks["x_component"] = actual.get("x_component") == exp_val
        elif key == "category":
            checks["category"] = actual.get("category") == exp_val

        # Boolean presence checks
        elif key == "has_reaction_rules":
            checks["has_reaction_rules"] = bool(actual.get("reaction_rules")) == exp_val
        elif key == "has_available_fields":
            checks["has_available_fields"] = bool(actual.get("available_fields")) == exp_val
        elif key == "has_component_sources":
            checks["has_component_sources"] = (
                bool(actual.get("component_sources")) == exp_val
            )
        elif key == "has_next_action":
            checks["has_next_action"] = bool(actual.get("next_action")) == exp_val
        elif key == "has_field_key":
            checks["has_field_key"] = bool(actual.get("field_key")) == exp_val
        elif key == "has_schema_path":
            checks["has_schema_path"] = bool(actual.get("schema_path")) == exp_val

        # Count checks
        elif key == "candidates_count":
            checks["candidates_count"] = (
                len(actual.get("candidates", [])) == exp_val
            )
        elif key == "field_count_min":
            checks["field_count_min"] = actual.get("field_count", 0) >= exp_val
        elif key == "locate_call_count":
            if locate_tool:
                checks["locate_call_count"] = (
                    locate_tool.invoke.call_count == exp_val
                )
            else:
                checks["locate_call_count"] = False
    return checks


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def evaluate(case: EvalCase, *, tmp_dir: Path | None = None, live: bool = False, **_kwargs) -> EvalResult:
    """Evaluate a single tool-level case.

    live=False (default): mock MCP, tests tool logic
    live=True: real MCP, tests API integration
    """
    tmp_dir = tmp_dir or Path(mkdtemp())

    if live:
        return _evaluate_live(case, tmp_dir)
    return _evaluate_mock(case, tmp_dir)


def _evaluate_mock(case: EvalCase, tmp_dir: Path) -> EvalResult:
    """Run with mock MCP tools."""
    mock_type = case.input.get("mock", "single_locate")

    start = time.monotonic()
    _orig_locate = None
    try:
        mcp_tools, locate_tool = _build_mcp_tools(mock_type)
        tool = SchemaLocatorTool(mcp_tools=mcp_tools, schema_dir=str(tmp_dir))

        if case.input.get("mock_component_code"):
            import tools.schema_locator as sl

            _orig_locate = sl.locate_component_code
            sl.locate_component_code = lambda name, *a, **kw: (
                "/repo/root",
                [
                    {
                        "name": name,
                        "kind": "const/fn",
                        "file": f"src/{name}.tsx",
                        "line": 10,
                        "span": 20,
                        "exported": True,
                    }
                ],
            )

        tool_params = {k: v for k, v in case.input.items() if k in TOOL_PARAM_KEYS}
        actual = tool._run(**tool_params)

    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return EvalResult(
            case_id=case.id, passed=False, score=0.0,
            details={}, actual={}, elapsed_ms=elapsed, error=str(e),
        )
    finally:
        if _orig_locate is not None:
            import tools.schema_locator as sl
            sl.locate_component_code = _orig_locate

    elapsed = (time.monotonic() - start) * 1000
    checks = _check_expected(actual, case.expected, locate_tool)
    passed = all(checks.values())
    score = sum(checks.values()) / len(checks) if checks else 0.0

    return EvalResult(
        case_id=case.id, passed=passed, score=score,
        details=checks, actual=actual, elapsed_ms=elapsed,
    )


def _evaluate_live(case: EvalCase, tmp_dir: Path) -> EvalResult:
    """Run with real MCP server — no mocks."""
    from cli.bootstrap import setup_env
    setup_env()

    start = time.monotonic()
    try:
        # No mcp_tools → lazy-loads from real MCP cache
        tool = SchemaLocatorTool(schema_dir=str(tmp_dir))
        tool_params = {k: v for k, v in case.input.items() if k in TOOL_PARAM_KEYS}
        actual = tool._run(**tool_params)

    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return EvalResult(
            case_id=case.id, passed=False, score=0.0,
            details={}, actual={}, elapsed_ms=elapsed, error=str(e),
        )

    elapsed = (time.monotonic() - start) * 1000
    checks = _check_expected(actual, case.expected, locate_tool=None)
    passed = all(checks.values())
    score = sum(checks.values()) / len(checks) if checks else 0.0

    return EvalResult(
        case_id=case.id, passed=passed, score=score,
        details=checks, actual=actual, elapsed_ms=elapsed,
    )
