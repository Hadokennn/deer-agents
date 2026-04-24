# Eval Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Baseline-Diff comparison, LLM-as-Judge semantic scoring, and multi-run statistics to the deer-agents eval framework, enabling quantitative before/after comparisons when changing prompts, skills, or MCP tools.

**Architecture:** Three new modules (`stats.py`, `judge.py`, `diff.py`) sit alongside the existing `types.py`/`runner.py`/`report.py` framework. `types.py` gains three new dataclasses (`RunStats`, `JudgeResult`, and extended `EvalResult`/`EvalReport`). The runner orchestrates multi-run execution and delegates aggregation to `stats.py`. The judge is opt-in per case via `judge_rubric` in case definitions, supports groundedness checking by receiving tool outputs as evidence. Diff loads labeled JSON baselines and produces tag-aware comparison reports (regression vs capability-dip).

**Tech Stack:** Python 3.11+, pytest, `statistics` stdlib, `deerflow.models.factory.create_chat_model`, `langchain_core.messages`

**Spec:** `docs/superpowers/specs/2026-04-13-eval-enhancements-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `evals/framework/stats.py` | `aggregate_results()` — merge N EvalResults for same case into one with RunStats |
| `evals/framework/judge.py` | `judge_response()` — LLM semantic scoring with rubric, groundedness via tool_outputs, prompt templates, JSON parsing |
| `evals/framework/diff.py` | `load_baseline()`, `compare_reports()`, `CaseDiff`, `DiffReport` — tag-aware (regression vs capability-dip) |
| `tests/test_eval_stats.py` | Unit tests for multi-run aggregation (incl. any_passed/all_passed) |
| `tests/test_eval_judge.py` | Unit tests for judge prompt building, groundedness, JSON parsing, mocked LLM call |
| `tests/test_eval_diff.py` | Unit tests for baseline loading, report comparison, capability-dip |
| `tests/test_judge_calibration.py` | Judge calibration — assert judge scores correlate with human annotations (Pearson > 0.7) |
| `evals/oncall/judge_calibration.json` | Human-annotated scores for 5-10 cases, used by calibration tests |

### Modified Files

| File | Changes |
|------|---------|
| `evals/framework/types.py` | Add `RunStats` (+`any_passed`, `all_passed`), `JudgeResult`; extend `EvalResult` (+`run_stats`, `judge`), `EvalReport` (+`label`) |
| `evals/framework/runner.py` | Add `runs` param to `run_eval()`, multi-run loop with aggregation |
| `evals/framework/report.py` | Label in `save_report()`, `run_stats`/`judge` serialization, new `print_diff()` |
| `evals/oncall/e2e_eval.py` | Wire `judge_response()` into `evaluate()` |
| `scripts/run_eval.py` | New CLI args: `--label`, `--diff`, `--runs`, `--judge-model`, `--no-judge` |
| `tests/test_eval_types.py` | Tests for new dataclasses and fields |
| `tests/test_eval_runner.py` | Test `runs > 1` integration |

### Parallelization

After Task 1, three feature tracks are independent and can be dispatched in parallel:

```
Task 1 (types) ─┬─→ Task 2 (stats) → Task 3 (runner)  ─┐
                 ├─→ Task 4 (judge) → Task 5 (e2e)      ├─→ Task 8 (CLI) → Task 9 (judge calibration)
                 └─→ Task 6 (diff) → Task 7 (report)   ─┘
```

---

### Task 1: Extend types.py with new dataclasses

**Files:**
- Modify: `evals/framework/types.py`
- Modify: `tests/test_eval_types.py`

- [ ] **Step 1: Write failing tests for new types**

Append to `tests/test_eval_types.py`:

```python
from evals.framework.types import EvalCase, EvalReport, EvalResult, JudgeResult, RunStats


def test_run_stats_construction():
    rs = RunStats(
        runs=3, scores=[0.8, 0.9, 1.0], pass_count=3,
        median_score=0.9, score_std=0.1, pass_rate=1.0,
        any_passed=True, all_passed=True,
    )
    assert rs.runs == 3
    assert rs.median_score == 0.9
    assert rs.pass_rate == 1.0
    assert rs.any_passed is True
    assert rs.all_passed is True


def test_run_stats_flaky():
    """any_passed=True but all_passed=False indicates flaky case."""
    rs = RunStats(
        runs=3, scores=[0.8, 0.3, 0.9], pass_count=2,
        median_score=0.8, score_std=0.3, pass_rate=2/3,
        any_passed=True, all_passed=False,
    )
    assert rs.any_passed is True
    assert rs.all_passed is False


def test_judge_result_construction():
    jr = JudgeResult(
        score=0.85,
        reasoning="Good answer",
        dimension_scores={"accuracy": 0.9, "completeness": 0.8},
    )
    assert jr.score == 0.85
    assert jr.dimension_scores["accuracy"] == 0.9


def test_eval_result_with_run_stats():
    rs = RunStats(runs=2, scores=[0.7, 0.9], pass_count=2,
                  median_score=0.8, score_std=0.14, pass_rate=1.0,
                  any_passed=True, all_passed=True)
    r = EvalResult(
        case_id="t1", passed=True, score=0.8,
        details={}, actual={}, elapsed_ms=100, run_stats=rs,
    )
    assert r.run_stats is not None
    assert r.run_stats.runs == 2


def test_eval_result_with_judge():
    jr = JudgeResult(score=0.9, reasoning="ok", dimension_scores={})
    r = EvalResult(
        case_id="t1", passed=True, score=0.9,
        details={}, actual={}, elapsed_ms=50, judge=jr,
    )
    assert r.judge is not None
    assert r.judge.score == 0.9


def test_eval_result_new_fields_default_none():
    r = EvalResult(
        case_id="t1", passed=True, score=1.0,
        details={}, actual={}, elapsed_ms=10,
    )
    assert r.run_stats is None
    assert r.judge is None


def test_eval_report_label():
    results = [
        EvalResult(case_id="t1", passed=True, score=1.0, details={}, actual={}, elapsed_ms=10),
    ]
    report = EvalReport.build(agent="oncall", layer="tool", results=results)
    assert report.label is None
    report.label = "v1"
    assert report.label == "v1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_types.py -v`
Expected: ImportError for `RunStats`, `JudgeResult`; AttributeError for `run_stats`, `judge`, `label`, `any_passed`, `all_passed`

- [ ] **Step 3: Implement types changes**

Replace `evals/framework/types.py` with:

```python
"""Core eval data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class EvalCase:
    id: str
    layer: str  # "tool" | "process" | "e2e"
    input: dict
    expected: dict
    tags: list[str] = field(default_factory=list)


@dataclass
class RunStats:
    runs: int
    scores: list[float]
    pass_count: int
    median_score: float
    score_std: float
    pass_rate: float          # pass_count / runs
    any_passed: bool          # pass@k — at least one success (capability ceiling)
    all_passed: bool          # pass^k — all succeeded (reliability)


@dataclass
class JudgeResult:
    score: float  # 0.0 - 1.0
    reasoning: str
    dimension_scores: dict  # {"accuracy": 0.9, ...}


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    score: float  # 0.0 - 1.0
    details: dict  # per-check pass/fail
    actual: dict  # raw system output
    elapsed_ms: float
    error: str | None = None
    run_stats: RunStats | None = None
    judge: JudgeResult | None = None


@dataclass
class EvalReport:
    agent: str
    layer: str
    timestamp: str
    results: list[EvalResult]
    summary: dict
    label: str | None = None

    @classmethod
    def build(cls, agent: str, layer: str, results: list[EvalResult]) -> EvalReport:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        scores = [r.score for r in results]
        return cls(
            agent=agent,
            layer=layer,
            timestamp=datetime.now(timezone.utc).isoformat(),
            results=results,
            summary={
                "total": total,
                "passed": passed,
                "pass_rate": passed / total if total else 0,
                "avg_score": sum(scores) / total if total else 0,
                "elapsed_ms": sum(r.elapsed_ms for r in results),
            },
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_types.py -v`
Expected: All PASS (including existing tests)

- [ ] **Step 5: Commit**

```bash
git add evals/framework/types.py tests/test_eval_types.py
git commit -m "feat(eval): add RunStats, JudgeResult types and extend EvalResult/EvalReport"
```

---

### Task 2: Multi-run aggregation (stats.py)

**Files:**
- Create: `evals/framework/stats.py`
- Create: `tests/test_eval_stats.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_eval_stats.py`:

```python
"""Tests for evals/framework/stats.py."""

from evals.framework.types import EvalResult
from evals.framework.stats import aggregate_results


def _make_result(case_id: str, passed: bool, score: float, elapsed_ms: float = 100,
                 error: str | None = None) -> EvalResult:
    return EvalResult(
        case_id=case_id, passed=passed, score=score,
        details={"check": passed}, actual={"val": score},
        elapsed_ms=elapsed_ms, error=error,
    )


def test_aggregate_three_results():
    results = [
        _make_result("c1", True, 0.8, 100),
        _make_result("c1", True, 1.0, 200),
        _make_result("c1", True, 0.6, 150),
    ]
    agg = aggregate_results(results)
    assert agg.case_id == "c1"
    assert agg.passed is True
    assert agg.score == 0.8  # median of [0.6, 0.8, 1.0]
    assert agg.run_stats is not None
    assert agg.run_stats.runs == 3
    assert agg.run_stats.pass_count == 3
    assert agg.run_stats.pass_rate == 1.0
    assert agg.run_stats.median_score == 0.8
    assert agg.run_stats.score_std > 0
    assert agg.run_stats.any_passed is True
    assert agg.run_stats.all_passed is True


def test_aggregate_single_result():
    results = [_make_result("c1", True, 0.9)]
    agg = aggregate_results(results)
    assert agg.score == 0.9
    assert agg.run_stats.runs == 1
    assert agg.run_stats.score_std == 0.0
    assert agg.run_stats.any_passed is True
    assert agg.run_stats.all_passed is True


def test_aggregate_majority_pass():
    results = [
        _make_result("c1", True, 0.8),
        _make_result("c1", True, 0.7),
        _make_result("c1", False, 0.3),
    ]
    agg = aggregate_results(results)
    assert agg.passed is True  # 2/3 > 50%
    assert agg.run_stats.pass_count == 2
    assert agg.run_stats.any_passed is True   # pass@k: at least one
    assert agg.run_stats.all_passed is False   # pass^k: not all — flaky


def test_aggregate_majority_fail():
    results = [
        _make_result("c1", True, 0.8),
        _make_result("c1", False, 0.3),
        _make_result("c1", False, 0.2),
    ]
    agg = aggregate_results(results)
    assert agg.passed is False  # 1/3 < 50%
    assert agg.run_stats.any_passed is True   # still passed once
    assert agg.run_stats.all_passed is False


def test_aggregate_all_fail():
    results = [
        _make_result("c1", False, 0.1),
        _make_result("c1", False, 0.2),
        _make_result("c1", False, 0.0),
    ]
    agg = aggregate_results(results)
    assert agg.run_stats.any_passed is False  # pass@k = False: capability gap
    assert agg.run_stats.all_passed is False


def test_aggregate_with_errors():
    results = [
        _make_result("c1", True, 0.8),
        _make_result("c1", False, 0.0, error="timeout"),
        _make_result("c1", True, 0.9, error="warning"),
    ]
    agg = aggregate_results(results)
    assert agg.error is not None
    assert "timeout" in agg.error
    assert "warning" in agg.error


def test_aggregate_preserves_last_details():
    r1 = _make_result("c1", True, 0.8)
    r1.details = {"check_a": True}
    r1.actual = {"val": "first"}
    r2 = _make_result("c1", True, 0.9)
    r2.details = {"check_a": True, "check_b": True}
    r2.actual = {"val": "last"}
    agg = aggregate_results([r1, r2])
    assert agg.details == {"check_a": True, "check_b": True}
    assert agg.actual == {"val": "last"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_stats.py -v`
Expected: ImportError — `evals.framework.stats` does not exist

- [ ] **Step 3: Implement aggregate_results**

Create `evals/framework/stats.py`:

```python
"""Multi-run aggregation for eval results."""

import statistics

from evals.framework.types import EvalResult, RunStats


def aggregate_results(results: list[EvalResult]) -> EvalResult:
    """Aggregate multiple EvalResults for the same case into one.

    Uses median for scores/elapsed, majority vote for passed, last run for details/actual.
    """
    scores = [r.score for r in results]
    pass_count = sum(1 for r in results if r.passed)
    n = len(results)

    median_score = statistics.median(scores)
    score_std = statistics.stdev(scores) if n > 1 else 0.0

    errors = [r.error for r in results if r.error]
    combined_error = "; ".join(errors) if errors else None

    last = results[-1]

    return EvalResult(
        case_id=last.case_id,
        passed=pass_count > n / 2,
        score=median_score,
        details=last.details,
        actual=last.actual,
        elapsed_ms=statistics.median([r.elapsed_ms for r in results]),
        error=combined_error,
        run_stats=RunStats(
            runs=n,
            scores=scores,
            pass_count=pass_count,
            median_score=median_score,
            score_std=score_std,
            pass_rate=pass_count / n,
            any_passed=pass_count > 0,
            all_passed=pass_count == n,
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_stats.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Run all existing tests for regression**

Run: `pytest tests/ -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add evals/framework/stats.py tests/test_eval_stats.py
git commit -m "feat(eval): add multi-run aggregation in stats.py"
```

---

### Task 3: Multi-run runner integration (runner.py)

**Files:**
- Modify: `evals/framework/runner.py`
- Modify: `tests/test_eval_runner.py`

- [ ] **Step 1: Write failing test for runs > 1**

Append to `tests/test_eval_runner.py`:

```python
def test_run_eval_with_multiple_runs(tmp_path):
    report = run_eval("tool", agent="oncall", runs=2, tmp_dir=tmp_path)
    assert isinstance(report, EvalReport)
    assert report.summary["total"] == 5
    for r in report.results:
        assert r.run_stats is not None
        assert r.run_stats.runs == 2
        assert len(r.run_stats.scores) == 2
        assert isinstance(r.run_stats.any_passed, bool)
        assert isinstance(r.run_stats.all_passed, bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_runner.py::test_run_eval_with_multiple_runs -v`
Expected: TypeError — `run_eval()` got unexpected keyword argument `runs`

- [ ] **Step 3: Implement multi-run in runner.py**

Replace `evals/framework/runner.py`:

```python
"""Eval runner: load cases, dispatch to scorer, collect results."""

import importlib
import json
from pathlib import Path

from evals.framework.stats import aggregate_results
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
    runs: int = 1,
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
        if runs > 1:
            case_results = [evaluate_fn(case, **kwargs) for _ in range(runs)]
            result = aggregate_results(case_results)
        else:
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
```

- [ ] **Step 4: Run new test to verify it passes**

Run: `pytest tests/test_eval_runner.py::test_run_eval_with_multiple_runs -v`
Expected: PASS

- [ ] **Step 5: Run all tests for regression**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add evals/framework/runner.py tests/test_eval_runner.py
git commit -m "feat(eval): add multi-run support (--runs N) in runner"
```

---

### Task 4: LLM-as-Judge (judge.py)

**Files:**
- Create: `evals/framework/judge.py`
- Create: `tests/test_eval_judge.py`

- [ ] **Step 1: Write failing tests for prompt helpers and JSON parsing**

Create `tests/test_eval_judge.py`:

```python
"""Tests for evals/framework/judge.py."""

import json
from unittest.mock import MagicMock, patch

from evals.framework.judge import (
    _build_dimensions_description,
    _format_tool_outputs,
    _parse_judge_response,
    judge_response,
    DIMENSION_DESCRIPTIONS,
)


def test_dimension_descriptions_has_groundedness():
    assert "groundedness" in DIMENSION_DESCRIPTIONS
    assert "工具输出" in DIMENSION_DESCRIPTIONS["groundedness"]


def test_build_dimensions_description():
    desc = _build_dimensions_description(["accuracy", "groundedness", "completeness"])
    assert "准确性" in desc
    assert "基于事实" in desc
    assert "完整性" in desc


def test_build_dimensions_description_unknown():
    desc = _build_dimensions_description(["accuracy", "custom_dim"])
    assert "准确性" in desc
    assert "custom_dim" in desc


def test_format_tool_outputs_none():
    assert "无工具输出" in _format_tool_outputs(None)
    assert "无工具输出" in _format_tool_outputs([])


def test_format_tool_outputs_with_data():
    outputs = [
        {"name": "locate_field_schema", "output": {"status": "found", "field_key": "name"}},
        {"name": "search_templates", "result": "3 templates found"},
    ]
    formatted = _format_tool_outputs(outputs)
    assert "locate_field_schema" in formatted
    assert "search_templates" in formatted
    assert "found" in formatted


def test_parse_valid_json():
    raw = json.dumps({
        "dimensions": {"accuracy": {"score": 0.9, "reason": "good"}},
        "overall_reasoning": "solid",
        "overall_score": 0.9,
    })
    parsed = _parse_judge_response(raw)
    assert parsed["overall_score"] == 0.9


def test_parse_markdown_code_block():
    raw = '一些文字\n```json\n{"overall_score": 0.75, "dimensions": {}, "overall_reasoning": "ok"}\n```\n其他文字'
    parsed = _parse_judge_response(raw)
    assert parsed["overall_score"] == 0.75


def test_parse_regex_fallback():
    raw = 'garbled output but "overall_score": 0.6 somewhere'
    parsed = _parse_judge_response(raw)
    assert parsed["overall_score"] == 0.6


def test_parse_total_failure():
    raw = "completely unparseable nonsense"
    parsed = _parse_judge_response(raw)
    assert parsed["overall_score"] == 0.0


def test_judge_response_with_mock_model():
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "dimensions": {
            "accuracy": {"score": 0.9, "reason": "准确"},
            "completeness": {"score": 0.8, "reason": "完整"},
        },
        "overall_reasoning": "回答质量良好",
        "overall_score": 0.85,
    })
    mock_model = MagicMock()
    mock_model.invoke.return_value = mock_response

    with patch("deerflow.models.factory.create_chat_model", return_value=mock_model):
        result = judge_response(
            query="测试查询",
            response="测试回答",
            rubric={
                "criteria": "检查准确性和完整性",
                "dimensions": ["accuracy", "completeness"],
            },
            tool_outputs=[{"name": "locate_field_schema", "output": {"status": "found"}}],
        )

    assert result.score == 0.85
    assert result.dimension_scores["accuracy"] == 0.9
    assert result.dimension_scores["completeness"] == 0.8
    assert "良好" in result.reasoning
    mock_model.invoke.assert_called_once()


def test_judge_response_model_error():
    mock_model = MagicMock()
    mock_model.invoke.side_effect = RuntimeError("API timeout")

    with patch("deerflow.models.factory.create_chat_model", return_value=mock_model):
        result = judge_response(
            query="q", response="r",
            rubric={"criteria": "test"},
        )

    assert result.score == 0.0
    assert "API timeout" in result.reasoning
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_judge.py -v`
Expected: ImportError — `evals.framework.judge` does not exist

- [ ] **Step 3: Implement judge.py**

Create `evals/framework/judge.py`:

```python
"""LLM-as-Judge: semantic quality scoring for agent responses."""

import json
import re

from evals.framework.types import JudgeResult

JUDGE_SYSTEM_PROMPT = """你是一个严格公正的 AI 评审员。你的任务是评估 AI Agent 的回答质量。

评分维度：
{dimensions_description}

每个维度打分 0.0 - 1.0：
- 1.0: 完美
- 0.8: 优秀，有微小瑕疵
- 0.6: 合格，基本满足要求
- 0.4: 部分满足，有明显缺失
- 0.2: 较差，大部分不满足
- 0.0: 完全不满足

输出格式（严格 JSON）：
{{
  "dimensions": {{
    "<dimension_name>": {{"score": <float>, "reason": "<简短理由>"}},
    ...
  }},
  "overall_reasoning": "<综合评判理由>",
  "overall_score": <float>
}}"""

JUDGE_USER_PROMPT = """## 用户查询
{query}

## Agent 使用的工具输出（事实来源）
{tool_outputs}

## Agent 回答
{response}

## 评判标准
{criteria}

{reference_section}

请按上述维度评分。"""

DIMENSION_DESCRIPTIONS = {
    "accuracy": "准确性 — 回答中的信息是否正确，是否有编造或错误",
    "groundedness": "基于事实 — 回答中的每个断言是否有工具输出支撑，未编造工具未返回的信息",
    "completeness": "完整性 — 回答是否覆盖了用户问题的所有方面",
    "conciseness": "简洁性 — 回答是否简洁清晰，没有不必要的冗余",
    "helpfulness": "有用性 — 回答是否真正帮助用户解决了问题",
    "safety": "安全性 — 回答是否避免了危险操作建议",
}

DEFAULT_DIMENSIONS = ["accuracy", "groundedness", "completeness", "conciseness"]


def _build_dimensions_description(dimensions: list[str]) -> str:
    return "\n".join(
        f"- {DIMENSION_DESCRIPTIONS.get(d, d)}"
        for d in dimensions
    )


def _parse_judge_response(text: str) -> dict:
    """Parse JSON from LLM response with multiple fallback strategies."""
    # Strategy 1: direct JSON parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: extract from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: regex extract overall_score
    score_match = re.search(r'"overall_score"\s*:\s*([\d.]+)', text)
    if score_match:
        return {
            "overall_score": float(score_match.group(1)),
            "dimensions": {},
            "overall_reasoning": "JSON parse failed, score extracted via regex",
        }

    # Strategy 4: give up
    return {
        "overall_score": 0.0,
        "dimensions": {},
        "overall_reasoning": "Failed to parse judge response",
    }


def _format_tool_outputs(tool_outputs: list[dict] | None) -> str:
    """Format tool outputs for judge prompt. Returns '（无工具输出）' if empty."""
    if not tool_outputs:
        return "（无工具输出）"
    parts = []
    for i, tc in enumerate(tool_outputs, 1):
        name = tc.get("name", "unknown")
        output = tc.get("output", tc.get("result", str(tc)))
        # Truncate long outputs to avoid blowing up judge context
        output_str = str(output)[:2000]
        parts.append(f"### Tool call {i}: {name}\n{output_str}")
    return "\n\n".join(parts)


def judge_response(
    query: str,
    response: str,
    rubric: dict,
    *,
    tool_outputs: list[dict] | None = None,
    model_name: str | None = None,
) -> JudgeResult:
    """Use LLM to score agent response quality against a rubric.

    tool_outputs: Tool call results from capture_run, used for groundedness checking.
    """
    try:
        from deerflow.models.factory import create_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage

        dimensions = rubric.get("dimensions", DEFAULT_DIMENSIONS)
        criteria = rubric["criteria"]
        reference_answer = rubric.get("reference_answer")

        dims_desc = _build_dimensions_description(dimensions)
        reference_section = ""
        if reference_answer:
            reference_section = (
                f"## 参考答案（仅供对照，Agent 不需要完全一致）\n{reference_answer}"
            )

        system_msg = JUDGE_SYSTEM_PROMPT.format(dimensions_description=dims_desc)
        user_msg = JUDGE_USER_PROMPT.format(
            query=query,
            tool_outputs=_format_tool_outputs(tool_outputs),
            response=response,
            criteria=criteria,
            reference_section=reference_section,
        )

        model = create_chat_model(name=model_name)
        result = model.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ])

        parsed = _parse_judge_response(result.content)

        dimension_scores = {}
        for dim_name, dim_data in parsed.get("dimensions", {}).items():
            if isinstance(dim_data, dict):
                dimension_scores[dim_name] = dim_data.get("score", 0.0)
            else:
                dimension_scores[dim_name] = float(dim_data)

        return JudgeResult(
            score=parsed.get("overall_score", 0.0),
            reasoning=parsed.get("overall_reasoning", ""),
            dimension_scores=dimension_scores,
        )

    except Exception as e:
        return JudgeResult(
            score=0.0,
            reasoning=f"Judge error: {e}",
            dimension_scores={},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_judge.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Run all tests for regression**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add evals/framework/judge.py tests/test_eval_judge.py
git commit -m "feat(eval): add LLM-as-Judge semantic scoring in judge.py"
```

---

### Task 5: Judge integration in e2e_eval.py

**Files:**
- Modify: `evals/oncall/e2e_eval.py`

- [ ] **Step 1: Write test for judge check threshold logic**

Append to `tests/test_e2e_eval.py`:

```python
def test_judge_check_threshold():
    """Verify judge_score check uses rubric threshold."""
    from evals.framework.types import JudgeResult

    # Simulate the judge integration logic from evaluate()
    judge_result = JudgeResult(score=0.7, reasoning="ok", dimension_scores={})
    rubric = {"criteria": "test", "pass_threshold": 0.6}
    threshold = rubric.get("pass_threshold", 0.6)
    check = judge_result.score >= threshold
    assert check is True

    rubric_strict = {"criteria": "test", "pass_threshold": 0.8}
    threshold_strict = rubric_strict.get("pass_threshold", 0.6)
    check_strict = judge_result.score >= threshold_strict
    assert check_strict is False
```

- [ ] **Step 2: Run test to verify it passes (pure logic test)**

Run: `pytest tests/test_e2e_eval.py::test_judge_check_threshold -v`
Expected: PASS (this tests the threshold logic, not the integration)

- [ ] **Step 3: Implement judge integration in evaluate()**

In `evals/oncall/e2e_eval.py`, replace the `evaluate` function (lines 184-228):

```python
def evaluate(case: EvalCase, **kwargs) -> EvalResult:
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

    # LLM Judge (opt-in per case via judge_rubric)
    no_judge = kwargs.get("no_judge", False)
    judge_model = kwargs.get("judge_model")
    judge_rubric = case.expected.get("judge_rubric")
    judge_result = None

    if judge_rubric and not no_judge:
        from evals.framework.judge import judge_response

        judge_result = judge_response(
            query=case.input["query"],
            response=run.final_response,
            rubric=judge_rubric,
            tool_outputs=run.tool_calls,  # Pass tool outputs for groundedness checking
            model_name=judge_model,
        )
        threshold = judge_rubric.get("pass_threshold", 0.6)
        checks["judge_score"] = judge_result.score >= threshold

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
        judge=judge_result,
    )
```

- [ ] **Step 4: Run all tests for regression**

Run: `pytest tests/ -v`
Expected: All tests PASS (no e2e tests actually run the full agent)

- [ ] **Step 5: Commit**

```bash
git add evals/oncall/e2e_eval.py tests/test_e2e_eval.py
git commit -m "feat(eval): integrate LLM judge into e2e evaluate()"
```

---

### Task 6: Baseline-Diff (diff.py)

**Files:**
- Create: `evals/framework/diff.py`
- Create: `tests/test_eval_diff.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_eval_diff.py`:

```python
"""Tests for evals/framework/diff.py."""

import json
from pathlib import Path

from evals.framework.diff import CaseDiff, DiffReport, compare_reports, load_baseline
from evals.framework.types import EvalReport, EvalResult


def _make_baseline(cases: list[dict], label: str = "v1") -> dict:
    """Build a baseline dict matching save_report JSON format."""
    return {
        "agent": "oncall",
        "layer": "e2e",
        "timestamp": "2026-04-13T14:30:00+00:00",
        "label": label,
        "summary": {},
        "cases": cases,
    }


def _make_report(results: list[EvalResult]) -> EvalReport:
    report = EvalReport.build(agent="oncall", layer="e2e", results=results)
    return report


def test_compare_improved():
    baseline = _make_baseline([
        {"case_id": "c1", "passed": False, "score": 0.5, "details": {}, "elapsed_ms": 100, "error": None},
    ])
    current = _make_report([
        EvalResult(case_id="c1", passed=True, score=0.9, details={}, actual={}, elapsed_ms=80),
    ])
    diff = compare_reports(baseline, current)
    assert len(diff.cases) == 1
    assert diff.cases[0].status == "improved"
    assert diff.cases[0].delta_score > 0
    assert diff.summary["improved"] == 1


def test_compare_regressed():
    baseline = _make_baseline([
        {"case_id": "c1", "passed": True, "score": 1.0, "details": {}, "elapsed_ms": 100, "error": None},
    ])
    current = _make_report([
        EvalResult(case_id="c1", passed=False, score=0.3, details={}, actual={}, elapsed_ms=80),
    ])
    diff = compare_reports(baseline, current)
    assert diff.cases[0].status == "regressed"
    assert diff.summary["regressed"] == 1


def test_compare_unchanged():
    baseline = _make_baseline([
        {"case_id": "c1", "passed": True, "score": 0.82, "details": {}, "elapsed_ms": 100, "error": None},
    ])
    current = _make_report([
        EvalResult(case_id="c1", passed=True, score=0.84, details={}, actual={}, elapsed_ms=80),
    ])
    diff = compare_reports(baseline, current)
    assert diff.cases[0].status == "unchanged"  # delta 0.02 < 0.10


def test_compare_capability_dip():
    """Capability-tagged cases get 'capability-dip' instead of 'regressed'."""
    baseline = _make_baseline([
        {"case_id": "c1", "passed": True, "score": 0.8, "details": {}, "elapsed_ms": 100, "error": None},
    ])
    current = _make_report([
        EvalResult(case_id="c1", passed=False, score=0.3, details={}, actual={}, elapsed_ms=80),
    ])
    diff = compare_reports(baseline, current, case_tags={"c1": ["capability", "multi-step"]})
    assert diff.cases[0].status == "capability-dip"
    assert diff.summary["capability-dip"] == 1
    assert diff.summary["regressed"] == 0


def test_compare_new_case():
    baseline = _make_baseline([])
    current = _make_report([
        EvalResult(case_id="c_new", passed=True, score=0.9, details={}, actual={}, elapsed_ms=50),
    ])
    diff = compare_reports(baseline, current)
    assert diff.cases[0].status == "new"
    assert diff.cases[0].baseline_score is None
    assert diff.summary["new"] == 1


def test_compare_removed_case():
    baseline = _make_baseline([
        {"case_id": "c_old", "passed": True, "score": 0.8, "details": {}, "elapsed_ms": 100, "error": None},
    ])
    current = _make_report([])
    diff = compare_reports(baseline, current)
    assert diff.cases[0].status == "removed"
    assert diff.cases[0].current_score is None
    assert diff.summary["removed"] == 1


def test_compare_mixed():
    baseline = _make_baseline([
        {"case_id": "c1", "passed": True, "score": 0.8, "details": {}, "elapsed_ms": 100, "error": None},
        {"case_id": "c2", "passed": True, "score": 1.0, "details": {}, "elapsed_ms": 100, "error": None},
        {"case_id": "c3", "passed": True, "score": 0.5, "details": {}, "elapsed_ms": 100, "error": None},
    ])
    current = _make_report([
        EvalResult(case_id="c1", passed=True, score=0.81, details={}, actual={}, elapsed_ms=80),  # unchanged
        EvalResult(case_id="c2", passed=False, score=0.3, details={}, actual={}, elapsed_ms=80),   # regressed
        EvalResult(case_id="c4", passed=True, score=0.9, details={}, actual={}, elapsed_ms=80),    # new
    ])
    diff = compare_reports(baseline, current)
    by_status = {cd.case_id: cd.status for cd in diff.cases}
    assert by_status["c1"] == "unchanged"  # delta +0.01 < 0.10
    assert by_status["c2"] == "regressed"  # delta -0.70 < -0.10
    assert by_status["c3"] == "removed"
    assert by_status["c4"] == "new"
    assert diff.summary == {"improved": 0, "regressed": 1, "capability-dip": 0, "unchanged": 1, "new": 1, "removed": 1}


def test_load_baseline_found(tmp_path, monkeypatch):
    monkeypatch.setattr("evals.framework.diff.REPORT_DIR", tmp_path)
    report_file = tmp_path / "oncall_e2e_20260413_143000_v1.json"
    report_file.write_text(json.dumps({
        "agent": "oncall", "layer": "e2e",
        "timestamp": "2026-04-13T14:30:00+00:00", "label": "v1",
        "summary": {}, "cases": [],
    }))
    baseline = load_baseline("oncall", "e2e", "v1")
    assert baseline["label"] == "v1"


def test_load_baseline_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr("evals.framework.diff.REPORT_DIR", tmp_path)
    import pytest
    with pytest.raises(FileNotFoundError, match="No baseline found"):
        load_baseline("oncall", "e2e", "nonexistent")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_diff.py -v`
Expected: ImportError — `evals.framework.diff` does not exist

- [ ] **Step 3: Implement diff.py**

Create `evals/framework/diff.py`:

```python
"""Baseline-diff: load labeled reports and compare eval runs."""

import json
from dataclasses import dataclass
from pathlib import Path

from evals.framework.report import REPORT_DIR
from evals.framework.types import EvalReport


@dataclass
class CaseDiff:
    case_id: str
    status: str  # "improved" | "regressed" | "capability-dip" | "unchanged" | "new" | "removed"
    baseline_score: float | None
    current_score: float | None
    baseline_passed: bool | None
    current_passed: bool | None
    delta_score: float | None
    details: dict


@dataclass
class DiffReport:
    baseline_label: str
    baseline_timestamp: str
    current_timestamp: str
    cases: list[CaseDiff]
    summary: dict  # {"improved": N, "regressed": N, "capability-dip": N, ...}


def load_baseline(agent: str, layer: str, label: str) -> dict:
    """Load the most recent report matching the given label."""
    pattern = f"{agent}_{layer}_*_{label}.json"
    matches = sorted(REPORT_DIR.glob(pattern), reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No baseline found with label '{label}' "
            f"(searched {REPORT_DIR / pattern})"
        )
    return json.loads(matches[0].read_text(encoding="utf-8"))


def compare_reports(
    baseline: dict,
    current: EvalReport,
    case_tags: dict[str, list[str]] | None = None,
) -> DiffReport:
    """Compare baseline (loaded JSON dict) with current (in-memory EvalReport).

    case_tags: optional mapping of case_id -> tags, used for capability vs regression
    classification. Cases tagged "capability" get "capability-dip" instead of "regressed".
    """
    baseline_by_id = {c["case_id"]: c for c in baseline.get("cases", [])}
    current_by_id = {r.case_id: r for r in current.results}
    case_tags = case_tags or {}

    all_ids = sorted(set(baseline_by_id) | set(current_by_id))
    cases: list[CaseDiff] = []

    for cid in all_ids:
        b = baseline_by_id.get(cid)
        c = current_by_id.get(cid)

        if b and not c:
            status = "removed"
        elif c and not b:
            status = "new"
        else:
            delta = c.score - b["score"]
            is_capability = "capability" in case_tags.get(cid, [])
            if delta > 0.10:
                status = "improved"
            elif delta < -0.10:
                status = "capability-dip" if is_capability else "regressed"
            else:
                status = "unchanged"

        cases.append(CaseDiff(
            case_id=cid,
            status=status,
            baseline_score=b["score"] if b else None,
            current_score=c.score if c else None,
            baseline_passed=b["passed"] if b else None,
            current_passed=c.passed if c else None,
            delta_score=(c.score - b["score"]) if (b and c) else None,
            details={},
        ))

    summary = {}
    for s in ("improved", "regressed", "capability-dip", "unchanged", "new", "removed"):
        summary[s] = sum(1 for cd in cases if cd.status == s)

    return DiffReport(
        baseline_label=baseline.get("label", "?"),
        baseline_timestamp=baseline.get("timestamp", "?"),
        current_timestamp=current.timestamp,
        cases=cases,
        summary=summary,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_diff.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Run all tests for regression**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add evals/framework/diff.py tests/test_eval_diff.py
git commit -m "feat(eval): add baseline-diff comparison in diff.py"
```

---

### Task 7: Report enhancements (report.py)

**Files:**
- Modify: `evals/framework/report.py`

- [ ] **Step 1: Write failing test for save_report with label**

Append to `tests/test_eval_types.py` (or create a small test inline — we test via report.py behavior):

Create `tests/test_eval_report.py`:

```python
"""Tests for evals/framework/report.py enhancements."""

import json
from pathlib import Path

from evals.framework.report import print_diff, save_report
from evals.framework.types import EvalReport, EvalResult, JudgeResult, RunStats
from evals.framework.diff import CaseDiff, DiffReport


def test_save_report_with_label(tmp_path, monkeypatch):
    monkeypatch.setattr("evals.framework.report.REPORT_DIR", tmp_path)
    results = [
        EvalResult(case_id="t1", passed=True, score=1.0, details={}, actual={}, elapsed_ms=10),
    ]
    report = EvalReport.build(agent="oncall", layer="tool", results=results)
    report.label = "v1"
    path = save_report(report)
    assert "_v1.json" in path.name
    data = json.loads(path.read_text())
    assert data["label"] == "v1"


def test_save_report_without_label(tmp_path, monkeypatch):
    monkeypatch.setattr("evals.framework.report.REPORT_DIR", tmp_path)
    results = [
        EvalResult(case_id="t1", passed=True, score=1.0, details={}, actual={}, elapsed_ms=10),
    ]
    report = EvalReport.build(agent="oncall", layer="tool", results=results)
    path = save_report(report)
    assert path.name.count("_") == 3  # oncall_tool_YYYYMMDD_HHMMSS
    data = json.loads(path.read_text())
    assert data["label"] is None


def test_save_report_with_run_stats(tmp_path, monkeypatch):
    monkeypatch.setattr("evals.framework.report.REPORT_DIR", tmp_path)
    rs = RunStats(runs=3, scores=[0.7, 0.8, 0.9], pass_count=3,
                  median_score=0.8, score_std=0.1, pass_rate=1.0,
                  any_passed=True, all_passed=True)
    results = [
        EvalResult(case_id="t1", passed=True, score=0.8, details={}, actual={},
                   elapsed_ms=100, run_stats=rs),
    ]
    report = EvalReport.build(agent="oncall", layer="tool", results=results)
    path = save_report(report)
    data = json.loads(path.read_text())
    assert data["cases"][0]["run_stats"]["runs"] == 3
    assert data["cases"][0]["run_stats"]["median_score"] == 0.8


def test_save_report_with_judge(tmp_path, monkeypatch):
    monkeypatch.setattr("evals.framework.report.REPORT_DIR", tmp_path)
    jr = JudgeResult(score=0.85, reasoning="good", dimension_scores={"accuracy": 0.9})
    results = [
        EvalResult(case_id="t1", passed=True, score=0.85, details={}, actual={},
                   elapsed_ms=50, judge=jr),
    ]
    report = EvalReport.build(agent="oncall", layer="tool", results=results)
    path = save_report(report)
    data = json.loads(path.read_text())
    assert data["cases"][0]["judge"]["score"] == 0.85


def test_print_diff(capsys):
    diff = DiffReport(
        baseline_label="v1",
        baseline_timestamp="2026-04-13T14:30:00+00:00",
        current_timestamp="2026-04-13T15:00:00+00:00",
        cases=[
            CaseDiff(case_id="c1", status="improved", baseline_score=0.5, current_score=0.9,
                     baseline_passed=False, current_passed=True, delta_score=0.4, details={}),
            CaseDiff(case_id="c2", status="regressed", baseline_score=1.0, current_score=0.3,
                     baseline_passed=True, current_passed=False, delta_score=-0.7, details={}),
            CaseDiff(case_id="c3", status="capability-dip", baseline_score=0.6, current_score=0.3,
                     baseline_passed=True, current_passed=False, delta_score=-0.3, details={}),
        ],
        summary={"improved": 1, "regressed": 1, "capability-dip": 1, "unchanged": 0, "new": 0, "removed": 0},
    )
    print_diff(diff)
    out = capsys.readouterr().out
    assert "improved" in out
    assert "regressed" in out
    assert "capability-dip" in out
    assert "REGRESSION DETECTED" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_report.py -v`
Expected: FAIL — `save_report` doesn't produce label in filename or JSON; `print_diff` doesn't exist

- [ ] **Step 3: Implement report.py enhancements**

Replace `evals/framework/report.py`:

```python
"""Eval report: console output + JSON persistence."""

import dataclasses
import json
from datetime import datetime
from pathlib import Path

from evals.framework.types import EvalReport

REPORT_DIR = Path(".deer-flow/eval-reports")

STATUS_ICONS = {
    "improved": "\u2705",
    "regressed": "\U0001f534",
    "capability-dip": "\U0001f7e1",
    "unchanged": "\u2796",
    "new": "\U0001f195",
    "removed": "\u274c",
}


def print_report(report: EvalReport) -> None:
    """Print report to console."""
    print(f"\nEval Report: {report.agent} / {report.layer}")
    print(f"  {report.timestamp}\n")

    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        line = f"  {r.case_id:40s} {status:4s}  {r.score:.2f}  ({r.elapsed_ms:.0f}ms)"
        if r.run_stats:
            rs = r.run_stats
            line += f"  [{rs.pass_count}/{rs.runs} passed, \u03c3={rs.score_std:.2f}]"
            if rs.any_passed and not rs.all_passed:
                line += " \u26a1flaky"
        if r.judge:
            line += f"  judge={r.judge.score:.2f}"
        print(line)
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
    suffix = f"_{report.label}" if report.label else ""
    path = REPORT_DIR / f"{report.agent}_{report.layer}_{ts}{suffix}.json"

    cases = []
    for r in report.results:
        case_data = {
            "case_id": r.case_id,
            "passed": r.passed,
            "score": r.score,
            "details": r.details,
            "elapsed_ms": r.elapsed_ms,
            "error": r.error,
        }
        if r.run_stats:
            case_data["run_stats"] = dataclasses.asdict(r.run_stats)
        if r.judge:
            case_data["judge"] = dataclasses.asdict(r.judge)
        cases.append(case_data)

    data = {
        "agent": report.agent,
        "layer": report.layer,
        "timestamp": report.timestamp,
        "label": report.label,
        "summary": report.summary,
        "cases": cases,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_diff(diff) -> None:
    """Print diff report to console.

    Args:
        diff: DiffReport instance from evals.framework.diff
    """
    print(f"\nDiff Report: baseline vs current")
    print(f"  Baseline: {diff.baseline_label} ({diff.baseline_timestamp})")
    print(f"  Current:  ({diff.current_timestamp})\n")

    print(f"  {'CASE':40s} {'BASE':>5s} \u2192 {'NOW':>5s}  {'DELTA':>6s}  STATUS")
    for cd in diff.cases:
        base_str = f"{cd.baseline_score:.2f}" if cd.baseline_score is not None else "    \u2014"
        now_str = f"{cd.current_score:.2f}" if cd.current_score is not None else "\u2014    "
        delta_str = f"{cd.delta_score:+.2f}" if cd.delta_score is not None else "   \u2014  "
        icon = STATUS_ICONS.get(cd.status, "?")
        print(f"  {cd.case_id:40s} {base_str} \u2192 {now_str}  {delta_str}  {icon} {cd.status}")

    s = diff.summary
    parts = [
        f"{s.get('improved', 0)} improved",
        f"{s.get('regressed', 0)} regressed",
    ]
    if s.get("capability-dip", 0):
        parts.append(f"{s['capability-dip']} capability-dip")
    parts.extend([
        f"{s.get('unchanged', 0)} unchanged",
        f"{s.get('new', 0)} new",
        f"{s.get('removed', 0)} removed",
    ])
    print(f"\n  Summary: {', '.join(parts)}")

    if s.get("regressed", 0) > 0:
        count = s["regressed"]
        print(f"\n  \u26a0\ufe0f  REGRESSION DETECTED: {count} case(s) regressed. Review before deploying.")
    if s.get("capability-dip", 0) > 0:
        count = s["capability-dip"]
        print(f"  \U0001f4a1 {count} capability case(s) dipped — expected variance, monitor trend.")
    print()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_report.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Run all existing tests for regression**

Run: `pytest tests/ -v`
Expected: All tests PASS (existing report tests in test_eval_runner.py still work)

- [ ] **Step 6: Commit**

```bash
git add evals/framework/report.py tests/test_eval_report.py
git commit -m "feat(eval): enhance report with label, run_stats/judge serialization, print_diff"
```

---

### Task 8: CLI integration (run_eval.py)

**Files:**
- Modify: `scripts/run_eval.py`

- [ ] **Step 1: Implement CLI changes**

Replace `scripts/run_eval.py`:

```python
"""Eval runner CLI.

Usage:
    python scripts/run_eval.py tool                                    # Layer 1 mock
    python scripts/run_eval.py tool --live                             # Layer 1 live
    python scripts/run_eval.py tool --case happy_path                  # Single case
    python scripts/run_eval.py tool --tag cold-start                   # Filter by tag
    python scripts/run_eval.py e2e                                     # Layer 3 (full pipeline)
    python scripts/run_eval.py all                                     # All layers
    python scripts/run_eval.py tool --save                             # Save JSON report
    python scripts/run_eval.py e2e --runs 3 --save --label v1         # Multi-run + label
    python scripts/run_eval.py e2e --runs 3 --save --diff v1          # Run + diff vs baseline
    python scripts/run_eval.py e2e --no-judge --diff v1               # Quick regression (no judge)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.framework.report import print_report, save_report
from evals.framework.runner import run_eval


def main():
    parser = argparse.ArgumentParser(
        prog="run_eval", description="Run deer-agents evaluations"
    )
    parser.add_argument(
        "layer",
        choices=["tool", "e2e", "all"],
        help="Which evaluation layer to run",
    )
    parser.add_argument(
        "--agent", default="oncall", help="Agent to evaluate (default: oncall)"
    )
    parser.add_argument("--case", dest="case_id", help="Run specific case by id")
    parser.add_argument("--tag", help="Filter cases by tag")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real MCP/agent instead of mocks",
    )
    parser.add_argument(
        "--json",
        dest="json_only",
        action="store_true",
        help="JSON output only (no console)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save report to .deer-flow/eval-reports/",
    )
    # --- New arguments ---
    parser.add_argument(
        "--label",
        help="Label this report (for later --diff comparison)",
    )
    parser.add_argument(
        "--diff",
        dest="diff_label",
        help="Compare with a labeled baseline report",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Run each case N times and aggregate (default: 1)",
    )
    parser.add_argument(
        "--judge-model",
        help="Model name for LLM Judge (default: config first model)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM Judge scoring globally",
    )
    args = parser.parse_args()

    layers = ["tool", "e2e"] if args.layer == "all" else [args.layer]
    case_ids = [args.case_id] if args.case_id else None
    tags = [args.tag] if args.tag else None

    for layer in layers:
        try:
            report = run_eval(
                layer,
                agent=args.agent,
                case_ids=case_ids,
                tags=tags,
                live=args.live,
                runs=args.runs,
                no_judge=args.no_judge,
                judge_model=args.judge_model,
            )
        except FileNotFoundError as e:
            if args.layer == "all":
                continue
            print(f"  Error: {e}")
            sys.exit(1)

        if args.label:
            report.label = args.label

        if not args.json_only:
            print_report(report)

        if args.save or args.json_only:
            path = save_report(report)
            print(f"  Report saved: {path}")

        if args.diff_label:
            from evals.framework.diff import compare_reports, load_baseline
            from evals.framework.report import print_diff
            from evals.framework.runner import load_cases

            try:
                baseline = load_baseline(args.agent, layer, args.diff_label)
                # Build case_tags for capability vs regression classification
                try:
                    all_cases = load_cases(args.agent, layer)
                    case_tags = {c.id: c.tags for c in all_cases}
                except FileNotFoundError:
                    case_tags = {}
                diff = compare_reports(baseline, report, case_tags=case_tags)
                print_diff(diff)
            except FileNotFoundError as e:
                print(f"  Diff error: {e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help shows new args**

Run: `python scripts/run_eval.py --help`
Expected: Shows `--label`, `--diff`, `--runs`, `--judge-model`, `--no-judge` in help output

- [ ] **Step 3: Run existing tool eval to verify no regression**

Run: `python scripts/run_eval.py tool`
Expected: Same output as before (5 cases, all pass)

- [ ] **Step 4: Test multi-run via CLI**

Run: `python scripts/run_eval.py tool --runs 2`
Expected: 5 cases, each showing `[N/2 passed, σ=X.XX]` suffix

- [ ] **Step 5: Commit**

```bash
git add scripts/run_eval.py
git commit -m "feat(eval): add --label/--diff/--runs/--judge-model/--no-judge CLI args"
```

---

### Task 9: Judge Calibration

**Files:**
- Create: `evals/oncall/judge_calibration.json`
- Create: `tests/test_judge_calibration.py`

- [ ] **Step 1: Create human-annotated calibration data**

Create `evals/oncall/judge_calibration.json` with 5-10 manually scored cases. Each entry has the query, the agent's response, tool outputs, and human-assigned scores per dimension:

```json
[
  {
    "case_id": "cal_overview_accurate",
    "query": "帮我看一下购物>果蔬生鲜>水果类目下的模板字段",
    "response": "该类目下包含 basicInfo（商品品类 CategorySelect）和 merchantInfo（商家名称 AccountName、商家平台商品ID PlatformProductId）等字段组。",
    "tool_outputs": [
      {"name": "locate_field_schema", "output": {"status": "found", "groups": ["basicInfo", "merchantInfo"], "fields": ["CategorySelect", "AccountName", "PlatformProductId"]}}
    ],
    "human_scores": {
      "accuracy": 0.9,
      "groundedness": 0.95,
      "completeness": 0.7,
      "conciseness": 0.9
    }
  },
  {
    "case_id": "cal_hallucinated_field",
    "query": "这个类目有价格字段吗？",
    "response": "是的，该类���下有一个 priceRange（价格区间 PriceField）字段。",
    "tool_outputs": [
      {"name": "locate_field_schema", "output": {"status": "not_found", "message": "No price field in this template"}}
    ],
    "human_scores": {
      "accuracy": 0.1,
      "groundedness": 0.0,
      "completeness": 0.5,
      "conciseness": 0.8
    }
  }
]
```

Fill in 5+ cases covering: accurate responses, hallucinated fields, incomplete answers, verbose answers, and partial matches.

- [ ] **Step 2: Write calibration test**

Create `tests/test_judge_calibration.py`:

```python
"""Judge calibration: verify LLM judge scores correlate with human annotations.

This test requires real LLM calls. Mark with @requires_llm or skip in CI.
Run manually: pytest tests/test_judge_calibration.py -v --run-llm
"""

import json
import os
from pathlib import Path

import pytest

CALIBRATION_FILE = Path(__file__).parent.parent / "evals" / "oncall" / "judge_calibration.json"


@pytest.fixture
def calibration_data():
    if not CALIBRATION_FILE.exists():
        pytest.skip("Calibration data not found")
    return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))


@pytest.mark.skipif(
    not os.environ.get("RUN_LLM_TESTS"),
    reason="Requires RUN_LLM_TESTS=1 (real LLM calls)",
)
def test_judge_correlates_with_human(calibration_data):
    """Judge dimension scores should correlate with human scores (Pearson > 0.7)."""
    from evals.framework.judge import judge_response

    human_scores_flat = []
    judge_scores_flat = []

    for entry in calibration_data:
        result = judge_response(
            query=entry["query"],
            response=entry["response"],
            rubric={
                "criteria": "评估回答的准确性、基于事实程度、完整性和简洁性。",
                "dimensions": list(entry["human_scores"].keys()),
            },
            tool_outputs=entry.get("tool_outputs"),
        )

        for dim, human_score in entry["human_scores"].items():
            if dim in result.dimension_scores:
                human_scores_flat.append(human_score)
                judge_scores_flat.append(result.dimension_scores[dim])

    assert len(human_scores_flat) >= 10, (
        f"Need at least 10 dimension scores for meaningful correlation, got {len(human_scores_flat)}"
    )

    # Pearson correlation
    n = len(human_scores_flat)
    mean_h = sum(human_scores_flat) / n
    mean_j = sum(judge_scores_flat) / n
    cov = sum((h - mean_h) * (j - mean_j) for h, j in zip(human_scores_flat, judge_scores_flat)) / n
    std_h = (sum((h - mean_h) ** 2 for h in human_scores_flat) / n) ** 0.5
    std_j = (sum((j - mean_j) ** 2 for j in judge_scores_flat) / n) ** 0.5

    if std_h == 0 or std_j == 0:
        pytest.fail("Zero variance in scores — calibration data lacks diversity")

    pearson_r = cov / (std_h * std_j)

    print(f"\n  Judge calibration: Pearson r = {pearson_r:.3f} (n={n} dimension scores)")
    print(f"  Human scores:  {[f'{s:.1f}' for s in human_scores_flat]}")
    print(f"  Judge scores:  {[f'{s:.1f}' for s in judge_scores_flat]}")

    assert pearson_r > 0.7, (
        f"Judge scores poorly correlated with human annotations: r={pearson_r:.3f}. "
        f"Adjust judge prompt or dimension descriptions."
    )
```

- [ ] **Step 3: Run calibration test (requires LLM)**

Run: `RUN_LLM_TESTS=1 pytest tests/test_judge_calibration.py -v -s`
Expected: PASS with Pearson r > 0.7. If FAIL, iterate on judge prompt in `judge.py`.

- [ ] **Step 4: Commit**

```bash
git add evals/oncall/judge_calibration.json tests/test_judge_calibration.py
git commit -m "feat(eval): add judge calibration data and correlation test"
```
