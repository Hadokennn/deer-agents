# 评测增强：Baseline-Diff / LLM-as-Judge / 多次运行统计 — 设计 Spec

> 为 deer-agents 评测框架增加三项关键能力，使其能够回答
> "改了 prompt/skill/MCP 之后，agent 表现是变好了还是变差了？"

## 问题

当前评测体系可以跑单次评分、存 JSON 快照，但无法：

1. **对比两次评测结果** — 没有 baseline 标签机制，没有 diff 报告
2. **评判回答语义质量** — 输出检查只有关键词/长度，prompt 优化效果无法量化
3. **消除随机波动** — E2E 涉及真实 LLM 调用，单次结果不可信

## 目标

| 能力 | 用户体验 |
|------|----------|
| Baseline-Diff | `run_eval e2e --save --label v1` → 改代码 → `run_eval e2e --diff v1` → 看到逐 case 对比 |
| LLM-as-Judge | E2E case 中声明 `judge_rubric`，自动用 LLM 评分回答质量 (0-1) |
| 多次运行统计 | `run_eval e2e --runs 3` → 每条 case 跑 3 次，取中位数，报告置信度 |

## 设计总览

```
┌─────────────────────────────────────────────────────────┐
│                     run_eval CLI                         │
│  新增参数:                                               │
│    --label <name>    给本次报告打标签                     │
│    --diff <label>    与指定 baseline 对比                 │
│    --runs <N>        每条 case 运行 N 次 (默认 1)         │
│    --judge-model <m> 指定评判 LLM (默认 config 首个模型)  │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
   ┌────────────┐ ┌──────────┐ ┌──────────────┐
   │ Multi-Run  │ │ LLM      │ │ Baseline     │
   │ Runner     │ │ Judge    │ │ Diff         │
   │ (runner.py)│ │(judge.py)│ │ (diff.py)    │
   └────────────┘ └──────────┘ └──────────────┘
          │          │                │
          ▼          ▼                ▼
   ┌──────────────────────────────────────────┐
   │           EvalReport + EvalResult         │
   │    (types.py — 扩展 stats / judge 字段)   │
   └──────────────────────────────────────────┘
          │                           │
          ▼                           ▼
   ┌─────────────┐           ┌──────────────┐
   │ report.py   │           │ 报告存储      │
   │ (增强打印)   │           │ (带 label)    │
   └─────────────┘           └──────────────┘
```

---

## Feature 1: 多次运行取统计值

### 动机

E2E 评测调用真实 LLM，同一 case 跑两次可能一次 PASS 一次 FAIL。单次结果无法区分"真的回归"和"随机波动"。

### 设计

#### Runner 层变更 (`evals/framework/runner.py`)

`run_eval` 新增 `runs: int = 1` 参数。当 `runs > 1` 时，每条 case 执行 N 次，收集所有 `EvalResult`，然后聚合为一个 `EvalResult`（取统计值）。

```python
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
```

#### 聚合逻辑

对同一 case 的 N 次结果：

| 指标 | 聚合方式 |
|------|----------|
| `score` | 取 **中位数**（比平均值更抗极端值） |
| `passed` | N 次中 **多数通过** 则为 True（> 50%） |
| `elapsed_ms` | 取中位数 |
| `details` | 取最后一次运行的 details（代表性） |
| `actual` | 取最后一次运行的 actual |
| `error` | 如有任何一次出错，收集所有 error |

#### 新增统计字段 (`EvalResult` 扩展)

```python
@dataclass
class RunStats:
    runs: int
    scores: list[float]
    pass_count: int
    median_score: float
    score_std: float
    pass_rate: float

@dataclass
class EvalResult:
    case_id: str
    passed: bool
    score: float          # 聚合后的中位数
    details: dict
    actual: dict
    elapsed_ms: float
    error: str | None = None
    run_stats: RunStats | None = None  # 新增, runs=1 时为 None
```

#### 聚合函数

新增 `evals/framework/stats.py`：

```python
import statistics
from evals.framework.types import EvalResult, RunStats

def aggregate_results(results: list[EvalResult]) -> EvalResult:
    """将同一 case 的多次 EvalResult 聚合为一个。"""
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
        ),
    )
```

#### Report 打印增强

当 `run_stats` 存在时，追加显示：

```
  e2e_overview_request  PASS  0.83  (1520ms)  [3/3 passed, σ=0.06]
```

### 约束

- `runs > 1` 仅对 `e2e` 和 `tool --live` 有意义；mock 模式结果确定性，默认 `runs=1`
- 多次运行 **串行执行**（同一 case 内），避免并发竞争状态（如 MCP 连接池）
- 后续可考虑进程级并行（跨 case），但不在本次 scope

---

## Feature 2: LLM-as-Judge 语义评分

### 动机

prompt/skill 变更主要影响回答的语义质量（准确性、完整性、简洁性），当前的结构性检查（关键词、长度）无法捕捉这些维度。

### 设计

#### 新增模块 `evals/framework/judge.py`

```python
from dataclasses import dataclass

@dataclass
class JudgeResult:
    score: float            # 0.0 - 1.0
    reasoning: str          # LLM 的评判理由
    dimension_scores: dict  # {"accuracy": 0.9, "completeness": 0.8, ...}
```

#### Rubric 定义（在 case 的 `expected` 中）

E2E case 中新增可选的 `judge_rubric` 字段：

```json
{
  "id": "e2e_overview_request",
  "layer": "e2e",
  "input": {
    "query": "帮我看一下购物>果蔬生鲜>水果类目下，团购商品的模板有哪些字段？"
  },
  "expected": {
    "process_rules": [...],
    "output_checks": {...},
    "judge_rubric": {
      "dimensions": ["accuracy", "completeness", "conciseness"],
      "criteria": "回答应准确列出该类目模板下的字段列表，包含字段名称和类型信息。不应编造不存在的字段。",
      "reference_answer": "该类目下包含 basicInfo（商品品类 CategorySelect）和 merchantInfo（商家名称 AccountName、商家平台商品ID PlatformProductId）等字段组。",
      "pass_threshold": 0.6
    }
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `dimensions` | `list[str]` | 否 | 评分维度，默认 `["accuracy", "completeness", "conciseness"]` |
| `criteria` | `str` | 是 | 评判标准的自然语言描述 |
| `reference_answer` | `str` | 否 | 参考答案（给 LLM judge 用作对照） |
| `pass_threshold` | `float` | 否 | 通过阈值，默认 0.6 |

#### Prompt 模板

```python
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

## Agent 回答
{response}

## 评判标准
{criteria}

{reference_section}

请按上述维度评分。"""
```

其中 `{reference_section}` 在 `reference_answer` 存在时注入：

```
## 参考答案（仅供对照，Agent 不需要完全一致）
{reference_answer}
```

#### 维度描述映射

```python
DIMENSION_DESCRIPTIONS = {
    "accuracy": "准确性 — 回答中的信息是否正确，是否有编造或错误",
    "completeness": "完整性 — 回答是否覆盖了用户问题的所有方面",
    "conciseness": "简洁性 — 回答是否简洁清晰，没有不必要的冗余",
    "helpfulness": "有用性 — 回答是否真正帮助用户解决了问题",
    "safety": "安全性 — 回答是否避免了危险操作建议",
}
```

#### 调用方式

```python
def judge_response(
    query: str,
    response: str,
    rubric: dict,
    *,
    model_name: str | None = None,
) -> JudgeResult:
    """用 LLM 评判 agent 回答质量。"""
```

使用项目现有的 `create_chat_model` 工厂函数创建 LLM 实例：

```python
from deerflow.models.factory import create_chat_model

def judge_response(query, response, rubric, *, model_name=None):
    model = create_chat_model(name=model_name)
    # ... 构建 prompt, 调用 model, 解析 JSON 响应
```

这样复用了项目已有的模型配置体系（`config.yaml` 中的 models 列表），无需额外配置。

#### 与 E2E Eval 集成

在 `evals/oncall/e2e_eval.py` 的 `evaluate` 函数中，当 case 有 `judge_rubric` 时，调用 judge：

```python
def evaluate(case: EvalCase, **kwargs) -> EvalResult:
    # ... 现有逻辑: capture_run, process_checks, output_checks ...

    # LLM Judge (如果 case 声明了 rubric)
    judge_rubric = case.expected.get("judge_rubric")
    judge_result = None
    if judge_rubric:
        from evals.framework.judge import judge_response
        judge_result = judge_response(
            query=case.input["query"],
            response=run.final_response,
            rubric=judge_rubric,
            model_name=kwargs.get("judge_model"),
        )
        # 将 judge 分数纳入 checks
        threshold = judge_rubric.get("pass_threshold", 0.6)
        checks["judge_score"] = judge_result.score >= threshold

    # ... 计算最终 passed/score ...
```

#### EvalResult 扩展

```python
@dataclass
class EvalResult:
    # ... 现有字段 ...
    judge: JudgeResult | None = None  # 新增
```

#### 容错

- LLM judge 调用失败时，**不阻塞评测**，`judge_score` check 标记为 False，error 中记录原因
- JSON 解析失败时，尝试 regex 提取 `overall_score`，仍失败则 score = 0.0

#### 成本控制

- 默认不启用 judge；只有 case 中声明了 `judge_rubric` 才触发
- CLI 增加 `--no-judge` 参数，可全局跳过 judge（快速回归时使用）
- judge model 可通过 `--judge-model` 指定，推荐用较便宜的模型（如 GPT-4o-mini）

---

## Feature 3: Baseline-Diff 对比报告

### 动机

跑两次评测后，需要自动对比哪些 case 变好/持平/变差，而不是人工翻 JSON 文件。

### 设计

#### 标签化报告存储

现有 `save_report` 按时间戳命名。扩展为支持 label：

```
.deer-flow/eval-reports/
├── oncall_e2e_20260413_143000.json          # 无 label
├── oncall_e2e_20260413_143000_v1.json       # 有 label "v1"
└── oncall_e2e_20260413_150000_v2.json       # 有 label "v2"
```

`EvalReport` 新增 `label` 字段：

```python
@dataclass
class EvalReport:
    agent: str
    layer: str
    timestamp: str
    results: list[EvalResult]
    summary: dict
    label: str | None = None  # 新增
```

`save_report` 增强：

```python
def save_report(report: EvalReport) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.fromisoformat(report.timestamp).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{report.label}" if report.label else ""
    path = REPORT_DIR / f"{report.agent}_{report.layer}_{ts}{suffix}.json"
    # ... 写 JSON（含 label 字段）...
```

#### 加载 baseline

新增 `evals/framework/diff.py`：

```python
def load_baseline(agent: str, layer: str, label: str) -> dict:
    """按 label 查找最新的匹配报告。"""
    pattern = f"{agent}_{layer}_*_{label}.json"
    matches = sorted(REPORT_DIR.glob(pattern), reverse=True)
    if not matches:
        raise FileNotFoundError(f"No baseline found with label '{label}'")
    return json.loads(matches[0].read_text(encoding="utf-8"))
```

按文件名排序取最新，确保同一 label 多次保存时取最后一次。

#### Diff 报告

```python
@dataclass
class CaseDiff:
    case_id: str
    status: str           # "improved" | "regressed" | "unchanged" | "new" | "removed"
    baseline_score: float | None
    current_score: float | None
    baseline_passed: bool | None
    current_passed: bool | None
    delta_score: float | None
    details: dict         # 各 check 的逐项对比


@dataclass
class DiffReport:
    baseline_label: str
    baseline_timestamp: str
    current_timestamp: str
    cases: list[CaseDiff]
    summary: dict         # {"improved": 2, "regressed": 1, "unchanged": 5, "new": 0, "removed": 0}
```

#### 对比逻辑

```python
def compare_reports(baseline: dict, current: EvalReport) -> DiffReport:
    """逐 case 对比两次评测结果。"""
    baseline_by_id = {c["case_id"]: c for c in baseline["cases"]}
    current_by_id = {r.case_id: r for r in current.results}

    all_ids = set(baseline_by_id) | set(current_by_id)
    cases = []

    for cid in sorted(all_ids):
        b = baseline_by_id.get(cid)
        c = current_by_id.get(cid)

        if b and not c:
            status = "removed"
        elif c and not b:
            status = "new"
        else:
            delta = c.score - b["score"]
            if delta > 0.05:
                status = "improved"
            elif delta < -0.05:
                status = "regressed"
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
            details={},  # 可扩展为逐 check 对比
        ))

    summary = {}
    for s in ["improved", "regressed", "unchanged", "new", "removed"]:
        summary[s] = sum(1 for cd in cases if cd.status == s)

    return DiffReport(
        baseline_label=baseline.get("label", "?"),
        baseline_timestamp=baseline["timestamp"],
        current_timestamp=current.timestamp,
        cases=cases,
        summary=summary,
    )
```

#### Delta 阈值

- `delta > +0.05` → improved（避免噪声被标记为改善）
- `delta < -0.05` → regressed
- 其余 → unchanged

阈值 0.05 是硬编码默认值，后续可配置。

#### Diff 控制台输出

新增 `print_diff` 函数：

```
Diff Report: oncall / e2e
  Baseline: v1 (2026-04-13T14:30:00Z)
  Current:       (2026-04-13T15:00:00Z)

  CASE                                     BASE → NOW   DELTA  STATUS
  e2e_overview_request                     0.75 → 1.00  +0.25  ✅ improved
  e2e_field_query                          1.00 → 0.33  -0.67  🔴 regressed
  e2e_not_found                            0.80 → 0.83  +0.03  ➖ unchanged
  e2e_new_case                                — → 0.90     —   🆕 new

  Summary: 1 improved, 1 regressed, 1 unchanged, 1 new, 0 removed
```

有回归时（`regressed > 0`），打印醒目警告：

```
  ⚠️  REGRESSION DETECTED: 1 case(s) regressed. Review before deploying.
```

---

## CLI 变更

### `scripts/run_eval.py` 新增参数

```python
parser.add_argument("--label", help="给本次报告打标签 (用于后续 --diff)")
parser.add_argument("--diff", dest="diff_label", help="与指定 label 的 baseline 对比")
parser.add_argument("--runs", type=int, default=1, help="每条 case 运行次数 (默认 1)")
parser.add_argument("--judge-model", help="LLM Judge 使用的模型名 (默认 config 首个)")
parser.add_argument("--no-judge", action="store_true", help="跳过 LLM Judge 评分")
```

### 典型工作流

```bash
# 1. 改之前：跑 baseline
python scripts/run_eval.py e2e --runs 3 --save --label baseline-v1

# 2. 修改 prompt / skill / MCP

# 3. 改之后：跑评测并对比
python scripts/run_eval.py e2e --runs 3 --save --label after-v1 --diff baseline-v1

# 4. 只想快速回归（不要 judge，不要多次运行）
python scripts/run_eval.py e2e --no-judge --diff baseline-v1

# 5. 只跑特定 tag
python scripts/run_eval.py e2e --tag cold-start --runs 3 --diff baseline-v1
```

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `evals/framework/types.py` | 修改 | 新增 `RunStats`, `JudgeResult`, `EvalResult.run_stats`, `EvalResult.judge`, `EvalReport.label` |
| `evals/framework/runner.py` | 修改 | `run_eval` 新增 `runs` 参数，多次运行 + 聚合 |
| `evals/framework/stats.py` | **新增** | `aggregate_results` 聚合函数 |
| `evals/framework/judge.py` | **新增** | `judge_response` LLM 评判函数 |
| `evals/framework/diff.py` | **新增** | `load_baseline`, `compare_reports`, `CaseDiff`, `DiffReport` |
| `evals/framework/report.py` | 修改 | `save_report` 支持 label；新增 `print_diff`；打印增强（显示 run_stats, judge） |
| `evals/oncall/e2e_eval.py` | 修改 | `evaluate` 集成 judge 调用 |
| `scripts/run_eval.py` | 修改 | 新增 CLI 参数 |

### 不变更的文件

- `evals/oncall/tool_eval.py` — Tool 层评估逻辑不变
- `evals/oncall/process_eval.py` — Process 层评估逻辑不变
- `evals/oncall/fixtures.py` — Mock 数据不变

---

## 依赖

| 依赖 | 来源 | 用途 |
|------|------|------|
| `statistics` (stdlib) | Python 标准库 | 中位数、标准差计算 |
| `deerflow.models.factory.create_chat_model` | 项目现有 | LLM Judge 创建模型实例 |
| `langchain_core.messages` | 项目现有依赖 | 构建 judge prompt messages |

**无新增外部依赖。**

---

## 测试计划

### 单元测试

| 测试文件 | 覆盖 |
|----------|------|
| `tests/test_eval_stats.py` | `aggregate_results` — 多结果聚合、中位数、pass 判定 |
| `tests/test_eval_judge.py` | `judge_response` — prompt 构建、JSON 解析、容错 |
| `tests/test_eval_diff.py` | `compare_reports` — improved/regressed/unchanged/new/removed 判定 |
| `tests/test_eval_types.py` (扩展) | 新增字段的序列化/构造 |
| `tests/test_eval_runner.py` (扩展) | `runs > 1` 的聚合集成 |

### 集成测试

- `python scripts/run_eval.py tool --save --label test-baseline`
- `python scripts/run_eval.py tool --save --diff test-baseline`
- `python scripts/run_eval.py e2e --runs 3 --save --label e2e-test` (需要真实 LLM)

---

## 实现顺序

| 阶段 | 内容 | 依赖 |
|------|------|------|
| Phase 1 | `types.py` 扩展（新增字段 + dataclass） | 无 |
| Phase 2 | `stats.py` + `runner.py` 多次运行 | Phase 1 |
| Phase 3 | `judge.py` + `e2e_eval.py` 集成 | Phase 1 |
| Phase 4 | `diff.py` + `report.py` 增强 | Phase 1 |
| Phase 5 | `run_eval.py` CLI 集成 | Phase 2-4 |
| Phase 6 | 测试用例 | Phase 2-5 |

Phase 2、3、4 之间无依赖，可并行实现。
