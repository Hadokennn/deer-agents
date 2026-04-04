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
class EvalResult:
    case_id: str
    passed: bool
    score: float  # 0.0 - 1.0
    details: dict  # per-check pass/fail
    actual: dict  # raw system output
    elapsed_ms: float
    error: str | None = None


@dataclass
class EvalReport:
    agent: str
    layer: str
    timestamp: str
    results: list[EvalResult]
    summary: dict

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
