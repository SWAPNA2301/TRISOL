"""The scorecard: one measured 0-100 score per area, plus the numbers behind it.

Every score is computed from measurements the checks already recorded, never
from a model's opinion, and each comes with the raw figures it was built from so
a reader can check the arithmetic. An area the target does not have (no routing
table, no corpus) is reported as not applicable rather than scored, because a
100 for "nothing to check" would inflate the overall number.

Areas
-----
safeguards  share of model calls protected by a timeout, error handling, retry
prompts     mean prompt score against the rubric in checks.prompts.RUBRIC
retrieval   recall@k of the retrieval strategy the code actually implements
routing     mean of unambiguous keyword share and having a fallback route
data        share of corpus documents with no defect
"""

from __future__ import annotations

from typing import Any

from .findings import AuditReport, CheckResult

__all__ = ["build_scorecard"]


def _result(report: AuditReport, name: str) -> CheckResult | None:
    for result in report.results:
        if result.name == name and not result.skipped:
            return result
    return None


def _safeguards(report: AuditReport) -> dict[str, Any] | None:
    result = _result(report, "reliability")
    if result is None:
        return None
    calls = int(result.metrics.get("model_calls", 0))
    if not calls:
        return None
    parts = {
        "timeout": calls - int(result.metrics.get("calls_without_timeout", 0)),
        "error handling": calls - int(result.metrics.get("calls_without_error_handling", 0)),
        "retry": calls - int(result.metrics.get("calls_without_retry", 0)),
    }
    score = round(100 * sum(parts.values()) / (3 * calls))
    return {
        "score": score,
        "summary": f"{sum(parts.values())} of {3 * calls} safeguards across {calls} model call(s)",
        "bars": [{"label": k, "value": v, "max": calls} for k, v in parts.items()],
    }


def _prompts(report: AuditReport) -> dict[str, Any] | None:
    result = _result(report, "prompts")
    rubric = result.metrics.get("rubric") if result else None
    if not rubric:
        return None
    score = round(sum(p["score"] for p in rubric) / len(rubric))
    return {
        "score": score,
        "summary": f"mean rubric score over {len(rubric)} prompt(s)",
        "prompts": rubric,
    }


def _retrieval(report: AuditReport) -> dict[str, Any] | None:
    result = _result(report, "benchmark")
    rows = result.metrics.get("results") if result else None
    if not rows or result is None:
        return None
    baseline = result.metrics.get("baseline")
    current = next((r for r in rows if r["name"] == baseline), rows[0])
    best = next((r for r in rows if r["name"] == result.metrics.get("best")), current)
    return {
        "score": round(100 * current["recall_at_k"]),
        "summary": (
            f"{baseline} recalls {current['recall_at_k']:.0%}; "
            f"{best['name']} would recall {best['recall_at_k']:.0%}"
        ),
        "strategies": rows,
        "baseline": baseline,
        "best": result.metrics.get("best"),
    }


def _routing(report: AuditReport) -> dict[str, Any] | None:
    result = _result(report, "routing")
    if result is None or "unambiguous_share" not in result.metrics:
        return None
    share = float(result.metrics["unambiguous_share"])
    fallback = bool(result.metrics.get("has_fallback"))
    return {
        "score": round(100 * (share + (1.0 if fallback else 0.0)) / 2),
        "summary": (
            f"{share:.0%} of keywords route unambiguously; "
            f"{'has' if fallback else 'no'} fallback route"
        ),
        "agents": result.metrics.get("per_agent", []),
        "has_fallback": fallback,
    }


def _data(report: AuditReport) -> dict[str, Any] | None:
    result = _result(report, "data")
    if result is None:
        return None
    docs = int(result.metrics.get("documents", 0))
    if not docs:
        return None
    affected = int(result.metrics.get("documents_with_issues", 0))
    return {
        "score": round(100 * (docs - affected) / docs),
        "summary": f"{docs - affected} of {docs} documents have no defect",
        "bars": [{"label": "documents without defects", "value": docs - affected, "max": docs}],
    }


_AREAS = (
    ("safeguards", "Call safeguards", _safeguards),
    ("prompts", "Prompt quality", _prompts),
    ("retrieval", "Retrieval recall", _retrieval),
    ("routing", "Routing", _routing),
    ("data", "Data health", _data),
)


def build_scorecard(report: AuditReport) -> dict[str, Any]:
    """Scores for every applicable area and their mean as the overall score."""
    areas = []
    for key, title, compute in _AREAS:
        detail = compute(report)
        if detail is not None:
            areas.append({"key": key, "title": title, **detail})
    overall = round(sum(a["score"] for a in areas) / len(areas)) if areas else None
    return {"overall": overall, "areas": areas}
