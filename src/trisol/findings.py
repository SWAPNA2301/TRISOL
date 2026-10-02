"""What a check produces, and how a run is summarised.

A :class:`Finding` is one problem in one place. It carries enough to act on --
file and line, why it matters, and where possible a concrete patch -- because a
report that only says "prompt quality is low" cannot be verified or fixed.

``Severity`` deliberately has four levels rather than three: the gap between
"this will break in production" (HIGH) and "this is worth tidying" (LOW) is wide
enough that squeezing them together makes a report either alarmist or ignorable.
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "AuditReport",
    "CheckResult",
    "Confidence",
    "Finding",
    "Patch",
    "Severity",
]


class Severity(enum.Enum):
    CRITICAL = "critical"  # the agent is broken or unsafe as written
    HIGH = "high"  # will fail on realistic input
    MEDIUM = "medium"  # degrades quality or cost
    LOW = "low"  # worth tidying

    @property
    def rank(self) -> int:
        return {"critical": 0, "high": 1, "medium": 2, "low": 3}[self.value]


class Confidence(enum.Enum):
    """How sure the check is.

    Separated from severity because the two are independent: a static parse can
    be CERTAIN about a LOW-severity nit, while a model's judgement about prompt
    quality may be a LIKELY read on a HIGH-severity problem. Collapsing them
    would hide which findings need a human look.
    """

    CERTAIN = "certain"  # proved by parsing or by running it
    LIKELY = "likely"  # strong signal, some interpretation
    POSSIBLE = "possible"  # a judgement call, review before acting


@dataclass
class Patch:
    """A concrete edit that resolves a finding.

    ``old`` must appear verbatim in the file so applying it is an exact,
    reviewable replacement rather than a regenerated file. A patch with an
    ``old`` that no longer matches is refused, not force-applied.
    """

    file: str
    old: str
    new: str
    explanation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Finding:
    check: str
    title: str
    detail: str
    severity: Severity
    confidence: Confidence = Confidence.LIKELY
    file: str | None = None
    line: int | None = None
    # What goes wrong, concretely. A finding without this tends to be a style
    # opinion dressed as a defect, so checks are expected to fill it in.
    impact: str = ""
    suggestion: str = ""
    patch: Patch | None = None
    # Set by the model layer when a finding came from or was reviewed by Claude.
    source: str = "static"
    evidence: list[str] = field(default_factory=list)

    @property
    def fixable(self) -> bool:
        return self.patch is not None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        data["confidence"] = self.confidence.value
        data["fixable"] = self.fixable
        return data


@dataclass
class CheckResult:
    """One check's outcome. A check that could not run is distinct from one that
    ran and found nothing -- conflating them silently turns a broken check into
    a clean bill of health."""

    name: str
    title: str
    findings: list[Finding] = field(default_factory=list)
    duration_ms: int = 0
    skipped: bool = False
    skip_reason: str = ""
    # Free-form numbers the check measured (latency, scores, counts), surfaced
    # in the report's benchmark section.
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.skipped and not self.findings

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "findings": [f.as_dict() for f in self.findings],
            "duration_ms": self.duration_ms,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "metrics": self.metrics,
            "ok": self.ok,
        }


@dataclass
class AuditReport:
    target: str
    started_at: float
    finished_at: float
    results: list[CheckResult] = field(default_factory=list)
    # Which Claude backend was used, if any, so a reader knows whether the
    # model-assisted checks actually ran.
    model: dict[str, Any] = field(default_factory=dict)
    trisol_version: str = ""

    @property
    def findings(self) -> list[Finding]:
        """Every finding, worst first, then by confidence, then by location --
        a total order, so two runs over the same code report in the same
        sequence and a diff of two reports is readable."""
        out = [f for r in self.results for f in r.findings]
        out.sort(
            key=lambda f: (
                f.severity.rank,
                {"certain": 0, "likely": 1, "possible": 2}[f.confidence.value],
                f.file or "",
                f.line or 0,
                f.title,
            )
        )
        return out

    @property
    def duration_ms(self) -> int:
        return int((self.finished_at - self.started_at) * 1000)

    def count(self, severity: Severity) -> int:
        return sum(1 for f in self.findings if f.severity is severity)

    @property
    def fixable(self) -> list[Finding]:
        return [f for f in self.findings if f.fixable]

    @property
    def exit_code(self) -> int:
        """0 clean, 1 findings that need attention, 2 nothing could be checked.

        CRITICAL/HIGH gate CI; MEDIUM/LOW report without failing a build, so
        teams can adopt the tool without it immediately blocking every merge.
        """
        if self._nothing_examined:
            return 2
        if self.count(Severity.CRITICAL) or self.count(Severity.HIGH):
            return 1
        return 0

    @property
    def _nothing_examined(self) -> bool:
        """True when the audit had nothing to look at.

        Not simply "every check skipped": a cheap check can run clean over an
        empty directory and make an audit of nothing look like a pass. The honest
        test is whether any check that ran actually examined something -- which
        it reports through its metrics.
        """
        if not self.results:
            return True
        for result in self.results:
            if result.skipped:
                continue
            if any(
                isinstance(value, (int, float)) and value > 0
                for value in result.metrics.values()
            ):
                return False
            if result.findings:
                return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "trisol_version": self.trisol_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "results": [r.as_dict() for r in self.results],
            # Imported here: scorecard reads AuditReport, so a module-level
            # import would be circular.
            "scorecard": _scorecard(self),
            "summary": {
                "total": len(self.findings),
                "critical": self.count(Severity.CRITICAL),
                "high": self.count(Severity.HIGH),
                "medium": self.count(Severity.MEDIUM),
                "low": self.count(Severity.LOW),
                "fixable": len(self.fixable),
                "checks_run": sum(1 for r in self.results if not r.skipped),
                "checks_skipped": sum(1 for r in self.results if r.skipped),
            },
        }

    def write_json(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2, default=str), encoding="utf-8")
        return target


def _scorecard(report: AuditReport) -> dict[str, Any]:
    from .scorecard import build_scorecard  # noqa: PLC0415 - avoids a cycle

    return build_scorecard(report)
