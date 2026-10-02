"""Trisol -- agent lifecycle verification.

Point it at an AI agent codebase and it tests what the code actually does:
reliability of its model calls, the quality of its prompts, the health of its
knowledge base, the soundness of its routing, and how its retrieval strategy
scores against stronger ones on its own data. Claude supplies the judgement
calls; everything else is proved by parsing or by measurement.

    from trisol import audit
    report = audit("path/to/agent")
    print(report.exit_code, len(report.findings))
"""

from __future__ import annotations

__version__ = "0.2.0"

from .audit import audit
from .findings import (
    AuditReport,
    CheckResult,
    Confidence,
    Finding,
    Patch,
    Severity,
)

__all__ = [
    "AuditReport",
    "CheckResult",
    "Confidence",
    "Finding",
    "Patch",
    "Severity",
    "__version__",
    "audit",
]
