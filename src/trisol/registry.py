"""The check registry: the single list the CLI, the report and the webview share.

Order matters for the report, not for correctness -- checks are independent, and
are listed cheapest-and-most-certain first so a reader meets provable defects
before judgement calls.
"""

from __future__ import annotations

from .checks.base import Check
from .checks.benchmark import BenchmarkCheck
from .checks.data import DataCheck
from .checks.prompts import PromptCheck
from .checks.reliability import ReliabilityCheck
from .checks.routing import RoutingCheck

__all__ = ["ALL_CHECKS", "build_checks", "check_names"]

ALL_CHECKS: tuple[Check, ...] = (
    ReliabilityCheck(),
    PromptCheck(),
    DataCheck(),
    RoutingCheck(),
    BenchmarkCheck(),
)


def check_names() -> list[str]:
    return [check.name for check in ALL_CHECKS]


def build_checks(only: list[str] | None = None, skip: list[str] | None = None) -> list[Check]:
    """Select checks by name.

    An unknown name raises rather than being ignored: a typo in ``--only`` would
    otherwise silently run everything, and the user would trust a report that
    did not do what they asked.
    """
    known = {check.name: check for check in ALL_CHECKS}

    if only:
        unknown = [name for name in only if name not in known]
        if unknown:
            raise ValueError(
                f"unknown check(s): {', '.join(unknown)}. "
                f"Available: {', '.join(known)}"
            )
        selected = [known[name] for name in only]
    else:
        selected = list(ALL_CHECKS)

    if skip:
        unknown = [name for name in skip if name not in known]
        if unknown:
            raise ValueError(
                f"unknown check(s) to skip: {', '.join(unknown)}. "
                f"Available: {', '.join(known)}"
            )
        selected = [check for check in selected if check.name not in skip]

    return selected
