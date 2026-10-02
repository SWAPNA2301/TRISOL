"""The check protocol.

A check is a small object with a name, a title and a ``run`` that returns a
:class:`~trisol.findings.CheckResult`. Keeping them uniform means the CLI, the
report and the webview never need to know what any individual check does, and a
new check is one file plus one registry entry.

A check must never raise. An exception inside one would abandon the rest of the
audit, so :func:`run_check` converts a crash into a skipped result with the
reason attached -- a broken check is visible, not silent, and not fatal.
"""

from __future__ import annotations

import time
import traceback
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..findings import CheckResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..claude import ClaudeBridge
    from ..discover import AgentProject

__all__ = ["Check", "CheckContext", "run_check"]


class CheckContext:
    """Everything a check is allowed to read.

    Passing one object rather than loose arguments means adding a capability
    later (a config file, a budget) does not change every check's signature.
    """

    def __init__(
        self,
        project: AgentProject,
        claude: ClaudeBridge | None = None,
        *,
        offline: bool = False,
        deep: bool = False,
    ):
        self.project = project
        self.claude = claude
        # Explicitly asked to stay local: checks must not call out even if a
        # usable binary was found.
        self.offline = offline
        # Opt in to the slower, more thorough passes (more model calls, running
        # benchmarks with more trials).
        self.deep = deep

    @property
    def model_available(self) -> bool:
        return bool(self.claude and self.claude.available and not self.offline)


@runtime_checkable
class Check(Protocol):
    name: str
    title: str
    #: True when the check cannot say anything without a model, so the CLI can
    #: report it as skipped up front rather than running it to no effect.
    needs_model: bool

    def run(self, context: CheckContext) -> CheckResult: ...


def run_check(check: Check, context: CheckContext) -> CheckResult:
    """Run one check, timing it and containing any failure."""
    started = time.perf_counter()
    try:
        result = check.run(context)
    except Exception as exc:  # a broken check must not abandon the audit
        return CheckResult(
            name=check.name,
            title=check.title,
            skipped=True,
            skip_reason=(
                f"{check.name} raised {exc.__class__.__name__}: {exc}. "
                f"{traceback.format_exc().strip().splitlines()[-1]}"
            ),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
    # Checks are not required to time themselves; fill it in if they didn't.
    if not result.duration_ms:
        result.duration_ms = int((time.perf_counter() - started) * 1000)
    return result
