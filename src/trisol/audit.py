"""Run the checks and assemble a report.

Kept separate from the CLI so the same entry point serves the terminal, the
webview and any programmatic caller: ``audit(path)`` is the whole API.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from . import __version__
from .checks.base import CheckContext, run_check
from .claude import ClaudeBridge
from .discover import discover
from .findings import AuditReport, CheckResult
from .registry import build_checks

__all__ = ["audit"]


def audit(
    target: Path | str,
    *,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    offline: bool = False,
    deep: bool = False,
    claude_bin: str | None = None,
    model: str | None = None,
    on_progress: Callable[[str, str], None] | None = None,
    on_result: Callable[[CheckResult], None] | None = None,
) -> AuditReport:
    """Audit an agent codebase.

    ``on_progress(stage, name)`` is called before each phase so a caller can show
    live progress; ``on_result(result)`` after each check finishes, so a live
    view can show findings as they arrive instead of all at the end. Both are
    optional and never affect the result.
    """
    started = time.time()
    root = Path(target).resolve()
    if not root.exists():
        raise FileNotFoundError(f"no such path: {root}")

    if on_progress:
        on_progress("discover", str(root))
    project = discover(root)

    bridge = ClaudeBridge.discover(claude_bin, model=model) if not offline else None
    context = CheckContext(project, bridge, offline=offline, deep=deep)

    results = []
    for check in build_checks(only=only, skip=skip):
        if on_progress:
            on_progress("check", check.name)
        result = run_check(check, context)
        results.append(result)
        if on_result:
            on_result(result)

    model_status = (
        {"available": False, "reason": "offline mode requested (--offline)"}
        if offline
        else (bridge.status() if bridge else {"available": False, "reason": "no bridge"})
    )

    return AuditReport(
        target=str(root),
        started_at=started,
        finished_at=time.time(),
        results=results,
        model=model_status,
        trisol_version=__version__,
    )
