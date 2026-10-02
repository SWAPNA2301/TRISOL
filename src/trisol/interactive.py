"""What each entry in the interactive menu does.

Kept apart from ``cli.py`` so the command-line wiring stays readable; the
actions reuse the same audit, report, fix and serve code the commands do, so the
menu can never drift into behaving differently from `trisol audit`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import tui
from .claude import ClaudeBridge
from .cli import _run_audit, _theme, cmd_checks, cmd_doctor
from .demo_agents import copy_demos, list_demos
from .findings import AuditReport
from .fixer import apply_fixes
from .report.terminal import render_fix_plan, render_report
from .theme import CYAN, TEXT

__all__ = ["run"]


def run(args: argparse.Namespace) -> int:
    theme = _theme(args)
    status = ClaudeBridge.discover(None).status()

    def audit_args(target: Path | str) -> argparse.Namespace:
        # Online whenever a claude binary was found, so the menu exercises the
        # model-assisted checks exactly like `trisol audit` would.
        return argparse.Namespace(
            target=str(target),
            offline=not status["available"],
            quiet=False,
            json=False,
            no_color=args.no_color,
            no_unicode=args.no_unicode,
        )

    def report_for(target: Path | str) -> AuditReport:
        return _run_audit(audit_args(target))

    def show(report: AuditReport) -> None:
        print(
            render_report(
                report,
                color=False if args.no_color else None,
                unicode=False if args.no_unicode else None,
            )
        )
        print(
            "  "
            + theme.fg(CYAN, "w")
            + theme.muted(" open in browser     ")
            + theme.fg(CYAN, "f")
            + theme.muted(" preview fixes     ")
            + theme.fg(CYAN, "any other key")
            + theme.muted(" back to menu")
        )
        if not sys.stdin.isatty():
            return
        try:
            key = tui.read_key()
        except KeyboardInterrupt:
            return
        if key in ("w", "W"):
            serve(report)
        elif key in ("f", "F"):
            fix(report)

    def serve(report: AuditReport) -> None:
        from .web.server import serve_report  # noqa: PLC0415 - optional

        print()
        print("  " + theme.muted("Ctrl+C stops the server and returns to the menu."))
        serve_report(report, open_browser=True, block=True)

    def fix(report: AuditReport) -> None:
        print(render_fix_plan(report))
        if not report.fixable:
            return
        if not tui.confirm(theme, f"Apply {len(report.fixable)} fix(es) to the files?"):
            print("  " + theme.muted("nothing changed"))
            return
        outcomes = apply_fixes(report.target, report.fixable, dry_run=False)
        done = sum(1 for o in outcomes if o.applied)
        print("  " + theme.ok(f"applied {done} of {len(outcomes)} fix(es)"))
        for outcome in outcomes:
            if not outcome.applied:
                print("  " + theme.warn("skipped  ") + outcome.reason)

    # -- the entries ------------------------------------------------------

    here = Path.cwd()

    def audit_here() -> None:
        # The common case: run `trisol` inside a project and audit it, no prompt.
        show(report_for(here))

    def audit_project() -> None:
        show(report_for(tui.prompt_path(theme, "Folder to audit", ".", must_be="dir")))

    def audit_file() -> None:
        show(report_for(tui.prompt_path(theme, "Python file to audit", "", must_be="file")))

    def try_demo() -> None:
        demos = list_demos()
        index = tui.prompt_choice(theme, "Which demo", [(d.name, d.description) for d in demos])
        if index is not None:
            show(report_for(demos[index].path))

    def dashboard() -> None:
        from .web.server import serve_live  # noqa: PLC0415 - optional

        target = tui.prompt_path(theme, "File or folder to audit", ".")
        print("  " + theme.muted("Ctrl+C stops the server and returns to the menu."))
        serve_live(target, offline=not status["available"], open_browser=True, block=True)

    def fix_issues() -> None:
        fix(report_for(tui.prompt_path(theme, "File or folder to fix", ".")))

    def create_demos() -> None:
        try:
            raw = input(
                "  " + theme.fg(TEXT, "Copy demos into") + theme.muted(" [./trisol-demos]")
                + theme.gold(" > ")
            )
        except EOFError:
            raw = ""
        target = Path(raw.strip().strip('"') or "./trisol-demos")
        for path in copy_demos(target):
            print("  " + theme.ok("created  ") + str(path))
        print()
        print("  " + theme.muted("Now choose ") + theme.fg(TEXT, "Audit a project")
              + theme.muted(" and point it at one of these."))

    def doctor() -> None:
        cmd_doctor(argparse.Namespace(claude_bin=None, no_color=args.no_color,
                                      no_unicode=args.no_unicode))

    def checks() -> None:
        cmd_checks(args)

    items = [
        tui.MenuItem("here", "Audit this folder", _shorten(str(here), 44), audit_here),
        tui.MenuItem("audit", "Audit another folder", "choose any folder to scan",
                     audit_project),
        tui.MenuItem("file", "Audit a single file", "check one agent module", audit_file),
        tui.MenuItem("demo", "Try a demo agent", "bundled agents with planted defects",
                     try_demo),
        tui.MenuItem("web", "Live web dashboard", "watch the audit run in your browser",
                     dashboard),
        tui.MenuItem("fix", "Fix issues", "preview, then apply, the safe fixes", fix_issues),
        tui.MenuItem("copy", "Create demo folders", "copy the demos here to experiment on",
                     create_demos),
        tui.MenuItem("doctor", "Environment check", "Python, Claude CLI, checks", doctor),
        tui.MenuItem("checks", "List checks", "everything Trisol looks for", checks),
        tui.MenuItem("quit", "Quit", "", None),
    ]
    return tui.run_menu(theme, items, status)


def _shorten(path: str, limit: int) -> str:
    """Keep the end of a long path, which is the part that identifies it."""
    return path if len(path) <= limit else "..." + path[-(limit - 3):]
