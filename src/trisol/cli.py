"""Trisol's command line.

Wiring only -- no analysis lives here. Argparse rather than a framework so the
package has no runtime dependencies at all: a tool meant to be dropped into
someone else's CI should not drag a dependency tree in with it.

Commands
--------
``audit``  run the checks and report
``fix``    apply the fixes the audit found
``serve``  open the report in a browser
``demo``   run or copy the bundled demo agents
``checks`` list what the checks are
``doctor`` is the environment set up

With no command, an interactive terminal gets a menu; anything else gets help.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .audit import audit as run_audit
from .banner import render as render_banner
from .claude import ClaudeBridge
from .demo_agents import copy_demos, list_demos
from .findings import Severity
from .fixer import apply_fixes
from .registry import ALL_CHECKS
from .report.terminal import render_fix_plan, render_report
from .theme import CYAN, TEXT, Theme, detect

__all__ = ["build_parser", "main"]

_REPORT_NAME = "report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trisol",
        description="Agent lifecycle verification: test, benchmark and fix AI agent codebases.",
    )
    parser.add_argument("--version", action="version", version=f"trisol {__version__}")
    # default=None for the same reason as the subparser copies below: the two
    # parses are merged by _merge_global_flags, and False would be
    # indistinguishable from "not given".
    parser.add_argument(
        "--no-color", action="store_const", const=True, default=None,
        help="never emit ANSI colour",
    )
    parser.add_argument(
        "--no-unicode", action="store_const", const=True, default=None,
        help="ASCII output only",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_const", const=True, default=None,
        help="suppress the banner",
    )

    # Repeated on every subcommand so `trisol audit . --no-color` works as well
    # as `trisol --no-color audit .`; argparse otherwise only accepts the latter.
    # default=None, not False: argparse applies subparser defaults after the
    # top-level parse, so a False default here would overwrite a flag the user
    # passed *before* the subcommand. None means "not given here", and
    # _merge_global_flags folds the two parses together.
    global_flags = argparse.ArgumentParser(add_help=False)
    global_flags.add_argument(
        "--no-color", action="store_const", const=True, default=None, help=argparse.SUPPRESS
    )
    global_flags.add_argument(
        "--no-unicode", action="store_const", const=True, default=None, help=argparse.SUPPRESS
    )
    global_flags.add_argument(
        "--quiet", "-q", action="store_const", const=True, default=None,
        help=argparse.SUPPRESS,
    )

    subparsers = parser.add_subparsers(dest="command")

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("target", nargs="?", default=".", help="file or directory to audit")
        sub.add_argument("--only", help="comma-separated checks to run")
        sub.add_argument("--skip", help="comma-separated checks to skip")
        sub.add_argument(
            "--offline", action="store_true", help="never call Claude; static checks only"
        )
        sub.add_argument(
            "--deep", action="store_true", help="slower, more thorough passes"
        )
        sub.add_argument("--claude-bin", help="path to the claude executable")
        sub.add_argument("--model", help="model to pass to claude")

    audit_parser = subparsers.add_parser(
        "audit", help="run the checks and report", parents=[global_flags]
    )
    add_common(audit_parser)
    audit_parser.add_argument(
        "--json", action="store_true", help="machine-readable output"
    )
    audit_parser.add_argument("--out", help="also write the JSON report to this path")
    audit_parser.add_argument("--verbose", "-v", action="store_true", help="show evidence")
    audit_parser.add_argument(
        "--serve", action="store_true", help="open the report in a browser when done"
    )
    audit_parser.add_argument("--port", type=int, default=8899, help="port for --serve")
    audit_parser.add_argument(
        "--fail-on",
        choices=[s.value for s in Severity],
        default="high",
        help="lowest severity that should exit non-zero (default: high)",
    )

    fix_parser = subparsers.add_parser(
        "fix", help="apply the fixes an audit found", parents=[global_flags]
    )
    add_common(fix_parser)
    fix_parser.add_argument(
        "--apply",
        action="store_true",
        help="actually write the changes (default: dry run)",
    )

    serve_parser = subparsers.add_parser(
        "serve", help="audit and open the webview", parents=[global_flags]
    )
    add_common(serve_parser)
    serve_parser.add_argument("--port", type=int, default=8899)
    serve_parser.add_argument(
        "--no-open",
        action="store_true",
        help="start the server without opening a browser",
    )

    demo_parser = subparsers.add_parser(
        "demo", help="run or copy the bundled demo agents", parents=[global_flags]
    )
    demo_parser.add_argument("name", nargs="?", help="demo to audit; omit to list them")
    demo_parser.add_argument("--copy", metavar="DIR", help="copy all demos into DIR")
    demo_parser.add_argument(
        "--force", action="store_true", help="replace existing folders when copying"
    )
    demo_parser.add_argument(
        "--offline", action="store_true", help="never call Claude; static checks only"
    )
    demo_parser.add_argument("--claude-bin", help="path to the claude executable")
    demo_parser.add_argument("--serve", action="store_true", help="open the report in a browser")
    demo_parser.add_argument("--port", type=int, default=8899)

    subparsers.add_parser("checks", help="list the available checks", parents=[global_flags])
    doctor_parser = subparsers.add_parser(
        "doctor", help="check the environment", parents=[global_flags]
    )
    doctor_parser.add_argument("--claude-bin", help="path to the claude executable")

    return parser


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _theme(args: argparse.Namespace) -> Theme:
    return detect(no_color=bool(args.no_color), no_unicode=bool(args.no_unicode))


def _header(args: argparse.Namespace, title: str) -> None:
    """One-line brand header for regular commands.

    The full trident belongs to the menu and the help page; repeating 19 rows of
    art above every audit pushes the findings off screen. Never printed in
    --json mode, where anything but JSON on stdout breaks the consumer.
    """
    if args.quiet or getattr(args, "json", False):
        return
    theme = _theme(args)
    mark = "Ψ" if theme.unicode else "Y"
    rule = ("─" if theme.unicode else "-") * min(theme.width - 4, 76)
    print()
    print(
        "  "
        + theme.bold(theme.gold(f"{mark} TRISOL"))
        + theme.faint(f"  v{__version__}")
        + theme.muted(f"   {title}")
    )
    print("  " + theme.faint(rule))


def _run_audit(args: argparse.Namespace):
    theme = _theme(args)

    def progress(stage: str, name: str) -> None:
        if args.quiet or getattr(args, "json", False):
            return
        if stage == "discover":
            print("  " + theme.muted("scanning ") + theme.fg(TEXT, name), flush=True)
        else:
            print("  " + theme.faint("  running ") + theme.muted(name), flush=True)

    # getattr: `demo` and the menu build namespaces without --only/--skip/--deep.
    return run_audit(
        args.target,
        only=_csv(getattr(args, "only", None)),
        skip=_csv(getattr(args, "skip", None)),
        offline=getattr(args, "offline", False),
        deep=getattr(args, "deep", False),
        claude_bin=getattr(args, "claude_bin", None),
        model=getattr(args, "model", None),
        on_progress=progress,
    )


def cmd_audit(args: argparse.Namespace) -> int:
    report = _run_audit(args)

    if args.out:
        path = report.write_json(args.out)
        if not args.json:
            print(f"  report written to {path}")

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        print(
            render_report(
                report,
                color=False if args.no_color else None,
                unicode=False if args.no_unicode else None,
                verbose=args.verbose,
            )
        )

    if args.serve:
        # Imported lazily: the webview is optional, and `trisol audit`
        # should not pay to import an HTTP server it will not start.
        from .web.server import serve_report  # noqa: PLC0415

        serve_report(report, port=args.port, open_browser=True, block=True)

    return _exit_code(report, args.fail_on)


def _exit_code(report, fail_on: str) -> int:
    """Exit 1 only at or above the chosen severity, so a team can adopt the tool
    at `--fail-on critical` and tighten later."""
    # AuditReport owns this rule; duplicating it here is how the two drifted.
    if report.exit_code == 2:
        return 2
    threshold = Severity(fail_on).rank
    return 1 if any(f.severity.rank <= threshold for f in report.findings) else 0


def cmd_fix(args: argparse.Namespace) -> int:
    report = _run_audit(args)
    fixable = report.fixable

    if not fixable:
        print(render_fix_plan(report))
        return 0

    outcomes = apply_fixes(report.target, fixable, dry_run=not args.apply)
    applied = [o.finding for o in outcomes if o.applied]

    print(render_fix_plan(report, applied=applied))
    for outcome in outcomes:
        if not outcome.applied and outcome.reason and outcome.reason != "dry run":
            print(f"  skipped: {outcome.finding.title} - {outcome.reason}")

    if not args.apply:
        print(f"  {len(fixable)} fix(es) available. Re-run with --apply to write them.\n")
        return 0

    print(f"  applied {len(applied)} of {len(fixable)} fix(es).\n")
    return 0 if len(applied) == len(fixable) else 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the dashboard first, print its link, then audit in the background.

    The browser shows each check as it runs; nothing waits for the whole audit
    before the page is reachable.
    """
    from .web.server import serve_live  # noqa: PLC0415

    target = Path(args.target)
    if not target.exists():
        raise FileNotFoundError(f"no such path: {target.resolve()}")
    serve_live(
        target,
        port=args.port,
        offline=args.offline,
        claude_bin=args.claude_bin,
        open_browser=not args.no_open,
        block=True,
    )
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    theme = _theme(args)
    demos = list_demos()

    if args.copy:
        created = copy_demos(args.copy, overwrite=args.force)
        print()
        for path in created:
            print("  " + theme.ok("copied  ") + str(path))
        print()
        print("  " + theme.muted("Now try:  ") + theme.fg(CYAN, f"trisol audit {created[0]}"))
        print()
        return 0

    if not args.name:
        print()
        for demo in demos:
            print("  " + theme.gold(demo.name.ljust(18)) + theme.muted(demo.description))
        print()
        print("  " + theme.muted("Run one:   ") + theme.fg(CYAN, "trisol demo rag_support_bot"))
        copy_hint = "trisol demo --copy ./trisol-demos"
        print("  " + theme.muted("Copy all:  ") + theme.fg(CYAN, copy_hint))
        print()
        return 0

    match = next((d for d in demos if d.name == args.name), None)
    if match is None:
        names = ", ".join(d.name for d in demos)
        raise ValueError(f"no demo named {args.name!r}. Available: {names}")

    args.target = str(match.path)
    report = _run_audit(args)
    print(
        render_report(
            report,
            color=False if args.no_color else None,
            unicode=False if args.no_unicode else None,
        )
    )
    if args.serve:
        from .web.server import serve_report  # noqa: PLC0415

        serve_report(report, port=args.port, open_browser=True, block=True)
    # Same exit code the report prints, so `trisol demo` and `trisol audit`
    # never disagree about the same code.
    return _exit_code(report, "high")


def cmd_checks(args: argparse.Namespace) -> int:
    theme = _theme(args)
    print()
    for check in ALL_CHECKS:
        note = "   uses Claude when available" if check.name == "prompts" else ""
        print(
            "  " + theme.gold(check.name.ljust(13)) + theme.fg(TEXT, check.title)
            + theme.muted(note)
        )
    print()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    theme = _theme(args)
    print()
    ok = True

    version = sys.version_info
    python_ok = version >= (3, 10)
    ok &= python_ok
    label = theme.ok("PASS") if python_ok else theme.bad("FAIL")
    print(
        f"  {label}  python {version.major}.{version.minor}.{version.micro}"
        + ("" if python_ok else " (3.10+ required)")
    )

    bridge = ClaudeBridge.discover(getattr(args, "claude_bin", None))
    status = bridge.status()
    if status["available"]:
        print(f"  {theme.ok('PASS')}  claude CLI at {status['binary']}")
    else:
        # Not a failure: every static check still runs. Reported so the user
        # knows which part of the audit will be missing.
        print(
            f"  {theme.warn('WARN')}  claude CLI not found - "
            "model-assisted checks will be skipped"
        )
        print("        " + theme.muted(status["reason"]))

    print(f"  {theme.ok('PASS')}  {len(ALL_CHECKS)} checks registered")
    print(f"  {theme.ok('PASS')}  {len(list_demos())} demo agents bundled")
    print()
    return 0 if ok else 1


# ----------------------------------------------------------------- help page


def render_help(theme: Theme) -> str:
    """The `trisol --help` page: banner, then commands laid out as a table.

    Hand-built rather than argparse's default, which leads with
    `{audit,fix,serve,...}` syntax before saying what any of it does.
    """

    def section(name: str) -> str:
        return "  " + theme.bold(theme.gold(name))

    def row(cmd: str, arg: str, text: str) -> str:
        return (
            "    " + theme.fg(CYAN, cmd.ljust(8)) + theme.fg(TEXT, arg.ljust(9))
            + theme.muted(text)
        )

    def opt(flag: str, text: str) -> str:
        return "    " + theme.fg(TEXT, flag.ljust(19)) + theme.muted(text)

    lines = [
        "",
        render_banner(theme, version=__version__),
        "",
        section("USAGE"),
        "    " + theme.fg(CYAN, "trisol".ljust(26)) + theme.muted("open the interactive menu"),
        "    "
        + theme.fg(CYAN, "trisol <command> ")
        + theme.fg(TEXT, "[path]".ljust(9))
        + theme.muted("run one command directly"),
        "",
        section("COMMANDS"),
        row("audit", "PATH", "test and benchmark an agent - a folder or a single .py file"),
        row("fix", "PATH", "preview the safe automatic fixes; --apply writes them"),
        row("serve", "PATH", "audit, then open the report in your browser"),
        row("demo", "[NAME]", "run a bundled demo agent, or --copy DIR to get your own"),
        row("checks", "", "list everything Trisol checks"),
        row("doctor", "", "check Python, the Claude CLI and the bundled checks"),
        "",
        section("OPTIONS"),
        opt("--offline", "never call Claude; static checks only"),
        opt("--claude-bin PATH", "use this claude executable"),
        opt("--json", "machine-readable output"),
        opt("--fail-on LEVEL", "critical | high | medium | low    (default: high)"),
        opt("--only / --skip", "comma-separated check names"),
        opt("--no-color", "plain output; also --no-unicode and -q"),
        "",
        section("EXAMPLES"),
        "    " + theme.fg(TEXT, "trisol audit ./my_agent"),
        "    " + theme.fg(TEXT, "trisol audit agent.py --offline"),
        "    " + theme.fg(TEXT, "trisol demo rag_support_bot --serve"),
        "    " + theme.fg(TEXT, "trisol fix ./my_agent --apply"),
        "",
        "  "
        + theme.muted("Options for one command:  ")
        + theme.fg(CYAN, "trisol <command> --help"),
        "",
    ]
    return "\n".join(lines)


#: Flags accepted both before and after the subcommand, and the argv spellings
#: that set them.
_GLOBAL_FLAGS = {
    "no_color": ("--no-color",),
    "no_unicode": ("--no-unicode",),
    "quiet": ("--quiet", "-q"),
}


def _merge_global_flags(args: argparse.Namespace, argv: list[str] | None = None) -> None:
    """Resolve each repeated global flag to a single boolean.

    argparse applies a subparser's defaults over the namespace it is given, so a
    flag passed *before* the subcommand is always reset by the subparser's own
    default -- whatever that default is. Reading argv directly is therefore the
    only position-independent answer, and it keeps `trisol --no-color audit .`
    and `trisol audit . --no-color` equivalent.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    for attr, spellings in _GLOBAL_FLAGS.items():
        setattr(args, attr, any(flag in raw for flag in spellings))


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    # Top-level help is hand-built; per-command help stays argparse's.
    if raw and raw[0] in ("-h", "--help"):
        theme = detect(no_color="--no-color" in raw, no_unicode="--no-unicode" in raw)
        print(render_help(theme))
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)
    # argparse applies the subparser's defaults *after* the top-level parse, so
    # `--no-color audit .` would be reset to False by the audit subparser's own
    # default. Re-reading the raw argv is the reliable way to honour both
    # positions without declaring the flags in only one of them.
    # argv, not sys.argv: main() is called with an explicit list by tests and by
    # any programmatic caller, and reading the process's own argv there would
    # resolve the flags from the wrong command line entirely.
    _merge_global_flags(args, argv)

    if args.command is None:
        # A real keyboard gets the menu; a pipe or a CI log gets the help page,
        # because a menu there would wait for keypresses that can never arrive.
        if sys.stdin.isatty() and sys.stdout.isatty():
            from . import interactive  # noqa: PLC0415 - imports cli; load late

            return interactive.run(args)
        print(render_help(_theme(args)))
        return 0

    titles = {
        "audit": "audit",
        "fix": "fix",
        "serve": "web dashboard",
        "demo": "demo agents",
        "checks": "checks",
        "doctor": "environment",
    }
    _header(args, titles.get(args.command, args.command))

    handlers = {
        "audit": cmd_audit,
        "fix": cmd_fix,
        "serve": cmd_serve,
        "demo": cmd_demo,
        "checks": cmd_checks,
        "doctor": cmd_doctor,
    }

    try:
        return handlers[args.command](args)
    except FileNotFoundError as exc:
        print(f"\n  error: {exc}\n", file=sys.stderr)
        return 2
    except (ValueError, FileExistsError) as exc:
        # An unknown --only/--skip or demo name, or copy_demos refusing to
        # overwrite a folder the user may have edited.
        print(f"\n  error: {exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n  interrupted\n", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
