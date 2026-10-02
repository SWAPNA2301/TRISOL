"""Render a report for a terminal.

Ordered by what someone reading a failing build needs: the verdict, then the
worst findings with file:line, then the benchmark numbers, then what was not
checked and why. Skipped checks are always printed -- a report that hides them
reads like a clean pass.

Built on the same Theme as the banner and menu, so colour depth, glyph set and
width are decided once. Every wrapped paragraph keeps its own hanging indent, so
continuation lines line up under their first line at any terminal width.
"""

from __future__ import annotations

import textwrap
from typing import TextIO

from ..findings import AuditReport, CheckResult, Confidence, Finding, Severity
from ..theme import AMBER, CYAN, GOLD_TOP, GREEN, MUTED, RED, TEXT, Theme, detect

__all__ = ["render_fix_plan", "render_report"]

#: (foreground, badge background) per severity.
_SEVERITY = {
    Severity.CRITICAL: ((255, 110, 100), (74, 24, 22)),
    Severity.HIGH: ((255, 150, 80), (70, 38, 16)),
    Severity.MEDIUM: ((236, 196, 84), (60, 50, 16)),
    Severity.LOW: ((170, 170, 170), (46, 46, 46)),
}

_INDENT = 4  # left margin of the whole report
_BADGE = 12  # width of the severity column, badge plus gap


def _theme(stream: TextIO | None, color: bool | None, unicode: bool | None) -> Theme:
    theme = detect(stream)
    if color is False:
        theme.color = "none"
    elif color is True and theme.color == "none":
        theme.color = "truecolor"
    if unicode is not None:
        theme.unicode = unicode
    return theme


def _rule(theme: Theme, width: int) -> str:
    return " " * _INDENT + theme.faint(("─" if theme.unicode else "-") * width)


def _wrap(text: str, right_edge: int, indent: int, first_prefix: str = "") -> list[str]:
    """Wrap ``text`` between column ``indent`` and column ``right_edge``.

    ``first_prefix`` is already-rendered text (it may carry colour) that takes
    the place of the indent on the first line only; every continuation line is
    indented to ``indent`` so the paragraph reads as one block.
    """
    body_width = max(20, right_edge - indent)
    lines = textwrap.wrap(str(text), body_width) or [""]
    return [
        (first_prefix if index == 0 and first_prefix else " " * indent) + line
        for index, line in enumerate(lines)
    ]


def _content_width(theme: Theme) -> int:
    # A readable measure rather than the full terminal: prose wider than ~92
    # columns is hard to scan, and very wide terminals should not stretch it.
    return max(56, min(theme.width - _INDENT - 2, 92))


def render_report(
    report: AuditReport,
    *,
    stream: TextIO | None = None,
    color: bool | None = None,
    unicode: bool | None = None,
    verbose: bool = False,
    max_findings: int = 40,
) -> str:
    """The full report as a string."""
    theme = _theme(stream, color, unicode)
    width = _content_width(theme)
    right = _INDENT + width
    pad = " " * _INDENT
    out: list[str] = [""]

    # -- header -------------------------------------------------------------
    out.append(pad + theme.muted("Target  ") + theme.fg(TEXT, report.target))
    model = report.model or {}
    if model.get("available"):
        out.append(
            pad + theme.muted("Claude  ") + theme.ok("connected  ")
            + theme.faint(str(model.get("binary", "")))
        )
    else:
        reason = str(model.get("reason", "")).split(". ")[0]
        out.append(
            pad + theme.muted("Claude  ") + theme.warn("not used  ")
            + theme.faint(reason[: max(10, width - 20)])
        )
    out.append("")
    out.append(_summary(theme, report))
    out.extend(_scorecard(theme, report))
    out.append("")

    # -- findings -----------------------------------------------------------
    findings = report.findings
    shown = findings[:max_findings]
    for finding in shown:
        out.extend(_finding(theme, finding, right, verbose=verbose))
    if len(findings) > len(shown):
        out.append(pad + theme.muted(
            f"... and {len(findings) - len(shown)} more   (--json shows everything)"
        ))
        out.append("")

    # -- benchmark ----------------------------------------------------------
    bench = _benchmark(theme, report)
    if bench:
        out.append(_rule(theme, width))
        out.extend(bench)
        out.append("")

    # -- not checked --------------------------------------------------------
    skipped = [r for r in report.results if r.skipped]
    if skipped:
        out.append(_rule(theme, width))
        out.append("")
        out.append(pad + theme.bold(theme.fg(TEXT, "Not checked")))
        for result in skipped:
            first = pad + "  " + theme.muted(result.name.ljust(12))
            out.extend(_wrap(result.skip_reason, right, _INDENT + 14, first_prefix=first))
        out.append("")

    # -- verdict ------------------------------------------------------------
    out.append(_rule(theme, width))
    out.append("")
    out.append(pad + _verdict(theme, report))
    summary = report.as_dict()["summary"]
    sep = " · " if theme.unicode else " - "
    out.append(pad + theme.faint(
        f"{summary['checks_run']} checks{sep}{report.duration_ms}ms{sep}exit {report.exit_code}"
    ))
    out.append("")
    return "\n".join(out)


def _score_color(score: int):
    return GREEN if score >= 80 else (AMBER if score >= 50 else RED)


def _scorecard(theme: Theme, report: AuditReport) -> list[str]:
    """Overall score and one bar per measured area."""
    card = report.as_dict()["scorecard"]
    if card["overall"] is None:
        return []
    pad = " " * _INDENT
    full, empty = ("━", "·") if theme.unicode else ("=", ".")
    out = [
        "",
        pad
        + theme.bold(theme.fg(TEXT, "Scorecard  "))
        + theme.bold(theme.fg(_score_color(card["overall"]), f"{card['overall']}"))
        + theme.muted(" / 100"),
    ]
    for area in card["areas"]:
        score = area["score"]
        filled = round(score / 5)
        bar = theme.fg(_score_color(score), full * filled) + theme.faint(empty * (20 - filled))
        out.append(
            pad + "  " + theme.fg(TEXT, f"{area['title']:<18}") + bar
            + theme.fg(_score_color(score), f"{score:>5}") + "   " + theme.muted(area["summary"])
        )
    return out


def _summary(theme: Theme, report: AuditReport) -> str:
    summary = report.as_dict()["summary"]
    dot = "●" if theme.unicode else "*"
    parts = []
    for severity in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW):
        count = report.count(severity)
        if count:
            fg, _ = _SEVERITY[severity]
            parts.append(theme.fg(fg, dot) + " " + theme.fg(TEXT, f"{count} {severity.value}"))
    if not parts:
        parts.append(theme.ok(dot) + " " + theme.fg(TEXT, "no findings"))
    line = "     ".join(parts)
    if summary["fixable"]:
        line += "        " + theme.fg(CYAN, f"{summary['fixable']} auto-fixable")
    return " " * _INDENT + line


def _finding(theme: Theme, finding: Finding, right: int, *, verbose: bool) -> list[str]:
    fg, bg = _SEVERITY[finding.severity]
    label = finding.severity.value.upper()
    if theme.colored:
        badge = theme.bg(bg, theme.bold(theme.fg(fg, f" {label} ".center(10))))
    else:
        badge = f"[{label}]".ljust(10)
    body = _INDENT + _BADGE  # every line in the block starts here

    where = ""
    if finding.file:
        location = finding.file + (f":{finding.line}" if finding.line else "")
        where = "   " + theme.fg(CYAN, location)

    out = [" " * _INDENT + badge + "  " + theme.bold(theme.fg(TEXT, finding.title)) + where]
    out.extend(_wrap(finding.detail, right, body))
    if finding.impact:
        out.extend(_wrap(finding.impact, right, body + 8,
                         first_prefix=" " * body + theme.muted("Impact  ")))
    if finding.suggestion:
        out.extend(_wrap(finding.suggestion, right, body + 8,
                         first_prefix=" " * body + theme.fg(GOLD_TOP, "Fix     ")))

    tags = []
    if finding.source == "claude":
        tags.append(theme.fg(GOLD_TOP, "claude"))
    if finding.confidence is not Confidence.CERTAIN:
        tags.append(theme.muted(finding.confidence.value))
    if finding.fixable:
        tags.append(theme.fg(CYAN, "auto-fixable"))
    if tags:
        sep = theme.faint(" · " if theme.unicode else " - ")
        out.append(" " * body + sep.join(tags))

    if verbose and finding.evidence:
        bullet = "•" if theme.unicode else "*"
        for item in finding.evidence[:6]:
            out.append(" " * body + theme.faint(f"{bullet} {str(item)[: right - body - 2]}"))
    out.append("")
    return out


def _benchmark(theme: Theme, report: AuditReport) -> list[str]:
    result: CheckResult | None = next(
        (r for r in report.results if r.metrics.get("results")), None
    )
    if result is None:
        return []
    m = result.metrics
    pad = " " * _INDENT
    sep = " · " if theme.unicode else " - "
    out = [
        "",
        pad + theme.bold(theme.fg(TEXT, "Benchmark  ")) + theme.muted(str(m.get("corpus", ""))),
        pad + theme.faint(
            f"{m.get('documents', 0)} documents{sep}{m.get('queries', 0)} queries"
            f"{sep}top {m.get('top_k', 0)}"
        ),
        "",
        pad + "  " + theme.faint(f"{'strategy':<18}{'recall':>7}{'MRR':>8}"),
    ]
    full, empty = ("━", "·") if theme.unicode else ("=", ".")
    for row in m["results"]:
        recall = row["recall_at_k"]
        filled = round(recall * 20)
        is_best = row["name"] == m.get("best")
        is_current = row["name"] == m.get("baseline")
        color = GREEN if is_best else (AMBER if is_current else MUTED)
        bar = theme.fg(color, full * filled) + theme.faint(empty * (20 - filled))
        tag = ""
        if is_best:
            tag = "  " + theme.ok("best")
        elif is_current:
            tag = "  " + theme.warn("your code")
        name = theme.fg(TEXT if (is_best or is_current) else MUTED, f"{row['name']:<18}")
        out.append(
            pad + "  " + name + theme.fg(TEXT, f"{recall:>7.0%}")
            + theme.muted(f"{row['mrr']:>8.2f}") + "   " + bar + tag
        )
    return out


def _verdict(theme: Theme, report: AuditReport) -> str:
    dot = "●" if theme.unicode else "*"
    code = report.exit_code
    if code == 2:
        return theme.warn(f"{dot} Nothing could be checked")
    if code == 1:
        critical = report.count(Severity.CRITICAL)
        high = report.count(Severity.HIGH)
        return theme.fg(RED if critical else AMBER, f"{dot} Needs attention") + theme.fg(
            TEXT, f"   {critical} critical, {high} high"
        )
    total = len(report.findings)
    if total:
        return theme.ok(f"{dot} No blocking issues") + theme.muted(f"   {total} advisory")
    return theme.ok(f"{dot} Clean")


def render_fix_plan(report: AuditReport, *, applied: list[Finding] | None = None) -> str:
    """What `trisol fix` is about to do, or just did."""
    theme = detect()
    pad = " " * _INDENT
    fixable = report.fixable
    if not fixable:
        return "\n" + pad + theme.muted("No findings carry an automatic fix.") + "\n"

    done = {id(f) for f in (applied or [])}
    dot = "●" if theme.unicode else "*"
    out = ["", pad + theme.bold(theme.fg(TEXT, "Automatic fixes")), ""]
    for finding in fixable:
        mark = theme.ok(dot) if id(finding) in done else theme.fg(CYAN, dot)
        out.append(pad + mark + " " + theme.fg(TEXT, finding.title))
        patch = finding.patch
        if patch:
            out.append(pad + "  " + theme.fg(CYAN, patch.file))
            out.append(pad + "  " + theme.bad("- " + patch.old.strip()[:70]))
            out.append(pad + "  " + theme.ok("+ " + patch.new.strip()[:70]))
        out.append("")
    return "\n".join(out)
