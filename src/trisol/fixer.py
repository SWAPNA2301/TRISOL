"""Apply a finding's patch to the working tree.

Rules, because automatic edits to someone else's code earn no benefit of the
doubt:

* a patch applies only when its ``old`` text is present **exactly once** -- two
  matches mean the intended site is ambiguous, so it is refused rather than
  guessed at
* the file is re-read at apply time, so a file edited since the audit is
  detected instead of silently clobbered
* every change is reported, including refusals and why
* ``dry_run`` is the default at the call site, so showing is cheaper than doing
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .findings import Finding

__all__ = ["FixOutcome", "apply_fixes"]


@dataclass
class FixOutcome:
    finding: Finding
    applied: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "title": self.finding.title,
            "file": self.finding.patch.file if self.finding.patch else None,
            "applied": self.applied,
            "reason": self.reason,
        }


def apply_fixes(
    root: Path | str,
    findings: list[Finding],
    *,
    dry_run: bool = True,
) -> list[FixOutcome]:
    """Apply every fixable finding. Returns one outcome per attempt."""
    base = Path(root).resolve()
    outcomes: list[FixOutcome] = []

    for finding in findings:
        patch = finding.patch
        if patch is None:
            continue

        path = (base / patch.file).resolve()
        # Refuse to write outside the audited tree, even if a patch's path tries.
        try:
            path.relative_to(base)
        except ValueError:
            outcomes.append(
                FixOutcome(finding, False, f"patch targets a path outside {base}")
            )
            continue

        if not path.is_file():
            outcomes.append(FixOutcome(finding, False, f"{patch.file} no longer exists"))
            continue

        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            outcomes.append(FixOutcome(finding, False, f"could not read {patch.file}: {exc}"))
            continue

        count = source.count(patch.old)
        if count == 0:
            outcomes.append(
                FixOutcome(
                    finding,
                    False,
                    f"the text to replace is no longer in {patch.file} "
                    "(the file changed since the audit)",
                )
            )
            continue
        if count > 1:
            outcomes.append(
                FixOutcome(
                    finding,
                    False,
                    f"{count} matches in {patch.file}; refusing to guess which to change",
                )
            )
            continue

        updated = source.replace(patch.old, patch.new, 1)

        # Last line of defence: a fix must never leave a Python file that no
        # longer parses. Checked before writing, so a bad patch changes nothing.
        if path.suffix == ".py":
            try:
                ast.parse(updated)
            except SyntaxError as exc:
                outcomes.append(
                    FixOutcome(
                        finding,
                        False,
                        f"refused: the edit would break {patch.file} (line {exc.lineno}: "
                        f"{exc.msg})",
                    )
                )
                continue

        if dry_run:
            outcomes.append(FixOutcome(finding, False, "dry run"))
            continue

        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            outcomes.append(FixOutcome(finding, False, f"could not write {patch.file}: {exc}"))
            continue

        outcomes.append(FixOutcome(finding, True))

    return outcomes
