"""The CLI surface: exit codes, output formats, and the fix loop.

Exit codes get the most attention because they are the contract with CI. A tool
that reports problems but exits 0 silently stops gating anything.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from trisol.audit import audit
from trisol.cli import _merge_global_flags, build_parser, main
from trisol.findings import Confidence, Finding, Patch, Severity
from trisol.fixer import apply_fixes


@pytest.fixture
def flawed(tmp_path):
    """An agent with one HIGH finding and one fixable MEDIUM one."""
    (tmp_path / "agent.py").write_text(
        textwrap.dedent(
            '''
            import urllib.request, json
            OLLAMA = "http://127.0.0.1:11434/api/generate"
            SYSTEM_PROMPT = "You are a helpful assistant. Answer well."
            def ask(prompt):
                body = json.dumps({"temperature": 1.3}).encode()
                return urllib.request.urlopen(OLLAMA, data=body)
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def clean(tmp_path):
    (tmp_path / "agent.py").write_text(
        textwrap.dedent(
            '''
            import urllib.request
            OLLAMA = "http://127.0.0.1:11434/api/generate"
            SYSTEM_PROMPT = (
                "You are a billing agent. Answer only from the context given. "
                "If it does not contain the answer, reply exactly: I don't know. "
                "Use at most three sentences of plain text."
            )
            def ask(body):
                try:
                    return urllib.request.urlopen(OLLAMA, data=body, timeout=30)
                except OSError:
                    return None
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    return tmp_path


class TestParser:
    def _parse(self, argv: list[str]):
        args = build_parser().parse_args(argv)
        _merge_global_flags(args, argv)
        return args

    def test_global_flags_work_before_the_subcommand(self) -> None:
        assert self._parse(["--no-color", "audit", "."]).no_color is True

    def test_global_flags_work_after_the_subcommand(self) -> None:
        """Users type it this way; argparse only allows it if declared on both."""
        assert self._parse(["audit", ".", "--no-color"]).no_color is True

    def test_unset_global_flag_is_false_not_none(self) -> None:
        """Downstream code treats these as booleans."""
        assert self._parse(["audit", "."]).no_color is False

    def test_target_defaults_to_cwd(self) -> None:
        assert build_parser().parse_args(["audit"]).target == "."

    def test_fail_on_defaults_to_high(self) -> None:
        assert build_parser().parse_args(["audit", "."]).fail_on == "high"


class TestExitCodes:
    def test_findings_exit_one(self, flawed) -> None:
        assert main(["audit", str(flawed), "--offline", "--quiet"]) == 1

    def test_clean_exits_zero(self, clean) -> None:
        assert main(["audit", str(clean), "--offline", "--quiet"]) == 0

    def test_nothing_checkable_exits_two(self, tmp_path) -> None:
        """An empty target is not a pass: every check skipped must be
        distinguishable from every check passing."""
        assert main(["audit", str(tmp_path), "--offline", "--quiet"]) == 2

    def test_missing_path_exits_two(self, tmp_path) -> None:
        assert main(["audit", str(tmp_path / "nope"), "--offline", "--quiet"]) == 2

    def test_fail_on_critical_tolerates_high(self, flawed) -> None:
        """Lets a team adopt the tool without it blocking every merge at once."""
        assert main(
            ["audit", str(flawed), "--offline", "--quiet", "--fail-on", "critical"]
        ) == 0

    def test_fail_on_medium_catches_medium(self, flawed) -> None:
        assert main(
            ["audit", str(flawed), "--offline", "--quiet", "--fail-on", "medium"]
        ) == 1

    def test_unknown_check_exits_two(self, flawed) -> None:
        """A typo in --only must fail loudly, not silently run everything."""
        assert main(
            ["audit", str(flawed), "--offline", "--quiet", "--only", "nonsense"]
        ) == 2


class TestOutput:
    def test_json_is_valid_and_complete(self, flawed, capsys) -> None:
        main(["audit", str(flawed), "--offline", "--quiet", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["total"] > 0
        assert payload["trisol_version"]
        assert payload["results"]

    def test_json_mode_emits_only_json(self, flawed, capsys) -> None:
        """Banner or progress text mixed into --json output would break any
        caller that pipes it into a parser."""
        main(["audit", str(flawed), "--offline", "--json"])
        json.loads(capsys.readouterr().out)

    def test_report_can_be_written_to_a_file(self, flawed, tmp_path) -> None:
        out = tmp_path / "sub" / "report.json"
        main(["audit", str(flawed), "--offline", "--quiet", "--json", "--out", str(out)])
        assert out.is_file()
        assert json.loads(out.read_text(encoding="utf-8"))["summary"]

    def test_terminal_report_names_skipped_checks(self, flawed, capsys) -> None:
        """Hiding skips would make a partial audit read as a full one."""
        main(["audit", str(flawed), "--offline", "--quiet", "--no-color"])
        assert "Not checked" in capsys.readouterr().out

    def test_ascii_mode_emits_no_escapes_or_wide_glyphs(self, flawed, capsys) -> None:
        main(["audit", str(flawed), "--offline", "--no-color", "--no-unicode"])
        out = capsys.readouterr().out
        assert "\033[" not in out
        out.encode("ascii")

    def test_checks_command_lists_every_check(self, capsys) -> None:
        assert main(["checks"]) == 0
        out = capsys.readouterr().out
        for name in ("reliability", "prompts", "data", "routing", "benchmark"):
            assert name in out

    def test_doctor_reports_without_claude(self, capsys) -> None:
        """No binary is a warning, not a failure: static checks still run."""
        assert main(["doctor", "--claude-bin", "/definitely/not/here"]) == 0
        assert "claude" in capsys.readouterr().out.lower()


class TestFix:
    def test_dry_run_leaves_the_file_alone(self, flawed) -> None:
        before = (flawed / "agent.py").read_text(encoding="utf-8")
        assert main(["fix", str(flawed), "--offline", "--quiet"]) == 0
        assert (flawed / "agent.py").read_text(encoding="utf-8") == before

    def test_apply_writes_the_change(self, flawed) -> None:
        assert main(["fix", str(flawed), "--offline", "--quiet", "--apply"]) == 0
        after = (flawed / "agent.py").read_text(encoding="utf-8")
        assert '"temperature": 0.0' in after

    def test_the_fix_removes_the_finding(self, flawed) -> None:
        """The loop that matters: audit, fix, re-audit, confirm."""
        first = audit(flawed, offline=True)
        assert any("temperature" in f.title for f in first.findings)
        main(["fix", str(flawed), "--offline", "--quiet", "--apply"])
        second = audit(flawed, offline=True)
        assert not any("temperature" in f.title for f in second.findings)


class TestFixer:
    def _finding(self, file: str, old: str, new: str) -> Finding:
        return Finding(
            check="t",
            title="t",
            detail="",
            severity=Severity.LOW,
            confidence=Confidence.CERTAIN,
            patch=Patch(file=file, old=old, new=new),
        )

    def test_refuses_an_ambiguous_patch(self, tmp_path) -> None:
        """Two matches means the intended site is unknown. Guessing would edit
        the wrong line in someone else's code."""
        (tmp_path / "a.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
        outcomes = apply_fixes(
            tmp_path, [self._finding("a.py", "x = 1", "x = 2")], dry_run=False
        )
        assert outcomes[0].applied is False
        assert "2 matches" in outcomes[0].reason
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\nx = 1\n"

    def test_refuses_when_the_file_changed_since_the_audit(self, tmp_path) -> None:
        (tmp_path / "a.py").write_text("y = 9\n", encoding="utf-8")
        outcomes = apply_fixes(
            tmp_path, [self._finding("a.py", "x = 1", "x = 2")], dry_run=False
        )
        assert outcomes[0].applied is False
        assert "changed since the audit" in outcomes[0].reason

    def test_refuses_to_escape_the_audited_tree(self, tmp_path) -> None:
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        outcomes = apply_fixes(
            tmp_path,
            [self._finding("../../outside.py", "x = 1", "x = 2")],
            dry_run=False,
        )
        assert outcomes[0].applied is False
        assert "outside" in outcomes[0].reason

    def test_applies_a_single_unambiguous_match(self, tmp_path) -> None:
        (tmp_path / "a.py").write_text("temp = 1.5\n", encoding="utf-8")
        outcomes = apply_fixes(
            tmp_path, [self._finding("a.py", "1.5", "0.0")], dry_run=False
        )
        assert outcomes[0].applied is True
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "temp = 0.0\n"

    def test_dry_run_reports_without_writing(self, tmp_path) -> None:
        (tmp_path / "a.py").write_text("temp = 1.5\n", encoding="utf-8")
        outcomes = apply_fixes(tmp_path, [self._finding("a.py", "1.5", "0.0")])
        assert outcomes[0].applied is False
        assert outcomes[0].reason == "dry run"
        assert "1.5" in (tmp_path / "a.py").read_text(encoding="utf-8")


class TestAuditApi:
    def test_audit_is_importable_and_returns_a_report(self, flawed) -> None:
        report = audit(flawed, offline=True)
        assert report.target
        assert report.trisol_version
        assert report.duration_ms >= 0

    def test_only_runs_the_named_check(self, flawed) -> None:
        report = audit(flawed, only=["reliability"], offline=True)
        assert [r.name for r in report.results] == ["reliability"]

    def test_skip_excludes_the_named_check(self, flawed) -> None:
        report = audit(flawed, skip=["benchmark"], offline=True)
        assert "benchmark" not in [r.name for r in report.results]

    def test_unknown_check_name_raises(self, flawed) -> None:
        with pytest.raises(ValueError, match="unknown check"):
            audit(flawed, only=["nope"], offline=True)

    def test_findings_are_ordered_worst_first(self, flawed) -> None:
        ranks = [f.severity.rank for f in audit(flawed, offline=True).findings]
        assert ranks == sorted(ranks)

    def test_offline_mode_is_recorded_in_the_report(self, flawed) -> None:
        report = audit(flawed, offline=True)
        assert report.model["available"] is False
        assert "offline" in report.model["reason"]

    def test_progress_callback_is_invoked(self, flawed) -> None:
        seen: list[tuple[str, str]] = []
        audit(flawed, offline=True, on_progress=lambda s, n: seen.append((s, n)))
        assert ("discover", str(flawed.resolve())) in seen
        assert any(stage == "check" for stage, _ in seen)
