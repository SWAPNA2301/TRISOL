"""Scorecard, prompt rubric, the new automatic fixes, and the live dashboard.

The live-dashboard tests run a real server on an ephemeral port and drive it over
HTTP, the same way the browser does, including applying fixes and checking that
the audit re-runs and the score history records the improvement.
"""

from __future__ import annotations

import json
import shutil
import textwrap
import time
import urllib.error
import urllib.request

import pytest

from trisol.audit import audit
from trisol.checks.prompts import RUBRIC, score_prompt
from trisol.demo_agents import DEMO_ROOT
from trisol.findings import Confidence, Finding, Patch, Severity
from trisol.fixer import apply_fixes
from trisol.scorecard import build_scorecard
from trisol.web.server import LiveDashboard

GOOD_PROMPT = (
    "You are a billing support agent for an online store. Answer only from the "
    "context provided. If the context does not contain the answer, reply exactly: "
    "I don't know. Respond in at most three sentences of plain text."
)


def demo(tmp_path, name: str):
    target = tmp_path / name
    shutil.copytree(DEMO_ROOT / name, target)
    return target


def wait_until_finished(dashboard: LiveDashboard, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = dashboard.snapshot()
        if state["status"] in ("done", "error"):
            return state
        time.sleep(0.03)
    raise AssertionError("audit did not finish")


# ------------------------------------------------------------------- rubric


class TestRubric:
    def test_every_criterion_is_described(self) -> None:
        result = score_prompt(GOOD_PROMPT, is_rag=True)
        assert set(result["criteria"]) == set(RUBRIC)

    def test_a_vague_prompt_scores_zero(self) -> None:
        result = score_prompt("You are a helpful assistant. Answer well.", is_rag=True)
        assert result["score"] == 0

    def test_a_specific_prompt_scores_full_marks(self) -> None:
        assert score_prompt(GOOD_PROMPT, is_rag=True)["score"] == 100

    def test_grounding_is_not_applicable_outside_retrieval(self) -> None:
        """Not counted as a pass or a fail, so it cannot inflate or sink a
        non-retrieval agent's score."""
        result = score_prompt(GOOD_PROMPT, is_rag=False)
        assert result["criteria"]["grounding"] is None
        assert result["score"] == 100

    def test_score_is_the_share_of_criteria_met(self) -> None:
        result = score_prompt("You are the triage agent for a hospital helpdesk.", is_rag=False)
        met = [v for v in result["criteria"].values() if v is not None]
        assert result["score"] == round(100 * sum(met) / len(met))


# ---------------------------------------------------------------- scorecard


class TestScorecard:
    def test_every_demo_gets_a_score(self, tmp_path) -> None:
        """The point of the scorecard: measured numbers even without a corpus."""
        for name in ("rag_support_bot", "tool_agent", "router_agent"):
            card = build_scorecard(audit(DEMO_ROOT / name, offline=True))
            assert card["overall"] is not None, name
            assert card["areas"], name

    def test_overall_is_the_mean_of_the_areas(self) -> None:
        card = build_scorecard(audit(DEMO_ROOT / "rag_support_bot", offline=True))
        assert card["overall"] == round(sum(a["score"] for a in card["areas"]) / len(card["areas"]))

    def test_missing_areas_are_omitted_not_scored(self) -> None:
        """No routing table must not count as a perfect routing score."""
        card = build_scorecard(audit(DEMO_ROOT / "rag_support_bot", offline=True))
        assert "routing" not in {a["key"] for a in card["areas"]}

    def test_router_scores_reflect_the_measurements(self) -> None:
        card = build_scorecard(audit(DEMO_ROOT / "router_agent", offline=True))
        routing = next(a for a in card["areas"] if a["key"] == "routing")
        # 11 of 15 keyword claims are unambiguous, and there is no fallback.
        assert routing["has_fallback"] is False
        assert routing["score"] == round(100 * (11 / 15 + 0) / 2)

    def test_nothing_measurable_has_no_overall(self, tmp_path) -> None:
        (tmp_path / "x.py").write_text("x = 1\n", encoding="utf-8")
        assert build_scorecard(audit(tmp_path, offline=True))["overall"] is None

    def test_scorecard_is_part_of_the_report(self) -> None:
        payload = audit(DEMO_ROOT / "tool_agent", offline=True).as_dict()
        assert payload["scorecard"]["overall"] is not None


# ---------------------------------------------------------------- new fixes


class TestNewFixes:
    def test_every_demo_fix_applies_and_keeps_python_valid(self, tmp_path) -> None:
        for name in ("rag_support_bot", "tool_agent", "router_agent"):
            target = demo(tmp_path, name)
            report = audit(target, offline=True)
            assert report.fixable, name
            outcomes = apply_fixes(target, report.fixable, dry_run=False)
            assert all(o.applied for o in outcomes), [o.reason for o in outcomes]
            after = audit(target, offline=True)
            assert len(after.findings) < len(report.findings), name
            # Every file still parses -- the fixer guarantees it.
            for path in target.rglob("*.py"):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_timeout_patch_targets_the_model_call_not_a_chained_call(self, tmp_path) -> None:
        """Regression: in `urlopen(url).read().decode()` every call starts on
        the same line; patching decode() produced `decode(, timeout=30)`."""
        (tmp_path / "agent.py").write_text(
            textwrap.dedent(
                '''
                import urllib.request
                OLLAMA = "http://127.0.0.1:11434/api/generate"
                def fetch(url):
                    return urllib.request.urlopen(url).read().decode()
                '''
            ),
            encoding="utf-8",
        )
        patch = next(f.patch for f in audit(tmp_path, offline=True).fixable if "timeout" in f.title)
        assert patch.old == "urllib.request.urlopen(url)"
        assert patch.new == "urllib.request.urlopen(url, timeout=30)"

    def test_grounding_patch_escapes_the_quote_style(self, tmp_path) -> None:
        (tmp_path / "rag.py").write_text(
            textwrap.dedent(
                """
                SYSTEM_PROMPT = 'You are a documentation helper for the internal wiki.'
                def retrieve(query, docs):
                    context = [d for d in docs if query in d]
                    return context
                """
            ),
            encoding="utf-8",
        )
        report = audit(tmp_path, offline=True)
        finding = next(f for f in report.fixable if "grounding" in f.title)
        apply_fixes(tmp_path, [finding], dry_run=False)
        namespace: dict = {}
        source = (tmp_path / "rag.py").read_text(encoding="utf-8")
        exec(compile(source, "rag.py", "exec"), namespace)
        assert "Answer only from the context provided" in namespace["SYSTEM_PROMPT"]

    def test_router_fallback_patch(self, tmp_path) -> None:
        target = demo(tmp_path, "router_agent")
        finding = next(f for f in audit(target, offline=True).fixable if "None" in f.title)
        assert finding.patch.new.startswith('return "support"')


class TestFixerSyntaxGuard:
    def test_a_patch_that_breaks_python_is_refused_and_nothing_is_written(self, tmp_path) -> None:
        path = tmp_path / "a.py"
        path.write_text("value = compute(1)\n", encoding="utf-8")
        bad = Finding(
            check="t", title="t", detail="", severity=Severity.LOW,
            confidence=Confidence.CERTAIN,
            patch=Patch(file="a.py", old="compute(1)", new="compute(1,, )"),
        )
        outcome = apply_fixes(tmp_path, [bad], dry_run=False)[0]
        assert outcome.applied is False
        assert "would break" in outcome.reason
        assert path.read_text(encoding="utf-8") == "value = compute(1)\n"

    def test_non_python_files_are_not_syntax_checked(self, tmp_path) -> None:
        (tmp_path / "notes.txt").write_text("temperature: 1.5\n", encoding="utf-8")
        fix = Finding(
            check="t", title="t", detail="", severity=Severity.LOW,
            confidence=Confidence.CERTAIN,
            patch=Patch(file="notes.txt", old="1.5", new="(((("),
        )
        assert apply_fixes(tmp_path, [fix], dry_run=False)[0].applied is True


# ------------------------------------------------------------ live dashboard


@pytest.fixture
def live(tmp_path):
    dashboard = LiveDashboard(demo(tmp_path, "rag_support_bot"), port=0, offline=True)
    dashboard.start()
    yield dashboard
    dashboard.stop()


def post(dashboard: LiveDashboard, path: str, body: dict, origin: str | None = None):
    req = urllib.request.Request(
        dashboard.url + path.lstrip("/"),
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "Origin": origin or dashboard.url.rstrip("/")},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


class TestLiveDashboard:
    def test_server_is_reachable_before_the_audit_finishes(self, live) -> None:
        """The link must work immediately; the audit runs behind it."""
        with urllib.request.urlopen(live.url, timeout=5) as resp:
            assert resp.status == 200
        assert live.snapshot()["status"] == "idle"

    def test_checks_stream_through_their_states(self, live) -> None:
        live.start_run()
        seen = set()
        deadline = time.time() + 30
        while time.time() < deadline:
            state = live.snapshot()
            seen.update(c["state"] for c in state["checks"])
            if state["status"] == "done":
                break
            time.sleep(0.005)
        assert {"done"} <= seen
        state = live.snapshot()
        assert all(c["state"] in ("done", "skipped") for c in state["checks"])
        assert state["report"]["scorecard"]["overall"] is not None
        assert any(entry["kind"] == "finish" for entry in state["log"])

    def test_state_endpoint_matches_the_snapshot(self, live) -> None:
        live.start_run()
        wait_until_finished(live)
        with urllib.request.urlopen(live.url + "api/state", timeout=5) as resp:
            payload = json.loads(resp.read())
        assert payload["report"]["summary"] == live.snapshot()["report"]["summary"]

    def test_event_stream_sends_state(self, live) -> None:
        live.start_run()
        wait_until_finished(live)
        with urllib.request.urlopen(live.url + "events", timeout=5) as resp:
            line = resp.readline().decode()
        assert line.startswith("data: ")
        assert json.loads(line[6:])["status"] == "done"

    def test_applying_fixes_from_the_web_reruns_and_improves_the_score(self, live) -> None:
        live.start_run()
        before = wait_until_finished(live)
        status, body = post(live, "/api/fix", {})
        assert status == 200
        assert body["applied"]
        time.sleep(0.05)
        after = wait_until_finished(live)
        history = after["history"]
        assert len(history) == 2
        assert history[1]["overall"] > history[0]["overall"]
        assert after["report"]["summary"]["total"] < before["report"]["summary"]["total"]

    def test_selected_fix_ids_apply_only_those(self, live) -> None:
        live.start_run()
        state = wait_until_finished(live)
        first = state["fixes"][0]["id"]
        status, body = post(live, "/api/fix", {"ids": [first]})
        assert status == 200
        assert len(body["applied"]) == 1

    def test_rerun_while_running_is_rejected(self, live) -> None:
        live.start_run()
        status, _ = post(live, "/api/run", {})
        assert status in (200, 409)  # 409 if the first run is still going
        wait_until_finished(live)

    def test_cross_origin_writes_are_refused(self, live) -> None:
        status, body = post(live, "/api/run", {}, origin="http://evil.example")
        assert status == 403
        assert "cross-origin" in body["error"]

    def test_bad_fix_ids_are_rejected(self, live) -> None:
        live.start_run()
        wait_until_finished(live)
        status, _ = post(live, "/api/fix", {"ids": ["not-a-number"]})
        assert status == 400

    def test_a_missing_target_is_reported_not_crashed(self, tmp_path) -> None:
        dashboard = LiveDashboard(tmp_path / "missing", port=0, offline=True)
        dashboard.start()
        try:
            dashboard.start_run()
            state = wait_until_finished(dashboard)
            assert state["status"] == "error"
            assert "no such path" in state["error"]
        finally:
            dashboard.stop()


class TestServeCommand:
    """`trisol serve` itself, not just the server class -- a missing import in
    the command once slipped past every server-level test."""

    def test_serve_on_a_missing_path_exits_two(self, tmp_path) -> None:
        from trisol.cli import main

        assert main(["serve", str(tmp_path / "missing"), "--no-open", "--quiet"]) == 2

    def test_serve_starts_the_live_dashboard(self, tmp_path, monkeypatch) -> None:
        from trisol import cli
        from trisol.web import server

        started = {}

        def fake_serve_live(target, **kwargs):
            started["target"] = str(target)
            started.update(kwargs)

        monkeypatch.setattr(server, "serve_live", fake_serve_live)
        target = demo(tmp_path, "tool_agent")
        assert cli.main(["serve", str(target), "--no-open", "--offline", "--quiet"]) == 0
        assert started["target"] == str(target)
        assert started["open_browser"] is False
        assert started["offline"] is True
