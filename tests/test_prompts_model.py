"""The model-assisted half of the prompt check.

Claude is stubbed, because what needs testing is Trisol's handling of the reply:
a model finding must be marked as model-sourced and never claimed as CERTAIN, a
malformed reply must not become an invented finding, and an unreachable model
must leave the static findings standing with the reason recorded.
"""

from __future__ import annotations

import pytest

from trisol.checks.base import CheckContext, run_check
from trisol.checks.prompts import PromptCheck
from trisol.claude import ClaudeError, ClaudeUnavailableError
from trisol.discover import discover
from trisol.findings import Confidence, Severity


class _StubBridge:
    """Stands in for ClaudeBridge: available, and returns whatever it is given."""

    def __init__(self, payload=None, raises=None):
        self._payload = payload
        self._raises = raises
        self.calls: list[str] = []
        self.available = True

    def status(self) -> dict:
        return {"available": True, "binary": "/stub/claude"}

    def ask_json(self, prompt: str, schema_hint: str = "", timeout_s=None):
        self.calls.append(prompt)
        if self._raises is not None:
            raise self._raises
        return self._payload


@pytest.fixture
def project(tmp_path):
    (tmp_path / "a.py").write_text(
        'SYSTEM_PROMPT = "You are a helpful assistant. Answer the question well."\n',
        encoding="utf-8",
    )
    return tmp_path


def _run(project_dir, bridge):
    return run_check(PromptCheck(), CheckContext(discover(project_dir), bridge))


class TestModelFindings:
    def test_model_finding_is_recorded_and_attributed(self, project) -> None:
        bridge = _StubBridge(
            {
                "findings": [
                    {
                        "title": "No output contract",
                        "detail": "The prompt never states a response format.",
                        "severity": "high",
                        "impact": "Downstream parsing breaks on free-form text.",
                        "suggestion": "Specify the exact output shape.",
                        "prompt_name": "SYSTEM_PROMPT",
                    }
                ]
            }
        )
        result = _run(project, bridge)
        model_findings = [f for f in result.findings if f.source == "claude"]
        assert len(model_findings) == 1
        finding = model_findings[0]
        assert finding.title == "No output contract"
        assert finding.severity is Severity.HIGH
        # Located back to the prompt's real site so it is actionable.
        assert finding.file == "a.py"
        assert finding.line == 1

    def test_model_findings_are_never_certain(self, project) -> None:
        """A model's judgement is an opinion worth reading, not a proof. Marking
        it CERTAIN would make a reviewer skip the check it needs."""
        bridge = _StubBridge(
            {"findings": [{"title": "x", "severity": "critical", "prompt_name": "SYSTEM_PROMPT"}]}
        )
        finding = next(f for f in _run(project, bridge).findings if f.source == "claude")
        assert finding.confidence is Confidence.POSSIBLE

    def test_unknown_severity_defaults_to_medium(self, project) -> None:
        """A badly labelled finding may still be real, so it is kept."""
        bridge = _StubBridge(
            {"findings": [{"title": "x", "severity": "spicy", "prompt_name": "SYSTEM_PROMPT"}]}
        )
        finding = next(f for f in _run(project, bridge).findings if f.source == "claude")
        assert finding.severity is Severity.MEDIUM

    def test_a_bare_list_reply_is_accepted(self, project) -> None:
        bridge = _StubBridge([{"title": "y", "severity": "low", "prompt_name": "SYSTEM_PROMPT"}])
        assert any(f.source == "claude" for f in _run(project, bridge).findings)

    def test_non_dict_items_are_ignored(self, project) -> None:
        """One malformed element must not discard the whole set."""
        bridge = _StubBridge(
            {"findings": ["nonsense", {"title": "real", "severity": "low",
                                       "prompt_name": "SYSTEM_PROMPT"}]}
        )
        titles = [f.title for f in _run(project, bridge).findings if f.source == "claude"]
        assert titles == ["real"]

    def test_empty_findings_list_is_a_valid_answer(self, project) -> None:
        result = _run(project, _StubBridge({"findings": []}))
        assert [f for f in result.findings if f.source == "claude"] == []
        assert result.metrics["model_reviewed"] is True

    def test_unknown_prompt_name_still_yields_a_finding(self, project) -> None:
        """Without a matching site it has no file/line, but dropping it would
        hide a real problem over a naming mismatch."""
        bridge = _StubBridge(
            {"findings": [{"title": "z", "severity": "high", "prompt_name": "NOPE"}]}
        )
        finding = next(f for f in _run(project, bridge).findings if f.source == "claude")
        assert finding.file is None


class TestModelFailures:
    def test_static_findings_survive_an_unavailable_model(self, project) -> None:
        bridge = _StubBridge(raises=ClaudeUnavailableError("not logged in"))
        result = _run(project, bridge)
        assert any(f.source == "static" for f in result.findings)
        assert "not logged in" in result.metrics["model_skipped_reason"]

    def test_a_malformed_reply_is_recorded_not_invented(self, project) -> None:
        """Asked and got nonsense is different from could not ask, and neither
        is a finding about the audited code."""
        bridge = _StubBridge(raises=ClaudeError("claude did not return JSON"))
        result = _run(project, bridge)
        assert "model_error" in result.metrics
        assert [f for f in result.findings if f.source == "claude"] == []

    def test_offline_never_calls_the_model(self, project) -> None:
        bridge = _StubBridge({"findings": [{"title": "x", "severity": "low"}]})
        result = run_check(
            PromptCheck(), CheckContext(discover(project), bridge, offline=True)
        )
        assert bridge.calls == []
        assert result.metrics["model_reviewed"] is False
        assert "offline" in result.metrics["model_skipped_reason"]

    def test_prompts_sent_to_the_model_are_bounded(self, tmp_path) -> None:
        """A large codebase must not produce one enormous call."""
        lines = [
            f'SYSTEM_PROMPT_{i} = "You are assistant number {i}. Answer well enough."'
            for i in range(30)
        ]
        (tmp_path / "many.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
        bridge = _StubBridge({"findings": []})
        run_check(PromptCheck(), CheckContext(discover(tmp_path), bridge))
        assert len(bridge.calls) == 1
        # 12 prompts is the cap; the 13th must not appear in the request.
        assert "SYSTEM_PROMPT_12" not in bridge.calls[0]
