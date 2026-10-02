"""The Claude bridge: discovery, JSON extraction, and failing safely.

The behaviour that matters most is the last one. Trisol must stay useful with no
`claude` binary, with an unauthenticated one, and with one that answers in prose
-- and it must distinguish "could not ask" (a skip) from "asked and got
nonsense" (worth saying out loud).
"""

from __future__ import annotations

import subprocess

import pytest

from trisol.claude import (
    ClaudeBridge,
    ClaudeError,
    ClaudeUnavailableError,
    _parse_json,
    find_claude_binary,
)


class TestParseJson:
    def test_bare_object(self) -> None:
        assert _parse_json('{"a": 1}') == {"a": 1}

    def test_bare_array(self) -> None:
        assert _parse_json("[1, 2]") == [1, 2]

    def test_fenced_block(self) -> None:
        text = 'Here you go:\n```json\n{"findings": []}\n```\nHope that helps.'
        assert _parse_json(text) == {"findings": []}

    def test_fence_without_a_language(self) -> None:
        assert _parse_json('```\n{"a": 2}\n```') == {"a": 2}

    def test_object_embedded_in_prose(self) -> None:
        """Models wrap JSON in commentary often enough that refusing it would
        make this layer needlessly brittle."""
        assert _parse_json('Sure thing. {"a": 3} Let me know.') == {"a": 3}

    def test_empty_output_raises(self) -> None:
        with pytest.raises(ClaudeError, match="no output"):
            _parse_json("   ")

    def test_prose_only_raises(self) -> None:
        with pytest.raises(ClaudeError, match="did not return JSON"):
            _parse_json("I think the prompt looks fine, honestly.")

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(ClaudeError):
            _parse_json("{not: valid,}")


class TestDiscovery:
    def test_missing_binary_is_none(self, monkeypatch) -> None:
        monkeypatch.delenv("TRISOL_CLAUDE_BIN", raising=False)
        monkeypatch.setattr("trisol.claude.shutil.which", lambda _n: None)
        monkeypatch.setattr("trisol.claude.Path.exists", lambda _self: False)
        assert find_claude_binary() is None

    def test_explicit_path_wins(self, tmp_path, monkeypatch) -> None:
        fake = tmp_path / "claude"
        fake.write_text("", encoding="utf-8")
        monkeypatch.setattr("trisol.claude.shutil.which", lambda _n: "/usr/bin/claude")
        assert find_claude_binary(str(fake)) == str(fake)

    def test_bad_explicit_path_is_not_silently_replaced(self, monkeypatch) -> None:
        """A wrong --claude-bin must surface, not fall back to PATH: otherwise
        the user believes they used a binary they did not."""
        monkeypatch.setattr("trisol.claude.shutil.which", lambda _n: None)
        assert find_claude_binary("/nope/claude") is None

    def test_env_var_is_honoured(self, tmp_path, monkeypatch) -> None:
        fake = tmp_path / "claude"
        fake.write_text("", encoding="utf-8")
        monkeypatch.setenv("TRISOL_CLAUDE_BIN", str(fake))
        assert find_claude_binary() == str(fake)


class TestStatus:
    def test_unavailable_status_explains_how_to_fix_it(self) -> None:
        status = ClaudeBridge(binary=None).status()
        assert status["available"] is False
        assert "claude" in status["reason"].lower()
        # The message has to be actionable, not just negative.
        assert "--claude-bin" in status["reason"] or "install" in status["reason"].lower()

    def test_available_status_names_the_binary(self) -> None:
        status = ClaudeBridge(binary="/usr/bin/claude").status()
        assert status["available"] is True
        assert status["binary"] == "/usr/bin/claude"


class TestAsking:
    def test_no_binary_raises_unavailable(self) -> None:
        with pytest.raises(ClaudeUnavailableError):
            ClaudeBridge(binary=None).ask_json("anything")

    def test_successful_call_returns_parsed_json(self, monkeypatch) -> None:
        class Result:
            returncode = 0
            stdout = '{"findings": [{"title": "x"}]}'
            stderr = ""

        monkeypatch.setattr("trisol.claude.subprocess.run", lambda *a, **k: Result())
        payload = ClaudeBridge(binary="/bin/claude").ask_json("review this")
        assert payload["findings"][0]["title"] == "x"

    def test_timeout_is_reported_as_an_error(self, monkeypatch) -> None:
        def boom(*_a, **_k):
            raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

        monkeypatch.setattr("trisol.claude.subprocess.run", boom)
        with pytest.raises(ClaudeError, match="timed out"):
            ClaudeBridge(binary="/bin/claude", timeout_s=1).ask_json("x")

    def test_auth_failure_disables_the_bridge(self, monkeypatch) -> None:
        """An auth failure repeats for every call, so it must stop the bridge
        rather than burn one timeout per check."""

        class Result:
            returncode = 1
            stdout = ""
            stderr = "Error: not logged in. Run `claude login`."

        monkeypatch.setattr("trisol.claude.subprocess.run", lambda *a, **k: Result())
        bridge = ClaudeBridge(binary="/bin/claude")
        with pytest.raises(ClaudeUnavailableError):
            bridge.ask_json("x")
        assert bridge.available is False
        # A second call must not shell out again.
        with pytest.raises(ClaudeUnavailableError):
            bridge.ask_json("y")

    def test_other_nonzero_exit_is_a_recoverable_error(self, monkeypatch) -> None:
        class Result:
            returncode = 3
            stdout = ""
            stderr = "transient upstream hiccup"

        monkeypatch.setattr("trisol.claude.subprocess.run", lambda *a, **k: Result())
        bridge = ClaudeBridge(binary="/bin/claude")
        with pytest.raises(ClaudeError):
            bridge.ask_json("x")
        # Not disabled: the next check may well succeed.
        assert bridge.available is True

    def test_oversized_prompt_is_truncated_with_a_marker(self, monkeypatch) -> None:
        captured: dict[str, str] = {}

        class Result:
            returncode = 0
            stdout = "{}"
            stderr = ""

        def fake_run(cmd, **kwargs):
            captured["prompt"] = kwargs["input"]
            return Result()

        monkeypatch.setattr("trisol.claude.subprocess.run", fake_run)
        monkeypatch.setattr("trisol.claude.MAX_PROMPT_CHARS", 200)
        ClaudeBridge(binary="/bin/claude").ask_json("line\n" * 500)
        assert len(captured["prompt"]) < 600
        # Silent truncation would let a model answer about a fragment as if it
        # were the whole input.
        assert "truncated by Trisol" in captured["prompt"]

    def test_schema_hint_is_appended(self, monkeypatch) -> None:
        captured: dict[str, str] = {}

        class Result:
            returncode = 0
            stdout = "{}"
            stderr = ""

        def fake_run(cmd, **kwargs):
            captured["prompt"] = kwargs["input"]
            return Result()

        monkeypatch.setattr("trisol.claude.subprocess.run", fake_run)
        ClaudeBridge(binary="/bin/claude").ask_json("q", schema_hint='{"a": str}')
        assert '{"a": str}' in captured["prompt"]
        assert "JSON only" in captured["prompt"]

    def test_prompt_travels_over_stdin_not_argv(self, monkeypatch) -> None:
        """Regression: on Windows `claude` is a .CMD file and cmd.exe cuts an
        argument at its first newline, so a multi-line prompt passed as argv
        arrived as one sentence. Measured against the real CLI: argv delivered
        1 line of 4, stdin all 4."""
        captured: dict = {}

        class Result:
            returncode = 0
            stdout = "{}"
            stderr = ""

        def fake_run(cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["input"] = kwargs.get("input")
            return Result()

        monkeypatch.setattr("trisol.claude.subprocess.run", fake_run)
        prompt = "first line\nsecond line\nthird line"
        ClaudeBridge(binary="/bin/claude").ask_json(prompt)
        assert all("second line" not in part for part in captured["cmd"])
        assert prompt in captured["input"]

    def test_model_flag_is_passed_through(self, monkeypatch) -> None:
        captured: dict[str, list[str]] = {}

        class Result:
            returncode = 0
            stdout = "{}"
            stderr = ""

        def fake_run(cmd, **_kwargs):
            captured["cmd"] = list(cmd)
            return Result()

        monkeypatch.setattr("trisol.claude.subprocess.run", fake_run)
        ClaudeBridge(binary="/bin/claude", model="claude-opus-5").ask_json("q")
        assert "--model" in captured["cmd"]
        assert "claude-opus-5" in captured["cmd"]
