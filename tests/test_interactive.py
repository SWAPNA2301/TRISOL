"""The interactive surface: menu drawing, prompts, the theme, demos and help.

Key reading itself needs a real console, so these drive everything around it:
what gets drawn, how input is validated, and that a non-interactive stdin never
leaves the tool waiting for a keypress.
"""

from __future__ import annotations

import io
import re

import pytest

from trisol import tui
from trisol.cli import main, render_help
from trisol.demo_agents import copy_demos, list_demos
from trisol.theme import Theme, _to_256, detect, lerp

_ESC = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    return _ESC.sub("", text)


def items() -> list[tui.MenuItem]:
    return [
        tui.MenuItem("a", "Audit a project", "scan a folder", lambda: None),
        tui.MenuItem("b", "Audit a single file", "one module", lambda: None),
        tui.MenuItem("q", "Quit", "", None),
    ]


class TestTheme:
    def test_highlight_survives_inner_resets(self) -> None:
        """A selection bar over coloured text must not stop at the first
        reset -- that was the half-highlighted menu row."""
        theme = Theme(color="truecolor", unicode=True, width=80)
        inner = theme.fg((255, 0, 0), "a") + theme.fg((0, 255, 0), "b")
        out = theme.highlight((1, 2, 3), inner)
        segments = out.split("\x1b[0m")
        # Every segment that carries text is preceded by the background code.
        for segment in segments:
            if plain(segment):
                assert "48;2;1;2;3" in segment

    def test_no_colour_theme_emits_nothing(self) -> None:
        theme = Theme(color="none", unicode=True, width=80)
        assert theme.fg((1, 1, 1), "x") == "x"
        assert theme.bg((1, 1, 1), "x") == "x"
        assert theme.highlight((1, 1, 1), "x") == "x"
        assert theme.bold("x") == "x"

    def test_256_colour_maps_into_the_cube(self) -> None:
        assert _to_256((0, 0, 0)) == 16
        assert _to_256((255, 255, 255)) == 231

    def test_lerp_endpoints_and_clamping(self) -> None:
        assert lerp((0, 0, 0), (10, 20, 30), 0.0) == (0, 0, 0)
        assert lerp((0, 0, 0), (10, 20, 30), 1.0) == (10, 20, 30)
        assert lerp((0, 0, 0), (10, 20, 30), 5.0) == (10, 20, 30)

    def test_detect_respects_no_color_flag(self) -> None:
        assert detect(io.StringIO(), no_color=True).color == "none"

    def test_detect_gives_a_pipe_no_colour(self, monkeypatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert detect(io.StringIO()).color == "none"

    def test_detect_respects_no_unicode_flag(self) -> None:
        assert detect(io.StringIO(), no_unicode=True).unicode is False

    def test_width_has_a_floor(self) -> None:
        """A tiny or unknown width must not produce zero-width layouts."""
        assert detect(io.StringIO()).width >= 40


class TestMenuDrawing:
    def test_every_item_is_listed_with_its_number(self, capsys) -> None:
        tui._draw(Theme("none", True, 100), items(), 0, {"available": False})
        out = capsys.readouterr().out
        for number, item in enumerate(items(), start=1):
            assert f"{number}  {item.label}" in out

    def test_selected_item_carries_the_pointer(self, capsys) -> None:
        tui._draw(Theme("none", True, 100), items(), 1, {"available": False})
        rows = capsys.readouterr().out.splitlines()
        line = next(row for row in rows if "Audit a single file" in row)
        assert "►" in line

    def test_status_reports_claude_state(self, capsys) -> None:
        tui._draw(Theme("none", True, 100), items(), 0, {"available": True})
        assert "Claude connected" in capsys.readouterr().out
        tui._draw(Theme("none", True, 100), items(), 0, {"available": False})
        assert "Claude not found" in capsys.readouterr().out

    def test_frame_edges_line_up(self, capsys) -> None:
        """Rows containing colour codes must still pad to the frame's width."""
        tui._draw(Theme("truecolor", True, 100), items(), 0, {"available": False})
        framed = [plain(row) for row in capsys.readouterr().out.splitlines() if "│" in row]
        assert framed
        assert len({len(row.rstrip()) for row in framed}) == 1

    def test_ascii_menu_is_ascii(self, capsys) -> None:
        tui._draw(Theme("none", False, 100), items(), 0, {"available": False})
        out = capsys.readouterr().out
        # Clear-screen escapes aside, the drawing is plain ASCII.
        plain(out).replace("\x1b[2J\x1b[3J\x1b[H", "").encode("ascii")

    def test_visible_len_ignores_escapes(self) -> None:
        assert tui._visible_len("\x1b[38;2;1;2;3mabc\x1b[0m") == 3


class TestNonInteractive:
    def test_numbered_fallback_never_waits_for_a_keypress(self, monkeypatch, capsys) -> None:
        """Piped stdin gets a one-shot numbered prompt, not a key loop."""
        monkeypatch.setattr(tui, "_interactive", lambda: False)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "3")
        assert tui.run_menu(Theme("none", True, 100), items(), {"available": False}) == 0
        assert "Audit a project" in capsys.readouterr().out

    def test_numbered_fallback_runs_the_choice(self, monkeypatch) -> None:
        ran = []
        menu = [tui.MenuItem("x", "Do it", "", lambda: ran.append(1))]
        monkeypatch.setattr(tui, "_interactive", lambda: False)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "1")
        tui.run_menu(Theme("none", True, 100), menu, {"available": False})
        assert ran == [1]

    def test_bare_trisol_without_a_tty_prints_help(self, capsys) -> None:
        """A menu in a CI log would hang; the help page is the safe answer."""
        assert main(["--no-color"]) == 0
        out = capsys.readouterr().out
        assert "COMMANDS" in out and "audit" in out


class TestPrompts:
    def test_prompt_path_retries_until_it_exists(self, monkeypatch, tmp_path, capsys) -> None:
        answers = iter([str(tmp_path / "missing"), str(tmp_path)])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
        got = tui.prompt_path(Theme("none", True, 80), "Folder", ".", must_be="dir")
        assert got == tmp_path.resolve()
        assert "not found" in capsys.readouterr().out

    def test_prompt_path_strips_windows_copy_as_path_quotes(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt="": f'"{tmp_path}"')
        assert tui.prompt_path(Theme("none", True, 80), "Folder", ".") == tmp_path.resolve()

    def test_prompt_path_rejects_a_folder_when_a_file_is_wanted(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        target = tmp_path / "agent.py"
        target.write_text("x = 1\n", encoding="utf-8")
        answers = iter([str(tmp_path), str(target)])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
        assert tui.prompt_path(Theme("none", True, 80), "File", "", must_be="file") == target
        assert "choose a single .py file" in capsys.readouterr().out

    def test_empty_answer_takes_the_default(self, monkeypatch, tmp_path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "")
        assert tui.prompt_path(Theme("none", True, 80), "Folder", ".") == tmp_path.resolve()

    @pytest.mark.parametrize(
        ("answer", "expected"), [("y", True), ("yes", True), ("n", False), ("", False)]
    )
    def test_confirm(self, monkeypatch, answer, expected) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt="": answer)
        assert tui.confirm(Theme("none", True, 80), "Sure?") is expected

    def test_prompt_choice_cancels_on_nonsense(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt="": "zzz")
        assert tui.prompt_choice(Theme("none", True, 80), "Pick", [("a", ""), ("b", "")]) is None


class TestDemos:
    def test_three_demos_are_bundled(self) -> None:
        assert {d.name for d in list_demos()} == {"rag_support_bot", "router_agent", "tool_agent"}

    def test_every_demo_has_a_description(self) -> None:
        assert all(d.description for d in list_demos())

    def test_copy_creates_editable_folders(self, tmp_path) -> None:
        created = copy_demos(tmp_path / "out")
        assert {p.name for p in created} == {"rag_support_bot", "router_agent", "tool_agent"}
        assert (tmp_path / "out" / "rag_support_bot" / "agent.py").is_file()
        assert (tmp_path / "out" / "rag_support_bot" / "knowledge" / "docs.json").is_file()

    def test_copy_refuses_to_overwrite_edited_work(self, tmp_path) -> None:
        copy_demos(tmp_path)
        with pytest.raises(FileExistsError, match="--force"):
            copy_demos(tmp_path)

    def test_copy_with_force_replaces(self, tmp_path) -> None:
        copy_demos(tmp_path)
        marker = tmp_path / "tool_agent" / "edited.txt"
        marker.write_text("mine", encoding="utf-8")
        copy_demos(tmp_path, overwrite=True)
        assert not marker.exists()

    def test_demo_command_lists(self, capsys) -> None:
        assert main(["demo", "--no-color", "--quiet"]) == 0
        assert "rag_support_bot" in capsys.readouterr().out

    def test_demo_command_audits_and_exits_like_audit(self) -> None:
        """`trisol demo X` and `trisol audit X` must agree on the exit code."""
        assert main(["demo", "router_agent", "--offline", "--quiet", "--no-color"]) == 1

    def test_unknown_demo_is_an_error(self) -> None:
        assert main(["demo", "nope", "--quiet", "--no-color"]) == 2

    def test_demo_copy_command(self, tmp_path) -> None:
        assert main(["demo", "--copy", str(tmp_path / "d"), "--quiet", "--no-color"]) == 0
        assert (tmp_path / "d" / "router_agent" / "router.py").is_file()


class TestHelp:
    def test_help_lists_every_command(self) -> None:
        out = plain(render_help(Theme("none", True, 100)))
        for command in ("audit", "fix", "serve", "demo", "checks", "doctor"):
            assert command in out

    def test_help_flag_prints_the_styled_page(self, capsys) -> None:
        assert main(["--help", "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "EXAMPLES" in out
        assert "\x1b[" not in out

    def test_subcommand_help_still_works(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["audit", "--help"])
        assert exc.value.code == 0
        assert "--fail-on" in capsys.readouterr().out
