"""The mark: alignment, layouts, and glyph safety.

The glyph test is the important one. Characters missing from the terminal's
font are drawn from a fallback font at a different width, which is what made the
first banner look broken; every glyph Trisol prints is checked against the set
Consolas (VS Code's default Windows terminal font) is known to contain.
"""

from __future__ import annotations

import re

import pytest

from trisol.banner import (
    TRIDENT,
    TRIDENT_ASCII,
    WORDMARK,
    WORDMARK_ASCII,
    display_width,
    render,
)
from trisol.theme import Theme

#: Every non-ASCII glyph Trisol is allowed to print. Verified present in
#: consola.ttf; extend only after checking the font's cmap.
CONSOLAS_SAFE = set("█▀▄▌▐░═║╔╗╚╝─│╭╮╰╯━┃·•→›►●○Ψ↑↓")

_ESC = re.compile(r"\x1b\[[0-9;]*m")


def strip(text: str) -> str:
    return _ESC.sub("", text)


def theme(width: int = 100, color: str = "none", unicode: bool = True) -> Theme:
    return Theme(color=color, unicode=unicode, width=width)


class TestArt:
    @pytest.mark.parametrize(
        "art", [TRIDENT, TRIDENT_ASCII, WORDMARK, WORDMARK_ASCII],
        ids=["trident", "trident-ascii", "wordmark", "wordmark-ascii"],
    )
    def test_every_row_is_the_same_width(self, art) -> None:
        """Uneven rows make the composition drift sideways."""
        assert len({display_width(row) for row in art}) == 1

    def test_both_tridents_share_a_width(self) -> None:
        assert display_width(TRIDENT[0]) == display_width(TRIDENT_ASCII[0])

    def test_shaft_sits_on_the_same_column_in_both_tiers(self) -> None:
        """Switching to ASCII must not move the centre line."""
        assert TRIDENT[14].index("██") == TRIDENT_ASCII[10].index("||")

    def test_the_trident_is_symmetric(self) -> None:
        """A lopsided trishul reads as a mistake. Mirror each row and compare
        shapes (left/right half-block glyphs swap under reflection)."""
        swap = str.maketrans("▌▐", "▐▌")
        for row in TRIDENT:
            assert row == row[::-1].translate(swap), f"asymmetric row: {row!r}"

    def test_ascii_art_is_pure_ascii(self) -> None:
        for row in TRIDENT_ASCII + WORDMARK_ASCII:
            row.encode("ascii")


class TestGlyphSafety:
    def _glyphs(self, text: str) -> set[str]:
        return {c for c in strip(text) if ord(c) > 127}

    def test_banner_uses_only_consolas_glyphs(self) -> None:
        unknown = self._glyphs(render(theme(), version="1.0")) - CONSOLAS_SAFE
        assert not unknown, f"glyphs missing from Consolas: {sorted(unknown)}"

    def test_menu_uses_only_consolas_glyphs(self, capsys) -> None:
        from trisol import tui

        items = [
            tui.MenuItem("a", "Audit", "hint", lambda: None),
            tui.MenuItem("q", "Quit", "", None),
        ]
        tui._draw(theme(), items, 0, {"available": True})
        unknown = self._glyphs(capsys.readouterr().out) - CONSOLAS_SAFE
        assert not unknown, f"glyphs missing from Consolas: {sorted(unknown)}"

    def test_report_uses_only_consolas_glyphs(self, tmp_path) -> None:
        from trisol.audit import audit
        from trisol.report.terminal import render_report

        (tmp_path / "a.py").write_text(
            'SYSTEM_PROMPT = "You are helpful."\nOPTS = {"temperature": 1.5}\n',
            encoding="utf-8",
        )
        out = render_report(audit(tmp_path, offline=True), color=False, unicode=True)
        unknown = self._glyphs(out) - CONSOLAS_SAFE
        assert not unknown, f"glyphs missing from Consolas: {sorted(unknown)}"


class TestLayouts:
    def test_wide_terminal_puts_the_wordmark_beside_the_trident(self) -> None:
        lines = strip(render(theme(width=100))).splitlines()
        # Some row carries both a trident stroke and wordmark blocks.
        assert any("██" in line[:34] and "███" in line[34:] for line in lines)

    def test_narrow_terminal_stacks_them(self) -> None:
        lines = strip(render(theme(width=60))).splitlines()
        assert len(lines) > len(TRIDENT) + len(WORDMARK)
        assert all(display_width(line) <= 60 for line in lines)

    def test_very_narrow_terminal_falls_back_to_a_title(self) -> None:
        out = strip(render(theme(width=40)))
        assert "T R I S O L" in out
        assert all(display_width(line) <= 40 for line in out.splitlines())

    def test_nothing_overflows_a_standard_terminal(self) -> None:
        for line in strip(render(theme(width=80), version="1.0")).splitlines():
            assert display_width(line) <= 80

    def test_ascii_mode_is_ascii(self) -> None:
        render(theme(unicode=False), version="1.0").encode("ascii")

    def test_version_is_shown_when_given(self) -> None:
        assert "v9.9.9" in strip(render(theme(), version="9.9.9"))

    def test_tagline_is_present(self) -> None:
        assert "agent lifecycle verification" in strip(render(theme()))


class TestColour:
    def test_no_colour_means_no_escapes(self) -> None:
        assert "\x1b[" not in render(theme(color="none"))

    def test_truecolor_uses_24_bit_escapes(self) -> None:
        assert "\x1b[38;2;" in render(theme(color="truecolor"))

    def test_256_colour_uses_palette_escapes(self) -> None:
        out = render(theme(color="256"))
        assert "\x1b[38;5;" in out
        assert "\x1b[38;2;" not in out

    def test_gradient_runs_top_to_bottom(self) -> None:
        """The first and last trident rows must be different colours, or the
        gradient silently degraded to a flat fill."""
        rows = render(theme(color="truecolor")).splitlines()
        colours = [re.findall(r"38;2;(\d+;\d+;\d+)", r) for r in rows if "38;2;" in r]
        assert colours[0][0] != colours[-1][0]

    def test_overrides_force_plain_output(self) -> None:
        out = render(theme(color="truecolor"), color=False, unicode=False)
        assert "\x1b[" not in out
        out.encode("ascii")


class TestDisplayWidth:
    def test_ascii(self) -> None:
        assert display_width("abc") == 3

    def test_wide_glyphs_count_double(self) -> None:
        assert display_width("中文") == 4

    def test_combining_marks_are_zero_width(self) -> None:
        assert display_width("é") == 1
