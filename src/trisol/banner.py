"""The Trisol mark: a trishul beside a block wordmark.

Drawn only with glyphs Consolas actually contains (``█ ▀ ▄ ▌ ▐`` and the
double box-drawing set). Consolas is VS Code's default terminal font on Windows
and has no quadrant blocks; any glyph it lacks is substituted from another font
at a different width, which is what makes terminal art look broken.

Layouts, chosen by terminal width:

* wide   -- trident left, wordmark and tagline right (needs ~80 columns)
* stacked -- trident above a centred wordmark
* plain  -- the same composition in pure ASCII for consoles without Unicode
"""

from __future__ import annotations

import unicodedata

from .theme import BRONZE, GOLD_BOTTOM, GOLD_TOP, MUTED, Theme, detect, lerp

__all__ = [
    "TAGLINE",
    "TRIDENT",
    "TRIDENT_ASCII",
    "WORDMARK",
    "WORDMARK_ASCII",
    "display_width",
    "render",
]

TRIDENT = (
    "▐▌            ▐▌            ▐▌",
    "██            ██            ██",
    "██           ▐██▌           ██",
    "██           ████           ██",
    "██           ████           ██",
    "▐█▌          ▐██▌          ▐█▌",
    " ██           ██           ██ ",
    " ▐█▄          ██          ▄█▌ ",
    "  ▀██▄▄▄▄▄▄▄▄▄██▄▄▄▄▄▄▄▄▄██▀  ",
    "     ▀▀▀▀▀▀▀▀▀██▀▀▀▀▀▀▀▀▀     ",
    "             ▄██▄             ",
    "             ████             ",
    "             ▀██▀             ",
    "              ██              ",
    "              ██              ",
    "              ██              ",
    "              ██              ",
    "             ▄██▄             ",
    "             ▀▀▀▀             ",
)

TRIDENT_ASCII = (
    '/\\            /\\            /\\',
    '||           /  \\           ||',
    '||           |  |           ||',
    '||           |  |           ||',
    '||           \\  /           ||',
    ' \\\\           ||           // ',
    '  \\\\          ||          //  ',
    "   '==========++=========='   ",
    '             [==]             ',
    '              ||              ',
    '              ||              ',
    '              ||              ',
    "             '--'             ",
)

WORDMARK = (
    "████████╗██████╗ ██╗███████╗ ██████╗ ██╗     ",
    "╚══██╔══╝██╔══██╗██║██╔════╝██╔═══██╗██║     ",
    "   ██║   ██████╔╝██║███████╗██║   ██║██║     ",
    "   ██║   ██╔══██╗██║╚════██║██║   ██║██║     ",
    "   ██║   ██║  ██║██║███████║╚██████╔╝███████╗",
    "   ╚═╝   ╚═╝  ╚═╝╚═╝╚══════╝ ╚═════╝ ╚══════╝",
)

WORDMARK_ASCII = (
    " _____ ____  ___ ____   ___  _     ",
    "|_   _|  _ \\|_ _/ ___| / _ \\| |    ",
    "  | | | |_) || |\\___ \\| | | | |    ",
    "  | | |  _ < | | ___) | |_| | |___ ",
    "  |_| |_| \\_\\___|____/ \\___/|_____|",
)

TAGLINE = "agent lifecycle verification"
SUBLINE_UNICODE = "test  ·  benchmark  ·  fix  ·  verify"
SUBLINE_ASCII = "test  -  benchmark  -  fix  -  verify"

_SHADOW = frozenset("╗╔═║╚╝")
_GAP = "    "


def display_width(text: str) -> int:
    """Printable width: wide East Asian glyphs count as two columns and
    combining marks as zero."""
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def _shade_row(theme: Theme, row: str, t: float) -> str:
    """Colour one row: gold gradient for solid strokes, bronze for the
    box-drawing 'shadow' that gives the wordmark its depth."""
    if not theme.colored:
        return row
    gold = lerp(GOLD_TOP, GOLD_BOTTOM, t)
    out: list[str] = []
    run = ""
    run_shadow: bool | None = None
    for char in row:
        if char == " ":
            if run:
                out.append(theme.fg(BRONZE if run_shadow else gold, run))
                run = ""
            out.append(char)
            run_shadow = None
            continue
        shadow = char in _SHADOW
        if run and shadow != run_shadow:
            out.append(theme.fg(BRONZE if run_shadow else gold, run))
            run = ""
        run += char
        run_shadow = shadow
    if run:
        out.append(theme.fg(BRONZE if run_shadow else gold, run))
    return "".join(out)


def _text_row(theme: Theme, content: str) -> str:
    """Tagline in gold, version faint, everything else muted."""
    if content == TAGLINE:
        return theme.gold(content)
    if content.startswith("v") and content[1:2].isdigit():
        return theme.faint(content)
    return theme.fg(MUTED, content)


def _side_by_side(
    theme: Theme, trident: tuple[str, ...], word: tuple[str, ...], text: list[str]
) -> list[str]:
    """Trident on the left; wordmark and tagline beside its crossbar, where the
    eye lands first."""
    top = 3 if theme.unicode else 2
    right: list[str] = [""] * len(trident)
    for i, row in enumerate(word):
        if top + i < len(right):
            right[top + i] = _shade_row(theme, row, i / max(1, len(word) - 1))
    for i, row in enumerate(text):
        index = top + len(word) + i
        if index < len(right) and row:
            right[index] = _text_row(theme, row)

    width = max(display_width(r) for r in trident)
    lines = []
    for index, row in enumerate(trident):
        left = _shade_row(theme, row.ljust(width), index / max(1, len(trident) - 1))
        lines.append(("  " + left + _GAP + right[index]).rstrip())
    return lines


def _stacked(
    theme: Theme, trident: tuple[str, ...], word: tuple[str, ...], text: list[str]
) -> list[str]:
    """Trident above the wordmark, both centred. On a terminal too narrow for
    the 45-column wordmark, a spaced title replaces it rather than wrapping."""
    lines = []
    trident_w = max(display_width(r) for r in trident)
    pad = " " * max(0, (theme.width - trident_w) // 2)
    for index, row in enumerate(trident):
        lines.append((pad + _shade_row(theme, row, index / max(1, len(trident) - 1))).rstrip())
    lines.append("")

    word_w = max(display_width(r) for r in word)
    if theme.width >= word_w + 2:
        word_pad = " " * max(0, (theme.width - word_w) // 2)
        for index, row in enumerate(word):
            shade = index / max(1, len(word) - 1)
            lines.append((word_pad + _shade_row(theme, row, shade)).rstrip())
    else:
        title = "T R I S O L"
        lines.append(" " * max(0, (theme.width - len(title)) // 2) + theme.gold(title))

    for row in text:
        lines.append(_text_row(theme, row.center(theme.width).rstrip()) if row else "")
    return lines


def render(
    theme: Theme | None = None,
    *,
    version: str = "",
    unicode: bool | None = None,
    color: bool | None = None,
) -> str:
    """The banner as one string, no trailing newline.

    ``unicode``/``color`` override detection, for --no-unicode / --no-color.
    """
    theme = theme or detect()
    if unicode is not None or color is not None:
        theme = Theme(
            color="none" if color is False else theme.color,
            unicode=theme.unicode if unicode is None else unicode,
            width=theme.width,
        )

    trident = TRIDENT if theme.unicode else TRIDENT_ASCII
    word = WORDMARK if theme.unicode else WORDMARK_ASCII
    subline = SUBLINE_UNICODE if theme.unicode else SUBLINE_ASCII
    text = ["", TAGLINE, subline, *(["", f"v{version}"] if version else [])]

    needed = display_width(trident[0]) + len(_GAP) + display_width(word[0]) + 4
    build = _side_by_side if theme.width >= needed else _stacked
    return "\n".join(build(theme, trident, word, text))
