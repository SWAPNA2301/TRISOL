"""Terminal capabilities and the colour palette.

Every visual surface (banner, menu, report) asks this module what the terminal
can do rather than guessing, so one decision governs colour depth and glyphs
everywhere and a piped or redirected run never receives escape codes.

Colour depth, best first:

* truecolor -- Windows Terminal, VS Code, iTerm2, most modern emulators
* 256       -- older xterm-compatible terminals
* none      -- not a TTY, NO_COLOR set, TERM=dumb, or --no-color
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from typing import TextIO

__all__ = ["RGB", "Theme", "detect", "enable_windows_vt"]

RGB = tuple[int, int, int]

# The palette. Gold for the mark and emphasis, bronze for depth, a cool grey for
# secondary text so the gold never has to compete with it.
GOLD_TOP: RGB = (255, 214, 92)
GOLD_BOTTOM: RGB = (196, 112, 18)
BRONZE: RGB = (122, 84, 30)
TEXT: RGB = (220, 220, 220)
MUTED: RGB = (140, 140, 140)
FAINT: RGB = (95, 95, 95)
GREEN: RGB = (98, 196, 120)
RED: RGB = (232, 92, 80)
AMBER: RGB = (232, 160, 60)
CYAN: RGB = (90, 196, 210)
SELECT_BG: RGB = (58, 46, 18)


def enable_windows_vt(stream: TextIO | None = None) -> bool:
    """Turn on ANSI escape processing in a classic Windows console.

    Windows Terminal and VS Code already interpret escapes; the legacy console
    host does not until ENABLE_VIRTUAL_TERMINAL_PROCESSING is set, and without
    it every colour prints as literal `←[38;2;...m` noise. Harmless elsewhere.
    """
    if sys.platform != "win32":
        return True
    with contextlib.suppress(Exception):
        import ctypes  # noqa: PLC0415 - Windows-only

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            return True
    return False


@dataclass
class Theme:
    color: str  # "truecolor" | "256" | "none"
    unicode: bool
    width: int

    # -- colour primitives -------------------------------------------------

    def fg(self, rgb: RGB, text: str) -> str:
        if self.color == "none" or not text:
            return text
        if self.color == "truecolor":
            return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}\033[0m"
        return f"\033[38;5;{_to_256(rgb)}m{text}\033[0m"

    def bg(self, rgb: RGB, text: str) -> str:
        if self.color == "none" or not text:
            return text
        if self.color == "truecolor":
            return f"\033[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}\033[0m"
        return f"\033[48;5;{_to_256(rgb)}m{text}\033[0m"

    def highlight(self, rgb: RGB, text: str) -> str:
        """Background across already-coloured text.

        A plain bg() wrap is cancelled by the first `\\033[0m` inside the text,
        so a selection bar stopped after its first coloured segment. Re-applying
        the background after every reset keeps the bar continuous.
        """
        if self.color == "none" or not text:
            return text
        if self.color == "truecolor":
            code = f"\033[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
        else:
            code = f"\033[48;5;{_to_256(rgb)}m"
        return code + text.replace("\033[0m", "\033[0m" + code) + "\033[0m"

    def bold(self, text: str) -> str:
        return text if self.color == "none" else f"\033[1m{text}\033[0m"

    # -- semantic helpers --------------------------------------------------

    def gold(self, text: str) -> str:
        return self.fg(GOLD_TOP, text)

    def muted(self, text: str) -> str:
        return self.fg(MUTED, text)

    def faint(self, text: str) -> str:
        return self.fg(FAINT, text)

    def ok(self, text: str) -> str:
        return self.fg(GREEN, text)

    def bad(self, text: str) -> str:
        return self.fg(RED, text)

    def warn(self, text: str) -> str:
        return self.fg(AMBER, text)

    @property
    def colored(self) -> bool:
        return self.color != "none"


def _to_256(rgb: RGB) -> int:
    """Nearest xterm-256 colour cube index."""
    r, g, b = (round(c / 255 * 5) for c in rgb)
    return 16 + 36 * r + 6 * g + b


def lerp(a: RGB, b: RGB, t: float) -> RGB:
    t = min(1.0, max(0.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def _supports_unicode(stream: TextIO) -> bool:
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        "█▀▄▌▐═║╔╗╚╝─│╭╮╰╯›".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _color_depth(stream: TextIO) -> str:
    if os.environ.get("NO_COLOR"):
        return "none"
    if not hasattr(stream, "isatty") or not stream.isatty():
        return "none"
    if os.environ.get("TERM") == "dumb":
        return "none"
    colorterm = os.environ.get("COLORTERM", "").lower()
    if colorterm in ("truecolor", "24bit"):
        return "truecolor"
    # Windows Terminal, VS Code and the modern console all do 24-bit once VT
    # processing is on; WT_SESSION / TERM_PROGRAM identify the first two.
    if os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM") in ("vscode", "iTerm.app"):
        return "truecolor"
    if sys.platform == "win32" and enable_windows_vt(stream):
        return "truecolor"
    return "256"


def detect(
    stream: TextIO | None = None,
    *,
    no_color: bool = False,
    no_unicode: bool = False,
) -> Theme:
    target = stream if stream is not None else sys.stdout
    try:
        width = os.get_terminal_size(target.fileno()).columns
    except (OSError, AttributeError, ValueError):
        width = int(os.environ.get("COLUMNS", "100") or 100)
    return Theme(
        color="none" if no_color else _color_depth(target),
        unicode=False if no_unicode else _supports_unicode(target),
        width=max(40, width),
    )
