"""The interactive front door: `trisol` with no arguments.

A full-screen menu driven by arrow keys (or number keys), redrawn in place.
Standard library only -- ``msvcrt`` on Windows, ``termios`` elsewhere -- so the
tool keeps its zero-dependency promise.

When stdin is not an interactive keyboard (piped, CI, an IDE "run" panel),
the menu degrades to a numbered prompt read with ``input()``; it never hangs
waiting for a keypress that cannot arrive.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .banner import render as render_banner
from .theme import AMBER, CYAN, GOLD_TOP, GREEN, SELECT_BG, TEXT, Theme

__all__ = [
    "MenuItem",
    "confirm",
    "pause",
    "prompt_choice",
    "prompt_path",
    "read_key",
    "run_menu",
]


@dataclass(frozen=True)
class MenuItem:
    key: str
    label: str
    hint: str
    action: Callable[[], object] | None  # None means quit


# ----------------------------------------------------------------- keyboard


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


_WINDOWS_KEYS = {"H": "up", "P": "down", "K": "left", "M": "right"}
_ANSI_KEYS = {"A": "up", "B": "down", "C": "right", "D": "left"}


def _decode(char: str) -> str:
    """Name for a single non-arrow character, shared by both platforms."""
    if char in ("\r", "\n"):
        return "enter"
    if char == "\x1b":
        return "esc"
    if char == "\x03":
        raise KeyboardInterrupt
    return char


# Branching on sys.platform rather than os.name: mypy understands the former,
# so each platform's half is type-checked only where its modules exist.
if sys.platform == "win32":
    import msvcrt

    def read_key() -> str:
        """One keypress as a name: 'up', 'down', 'enter', 'esc', or the character."""
        char = msvcrt.getwch()
        if char in ("\x00", "\xe0"):  # arrows arrive as a two-key pair
            return _WINDOWS_KEYS.get(msvcrt.getwch(), "")
        return _decode(char)

else:
    import termios
    import tty

    def read_key() -> str:
        """One keypress as a name: 'up', 'down', 'enter', 'esc', or the character."""
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            char = sys.stdin.read(1)
            if char == "\x1b" and sys.stdin.read(1) == "[":
                # Arrow keys send ESC [ A..D.
                return _ANSI_KEYS.get(sys.stdin.read(1), "")
            return _decode(char)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _clear() -> None:
    # Clear screen and scrollback position, then home the cursor.
    sys.stdout.write("\033[2J\033[3J\033[H")
    sys.stdout.flush()


# ---------------------------------------------------------------- drawing


def _box(theme: Theme, rows: list[str], width: int) -> list[str]:
    """Wrap pre-rendered rows in a rounded frame. ``rows`` may contain colour
    escapes, so padding is computed from each row's visible width."""
    tl, tr, bl, br, h, v = (
        ("╭", "╮", "╰", "╯", "─", "│") if theme.unicode else ("+", "+", "+", "+", "-", "|")
    )
    border = theme.faint
    out = [border(tl + h * (width - 2) + tr)]
    for row in rows:
        pad = width - 4 - _visible_len(row)
        out.append(border(v) + " " + row + " " * max(0, pad) + " " + border(v))
    out.append(border(bl + h * (width - 2) + br))
    return out


def _visible_len(text: str) -> int:
    length = 0
    skipping = False
    for char in text:
        if char == "\033":
            skipping = True
        elif skipping:
            if char == "m":
                skipping = False
        else:
            length += 1
    return length


def _status_line(theme: Theme, claude_status: dict) -> str:
    dot = "●" if theme.unicode else "*"
    hollow = "○" if theme.unicode else "o"
    if claude_status.get("available"):
        claude = theme.fg(GREEN, dot) + " " + theme.fg(TEXT, "Claude connected")
    else:
        claude = theme.fg(AMBER, hollow) + " " + theme.muted(
            "Claude not found - static checks only"
        )
    py = f"Python {sys.version_info.major}.{sys.version_info.minor}"
    sep = theme.faint("   |   ")
    return claude + sep + theme.muted(py) + sep + theme.muted(f"v{__version__}")


def _draw(theme: Theme, items: list[MenuItem], selected: int, claude_status: dict) -> None:
    _clear()
    print()
    print(render_banner(theme, version=""))
    print()

    width = min(max(theme.width - 4, 60), 86)
    pointer = "►" if theme.unicode else ">"
    label_w = max(len(i.label) for i in items) + 2

    rows: list[str] = [""]
    for index, item in enumerate(items):
        number = f"{index + 1}" if index < 9 else "0"
        if index == selected:
            body = (
                theme.fg(GOLD_TOP, f" {pointer} ")
                + theme.bold(theme.fg(GOLD_TOP, f"{number}  {item.label.ljust(label_w)}"))
                + theme.fg(TEXT, item.hint)
            )
            visible = _visible_len(body)
            body = theme.highlight(SELECT_BG, body + " " * max(0, width - 4 - visible))
        else:
            body = (
                "   "
                + theme.faint(f"{number}  ")
                + theme.fg(TEXT, item.label.ljust(label_w))
                + theme.muted(item.hint)
            )
        rows.append(body)
        rows.append("")

    for line in _box(theme, rows, width):
        print("  " + line)

    up_down = "↑ ↓" if theme.unicode else "up/down"
    print()
    print(
        "  "
        + theme.fg(CYAN, up_down)
        + theme.muted(" move    ")
        + theme.fg(CYAN, "enter")
        + theme.muted(" select    ")
        + theme.fg(CYAN, f"1-{min(len(items), 9)}")
        + theme.muted(" jump    ")
        + theme.fg(CYAN, "q")
        + theme.muted(" quit")
    )
    print()
    print("  " + _status_line(theme, claude_status))
    sys.stdout.flush()


# ------------------------------------------------------------------ loop


def run_menu(
    theme: Theme,
    items: list[MenuItem],
    claude_status: dict,
) -> int:
    """Show the menu until the user quits. Returns a process exit code."""
    if not _interactive():
        return _run_numbered(theme, items, claude_status)

    selected = 0
    while True:
        _draw(theme, items, selected, claude_status)
        try:
            key = read_key()
        except KeyboardInterrupt:
            print()
            return 0

        if key in ("up", "k"):
            selected = (selected - 1) % len(items)
        elif key in ("down", "j", "\t"):
            selected = (selected + 1) % len(items)
        elif key.isdigit():
            index = (int(key) - 1) if key != "0" else 9
            if 0 <= index < len(items):
                selected = index
                if _activate(theme, items[selected]):
                    return 0
        elif key == "enter":
            if _activate(theme, items[selected]):
                return 0
        elif key in ("q", "Q", "esc"):
            _clear()
            return 0


def _activate(theme: Theme, item: MenuItem) -> bool:
    """Run an item. True means the user chose to quit."""
    if item.action is None:
        _clear()
        return True
    _clear()
    print()
    print("  " + theme.bold(theme.gold(item.label)))
    print("  " + theme.muted(item.hint))
    print()
    try:
        item.action()
    except KeyboardInterrupt:
        print("\n  " + theme.muted("cancelled"))
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print("\n  " + theme.bad("error: ") + str(exc))
    pause(theme)
    return False


def _run_numbered(theme: Theme, items: list[MenuItem], claude_status: dict) -> int:
    """Fallback for a non-interactive stdin: print once, read a number."""
    print(render_banner(theme, version=__version__))
    print()
    for index, item in enumerate(items, start=1):
        print(f"  {index}. {item.label.ljust(26)} {item.hint}")
    print()
    print("  " + _status_line(theme, claude_status))
    print()
    try:
        choice = input(f"  choose 1-{len(items)}: ").strip()
    except (EOFError, KeyboardInterrupt):
        return 0
    if not choice.isdigit() or not 1 <= int(choice) <= len(items):
        return 0
    item = items[int(choice) - 1]
    if item.action is not None:
        item.action()
    return 0


# --------------------------------------------------------------- prompts


def prompt_path(theme: Theme, label: str, default: str = ".", must_be: str = "any") -> Path:
    """Ask for a path, with a default, and validate it exists.

    ``must_be`` is "file", "dir" or "any". Quotes are stripped because Windows
    "Copy as path" wraps paths in them.
    """
    while True:
        shown = theme.muted(f" [{default}]") if default else ""
        try:
            raw = input("  " + theme.fg(TEXT, label) + shown + theme.gold(" › ")).strip()
        except EOFError:
            raw = ""
        raw = raw.strip('"').strip("'") or default
        path = Path(raw).expanduser()
        if not path.exists():
            print("  " + theme.bad("not found: ") + str(path))
            continue
        if must_be == "file" and not path.is_file():
            print("  " + theme.bad("that is a folder - choose a single .py file"))
            continue
        if must_be == "dir" and not path.is_dir():
            print("  " + theme.bad("that is a file - choose a folder"))
            continue
        return path.resolve()


def prompt_choice(theme: Theme, label: str, options: list[tuple[str, str]]) -> int | None:
    """Numbered choice from a short list. Returns the index, or None to cancel."""
    for index, (name, hint) in enumerate(options, start=1):
        print(
            "   " + theme.gold(f"{index}") + "  " + theme.fg(TEXT, name.ljust(18))
            + theme.muted(hint)
        )
    print()
    try:
        raw = input("  " + theme.fg(TEXT, label) + theme.gold(" › ")).strip()
    except EOFError:
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return int(raw) - 1
    return None


def confirm(theme: Theme, question: str, default: bool = False) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        raw = input("  " + theme.fg(TEXT, question) + theme.muted(suffix) + " ").strip().lower()
    except EOFError:
        return default
    if not raw:
        return default
    return raw in ("y", "yes")


def pause(theme: Theme) -> None:
    if not _interactive():
        return
    print()
    print("  " + theme.muted("press any key to return to the menu"), end="", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        read_key()

