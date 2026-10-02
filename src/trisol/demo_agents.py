"""Find and copy the bundled demo agents."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DEMO_ROOT", "DemoAgent", "copy_demos", "list_demos"]

DEMO_ROOT = Path(__file__).parent / "demos"

#: One line each, shown in the menu and `trisol demo --list`.
_DESCRIPTIONS = {
    "rag_support_bot": "RAG support bot: vague prompt, substring retrieval, unguarded API call",
    "tool_agent": "Tool-calling agent: unbounded loop, unhandled tool errors, temperature 1.4",
    "router_agent": "Multi-agent router: overlapping keywords, no fallback agent",
}


@dataclass(frozen=True)
class DemoAgent:
    name: str
    path: Path
    description: str


def list_demos() -> list[DemoAgent]:
    if not DEMO_ROOT.is_dir():
        return []
    return [
        DemoAgent(entry.name, entry, _DESCRIPTIONS.get(entry.name, ""))
        for entry in sorted(DEMO_ROOT.iterdir())
        if entry.is_dir() and not entry.name.startswith(("_", "."))
    ]


def copy_demos(destination: Path | str, *, overwrite: bool = False) -> list[Path]:
    """Copy every demo into ``destination``. Returns the created folders.

    Refuses to overwrite an existing folder unless asked, because a user may
    have applied fixes there and a silent copy would undo their work.
    """
    target = Path(destination).resolve()
    target.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for demo in list_demos():
        out = target / demo.name
        if out.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{out} already exists. Pass --force to replace it, or choose "
                    "another folder."
                )
            shutil.rmtree(out)
        shutil.copytree(demo.path, out, ignore=shutil.ignore_patterns("__pycache__"))
        created.append(out)
    return created
