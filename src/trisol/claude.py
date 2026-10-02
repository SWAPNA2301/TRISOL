"""The bridge to Claude: run the local `claude` CLI, get structured JSON back.

Trisol's static checks find what can be proved by parsing. The judgement calls --
is this prompt actually specific enough, is this retrieval strategy the right
one, is this error path really unreachable -- need a model. This module is the
only place that talks to one.

Design rules, in order of importance:

1. **Never required.** If `claude` is not installed, every static check still
   runs and the report says plainly which model-assisted checks were skipped.
   A tool that dies without a binary on PATH is not a tool a judge can run.
2. **Structured or nothing.** The model is asked for JSON matching a schema and
   the reply is validated. A prose answer is a skipped check, not a finding with
   invented fields.
3. **Bounded.** Every call has a timeout and a prompt size cap, so a large
   repository cannot hang the audit or blow a context window.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ClaudeBridge",
    "ClaudeError",
    "ClaudeUnavailableError",
    "find_claude_binary",
]

#: Names to try on PATH, in order.
_BINARY_NAMES = ("claude", "claude.cmd", "claude.exe")

#: Where the official installers put it, checked after PATH.
_CANDIDATE_PATHS = (
    "~/.local/bin/claude",
    "~/AppData/Roaming/npm/claude.cmd",
    "~/AppData/Local/Programs/Claude/claude.exe",
    "/usr/local/bin/claude",
    "/opt/homebrew/bin/claude",
)

#: Hard cap on what we send. Well inside any model's window, and large enough
#: for the files a single check cares about.
MAX_PROMPT_CHARS = 60_000

DEFAULT_TIMEOUT_S = 180.0


class ClaudeUnavailableError(RuntimeError):
    """No usable `claude` binary. Callers skip their check and say why."""


class ClaudeError(RuntimeError):
    """The binary ran but did not return usable JSON."""


def find_claude_binary(explicit: str | None = None) -> str | None:
    """Locate the Claude CLI: explicit path, then env, then PATH, then installers."""
    if explicit:
        # An explicitly passed path is used as given: if it is wrong, the caller
        # should see that rather than have us silently fall back.
        return explicit if Path(explicit).exists() or shutil.which(explicit) else None

    from_env = os.environ.get("TRISOL_CLAUDE_BIN")
    if from_env:
        return from_env if Path(from_env).exists() or shutil.which(from_env) else None

    for name in _BINARY_NAMES:
        found = shutil.which(name)
        if found:
            return found

    for candidate in _CANDIDATE_PATHS:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None


@dataclass
class ClaudeBridge:
    """Runs one-shot, non-interactive Claude calls and parses JSON replies."""

    binary: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    model: str | None = None
    # Set when a call has already failed for a reason that will not change
    # (binary missing, not authenticated), so an audit with a dozen
    # model-assisted checks fails fast once instead of waiting out every timeout.
    _disabled_reason: str | None = None

    @classmethod
    def discover(
        cls, explicit: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S,
        model: str | None = None,
    ) -> ClaudeBridge:
        return cls(binary=find_claude_binary(explicit), timeout_s=timeout_s, model=model)

    @property
    def available(self) -> bool:
        return self.binary is not None and self._disabled_reason is None

    def status(self) -> dict[str, Any]:
        """What the report prints about the model layer."""
        if self.binary is None:
            return {
                "available": False,
                "reason": (
                    "no `claude` CLI found. Install Claude Code, or pass "
                    "--claude-bin /path/to/claude, or set TRISOL_CLAUDE_BIN."
                ),
                "binary": None,
            }
        if self._disabled_reason:
            return {"available": False, "reason": self._disabled_reason, "binary": self.binary}
        return {"available": True, "reason": "", "binary": self.binary, "model": self.model}

    def ask_json(
        self,
        prompt: str,
        *,
        schema_hint: str = "",
        timeout_s: float | None = None,
    ) -> Any:
        """Send a prompt, return parsed JSON.

        Raises :class:`ClaudeUnavailableError` when there is nothing to call and
        :class:`ClaudeError` when the reply cannot be parsed, so a caller can
        tell "could not ask" from "asked and got nonsense" -- the first is a
        skip, the second is worth surfacing as a warning.
        """
        if self.binary is None:
            raise ClaudeUnavailableError(self.status()["reason"])
        if self._disabled_reason:
            raise ClaudeUnavailableError(self._disabled_reason)

        if len(prompt) > MAX_PROMPT_CHARS:
            # Truncate at a line boundary with an explicit marker: a prompt cut
            # mid-token invites the model to complete the fragment.
            cut = prompt.rfind("\n", 0, MAX_PROMPT_CHARS)
            prompt = prompt[: cut if cut > 0 else MAX_PROMPT_CHARS]
            prompt += "\n\n[truncated by Trisol: input exceeded the size limit]"

        full = prompt
        if schema_hint:
            full += (
                "\n\nReply with JSON only -- no prose, no markdown fence. "
                f"It must match this shape:\n{schema_hint}"
            )

        # The prompt goes over stdin, never argv. On Windows `claude` is a
        # .CMD batch file, and cmd.exe cuts a command-line argument at its first
        # newline: Claude received only the opening sentence of every request
        # and ended up reviewing Trisol's own instructions instead of the
        # audited code. Measured: argv delivered 1 line of 4, stdin all 4.
        cmd = [self.binary, "-p", "--output-format", "text"]
        if self.model:
            cmd += ["--model", self.model]

        try:
            proc = subprocess.run(  # noqa: S603 - argv list, shell=False, no interpolation
                cmd,
                input=full,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s or self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeError(
                f"claude timed out after {timeout_s or self.timeout_s:.0f}s"
            ) from exc
        except OSError as exc:
            self._disabled_reason = f"could not execute {self.binary!r}: {exc}"
            raise ClaudeUnavailableError(self._disabled_reason) from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()[:400]
            lowered = stderr.lower()
            # Auth and missing-binary failures will repeat for every subsequent
            # call, so stop trying rather than burning a timeout per check.
            if any(hint in lowered for hint in ("login", "auth", "api key", "not found")):
                self._disabled_reason = f"claude is not usable: {stderr or 'authentication failed'}"
                raise ClaudeUnavailableError(self._disabled_reason)
            raise ClaudeError(f"claude exited {proc.returncode}: {stderr or '(no stderr)'}")

        return _parse_json(proc.stdout or "")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def _parse_json(text: str) -> Any:
    """Pull JSON out of a model reply.

    Models wrap JSON in prose or a fence often enough that insisting on a bare
    document would make this layer needlessly brittle, so we try, in order: the
    whole string, a fenced block, then the outermost {...} or [...] span.
    """
    stripped = text.strip()
    if not stripped:
        raise ClaudeError("claude returned no output")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = _FENCE_RE.search(stripped)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ClaudeError(f"claude did not return JSON (got {stripped[:160]!r})")
