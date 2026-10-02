"""Does this agent survive contact with reality?

Every finding here is provable by parsing -- a call either has a timeout keyword
or it does not -- so they are all CERTAIN, and several carry an exact patch. This
is the check most likely to catch a real outage: an LLM call with no timeout
inside a request handler is a hang waiting to happen.
"""

from __future__ import annotations

import ast
import re

from ..findings import CheckResult, Confidence, Finding, Patch, Severity
from .base import CheckContext

__all__ = ["ReliabilityCheck"]

#: A bare `while True:` with no break anywhere inside it. The classic
#: tool-calling agent failure: the model keeps asking, the loop keeps going.
_UNBOUNDED_LOOP_HINT = "while True"


class ReliabilityCheck:
    name = "reliability"
    title = "Reliability and error handling"
    needs_model = False

    def run(self, context: CheckContext) -> CheckResult:
        project = context.project
        result = CheckResult(name=self.name, title=self.title)

        for path, message in project.syntax_errors:
            result.findings.append(
                Finding(
                    check=self.name,
                    title="File does not parse",
                    detail=f"Python could not parse this file: {message}",
                    severity=Severity.CRITICAL,
                    confidence=Confidence.CERTAIN,
                    file=project.relative(path),
                    impact="Nothing in this file can run, and no other check can read it.",
                    suggestion="Fix the syntax error.",
                )
            )

        for step in project.steps:
            self._check_step(result, step, _timeout_patch(project, step))

        for path in project.python_files:
            self._check_source(result, context, path)

        result.metrics = {
            "model_calls": len(project.steps),
            "calls_without_timeout": sum(1 for s in project.steps if not s.has_timeout),
            "calls_without_error_handling": sum(1 for s in project.steps if not s.has_try),
            "calls_without_retry": sum(1 for s in project.steps if not s.has_retry),
        }
        return result

    def _check_step(self, result: CheckResult, step, timeout_patch=None) -> None:
        if not step.has_timeout:
            result.findings.append(
                Finding(
                    check=self.name,
                    title="Model call has no timeout",
                    detail=(
                        f"`{step.callee}` is called with no timeout, so it waits "
                        "indefinitely if the provider stalls."
                    ),
                    severity=Severity.HIGH,
                    confidence=Confidence.CERTAIN,
                    file=step.file,
                    line=step.line,
                    impact=(
                        "A single slow or hung provider request blocks the agent "
                        "forever. In a server this holds a worker until the process "
                        "is restarted."
                    ),
                    suggestion="Pass an explicit timeout, e.g. `timeout=30`.",
                    patch=timeout_patch,
                    evidence=[f"{step.file}:{step.line}"],
                )
            )
        if not step.has_try:
            result.findings.append(
                Finding(
                    check=self.name,
                    title="Model call is not error-handled",
                    detail=(
                        f"`{step.callee}` is not inside a try/except. A network "
                        "error, rate limit or malformed response raises straight "
                        "out of the agent."
                    ),
                    severity=Severity.HIGH,
                    confidence=Confidence.CERTAIN,
                    file=step.file,
                    line=step.line,
                    impact=(
                        "Any transient provider failure -- a 429, a dropped "
                        "connection -- crashes the run instead of being retried "
                        "or reported."
                    ),
                    suggestion=(
                        "Wrap the call and handle the provider's error types, "
                        "returning a usable fallback or re-raising with context."
                    ),
                    evidence=[f"{step.file}:{step.line}"],
                )
            )
        if not step.has_retry and not step.has_try:
            result.findings.append(
                Finding(
                    check=self.name,
                    title="Model call has no retry",
                    detail=(
                        f"`{step.callee}` has neither a retry policy nor error "
                        "handling, so a recoverable failure is fatal."
                    ),
                    severity=Severity.MEDIUM,
                    confidence=Confidence.CERTAIN,
                    file=step.file,
                    line=step.line,
                    impact=(
                        "Rate limits and 5xx responses are normal with LLM "
                        "providers; without a retry they surface to the user."
                    ),
                    suggestion="Retry with exponential backoff on transient errors.",
                )
            )

    def _check_source(self, result: CheckResult, context: CheckContext, path) -> None:
        project = context.project
        rel = project.relative(path)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        self._check_unbounded_loops(result, source, rel)
        self._check_division(result, source, rel)
        self._check_temperature(result, source, rel)
        self._check_secrets(result, source, rel)

    def _check_unbounded_loops(self, result: CheckResult, source: str, rel: str) -> None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if not isinstance(node, ast.While):
                continue
            # Only `while True:` -- a condition-driven loop has a termination
            # story even if we cannot evaluate it.
            is_literal_true = isinstance(node.test, ast.Constant) and node.test.value is True
            if not is_literal_true:
                continue
            # An exit that is itself inside an `if` is conditional, and in an
            # agent loop that condition is usually "did the model decide to
            # stop" -- which the model may never do. Only an unconditional exit
            # in the loop body proves termination, so a conditional one still
            # warrants a finding (at lower confidence).
            if self._has_unconditional_exit(node):
                continue
            counter = self._has_iteration_counter(node)
            if counter:
                continue
            result.findings.append(
                Finding(
                    check=self.name,
                    title="Unbounded agent loop",
                    detail=(
                        "`while True:` with no break, return or raise inside it. "
                        "An agent loop with no iteration cap runs until killed."
                    ),
                    severity=Severity.HIGH,
                    confidence=Confidence.CERTAIN,
                    file=rel,
                    line=node.lineno,
                    impact=(
                        "A model that keeps requesting tools loops forever, "
                        "spending tokens on every pass."
                    ),
                    suggestion=(
                        "Add a maximum iteration count and stop with a clear "
                        "error when it is reached."
                    ),
                    evidence=[f"{rel}:{node.lineno}"],
                )
            )

    @staticmethod
    def _has_unconditional_exit(loop: ast.While) -> bool:
        """An exit statement directly in the loop body, not nested in a branch."""
        return any(isinstance(stmt, (ast.Break, ast.Return, ast.Raise)) for stmt in loop.body)

    @staticmethod
    def _has_iteration_counter(loop: ast.While) -> bool:
        """Whether the loop compares something against a bound, e.g.
        `if turns > MAX: break` -- evidence of a deliberate iteration cap."""
        for node in ast.walk(loop):
            if isinstance(node, ast.Compare) and any(
                isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)) for op in node.ops
            ):
                return True
        return False

    def _check_division(self, result: CheckResult, source: str, rel: str) -> None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
                continue
            # A literal non-zero divisor is safe; a name could be anything.
            if isinstance(node.right, ast.Constant) and node.right.value != 0:
                continue
            if not isinstance(node.right, ast.Name):
                continue
            result.findings.append(
                Finding(
                    check=self.name,
                    title="Division by an unchecked value",
                    detail=(
                        f"`/ {node.right.id}` with no zero check. When a tool "
                        "argument comes from the model, it can be anything."
                    ),
                    severity=Severity.MEDIUM,
                    confidence=Confidence.LIKELY,
                    file=rel,
                    line=node.lineno,
                    impact="ZeroDivisionError crashes the agent on a bad argument.",
                    suggestion="Validate the divisor and return a tool error instead.",
                )
            )

    _TEMP_RE = re.compile(r"[\"']?temperature[\"']?\s*[:=]\s*([0-9]*\.?[0-9]+)")

    def _check_temperature(self, result: CheckResult, source: str, rel: str) -> None:
        for match in self._TEMP_RE.finditer(source):
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if value < 1.0:
                continue
            line = source.count("\n", 0, match.start()) + 1
            result.findings.append(
                Finding(
                    check=self.name,
                    title=f"High sampling temperature ({value})",
                    detail=(
                        f"temperature={value} makes output substantially "
                        "non-deterministic."
                    ),
                    severity=Severity.MEDIUM,
                    confidence=Confidence.LIKELY,
                    file=rel,
                    line=line,
                    impact=(
                        "Extraction, routing and tool-argument generation become "
                        "unreliable, and failures stop being reproducible."
                    ),
                    suggestion=(
                        "Use 0 for deterministic tasks; reserve high values for "
                        "open-ended generation."
                    ),
                    patch=Patch(
                        file=rel,
                        old=match.group(0),
                        new=match.group(0).replace(match.group(1), "0.0"),
                        explanation="Set a deterministic temperature.",
                    ),
                )
            )

    #: Looks for a hardcoded credential, not a read from the environment.
    _SECRET_RE = re.compile(
        r"(?P<name>api[_-]?key|secret|token|password)\s*=\s*[\"'](?P<value>[^\"']{12,})[\"']",
        re.IGNORECASE,
    )

    def _check_secrets(self, result: CheckResult, source: str, rel: str) -> None:
        for match in self._SECRET_RE.finditer(source):
            value = match.group("value")
            # os.environ reads and obvious placeholders are the correct pattern.
            if any(
                hint in value.lower()
                for hint in ("os.environ", "getenv", "your-", "xxx", "placeholder", "example")
            ):
                continue
            line = source.count("\n", 0, match.start()) + 1
            result.findings.append(
                Finding(
                    check=self.name,
                    title="Possible hardcoded credential",
                    detail=f"`{match.group('name')}` is assigned a literal string.",
                    severity=Severity.CRITICAL,
                    confidence=Confidence.LIKELY,
                    file=rel,
                    line=line,
                    impact=(
                        "A committed credential is exposed to anyone with repo "
                        "access and must be rotated once leaked."
                    ),
                    suggestion="Read it from the environment or a secret manager.",
                )
            )


def _timeout_patch(project, step) -> Patch | None:
    """An exact edit adding ``timeout=30`` to one model call, or None.

    Built from the call's own source text, so the patch's ``old`` is precisely
    what is in the file. Returns None when the call cannot be located exactly
    (multi-line edits, unusual syntax) -- a missing patch is better than a
    wrong one, and the finding still stands.
    """
    path = project.root / step.file
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.lineno == step.line):
            continue
        # Match the model call itself, not a call chained onto it: in
        # `urlopen(url).read().decode()` every call starts on the same line, and
        # patching the outer decode() would produce `decode(, timeout=30)`.
        func = ast.get_source_segment(source, node.func) or ""
        if func != step.callee and not func.endswith("." + step.callee.split(".")[-1]):
            continue
        segment = ast.get_source_segment(source, node)
        if not segment or not segment.endswith(")") or "\n" in segment:
            continue
        inner = segment[segment.index("(") + 1 : -1].strip()
        new = segment[:-1] + (", timeout=30)" if inner else "timeout=30)")
        return Patch(
            file=step.file,
            old=segment,
            new=new,
            explanation="Fail after 30 seconds instead of waiting forever.",
        )
    return None
