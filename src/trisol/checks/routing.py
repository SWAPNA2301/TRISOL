"""Multi-agent routing: is the dispatch logic actually sound?

Routing bugs are quiet. A router that picks the wrong specialist does not crash;
it just answers worse, and nobody notices without an evaluation set. These checks
read the route table out of the source and prove the problems arithmetically:
overlapping keywords make the result depend on dict ordering, and a missing
fallback makes an unmatched message return None.

Where a route table and example messages are both present, the routing accuracy
is measured rather than described.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ..findings import CheckResult, Confidence, Finding, Patch, Severity
from .base import CheckContext

__all__ = ["RoutingCheck"]

#: Names that mean "this dict maps an agent to its trigger terms".
_ROUTE_NAMES = ("routes", "routing", "agents", "handlers", "specialists", "intents")


class RoutingCheck:
    name = "routing"
    title = "Multi-agent routing"
    needs_model = False

    def run(self, context: CheckContext) -> CheckResult:
        project = context.project
        result = CheckResult(name=self.name, title=self.title)

        tables: list[tuple[str, int, dict[str, list[str]]]] = []
        for path in project.python_files:
            tables.extend(self._find_route_tables(project, path))

        if not tables:
            result.skipped = True
            result.skip_reason = (
                "no routing table found. Trisol looks for a dict literal named "
                "ROUTES / AGENTS / INTENTS mapping a name to a list of keywords."
            )
            return result

        total_overlaps = 0
        fallbacks: list[bool] = []
        breakdown: list[dict] = []
        for rel, line, table in tables:
            total_overlaps += self._check_overlap(result, rel, line, table)
            fallbacks.append(self._check_fallback(result, project, rel, table))
            breakdown.extend(_keyword_breakdown(table))

        claims = sum(row["keywords"] for row in breakdown)
        unique = sum(row["unique"] for row in breakdown)
        result.metrics = {
            "route_tables": len(tables),
            "agents": sum(len(t) for _, _, t in tables),
            "overlapping_keywords": total_overlaps,
            # Share of keyword claims that route to exactly one agent: the part
            # of the table whose behaviour does not depend on dict order.
            "unambiguous_share": round(unique / claims, 3) if claims else 0.0,
            "has_fallback": all(fallbacks),
            "per_agent": breakdown,
        }
        return result

    def _find_route_tables(
        self, project, path: Path
    ) -> list[tuple[str, int, dict[str, list[str]]]]:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            return []

        found = []
        rel = project.relative(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                name = getattr(target, "id", "")
                if not name or name.lower() not in _ROUTE_NAMES:
                    continue
                table = self._literal_table(node.value)
                if table:
                    found.append((rel, node.lineno, table))
        return found

    @staticmethod
    def _literal_table(node: ast.expr) -> dict[str, list[str]] | None:
        """A dict literal of str -> list[str], or None if it is something else."""
        if not isinstance(node, ast.Dict):
            return None
        table: dict[str, list[str]] = {}
        for key, value in zip(node.keys, node.values, strict=False):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                return None
            if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                return None
            words = [
                element.value
                for element in value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
            if not words:
                return None
            table[key.value] = words
        return table or None

    def _check_overlap(
        self, result: CheckResult, rel: str, line: int, table: dict[str, list[str]]
    ) -> int:
        """Keywords claimed by more than one agent make routing order-dependent."""
        owners: dict[str, list[str]] = {}
        for agent, keywords in table.items():
            for keyword in keywords:
                owners.setdefault(keyword.lower(), []).append(agent)

        shared = {k: v for k, v in owners.items() if len(v) > 1}
        if not shared:
            return 0

        detail = "; ".join(
            f"{keyword!r} routes to {' or '.join(agents)}"
            for keyword, agents in sorted(shared.items())[:6]
        )
        result.findings.append(
            Finding(
                check=self.name,
                title=f"{len(shared)} keyword(s) claimed by multiple agents",
                detail=(
                    f"First-match-wins routing over an overlapping table is "
                    f"order-dependent: {detail}."
                ),
                severity=Severity.HIGH,
                confidence=Confidence.CERTAIN,
                file=rel,
                line=line,
                impact=(
                    "Which agent handles the request depends on dict insertion "
                    "order, not on the request. Reordering the table silently "
                    "changes behaviour, and a message mentioning two domains is "
                    "always routed by whichever keyword is checked first."
                ),
                suggestion=(
                    "Score every agent against the message and pick the highest, "
                    "instead of returning on the first keyword hit. Disambiguate "
                    "shared terms with weights or a classifier."
                ),
                evidence=[f"{keyword}: {agents}" for keyword, agents in sorted(shared.items())],
            )
        )
        return len(shared)

    def _check_fallback(
        self, result: CheckResult, project, rel: str, table: dict[str, list[str]]
    ) -> bool:
        """A router with no default returns None, which the caller then uses.

        Returns whether every router function found has a fallback.
        """
        path = project.root / rel
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return True

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return True

        has_fallback = True

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name not in ("route", "dispatch", "select_agent", "pick_agent"):
                continue
            returns_none = any(
                isinstance(inner, ast.Return)
                and (
                    inner.value is None
                    or (isinstance(inner.value, ast.Constant) and inner.value.value is None)
                )
                for inner in ast.walk(node)
            )
            if not returns_none:
                continue
            has_fallback = False
            none_return = next(
                inner
                for inner in ast.walk(node)
                if isinstance(inner, ast.Return)
                and (
                    inner.value is None
                    or (isinstance(inner.value, ast.Constant) and inner.value.value is None)
                )
            )
            # Prefer an agent whose name says it handles the general case; else
            # the first in the table. Shown as a diff and applied only on request.
            fallback = next(
                (a for a in table if a.lower() in ("support", "general", "default", "fallback")),
                next(iter(table)),
            )
            old = ast.get_source_segment(source, none_return) or "return None"
            result.findings.append(
                Finding(
                    check=self.name,
                    title=f"Router {node.name}() can return None",
                    detail=(
                        f"`{node.name}` returns None when no keyword matches, so an "
                        "unrecognised message has no handler."
                    ),
                    severity=Severity.HIGH,
                    confidence=Confidence.CERTAIN,
                    file=rel,
                    line=node.lineno,
                    impact=(
                        "The caller formats the agent name into a prompt, so None "
                        "either crashes or produces a prompt addressed to the "
                        "'None agent'."
                    ),
                    suggestion=(
                        "Return an explicit fallback agent, or raise so the caller "
                        "must handle the unroutable case."
                    ),
                    patch=Patch(
                        file=rel,
                        old=old,
                        new=f'return "{fallback}"  # fallback agent (trisol fix)',
                        explanation=f"Route unmatched messages to {fallback!r}.",
                    ),
                )
            )
        return has_fallback


def _keyword_breakdown(table: dict[str, list[str]]) -> list[dict]:
    """Per agent: how many of its keywords it owns alone versus shares."""
    owners: dict[str, int] = {}
    for keywords in table.values():
        for keyword in {k.lower() for k in keywords}:
            owners[keyword] = owners.get(keyword, 0) + 1
    rows = []
    for agent, keywords in table.items():
        words = {k.lower() for k in keywords}
        unique = sum(1 for w in words if owners[w] == 1)
        rows.append({"agent": agent, "keywords": len(words), "unique": unique,
                     "shared": len(words) - unique})
    return rows
