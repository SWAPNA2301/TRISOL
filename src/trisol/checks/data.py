"""Data and knowledge-base health.

An agent is only as good as what it retrieves from. These checks open the corpus
and the database the agent actually uses and measure them: duplicate documents,
empty records, chunks too long to be useful, a schema with no index on the column
being queried.

Everything here is measured from real files, so the findings are CERTAIN. A
corpus that cannot be found is a skip, not a silent pass.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from ..findings import CheckResult, Confidence, Finding, Severity
from .base import CheckContext

__all__ = ["DataCheck"]

#: Corpus filenames worth opening, in preference order.
_CORPUS_GLOBS = (
    "**/knowledge/*.json",
    "**/docs.json",
    "**/documents.json",
    "**/corpus.json",
    "**/data/*.json",
    "**/*_docs.json",
)

#: A retrieval chunk longer than this wastes context and dilutes the embedding:
#: one vector has to represent too many ideas at once.
_LONG_CHUNK_CHARS = 2000

#: Shorter than this and a chunk rarely carries a whole fact.
_SHORT_CHUNK_CHARS = 50


class DataCheck:
    name = "data"
    title = "Data and knowledge base"
    needs_model = False
    _affected = 0

    def run(self, context: CheckContext) -> CheckResult:
        project = context.project
        result = CheckResult(name=self.name, title=self.title)

        corpora = self._find_corpora(project.root)
        databases = sorted(project.root.rglob("*.db")) + sorted(project.root.rglob("*.sqlite"))
        databases = [d for d in databases if ".trisol" not in d.parts][:5]

        if not corpora and not databases:
            result.skipped = True
            result.skip_reason = (
                "no knowledge corpus (knowledge/*.json, docs.json, ...) and no "
                "SQLite database found under the target."
            )
            return result

        total_docs = 0
        # Documents with at least one defect, counted once each, for the scorecard.
        self._affected = 0
        for path in corpora:
            total_docs += self._check_corpus(result, project, path)

        for path in databases:
            self._check_database(result, project, path)

        result.metrics = {
            "corpus_files": len(corpora),
            "documents": total_docs,
            "documents_with_issues": self._affected,
            "databases": len(databases),
        }
        return result

    @staticmethod
    def _find_corpora(root: Path) -> list[Path]:
        found: list[Path] = []
        for pattern in _CORPUS_GLOBS:
            for path in root.glob(pattern):
                if path.is_file() and path not in found and ".trisol" not in path.parts:
                    found.append(path)
        return found[:10]

    def _check_corpus(self, result: CheckResult, project, path: Path) -> int:
        rel = project.relative(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result.findings.append(
                Finding(
                    check=self.name,
                    title="Knowledge corpus will not parse",
                    detail=f"{rel} is not readable JSON: {exc}",
                    severity=Severity.HIGH,
                    confidence=Confidence.CERTAIN,
                    file=rel,
                    impact="Retrieval has nothing to search, so every answer is ungrounded.",
                    suggestion="Fix the JSON, or point the loader at the right file.",
                )
            )
            return 0

        docs = _as_documents(raw)
        if not docs:
            result.findings.append(
                Finding(
                    check=self.name,
                    title="Knowledge corpus is empty",
                    detail=f"{rel} contains no usable documents.",
                    severity=Severity.HIGH,
                    confidence=Confidence.CERTAIN,
                    file=rel,
                    impact="Retrieval always returns nothing.",
                    suggestion="Populate the corpus or correct the path.",
                )
            )
            return 0

        texts: list[str] = []
        empty = 0
        long_chunks: list[int] = []
        short_chunks: list[int] = []
        seen: dict[str, int] = {}
        duplicates: list[tuple[int, int]] = []

        for index, doc in enumerate(docs):
            text = _document_text(doc)
            if not text.strip():
                empty += 1
                self._affected += 1
                continue
            texts.append(text)
            if len(text) > _LONG_CHUNK_CHARS:
                long_chunks.append(index)
                self._affected += 1
            elif len(text) < _SHORT_CHUNK_CHARS:
                short_chunks.append(index)
            digest = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()
            if digest in seen:
                duplicates.append((seen[digest], index))
                self._affected += 1
            else:
                seen[digest] = index

        if empty:
            result.findings.append(
                Finding(
                    check=self.name,
                    title=f"{empty} empty document(s) in the corpus",
                    detail=f"{rel} has {empty} entries with no text.",
                    severity=Severity.MEDIUM,
                    confidence=Confidence.CERTAIN,
                    file=rel,
                    impact=(
                        "Empty documents occupy a retrieval slot that a real "
                        "document could have used, and embed to a meaningless vector."
                    ),
                    suggestion="Remove them, or fill in the missing text.",
                )
            )

        if duplicates:
            pairs = ", ".join(f"#{a}=#{b}" for a, b in duplicates[:5])
            result.findings.append(
                Finding(
                    check=self.name,
                    title=f"{len(duplicates)} duplicate document(s)",
                    detail=f"{rel} contains identical texts ({pairs}).",
                    severity=Severity.MEDIUM,
                    confidence=Confidence.CERTAIN,
                    file=rel,
                    impact=(
                        "Duplicates crowd out distinct documents in the top-k, so "
                        "the model sees the same fact repeated instead of more context."
                    ),
                    suggestion="Deduplicate by content hash when loading the corpus.",
                )
            )

        if long_chunks:
            result.findings.append(
                Finding(
                    check=self.name,
                    title=f"{len(long_chunks)} document(s) over {_LONG_CHUNK_CHARS} chars",
                    detail=(
                        f"{rel} has documents of up to "
                        f"{max(len(t) for t in texts)} characters."
                    ),
                    severity=Severity.MEDIUM,
                    confidence=Confidence.CERTAIN,
                    file=rel,
                    impact=(
                        "One embedding has to represent many separate ideas, which "
                        "blurs similarity, and each retrieved chunk eats context."
                    ),
                    suggestion=(
                        f"Split documents into ~{_LONG_CHUNK_CHARS // 2}-character "
                        "chunks on paragraph boundaries."
                    ),
                )
            )

        if short_chunks and len(short_chunks) > len(docs) // 2:
            result.findings.append(
                Finding(
                    check=self.name,
                    title="Most documents are very short",
                    detail=(
                        f"{len(short_chunks)} of {len(docs)} documents in {rel} are "
                        f"under {_SHORT_CHUNK_CHARS} characters."
                    ),
                    severity=Severity.LOW,
                    confidence=Confidence.CERTAIN,
                    file=rel,
                    impact="Fragments rarely contain a complete answer.",
                    suggestion="Merge related fragments into whole statements.",
                )
            )

        return len(docs)

    def _check_database(self, result: CheckResult, project, path: Path) -> None:
        rel = project.relative(path)
        try:
            # read-only URI: auditing must never modify the target's data.
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            result.findings.append(
                Finding(
                    check=self.name,
                    title="Database will not open",
                    detail=f"{rel}: {exc}",
                    severity=Severity.MEDIUM,
                    confidence=Confidence.CERTAIN,
                    file=rel,
                    impact="Anything the agent stores here is unavailable.",
                    suggestion="Check the file is a valid SQLite database.",
                )
            )
            return

        try:
            tables = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'"
                )
            ]
            if not tables:
                result.findings.append(
                    Finding(
                        check=self.name,
                        title="Database has no tables",
                        detail=f"{rel} contains no user tables.",
                        severity=Severity.LOW,
                        confidence=Confidence.CERTAIN,
                        file=rel,
                        impact="The agent's store is empty; reads return nothing.",
                        suggestion="Confirm the schema is created before first use.",
                    )
                )
            for table in tables[:20]:
                self._check_table(result, connection, rel, table)
        except sqlite3.Error as exc:
            # SQLite opens lazily, so a corrupt or non-database file only fails
            # on the first real query. Swallowing that silently reported a
            # broken store as healthy, which is the one outcome this check must
            # never produce. Findings gathered before the failure are kept.
            result.findings.append(
                Finding(
                    check=self.name,
                    title="Database could not be read",
                    detail=f"{rel}: {exc}",
                    severity=Severity.MEDIUM,
                    confidence=Confidence.CERTAIN,
                    file=rel,
                    impact=(
                        "Whatever the agent stores here is unreadable, so any "
                        "step that depends on it fails at run time."
                    ),
                    suggestion=(
                        "Check the file is a valid SQLite database and is not "
                        "truncated or partially written."
                    ),
                )
            )
        finally:
            connection.close()

    def _check_table(
        self, result: CheckResult, connection: sqlite3.Connection, rel: str, table: str
    ) -> None:
        # Table names come from sqlite_master, not user input, but they still
        # cannot be bound as parameters, so they are quoted defensively.
        quoted = '"' + table.replace('"', '""') + '"'
        try:
            # S608: `quoted` is a table name read from sqlite_master and
            # escaped above. SQLite cannot bind an identifier as a parameter,
            # so quoting is the only available defence, and no user input
            # reaches this string.
            query = f"SELECT COUNT(*) FROM {quoted}"  # noqa: S608
            count = connection.execute(query).fetchone()[0]
            indexes = connection.execute(f"PRAGMA index_list({quoted})").fetchall()
        except sqlite3.Error:
            return

        if count > 1000 and not indexes:
            result.findings.append(
                Finding(
                    check=self.name,
                    title=f"Table {table!r} has {count} rows and no index",
                    detail=(
                        f"{rel}: every query against {table!r} scans all {count} rows."
                    ),
                    severity=Severity.MEDIUM,
                    confidence=Confidence.CERTAIN,
                    file=rel,
                    impact=(
                        "Lookup latency grows with the table, and an agent that "
                        "reads memory on every step pays it on every step."
                    ),
                    suggestion="Add an index on the column the agent filters by.",
                )
            )


def _as_documents(raw: Any) -> list[Any]:
    """Normalise the shapes a corpus is usually stored in."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("documents", "docs", "data", "items", "chunks"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
        # A mapping of id -> document is also common.
        if raw and all(isinstance(v, (dict, str)) for v in raw.values()):
            return list(raw.values())
    return []


def _document_text(doc: Any) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in ("text", "content", "body", "page_content", "chunk"):
            value = doc.get(key)
            if isinstance(value, str):
                return value
    return ""
