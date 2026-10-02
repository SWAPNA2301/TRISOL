"""The database half of the data check.

Opening someone else's database is the most invasive thing Trisol does, so the
first test here is that it cannot write to it.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from trisol.checks.base import CheckContext, run_check
from trisol.checks.data import DataCheck
from trisol.discover import discover


def _audit(root):
    return run_check(DataCheck(), CheckContext(discover(root), offline=True))


def _titles(result) -> list[str]:
    return [f.title for f in result.findings]


@pytest.fixture
def db(tmp_path):
    def build(statements: list[str], name: str = "memory.db"):
        path = tmp_path / name
        connection = sqlite3.connect(path)
        try:
            for statement in statements:
                connection.execute(statement)
            connection.commit()
        finally:
            connection.close()
        return path

    return build


class TestDatabaseInspection:
    def test_opens_read_only(self, tmp_path, db) -> None:
        """The audit must never modify the target's data."""
        path = db(["CREATE TABLE notes (id INTEGER, body TEXT)"])
        before = path.read_bytes()
        _audit(tmp_path)
        assert path.read_bytes() == before

    def test_reports_a_database_with_no_tables(self, tmp_path, db) -> None:
        db([])
        assert "Database has no tables" in _titles(_audit(tmp_path))

    def test_accepts_a_small_table_without_an_index(self, tmp_path, db) -> None:
        """An index only matters once a table is big enough to scan slowly."""
        statements = ["CREATE TABLE notes (id INTEGER, body TEXT)"]
        statements += [f"INSERT INTO notes VALUES ({i}, 'x')" for i in range(5)]
        db(statements)
        assert not any("no index" in t for t in _titles(_audit(tmp_path)))

    def test_flags_a_large_table_with_no_index(self, tmp_path) -> None:
        path = tmp_path / "memory.db"
        connection = sqlite3.connect(path)
        try:
            connection.execute("CREATE TABLE notes (id INTEGER, body TEXT)")
            connection.executemany(
                "INSERT INTO notes VALUES (?, ?)",
                [(i, "body") for i in range(1200)],
            )
            connection.commit()
        finally:
            connection.close()
        assert any("no index" in t for t in _titles(_audit(tmp_path)))

    def test_accepts_a_large_indexed_table(self, tmp_path) -> None:
        path = tmp_path / "memory.db"
        connection = sqlite3.connect(path)
        try:
            connection.execute("CREATE TABLE notes (id INTEGER, body TEXT)")
            connection.executemany(
                "INSERT INTO notes VALUES (?, ?)",
                [(i, "body") for i in range(1200)],
            )
            connection.execute("CREATE INDEX idx_notes_id ON notes(id)")
            connection.commit()
        finally:
            connection.close()
        assert not any("no index" in t for t in _titles(_audit(tmp_path)))

    def test_reports_a_corrupt_database(self, tmp_path) -> None:
        (tmp_path / "broken.db").write_bytes(b"this is not a database at all")
        result = _audit(tmp_path)
        # Either it will not open, or it opens with nothing in it; both are
        # reported rather than crashing the check.
        assert result.findings or result.skipped

    def test_a_table_name_with_a_quote_cannot_break_the_query(self, tmp_path) -> None:
        """Identifiers cannot be bound as parameters, so they are quoted. A name
        containing a quote character must still be handled safely."""
        path = tmp_path / "memory.db"
        connection = sqlite3.connect(path)
        try:
            connection.execute('CREATE TABLE "odd""name" (id INTEGER)')
            connection.commit()
        finally:
            connection.close()
        result = _audit(tmp_path)  # must not raise
        assert not result.skipped


class TestCorpusShapes:
    """A corpus is stored in several shapes in the wild; all must be read."""

    def _write(self, tmp_path, payload):
        path = tmp_path / "knowledge" / "docs.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_a_bare_list_of_strings(self, tmp_path) -> None:
        self._write(tmp_path, ["first document text", "second document text"])
        assert _audit(tmp_path).metrics["documents"] == 2

    def test_a_documents_key(self, tmp_path) -> None:
        self._write(tmp_path, {"documents": [{"text": "a"}, {"text": "b"}]})
        assert _audit(tmp_path).metrics["documents"] == 2

    def test_an_id_to_document_mapping(self, tmp_path) -> None:
        self._write(tmp_path, {"d1": {"text": "a"}, "d2": {"text": "b"}})
        assert _audit(tmp_path).metrics["documents"] == 2

    def test_page_content_key_from_langchain(self, tmp_path) -> None:
        self._write(tmp_path, [{"page_content": "langchain style document"}])
        assert _audit(tmp_path).metrics["documents"] == 1

    def test_an_empty_corpus_is_reported(self, tmp_path) -> None:
        self._write(tmp_path, [])
        assert "Knowledge corpus is empty" in _titles(_audit(tmp_path))

    def test_mostly_short_fragments_are_flagged(self, tmp_path) -> None:
        self._write(tmp_path, [{"text": "tiny"} for _ in range(6)])
        assert any("very short" in t for t in _titles(_audit(tmp_path)))
