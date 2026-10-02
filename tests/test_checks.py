"""The checks, against code written to be found and code written to be clean.

Every check gets both: a flawed sample it must flag, and a correct sample it must
stay silent on. The second half matters more -- a linter that cries wolf gets
turned off, so each check has an explicit no-false-positive test.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from trisol.checks.base import CheckContext, run_check
from trisol.checks.benchmark import BenchmarkCheck
from trisol.checks.data import DataCheck
from trisol.checks.prompts import PromptCheck
from trisol.checks.reliability import ReliabilityCheck
from trisol.checks.routing import RoutingCheck
from trisol.discover import discover
from trisol.findings import Severity


def write(root, name: str, source: str):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    return path


def audit_with(check, root) -> list:
    result = run_check(check, CheckContext(discover(root), offline=True))
    assert not result.skipped or result.skip_reason, "a skip must explain itself"
    return result


def titles(result) -> list[str]:
    return [f.title for f in result.findings]


# ----------------------------------------------------------------- reliability


class TestReliability:
    def test_flags_a_call_with_no_timeout_or_handler(self, tmp_path) -> None:
        write(
            tmp_path,
            "agent.py",
            """
            import urllib.request, json
            OLLAMA = "http://127.0.0.1:11434/api/generate"
            def ask(prompt):
                return urllib.request.urlopen(OLLAMA, data=prompt.encode())
            """,
        )
        found = titles(audit_with(ReliabilityCheck(), tmp_path))
        assert "Model call has no timeout" in found
        assert "Model call is not error-handled" in found

    def test_stays_silent_on_a_guarded_call(self, tmp_path) -> None:
        write(
            tmp_path,
            "agent.py",
            """
            import urllib.request
            OLLAMA = "http://127.0.0.1:11434/api/generate"
            def ask(prompt):
                try:
                    return urllib.request.urlopen(OLLAMA, data=b"x", timeout=30)
                except OSError:
                    return None
            """,
        )
        found = titles(audit_with(ReliabilityCheck(), tmp_path))
        assert "Model call has no timeout" not in found
        assert "Model call is not error-handled" not in found

    def test_flags_an_unbounded_loop(self, tmp_path) -> None:
        write(
            tmp_path,
            "loop.py",
            """
            def run(model):
                while True:
                    reply = model()
                    if "TOOL:" not in reply:
                        return reply
            """,
        )
        assert "Unbounded agent loop" in titles(audit_with(ReliabilityCheck(), tmp_path))

    def test_accepts_a_loop_with_an_iteration_cap(self, tmp_path) -> None:
        """The conditional return is the same shape as the flagged case; the
        difference is the explicit bound, which is what proves termination."""
        write(
            tmp_path,
            "loop.py",
            """
            def run(model):
                turns = 0
                while True:
                    turns += 1
                    if turns > 8:
                        raise RuntimeError("too many turns")
                    reply = model()
                    if "TOOL:" not in reply:
                        return reply
            """,
        )
        assert "Unbounded agent loop" not in titles(audit_with(ReliabilityCheck(), tmp_path))

    def test_accepts_a_loop_with_an_unconditional_break(self, tmp_path) -> None:
        write(
            tmp_path,
            "loop.py",
            """
            def run():
                while True:
                    work()
                    break
            """,
        )
        assert "Unbounded agent loop" not in titles(audit_with(ReliabilityCheck(), tmp_path))

    def test_flags_high_temperature_with_a_patch(self, tmp_path) -> None:
        write(tmp_path, "cfg.py", 'OPTS = {"temperature": 1.5}\n')
        result = audit_with(ReliabilityCheck(), tmp_path)
        finding = next(f for f in result.findings if "temperature" in f.title)
        assert finding.fixable
        assert finding.patch is not None
        assert "0.0" in finding.patch.new

    def test_accepts_a_deterministic_temperature(self, tmp_path) -> None:
        write(tmp_path, "cfg.py", 'OPTS = {"temperature": 0.0}\n')
        assert not any("temperature" in t for t in titles(audit_with(ReliabilityCheck(), tmp_path)))

    def test_flags_a_hardcoded_credential(self, tmp_path) -> None:
        write(tmp_path, "keys.py", 'api_key = "sk-liveAABBCCDDEEFF112233"\n')
        result = audit_with(ReliabilityCheck(), tmp_path)
        assert any(f.severity is Severity.CRITICAL for f in result.findings)

    def test_accepts_a_credential_read_from_the_environment(self, tmp_path) -> None:
        write(tmp_path, "keys.py", 'import os\napi_key = os.environ.get("KEY", "")\n')
        assert titles(audit_with(ReliabilityCheck(), tmp_path)) == []

    def test_reports_a_file_that_does_not_parse(self, tmp_path) -> None:
        write(tmp_path, "broken.py", "def oops(:\n    pass\n")
        result = audit_with(ReliabilityCheck(), tmp_path)
        assert "File does not parse" in titles(result)
        assert result.findings[0].severity is Severity.CRITICAL

    def test_flags_unchecked_division(self, tmp_path) -> None:
        write(tmp_path, "tool.py", "def divide(a, b):\n    return a / b\n")
        assert "Division by an unchecked value" in titles(audit_with(ReliabilityCheck(), tmp_path))

    def test_accepts_division_by_a_literal(self, tmp_path) -> None:
        write(tmp_path, "tool.py", "def half(a):\n    return a / 2\n")
        assert titles(audit_with(ReliabilityCheck(), tmp_path)) == []


# --------------------------------------------------------------------- prompts


class TestPrompts:
    def test_flags_a_thin_prompt(self, tmp_path) -> None:
        write(tmp_path, "a.py", 'SYSTEM_PROMPT = "You are a helpful assistant. Be nice."\n')
        assert any("Thin system prompt" in t for t in titles(audit_with(PromptCheck(), tmp_path)))

    def test_accepts_a_specific_prompt(self, tmp_path) -> None:
        write(
            tmp_path,
            "a.py",
            '''
            SYSTEM_PROMPT = (
                "You are a billing support agent for an online store. "
                "Answer only from the context provided below. If the context does "
                "not contain the answer, reply exactly: I don't know. "
                "Respond in at most three sentences of plain text, with no markdown."
            )
            ''',
        )
        assert titles(audit_with(PromptCheck(), tmp_path)) == []

    def test_flags_a_retrieval_prompt_with_no_grounding_rule(self, tmp_path) -> None:
        write(
            tmp_path,
            "rag.py",
            '''
            SYSTEM_PROMPT = (
                "You are a documentation assistant. Use the material supplied to "
                "produce a thorough, well-organised explanation for the reader."
            )
            def retrieve(query, docs):
                context = [d for d in docs if query in d]
                return context
            ''',
        )
        assert "No grounding rule in a retrieval prompt" in titles(
            audit_with(PromptCheck(), tmp_path)
        )

    def test_no_grounding_finding_without_retrieval(self, tmp_path) -> None:
        """The same prompt in a non-retrieval agent is not this defect."""
        write(
            tmp_path,
            "chat.py",
            '''
            SYSTEM_PROMPT = (
                "You are a documentation assistant. Use the material supplied to "
                "produce a thorough, well-organised explanation for the reader."
            )
            ''',
        )
        assert "No grounding rule in a retrieval prompt" not in titles(
            audit_with(PromptCheck(), tmp_path)
        )

    def test_skips_with_a_reason_when_there_are_no_prompts(self, tmp_path) -> None:
        write(tmp_path, "empty.py", "x = 1\n")
        result = run_check(PromptCheck(), CheckContext(discover(tmp_path), offline=True))
        assert result.skipped
        assert "prompt" in result.skip_reason.lower()

    def test_records_why_the_model_was_not_consulted(self, tmp_path) -> None:
        """Offline must be visible in the output, not silently thinner."""
        write(tmp_path, "a.py", 'SYSTEM_PROMPT = "You are a helpful assistant. Be nice."\n')
        result = audit_with(PromptCheck(), tmp_path)
        assert result.metrics["model_reviewed"] is False
        assert result.metrics["model_skipped_reason"]


# ------------------------------------------------------------------------ data


class TestData:
    def _corpus(self, tmp_path, docs) -> None:
        path = tmp_path / "knowledge" / "docs.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(docs), encoding="utf-8")

    def test_flags_duplicates_and_empties(self, tmp_path) -> None:
        self._corpus(
            tmp_path,
            [
                {"id": "1", "text": "Refunds are issued within 30 days of purchase."},
                {"id": "2", "text": "Refunds are issued within 30 days of purchase."},
                {"id": "3", "text": ""},
            ],
        )
        found = titles(audit_with(DataCheck(), tmp_path))
        assert any("duplicate" in t for t in found)
        assert any("empty" in t for t in found)

    def test_accepts_a_healthy_corpus(self, tmp_path) -> None:
        self._corpus(
            tmp_path,
            [
                {"id": "1", "text": "Refunds are issued within thirty days of purchase."},
                {"id": "2", "text": "Standard shipping takes three to five working days."},
                {"id": "3", "text": "Passwords reset via the link on the sign-in page."},
            ],
        )
        assert titles(audit_with(DataCheck(), tmp_path)) == []

    def test_flags_an_oversized_chunk(self, tmp_path) -> None:
        self._corpus(tmp_path, [{"id": "1", "text": "word " * 600}])
        assert any("over" in t for t in titles(audit_with(DataCheck(), tmp_path)))

    def test_reports_unparseable_json(self, tmp_path) -> None:
        path = tmp_path / "knowledge" / "docs.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        assert "Knowledge corpus will not parse" in titles(audit_with(DataCheck(), tmp_path))

    def test_skips_with_a_reason_when_there_is_no_data(self, tmp_path) -> None:
        write(tmp_path, "a.py", "x = 1\n")
        result = run_check(DataCheck(), CheckContext(discover(tmp_path), offline=True))
        assert result.skipped
        assert "corpus" in result.skip_reason.lower()


# --------------------------------------------------------------------- routing


class TestRouting:
    def test_flags_overlapping_keywords(self, tmp_path) -> None:
        write(
            tmp_path,
            "router.py",
            '''
            ROUTES = {
                "billing": ["invoice", "account"],
                "support": ["broken", "account"],
            }
            def route(message):
                for agent, words in ROUTES.items():
                    for word in words:
                        if word in message:
                            return agent
                return None
            ''',
        )
        found = titles(audit_with(RoutingCheck(), tmp_path))
        assert any("claimed by multiple agents" in t for t in found)
        assert any("can return None" in t for t in found)

    def test_accepts_a_disjoint_table_with_a_fallback(self, tmp_path) -> None:
        write(
            tmp_path,
            "router.py",
            '''
            ROUTES = {
                "billing": ["invoice", "payment"],
                "support": ["broken", "login"],
            }
            def route(message):
                for agent, words in ROUTES.items():
                    for word in words:
                        if word in message:
                            return agent
                return "support"
            ''',
        )
        assert titles(audit_with(RoutingCheck(), tmp_path)) == []

    def test_skips_with_a_reason_when_there_is_no_router(self, tmp_path) -> None:
        write(tmp_path, "a.py", "x = 1\n")
        result = run_check(RoutingCheck(), CheckContext(discover(tmp_path), offline=True))
        assert result.skipped
        assert "routing" in result.skip_reason.lower()


# ------------------------------------------------------------------- benchmark


class TestBenchmark:
    def _corpus(self, tmp_path, docs) -> None:
        path = tmp_path / "knowledge" / "docs.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(docs), encoding="utf-8")

    def test_measures_every_strategy_on_the_real_corpus(self, tmp_path) -> None:
        self._corpus(
            tmp_path,
            [
                {"title": "Refunds", "text": "Customers receive reimbursement within thirty days."},
                {"title": "Shipping", "text": "Courier delivery completes in three working days."},
                {"title": "Passwords", "text": "Credentials reset through the verification link."},
                {
                    "title": "Warranty",
                    "text": "Hardware carries a biennial manufacturer guarantee.",
                },
            ],
        )
        write(
            tmp_path,
            "agent.py",
            '''
            def retrieve(query, docs):
                return [d for d in docs if query.lower() in d["text"].lower()]
            ''',
        )
        result = audit_with(BenchmarkCheck(), tmp_path)
        rows = result.metrics["results"]
        assert {r["name"] for r in rows} == {
            "substring", "keyword overlap", "char 4-gram", "BM25",
        }
        # Recall is a proportion, and MRR cannot exceed it.
        for row in rows:
            assert 0.0 <= row["recall_at_k"] <= 1.0
            assert 0.0 <= row["mrr"] <= row["recall_at_k"] + 1e-9

    def test_identifies_substring_as_the_implemented_strategy(self, tmp_path) -> None:
        self._corpus(
            tmp_path,
            [
                {"title": "Refunds", "text": "Customers receive reimbursement within thirty days."},
                {"title": "Shipping", "text": "Courier delivery completes in three working days."},
                {"title": "Passwords", "text": "Credentials reset through the verification link."},
            ],
        )
        write(
            tmp_path,
            "agent.py",
            '''
            def retrieve(query, docs):
                return [d for d in docs if query.lower() in d["text"].lower()]
            ''',
        )
        assert audit_with(BenchmarkCheck(), tmp_path).metrics["baseline"] == "substring"

    def test_prefers_bm25_when_accuracy_ties(self, tmp_path) -> None:
        """Several strategies score the same on a small corpus; recommending
        whichever came first in a list would be arbitrary."""
        self._corpus(
            tmp_path,
            [
                {"title": "Refunds", "text": "Customers receive reimbursement within thirty days."},
                {"title": "Shipping", "text": "Courier delivery completes in three working days."},
                {"title": "Passwords", "text": "Credentials reset through the verification link."},
                {
                    "title": "Warranty",
                    "text": "Hardware carries a biennial manufacturer guarantee.",
                },
            ],
        )
        write(
            tmp_path,
            "agent.py",
            "def retrieve(q, docs):\n"
            '    return [d for d in docs if q.lower() in d["text"].lower()]\n',
        )
        assert audit_with(BenchmarkCheck(), tmp_path).metrics["best"] == "BM25"

    def test_skips_a_corpus_too_small_to_measure(self, tmp_path) -> None:
        self._corpus(tmp_path, [{"text": "only one document here"}])
        result = run_check(BenchmarkCheck(), CheckContext(discover(tmp_path), offline=True))
        assert result.skipped
        assert "3" in result.skip_reason


# ------------------------------------------------------------- check protocol


class TestCheckProtocol:
    def test_a_crashing_check_is_contained(self, tmp_path) -> None:
        """One broken check must not abandon the rest of the audit."""

        class Exploding:
            name = "boom"
            title = "Always fails"
            needs_model = False

            def run(self, context):
                raise RuntimeError("deliberate")

        result = run_check(Exploding(), CheckContext(discover(tmp_path), offline=True))
        assert result.skipped
        assert "RuntimeError" in result.skip_reason
        assert "deliberate" in result.skip_reason

    def test_every_check_times_itself(self, tmp_path) -> None:
        write(tmp_path, "a.py", "x = 1\n")
        context = CheckContext(discover(tmp_path), offline=True)
        for check in (ReliabilityCheck(), PromptCheck(), DataCheck(), RoutingCheck()):
            assert run_check(check, context).duration_ms >= 0


@pytest.mark.parametrize(
    "check",
    [ReliabilityCheck(), PromptCheck(), DataCheck(), RoutingCheck(), BenchmarkCheck()],
    ids=lambda c: c.name,
)
def test_check_survives_an_empty_directory(check, tmp_path) -> None:
    """Pointed at nothing, a check skips with a reason -- it never crashes and
    never reports a clean pass it did not earn."""
    result = run_check(check, CheckContext(discover(tmp_path), offline=True))
    assert result.findings == []
    if result.skipped:
        assert result.skip_reason
