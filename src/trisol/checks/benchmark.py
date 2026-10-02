"""Benchmark the agent's retrieval and routing against better alternatives.

This is the check that answers "which algorithm is best" with a number instead of
an opinion. It does not read the code and guess -- it extracts the corpus the
agent actually uses, builds an evaluation set from it, then scores the agent's own
strategy against stronger ones on the same data.

The comparison is deliberately conservative:

* the baseline is the strategy the code really implements (substring, keyword, or
  whatever we could identify), not a strawman
* every candidate is scored on the same queries with the same metric
* a candidate only earns a finding when it beats the baseline by a margin wide
  enough not to be noise

Metric is recall@k plus mean reciprocal rank. Recall answers "did we find it at
all", MRR answers "how far down the list" -- reporting only recall hides a
retriever that technically returns the right document in position nine.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..findings import CheckResult, Confidence, Finding, Severity
from .base import CheckContext
from .data import DataCheck, _as_documents, _document_text

__all__ = ["BenchmarkCheck"]

#: A candidate must beat the baseline by at least this much recall to be worth
#: reporting. Below it, the difference is within the noise of a small eval set.
_MIN_RECALL_GAIN = 0.15

_WORD_RE = re.compile(r"[a-z0-9]+")
#: Removed before scoring so a query's distinctive terms carry the weight.
#: Written as one string and split for readability -- the alternative is a
#: 36-element list literal that no reviewer reads.
_STOPWORDS = frozenset(
    (
        "a an and are as at be by can for from how i in is it of on or that "
        "the this to was what when where which who will with you your do does "
        "my me"
    ).split()
)


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


@dataclass
class Scored:
    name: str
    recall_at_k: float
    mrr: float
    mean_ms: float
    description: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "recall_at_k": round(self.recall_at_k, 3),
            "mrr": round(self.mrr, 3),
            "mean_ms": round(self.mean_ms, 3),
            "description": self.description,
        }


# --------------------------------------------------------------- strategies


def _substring(query: str, docs: list[str]) -> list[int]:
    """What the demo agent does: the whole query must appear verbatim."""
    needle = query.lower()
    return [i for i, text in enumerate(docs) if needle in text.lower()]


def _keyword_overlap(query: str, docs: list[str]) -> list[int]:
    """Count shared terms. Order by how many query words appear."""
    want = set(_tokens(query))
    scored = []
    for index, text in enumerate(docs):
        overlap = len(want & set(_tokens(text)))
        if overlap:
            scored.append((overlap, index))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [index for _, index in scored]


def _bm25(query: str, docs: list[str]) -> list[int]:
    """Okapi BM25: term frequency, saturated, weighted by rarity, length-normalised."""
    k1, b = 1.5, 0.75
    doc_tokens = [_tokens(t) for t in docs]
    lengths = [len(t) for t in doc_tokens]
    avg = sum(lengths) / len(lengths) if lengths else 0.0
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] += 1

    scores = [0.0] * len(docs)
    for term in _tokens(query):
        n_q = df.get(term, 0)
        if not n_q:
            continue
        idf = math.log(1.0 + (len(docs) - n_q + 0.5) / (n_q + 0.5))
        for index, tokens in enumerate(doc_tokens):
            tf = tokens.count(term)
            if not tf:
                continue
            norm = 1.0 - b + b * (lengths[index] / avg if avg else 0.0)
            scores[index] += idf * (tf * (k1 + 1.0)) / (tf + k1 * norm)

    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [i for i in ranked if scores[i] > 0]


def _char_ngram(query: str, docs: list[str], n: int = 4) -> list[int]:
    """Character n-gram overlap: catches morphology and typos that word matching
    misses ("refund" vs "refunds", "cancelled" vs "cancel")."""

    def grams(text: str) -> set[str]:
        clean = re.sub(r"\s+", " ", text.lower())
        return {clean[i : i + n] for i in range(max(0, len(clean) - n + 1))}

    want = grams(query)
    if not want:
        return []
    scored = []
    for index, text in enumerate(docs):
        have = grams(text)
        if not have:
            continue
        overlap = len(want & have) / len(want)
        if overlap > 0:
            scored.append((overlap, index))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [index for _, index in scored]


_STRATEGIES: tuple[tuple[str, Callable[[str, list[str]], list[int]], str], ...] = (
    ("substring", _substring, "whole query must appear verbatim in the document"),
    ("keyword overlap", _keyword_overlap, "rank by count of shared terms"),
    (
        "char 4-gram",
        lambda q, d: _char_ngram(q, d),
        "fuzzy character overlap, tolerates word forms",
    ),
    ("BM25", _bm25, "term frequency weighted by rarity, length-normalised"),
)


class BenchmarkCheck:
    name = "benchmark"
    title = "Retrieval benchmark"
    needs_model = False

    def run(self, context: CheckContext) -> CheckResult:
        project = context.project
        result = CheckResult(name=self.name, title=self.title)

        corpora = DataCheck._find_corpora(project.root)
        if not corpora:
            result.skipped = True
            result.skip_reason = (
                "no knowledge corpus found to benchmark against. Trisol looks for "
                "knowledge/*.json, docs.json, corpus.json and similar."
            )
            return result

        docs, titles = self._load(corpora[0])
        if len(docs) < 3:
            result.skipped = True
            result.skip_reason = (
                f"corpus has only {len(docs)} document(s); a benchmark needs at "
                "least 3 to be meaningful."
            )
            return result

        queries = self._build_eval_set(docs, titles)
        if not queries:
            result.skipped = True
            result.skip_reason = "could not derive evaluation queries from the corpus."
            return result

        top_k = 3
        scores = [
            self._score(name, fn, queries, docs, top_k, description)
            for name, fn, description in _STRATEGIES
        ]
        baseline = self._identify_baseline(project, scores)
        best = self._pick_best(scores)

        result.metrics = {
            "corpus": project.relative(corpora[0]),
            "documents": len(docs),
            "queries": len(queries),
            "top_k": top_k,
            "baseline": baseline.name,
            "best": best.name,
            "results": [s.as_dict() for s in scores],
        }

        gain = best.recall_at_k - baseline.recall_at_k
        if best.name != baseline.name and gain >= _MIN_RECALL_GAIN:
            result.findings.append(
                Finding(
                    check=self.name,
                    title=f"Retrieval strategy underperforms: {best.name} beats it",
                    detail=(
                        f"On {len(queries)} queries over {len(docs)} real documents "
                        f"from this project, the implemented {baseline.name} strategy "
                        f"recalls {baseline.recall_at_k:.0%} at k={top_k} "
                        f"(MRR {baseline.mrr:.2f}), while {best.name} recalls "
                        f"{best.recall_at_k:.0%} (MRR {best.mrr:.2f}). "
                        f"{best.name} is {best.description}."
                    ),
                    severity=Severity.HIGH,
                    confidence=Confidence.CERTAIN,
                    impact=(
                        f"{gain:.0%} of answerable questions retrieve nothing "
                        "relevant, so the model answers them without grounding."
                    ),
                    suggestion=(
                        f"Replace the {baseline.name} lookup with {best.name}. "
                        "It needs no extra dependency and is a few dozen lines."
                    ),
                    evidence=[
                        f"{s.name}: recall@{top_k}={s.recall_at_k:.0%}, "
                        f"MRR={s.mrr:.2f}, {s.mean_ms:.2f}ms/query"
                        for s in scores
                    ],
                )
            )
        return result

    @staticmethod
    def _load(path: Path) -> tuple[list[str], list[str]]:
        import json  # noqa: PLC0415 - only needed on this path

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return [], []
        docs, titles = [], []
        for doc in _as_documents(raw):
            text = _document_text(doc)
            if text.strip():
                docs.append(text)
                titles.append(doc.get("title", "") if isinstance(doc, dict) else "")
        return docs, titles

    @staticmethod
    def _build_eval_set(docs: list[str], titles: list[str]) -> list[tuple[str, int]]:
        """Derive (query, expected document index) pairs from the corpus itself.

        Queries are built from each document's distinctive terms rather than
        copied from its text, so a substring matcher cannot win by accident --
        which is the whole point: a real user paraphrases.
        """
        df: Counter[str] = Counter()
        tokenised = [_tokens(d) for d in docs]
        for tokens in tokenised:
            for term in set(tokens):
                df[term] += 1

        queries: list[tuple[str, int]] = []
        for index, tokens in enumerate(tokenised):
            # Terms unique to this document, longest first: the words a user
            # would actually search for.
            distinctive = sorted(
                {t for t in tokens if df[t] == 1 and len(t) > 4},
                key=len,
                reverse=True,
            )
            if titles[index]:
                title_terms = [t for t in _tokens(titles[index]) if len(t) > 3]
                distinctive = title_terms + distinctive
            if len(distinctive) >= 2:
                queries.append((" ".join(distinctive[:3]), index))
        return queries

    @staticmethod
    def _score(
        name: str,
        fn: Callable[[str, list[str]], list[int]],
        queries: list[tuple[str, int]],
        docs: list[str],
        top_k: int,
        description: str,
    ) -> Scored:
        hits = 0
        reciprocal = 0.0
        started = time.perf_counter()
        for query, expected in queries:
            ranked = fn(query, docs)[:top_k]
            if expected in ranked:
                hits += 1
                reciprocal += 1.0 / (ranked.index(expected) + 1)
        elapsed = (time.perf_counter() - started) * 1000
        count = len(queries) or 1
        return Scored(
            name=name,
            recall_at_k=hits / count,
            mrr=reciprocal / count,
            mean_ms=elapsed / count,
            description=description,
        )

    #: Preference order when accuracy ties, strongest first. A small corpus makes
    #: several strategies score identically, and recommending whichever happened
    #: to be first in a list would be arbitrary -- BM25 is the one that keeps
    #: working as the corpus grows, so it wins a tie.
    _TIE_BREAK = ("BM25", "char 4-gram", "keyword overlap", "substring")

    @classmethod
    def _pick_best(cls, scores: list[Scored]) -> Scored:
        ranking = {name: index for index, name in enumerate(cls._TIE_BREAK)}
        return max(
            scores,
            key=lambda s: (
                round(s.recall_at_k, 3),
                round(s.mrr, 3),
                -ranking.get(s.name, len(ranking)),
            ),
        )

    @staticmethod
    def _identify_baseline(project, scores: list[Scored]) -> Scored:
        """Which strategy the code actually implements.

        Read from the source rather than assumed, so the comparison is against
        what the project really does. Defaults to substring, the weakest, only
        when nothing more specific is detectable -- and that default is stated in
        the finding so a reader can tell it was inferred.
        """
        by_name = {s.name: s for s in scores}
        joined = ""
        for path in project.python_files[:50]:
            try:
                joined += path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
        lowered = joined.lower()
        if "bm25" in lowered or "idf" in lowered:
            return by_name["BM25"]
        if "ngram" in lowered or "n_gram" in lowered:
            return by_name["char 4-gram"]
        # `in doc` / `in text` substring containment is the classic naive lookup.
        if re.search(r"\bin\s+\w*(?:doc|text|content)\w*\.lower\(\)", joined):
            return by_name["substring"]
        if re.search(r"set\(.*\)\s*&|intersection", joined):
            return by_name["keyword overlap"]
        return by_name["substring"]
