"""Source-level signals that an agent retrieves context before generating.

Separate module so the patterns live in a file written once, cleanly: a missing
grounding rule is only a real defect when the agent actually looks something up,
so getting this detection right matters more than its size suggests.

Matched against source text rather than imports, because plenty of agents do
keyword lookup over a JSON file and have exactly the same hallucination problem
as one using a vector database.
"""

from __future__ import annotations

import re

__all__ = ["MIN_SCORE", "RETRIEVAL_HINTS", "score_source"]

#: Each hint is (pattern, label, weight). Retrieval is only reported when the
#: weights sum to at least MIN_SCORE, because any single signal is too weak on
#: its own: `def search` is a web tool as often as a retriever, and the word
#: "similarity" appears in comments. Requiring agreement between two signals, or
#: one unambiguous one, keeps a missing-grounding finding off agents that never
#: retrieve anything.
RETRIEVAL_HINTS: tuple[tuple[re.Pattern[str], str, int], ...] = (
    # Unambiguous on its own: naming a function for context retrieval.
    (
        re.compile(r"\bdef\s+(?:retrieve|fetch_context|get_context|get_relevant)\w*\s*\("),
        "defines a retrieval function",
        2,
    ),
    # Building a context/passages variable AND feeding it to a prompt.
    (
        re.compile(r"\b(?:context|passages|chunks|retrieved)\s*=(?!=)"),
        "builds a context variable",
        1,
    ),
    (
        re.compile(r"\b(?:top_k|cosine_similarity|rerank|embedding[s_]|vector_store)\b"),
        "uses retrieval parameters",
        1,
    ),
    (
        re.compile(
            r"\b(?:knowledge|corpus)\b[^\n]{0,30}?(?:json|load|open|read)",
            re.IGNORECASE,
        ),
        "loads a knowledge corpus",
        1,
    ),
    # A generic `def search` only counts alongside something else.
    (re.compile(r"\bdef\s+(?:search|lookup)\w*\s*\("), "defines a search function", 1),
)

#: Two weak signals, or one strong one.
MIN_SCORE = 2


def score_source(source: str) -> tuple[int, list[str]]:
    """Total retrieval weight for one file, with the labels that matched."""
    score = 0
    labels: list[str] = []
    for pattern, label, weight in RETRIEVAL_HINTS:
        if pattern.search(source):
            score += weight
            labels.append(label)
    return score, labels
