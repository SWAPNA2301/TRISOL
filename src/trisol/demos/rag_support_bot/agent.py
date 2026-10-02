"""Customer-support RAG bot.

A deliberately flawed demo target for `trisol audit`. The defects here are the
ones that actually show up in real RAG agents, not synthetic tripwires:

* the system prompt is vague and sets no grounding rule, so the model is free
  to answer from memory instead of the retrieved context
* retrieval is pure substring matching, which misses any paraphrase
* no top-k cap, so the whole corpus can land in the prompt
* the model call has no timeout, no retry and no error handling
* the API key is read at import time and interpolated into an f-string prompt
* there are no tests and no retrieval evaluation at all
"""

from __future__ import annotations

import json
import os
import urllib.request

API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Vague: no grounding instruction, no refusal path, no output format.
SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's questions well."

DOCS_PATH = "knowledge/docs.json"


def load_docs():
    with open(DOCS_PATH) as handle:
        return json.load(handle)


def retrieve(query, docs):
    # Substring matching only: "refund" will not match "money back".
    hits = []
    for doc in docs:
        if query.lower() in doc["text"].lower():
            hits.append(doc)
    return hits


def answer(question):
    docs = load_docs()
    context = retrieve(question, docs)
    joined = "\n".join(d["text"] for d in context)

    prompt = f"""{SYSTEM_PROMPT}

Context:
{joined}

Question: {question}
"""

    body = json.dumps(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.9,
        }
    ).encode()

    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    # No timeout, no try/except, no retry: one blip takes the whole bot down.
    response = urllib.request.urlopen(request, timeout=30)
    payload = json.loads(response.read())
    return payload["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(answer("how do I get a refund"))
