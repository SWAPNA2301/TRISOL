"""Tool-calling research agent.

Second demo target. Its defects are about control flow and tool contracts
rather than retrieval:

* the tool dispatch loop has no iteration cap, so a model that keeps calling
  tools loops forever
* an unknown tool name raises KeyError instead of being reported back to the
  model, which ends the run on a recoverable mistake
* `divide` has an unguarded division, so one bad argument crashes the agent
* tool results are interpolated into the prompt with no length cap
* the prompt never tells the model what the tools are for or when to stop
* temperature 1.4 on a task that needs determinism
"""

from __future__ import annotations

import json
import os
import urllib.request

OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = os.environ.get("MODEL", "qwen2.5:0.5b")

# Tells the model nothing about the tools, the format, or when to stop.
INSTRUCTIONS = "You are a research agent. Use tools to find answers to questions."


def search(query):
    return f"results for {query}"


def divide(a, b):
    # No zero check: a single bad argument from the model crashes the run.
    return a / b


def fetch(url):
    # No timeout and no scheme validation.
    return urllib.request.urlopen(url).read().decode()


TOOLS = {"search": search, "divide": divide, "fetch": fetch}


def call_model(prompt):
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 1.4},
        }
    ).encode()
    request = urllib.request.Request(
        OLLAMA, data=body, headers={"Content-Type": "application/json"}
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read())["response"]


def run(question):
    transcript = f"{INSTRUCTIONS}\n\nQuestion: {question}\n"

    # No iteration cap: if the model never stops asking for tools, neither does
    # this loop.
    while True:
        reply = call_model(transcript)
        if "TOOL:" not in reply:
            return reply

        name = reply.split("TOOL:")[1].split()[0].strip()
        argument = reply.split("TOOL:")[1].split(maxsplit=1)[1] if " " in reply else ""

        # KeyError on an unknown tool: the model's mistake kills the agent
        # instead of being handed back as a correctable error.
        result = TOOLS[name](argument)

        # No cap on result length: a large tool output can blow the context.
        transcript += f"\nTOOL {name} -> {result}\n"


if __name__ == "__main__":
    print(run("what is the population of France divided by 2"))
