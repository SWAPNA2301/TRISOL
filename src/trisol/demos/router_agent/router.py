"""Multi-agent router: picks which specialist agent handles a request.

Third demo target. The interesting defect is algorithmic rather than a missing
try/except: routing is done by first-keyword-wins over an unordered dict, which
is both order-dependent and unable to handle a request mentioning two domains.
A scored or embedding-based router would measurably beat it, which is exactly
what `trisol bench` is for.

Also planted:

* routing keywords overlap between agents, so the result depends on dict order
* no fallback agent, so an unmatched request returns None and the caller crashes
* every specialist shares one prompt template with the role string swapped in,
  so none of them get real domain instruction
* no evaluation set, so nobody knows the routing accuracy
"""

from __future__ import annotations

# "billing" appears under both billing and sales; "account" under both support
# and billing. First match wins, so the answer depends on iteration order.
ROUTES = {
    "billing": ["invoice", "payment", "billing", "charge", "account"],
    "support": ["broken", "error", "help", "account", "login"],
    "sales": ["price", "quote", "billing", "demo", "upgrade"],
}

# One generic template for every specialist: the role name is the only thing
# that changes, so a billing agent gets no billing-specific instruction.
AGENT_PROMPT = "You are the {role} agent. Help the user with their request."


def route(message):
    words = message.lower().split()
    for agent, keywords in ROUTES.items():
        for keyword in keywords:
            if keyword in words:
                return agent
    # No fallback: an unmatched message returns None and the caller breaks.
    return None


def handle(message):
    agent = route(message)
    prompt = AGENT_PROMPT.format(role=agent)
    return {"agent": agent, "prompt": prompt, "message": message}


EXAMPLES = [
    "my invoice is wrong",
    "the login page is broken",
    "what is the price for an upgrade",
    "my account has a billing error",
    "hello",
]

if __name__ == "__main__":
    for example in EXAMPLES:
        print(f"{example!r} -> {route(example)}")
