"""Small deterministic query normalization for customer support policy terms."""

import re

_EXPANSIONS = {
    "ship": "shipping delivery dispatch",
    "shipped": "shipping delivery dispatch",
    "arrive": "delivery shipping timeline",
    "late": "delay delayed delivery",
    "return": "returns refund exchange",
    "exchange": "return replacement",
    "money": "refund payment",
    "cancel": "cancellation order",
}


def optimize_query(query: str) -> str:
    words = re.findall(r"[a-z0-9]+", query.lower())
    expanded = list(words)
    for word in words:
        expanded.extend(_EXPANSIONS.get(word, "").split())
    return " ".join(dict.fromkeys(expanded))
