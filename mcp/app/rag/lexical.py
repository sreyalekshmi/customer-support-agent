"""Dependency-light BM25 scoring over the local policy chunks."""

import math
import re
from collections import Counter

_STOP = {
    "the", "a", "an", "is", "are", "do", "does", "what", "how", "your",
    "you", "i", "my", "on", "of", "to", "for", "and", "or", "in", "it",
    "me", "can", "if", "be", "with", "please", "tell", "about", "policy",
    "support", "help",
}


def tokens(text: str) -> list[str]:
    return [
        word[:-1] if len(word) > 3 and word.endswith("s") else word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in _STOP
    ]


def lexical_search(
    query: str, chunks: list[tuple[str, str]], limit: int = 6
) -> list[tuple[str, str, float]]:
    query_terms = set(tokens(query))
    if not query_terms or not chunks:
        return []
    docs = [tokens(text) for text, _ in chunks]
    avg_len = sum(map(len, docs)) / len(docs) or 1
    df = Counter(term for doc in docs for term in set(doc))
    scored = []
    for (text, source), doc in zip(chunks, docs):
        counts = Counter(doc)
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if frequency:
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * frequency * 2.2 / (
                    frequency + 1.2 * (0.25 + 0.75 * len(doc) / avg_len)
                )
        if score:
            scored.append((text, source, score))
    return sorted(scored, key=lambda item: item[2], reverse=True)[:limit]
