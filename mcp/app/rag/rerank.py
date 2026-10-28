"""Optional cross-encoder reranking of fused retrieval candidates.

Off by default. Enable with RAG_RERANKER=BAAI/bge-reranker-base (any
sentence-transformers CrossEncoder id). On the 46-query golden set it lifted
hit@1 0.85 -> 0.90, which is within noise; re-measure with scripts.eval_rag
before turning it on.
"""

import logging
import os
from functools import lru_cache

log = logging.getLogger(__name__)


def reranker_name() -> str | None:
    return os.getenv("RAG_RERANKER", "").strip() or None


@lru_cache(maxsize=1)
def _model(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


def rerank(query: str, candidates: list[tuple[str, str, float]], limit: int):
    """Reorder (text, source, score) candidates; keep the input order on any failure."""
    name = reranker_name()
    if not name or len(candidates) < 2:
        return candidates[:limit]
    try:
        scores = _model(name).predict([(query, text) for text, _, _ in candidates])
    except Exception as exc:
        log.warning("Reranker unavailable (%s); keeping fused order", type(exc).__name__)
        return candidates[:limit]
    ranked = sorted(zip(scores, candidates), key=lambda pair: -pair[0])
    return [(text, source, float(score)) for score, (text, source, _) in ranked[:limit]]
