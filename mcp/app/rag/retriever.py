"""Hybrid local RAG entry points; see the focused modules in this package."""

import os
from functools import lru_cache

from app.rag.documents import load_policy_chunks
from app.rag.generation import generate_grounded_answer
from app.rag.hybrid import reciprocal_rank_fusion
from app.rag.lexical import lexical_search
from app.rag.query import optimize_query
from app.rag.rerank import rerank, reranker_name
from app.rag.vectorstore import semantic_search
from app.schemas import RetrievalMode, RetrievedChunk, RetrievalResult


# RRF weight of semantic vs BM25 (weight 1.0). Swept on tests/rag_golden.jsonl.
DEFAULT_SEMANTIC_WEIGHT = 30.0


@lru_cache(maxsize=1)
def _load_chunks():
    return load_policy_chunks()


def _lexical_context(query: str, k: int, *, fallback: bool = False) -> RetrievalResult:
    results = lexical_search(query, _load_chunks(), k)
    return RetrievalResult(
        query=query,
        chunks=[RetrievedChunk(text=text, source=source, score=score)
                for text, source, score in results],
        is_mock=True,
        retrieval_mode="bm25",
        retrieval_fallback=fallback,
    )


def retrieve_context(
    query: str, k: int = 3, mode: RetrievalMode | None = None
) -> RetrievalResult:
    """Run one selected strategy: hybrid, semantic, or BM25 keyword retrieval."""
    if not query.strip() or k < 1:
        return RetrievalResult(query=query, chunks=[], retrieval_mode=mode or "hybrid")

    selected = mode or os.getenv("RAG_RETRIEVAL_MODE", "hybrid").strip().lower()
    if selected not in {"hybrid", "semantic", "bm25"}:
        raise ValueError("RAG_RETRIEVAL_MODE must be hybrid, semantic, or bm25.")
    if selected == "bm25":
        return _lexical_context(query, k)

    if selected == "semantic":
        semantic = semantic_search(query, k)
        if semantic is None:
            return _lexical_context(query, k, fallback=True)
        return RetrievalResult(
            query=query,
            chunks=[RetrievedChunk(text=text, source=source, score=score)
                    for text, source, score in semantic],
            retrieval_mode="semantic",
        )

    chunks = _load_chunks()
    optimized = optimize_query(query)
    lexical = lexical_search(optimized, chunks, max(k * 3, 6))
    semantic = semantic_search(query, max(k * 3, 6))
    if semantic is None:
        return _lexical_context(query, k, fallback=True)
    if not semantic:
        # The semantic index answered but nothing cleared the relevance cutoff:
        # BM25 alone cannot tell "relevant" from "shares a common word".
        return RetrievalResult(query=query, chunks=[], retrieval_mode="hybrid")
    pool = max(k * 3, 10) if reranker_name() else k
    fused = reciprocal_rank_fusion(
        lexical, semantic, pool,
        semantic_weight=float(os.getenv("RAG_SEMANTIC_WEIGHT", DEFAULT_SEMANTIC_WEIGHT)),
    )
    fused = rerank(query, fused, k)
    return RetrievalResult(
        query=query,
        chunks=[RetrievedChunk(text=text, source=source, score=score)
                for text, source, score in fused],
        retrieval_mode="hybrid",
    )


def generate_rag_answer(retrieval: RetrievalResult) -> str | None:
    """Generate a grounded Groq response; LangChain tracing is env-configured."""
    return generate_grounded_answer(retrieval)
