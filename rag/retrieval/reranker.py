"""Cross-encoder reranking for improving retrieval quality"""

from typing import List, Dict
import numpy as np
from .base import RetrievalResult


class SimpleRerankerModel:
    """
    Simple relevance model for reranking.
    In production: use CrossEncoder from sentence-transformers.
    """

    def __init__(self):
        self.cache = {}

    def score(self, query: str, document: str) -> float:
        """
        Score relevance of document to query.
        Returns 0.0-1.0 relevance score.
        """
        cache_key = (query, document)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Simple scoring heuristics (production: replace with neural model)
        score = 0.0

        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())

        # Jaccard similarity
        if query_words and doc_words:
            intersection = len(query_words & doc_words)
            union = len(query_words | doc_words)
            jaccard = intersection / union if union > 0 else 0.0
            score += 0.3 * jaccard

        # Query word coverage in document
        coverage = len(query_words & doc_words) / len(query_words) if query_words else 0
        score += 0.4 * coverage

        # Document length penalty (avoid too short docs)
        doc_length = len(document.split())
        query_length = len(query.split())
        length_factor = min(1.0, doc_length / max(query_length * 3, 50))
        score += 0.3 * length_factor

        # Normalize to 0-1
        score = min(1.0, score / 1.0)

        self.cache[cache_key] = score
        return score


class CrossEncoderReranker:
    """
    Reranking using cross-encoder scores.

    Takes query + document pairs and scores their relevance.
    Used to rerank initial retrieval results for better quality.
    """

    def __init__(self):
        self.model = SimpleRerankerModel()

    def rerank(self, query: str, candidates: List[RetrievalResult], k: int = None) -> List[RetrievalResult]:
        """
        Rerank candidates using cross-encoder model.

        Args:
            query: Search query
            candidates: Initial retrieval results
            k: Number of results to return (if None, return all)

        Returns:
            Reranked results
        """
        if not candidates:
            return []

        # Score each candidate
        scored = []
        for result in candidates:
            relevance_score = self.model.score(query, result.content)
            scored.append((result, relevance_score))

        # Sort by relevance (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Create new results with updated scores
        reranked = []
        for result, relevance in scored:
            # Keep original retrieval score but use relevance for sorting
            reranked_result = RetrievalResult(
                document_id=result.document_id,
                content=result.content,
                score=relevance,  # Update score to relevance
                metadata={
                    **(result.metadata or {}),
                    "original_score": result.score,
                    "reranker_score": relevance
                }
            )
            reranked.append(reranked_result)

        # Return top-k
        return reranked[:k] if k else reranked

    def batch_rerank(
        self,
        query: str,
        candidates_per_step: List[List[RetrievalResult]],
        k: int = 5
    ) -> List[RetrievalResult]:
        """
        Multi-stage reranking with progressive filtering.

        Useful for expensive cross-encoder models on large result sets.
        """
        current_candidates = candidates_per_step[0] if candidates_per_step else []

        for candidates in candidates_per_step[1:]:
            # Rerank current to keep top-k
            current_candidates = self.rerank(query, current_candidates, k=k)
            # Add new candidates
            current_candidates.extend(candidates)

        return self.rerank(query, current_candidates, k=k)
