"""Hybrid retrieval combining semantic and lexical search"""

from typing import List, Dict
from .base import BaseRetriever, RetrievalResult
from .semantic import SemanticRetriever
from .bm25_retriever import BM25Retriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining semantic and BM25.

    Uses Reciprocal Rank Fusion (RRF) to combine results:
    RRF_score = sum(1 / (k + rank))

    Semantic search captures meaning/intent.
    BM25 captures exact keyword matches and entities.
    Together: better recall and precision.
    """

    def __init__(self, semantic_weight: float = 0.5, k_rrf: int = 60):
        super().__init__(name="HybridRetriever")
        self.semantic_weight = semantic_weight
        self.bm25_weight = 1.0 - semantic_weight
        self.k_rrf = k_rrf

        self.semantic = SemanticRetriever()
        self.bm25 = BM25Retriever()

    def _build_index(self):
        """Build indexes for both retrievers"""
        self.semantic.index_documents(self.documents, self.metadata)
        self.bm25.index_documents(self.documents, self.metadata)

    def retrieve(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve using hybrid strategy with RRF fusion.

        Args:
            query: Search query
            k: Number of results

        Returns:
            Top-k fused results
        """
        if not self.documents:
            return []

        # Get results from both retrievers
        # Use larger k to ensure good fusion
        retrieve_k = max(k * 5, 50)

        semantic_results = self.semantic.retrieve(query, k=retrieve_k)
        bm25_results = self.bm25.retrieve(query, k=retrieve_k)

        # Combine using RRF
        fused_scores = {}

        # Add semantic results
        for rank, result in enumerate(semantic_results, 1):
            rrf_score = 1.0 / (self.k_rrf + rank)
            if result.document_id not in fused_scores:
                fused_scores[result.document_id] = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "semantic_score": result.score,
                    "bm25_score": 0.0,
                    "rrf_score": 0.0
                }
            fused_scores[result.document_id]["rrf_score"] += self.semantic_weight * rrf_score
            fused_scores[result.document_id]["semantic_rank"] = rank

        # Add BM25 results
        for rank, result in enumerate(bm25_results, 1):
            rrf_score = 1.0 / (self.k_rrf + rank)
            if result.document_id not in fused_scores:
                fused_scores[result.document_id] = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "semantic_score": 0.0,
                    "bm25_score": result.score,
                    "rrf_score": 0.0
                }
            fused_scores[result.document_id]["rrf_score"] += self.bm25_weight * rrf_score
            fused_scores[result.document_id]["bm25_rank"] = rank

        # Sort by fused RRF score
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True
        )

        # Create result objects
        results = []
        for doc_id, scores in sorted_results[:k]:
            # Combined score as weighted average
            combined_score = (
                self.semantic_weight * scores["semantic_score"] +
                self.bm25_weight * scores["bm25_score"]
            )

            results.append(RetrievalResult(
                document_id=doc_id,
                content=scores["content"],
                score=combined_score,
                metadata=scores["metadata"]
            ))

        return results

    def set_weights(self, semantic: float = 0.5, bm25: float = 0.5):
        """Adjust retrieval weights dynamically"""
        total = semantic + bm25
        self.semantic_weight = semantic / total
        self.bm25_weight = bm25 / total
