"""Semantic retrieval using embeddings and cosine similarity"""

from typing import List, Dict
import numpy as np
from .base import BaseRetriever, RetrievalResult


class SimpleEmbedder:
    """
    Simple embedding function for demo purposes.
    In production, use: sentence-transformers, OpenAI embeddings, etc.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.cache = {}

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if text in self.cache:
            return self.cache[text]

        # Simple hash-based embedding (deterministic, reproducible)
        # Production: replace with real embedding model
        hash_val = hash(text)
        np.random.seed(hash_val % (2**31))
        embedding = np.random.randn(self.dim).astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        self.cache[text] = embedding
        return embedding


class SemanticRetriever(BaseRetriever):
    """
    Semantic retrieval using embeddings.

    Computes cosine similarity between query and documents.
    Requires embedding model for generating vectors.
    """

    def __init__(self, embedding_dim: int = 384):
        super().__init__(name="SemanticRetriever")
        self.embedding_dim = embedding_dim
        self.embedder = SimpleEmbedder(dim=embedding_dim)
        self.doc_embeddings = {}  # {doc_id: embedding}

    def _build_index(self):
        """Generate embeddings for all documents"""
        self.doc_embeddings = {}
        for doc_id, content in self.documents.items():
            self.doc_embeddings[doc_id] = self.embedder.embed(content)

    def retrieve(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve documents using semantic similarity.

        Args:
            query: Search query
            k: Number of results

        Returns:
            Top-k most similar documents
        """
        if not self.documents:
            return []

        # Embed query
        query_embedding = self.embedder.embed(query)

        # Compute similarity with all documents
        scores = []
        for doc_id, doc_embedding in self.doc_embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding)
            scores.append((doc_id, similarity))

        # Sort by score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Create results
        results = []
        for doc_id, score in scores[:k]:
            results.append(RetrievalResult(
                document_id=doc_id,
                content=self.documents[doc_id],
                score=float(score),
                metadata=self.metadata.get(doc_id)
            ))

        return results

    def estimate_tokens(self, text: str) -> int:
        """Rough token count (for evaluation)"""
        return len(text.split()) // 2 + 1
