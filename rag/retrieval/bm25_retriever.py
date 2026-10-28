"""BM25 retrieval using lexical matching"""

from typing import List, Dict, Set
from collections import defaultdict
import math
from .base import BaseRetriever, RetrievalResult


class BM25Retriever(BaseRetriever):
    """
    BM25 (Best Matching 25) retrieval algorithm.

    Lexical search using TF-IDF with BM25 weighting.
    Great for exact term matching, keywords, entities.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__(name="BM25Retriever")
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter

        # Index structures
        self.vocabulary = set()  # All unique terms
        self.doc_freq = defaultdict(int)  # {term: doc_count}
        self.term_positions = defaultdict(dict)  # {term: {doc_id: positions}}
        self.doc_lengths = {}  # {doc_id: token_count}
        self.avg_doc_length = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()

    def _build_index(self):
        """Build BM25 index from documents"""
        self.vocabulary = set()
        self.doc_freq = defaultdict(int)
        self.term_positions = defaultdict(dict)
        self.doc_lengths = {}

        total_length = 0

        # First pass: collect statistics
        for doc_id, content in self.documents.items():
            tokens = self._tokenize(content)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # Track which terms appear in this document
            doc_terms = set()

            for pos, token in enumerate(tokens):
                self.vocabulary.add(token)

                if doc_id not in self.term_positions[token]:
                    self.term_positions[token][doc_id] = []
                self.term_positions[token][doc_id].append(pos)

                doc_terms.add(token)

            # Update document frequency
            for term in doc_terms:
                self.doc_freq[term] += 1

        # Calculate average document length
        num_docs = len(self.documents)
        self.avg_doc_length = total_length / max(num_docs, 1)

    def _bm25_score(self, query_tokens: List[str], doc_id: str) -> float:
        """Calculate BM25 score for document"""
        score = 0.0
        doc_length = self.doc_lengths.get(doc_id, 0)

        for term in set(query_tokens):  # Unique terms
            if term not in self.term_positions or doc_id not in self.term_positions[term]:
                continue

            # Term frequency in document
            term_freq = len(self.term_positions[term][doc_id])

            # Inverse document frequency
            num_docs = len(self.documents)
            doc_count = self.doc_freq.get(term, 0)
            idf = math.log((num_docs - doc_count + 0.5) / (doc_count + 0.5) + 1.0)

            # BM25 formula
            norm_factor = 1 - self.b + self.b * (doc_length / max(self.avg_doc_length, 1))
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * norm_factor

            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve documents using BM25 scoring.

        Args:
            query: Search query
            k: Number of results

        Returns:
            Top-k highest-scoring documents
        """
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Score all documents
        scores = []
        for doc_id in self.documents:
            score = self._bm25_score(query_tokens, doc_id)
            scores.append((doc_id, score))

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
