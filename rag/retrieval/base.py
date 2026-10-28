"""Base retriever interface"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Single retrieval result"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = None

    def __repr__(self):
        return f"RetrievalResult(id={self.document_id}, score={self.score:.3f})"


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval strategies.

    All retrievers must implement retrieve() method.
    """

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.documents = {}  # {doc_id: content}
        self.metadata = {}   # {doc_id: metadata}

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for query.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of RetrievalResult sorted by score (highest first)
        """
        pass

    def index_documents(self, documents: Dict[str, str], metadata: Dict[str, Dict] = None):
        """
        Index documents for retrieval.

        Args:
            documents: {doc_id: content}
            metadata: {doc_id: {key: value}}
        """
        self.documents = documents
        self.metadata = metadata or {}
        self._build_index()

    @abstractmethod
    def _build_index(self):
        """Build index from documents. Called after indexing."""
        pass

    def get_document(self, doc_id: str) -> str:
        """Get document content by ID"""
        return self.documents.get(doc_id)

    def get_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get document metadata"""
        return self.metadata.get(doc_id, {})

    def __repr__(self):
        return f"{self.name}(documents={len(self.documents)})"
