"""
Production RAG System - Retrieval-Augmented Generation with evaluation framework
"""

__version__ = "1.0.0"
__author__ = "Artem Kazakov Kozlov (KazKozDev)"

from .core import RAGPipeline
from .retrieval import SemanticRetriever, BM25Retriever, HybridRetriever

__all__ = [
    "RAGPipeline",
    "SemanticRetriever",
    "BM25Retriever",
    "HybridRetriever",
]
