"""Core RAG pipeline orchestrator"""

from typing import List, Dict, Optional
from .retrieval import BaseRetriever, CrossEncoderReranker
from .retrieval.base import RetrievalResult
from .query import QueryOptimizer, QueryAnalyzer


class RAGPipeline:
    """
    Production RAG Pipeline.

    Orchestrates:
    1. Query optimization
    2. Multi-strategy retrieval
    3. Cross-encoder reranking
    4. Result assembly
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        query_optimizer: Optional[QueryOptimizer] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
    ):
        """
        Initialize RAG pipeline.

        Args:
            retriever: Retriever strategy (Semantic, BM25, or Hybrid)
            reranker: Optional cross-encoder for reranking
            query_optimizer: Optional query optimization
            query_analyzer: Optional query analysis
        """
        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.query_optimizer = query_optimizer or QueryOptimizer()
        self.query_analyzer = query_analyzer or QueryAnalyzer()

        self.retrieval_stats = {
            "queries_processed": 0,
            "total_retrieval_time": 0,
            "total_reranking_time": 0,
        }

    def index_documents(self, documents: Dict[str, str], metadata: Dict = None):
        """Index documents for retrieval"""
        self.retriever.index_documents(documents, metadata)
        print(f"✓ Indexed {len(documents)} documents")

    def query(
        self,
        query: str,
        k: int = 10,
        optimize_query: bool = True,
        rerank: bool = True,
        rerank_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Execute RAG query.

        Args:
            query: User query
            k: Number of retrieval candidates
            optimize_query: Whether to optimize query
            rerank: Whether to apply reranking
            rerank_k: Number of results after reranking

        Returns:
            Top-k reranked results
        """
        self.retrieval_stats["queries_processed"] += 1

        # 1. Analyze query
        analysis = self.query_analyzer.analyze(query)
        print(f"Query intent: {analysis.intent}")
        print(f"Keywords: {analysis.keywords}")

        # 2. Optimize query if needed
        queries_to_run = [query]
        if optimize_query:
            strategy = self.query_analyzer.suggest_strategy(query)
            queries_to_run = self.query_optimizer.optimize(query, strategy)
            if len(queries_to_run) > 1:
                print(f"Query expansion: {len(queries_to_run)} variants")

        # 3. Retrieve with each optimized query (ensemble)
        all_results = {}

        for q in queries_to_run:
            results = self.retriever.retrieve(q, k=k)

            for result in results:
                if result.document_id not in all_results:
                    all_results[result.document_id] = result
                else:
                    # Average score if seen multiple times
                    existing = all_results[result.document_id]
                    avg_score = (existing.score + result.score) / 2
                    all_results[result.document_id].score = avg_score

        # Convert to list and sort
        results_list = list(all_results.values())
        results_list.sort(key=lambda x: x.score, reverse=True)
        results_list = results_list[:k]

        # 4. Rerank if requested
        if rerank and results_list:
            results_list = self.reranker.rerank(query, results_list, k=rerank_k)
            print(f"✓ Reranked to top-{rerank_k}")

        return results_list

    def batch_query(
        self,
        queries: List[str],
        k: int = 10,
        **kwargs
    ) -> List[List[RetrievalResult]]:
        """
        Process multiple queries.

        Args:
            queries: List of queries
            k: Results per query
            **kwargs: Arguments to pass to query()

        Returns:
            List of result lists
        """
        results = []

        for i, query in enumerate(queries):
            if i % 10 == 0:
                print(f"Processing query {i+1}/{len(queries)}...")

            result = self.query(query, k=k, **kwargs)
            results.append(result)

        return results

    def format_results(
        self,
        results: List[RetrievalResult],
        include_metadata: bool = False
    ) -> str:
        """Format results for display"""
        output = []

        for rank, result in enumerate(results, 1):
            output.append(f"\n[{rank}] Score: {result.score:.3f}")
            output.append(f"ID: {result.document_id}")

            # Show snippet of document
            snippet = result.content[:200] + "..." if len(result.content) > 200 else result.content
            output.append(f"Content: {snippet}")

            if include_metadata and result.metadata:
                output.append(f"Metadata: {result.metadata}")

        return "\n".join(output)

    def get_stats(self) -> Dict:
        """Get retrieval statistics"""
        return {
            **self.retrieval_stats,
            "retriever": self.retriever.name,
            "documents_indexed": len(self.retriever.documents),
        }

    def __repr__(self):
        return (
            f"RAGPipeline(\n"
            f"  retriever={self.retriever.name},\n"
            f"  documents={len(self.retriever.documents)}\n"
            f")"
        )
