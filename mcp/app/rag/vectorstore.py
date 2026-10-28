"""Local Chroma semantic index; no hosted vector database is used."""

import hashlib
import json
import logging
import os
from functools import lru_cache

from app.config import REPO_ROOT

INDEX_DIR = REPO_ROOT / "data" / "chroma_index"
log = logging.getLogger(__name__)

# Calibrated with scripts/eval_rag.py on MiniLM + cosine + aliases (both golden sets):
# 0.74 refuses ~7% of answerable and ~66% of unanswerable queries. Re-tune whenever
# the embedding model, aliases or knowledge base change.
DEFAULT_MIN_SCORE = 0.74


def min_score() -> float:
    return float(os.getenv("RAG_MIN_SEMANTIC_SCORE", DEFAULT_MIN_SCORE))


def _cosine_relevance(distance: float) -> float:
    """Scale cosine distance (0..2) to the bounded relevance range (0..1)."""
    return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))


@lru_cache(maxsize=1)
def _vectorstore():
    from langchain_community.vectorstores import Chroma

    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from app.rag.documents import load_aliases, load_policy_chunks

    model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunks = load_policy_chunks()
    aliases = load_aliases()
    # The collection name hashes the content and model, so editing the knowledge
    # base, the aliases or the model builds a fresh index instead of serving a stale one.
    fingerprint = hashlib.sha256(
        json.dumps([model, chunks, aliases], sort_keys=True).encode()
    ).hexdigest()[:10]

    embeddings = HuggingFaceEmbeddings(
        model_name=model
    )
    store = Chroma(
        collection_name=f"support_policies_{fingerprint}",
        embedding_function=embeddings,
        persist_directory=str(INDEX_DIR),
        collection_metadata={"hnsw:space": "cosine"},
        relevance_score_fn=_cosine_relevance,
    )
    if store._collection.count() == 0:
        split_docs = []
        for text, source in chunks:
            meta = {"source": source, "chunk": text}
            split_docs.append(Document(page_content=text, metadata=meta))
            # Alias entries are embedded on their own but resolve back to the chunk.
            for alias in aliases.get(text.split("\n", 1)[0].strip(), []):
                split_docs.append(Document(page_content=alias, metadata=meta))
        if split_docs:
            store.add_documents(split_docs)
    return store


def semantic_search(query: str, limit: int = 6):
    try:
        matches = _vectorstore().similarity_search_with_relevance_scores(
            query, k=limit * 4  # aliases mean several hits can resolve to one chunk
        )
    except Exception as exc:
        # Keep the policy experience available offline or without model weights.
        log.info("Local semantic index unavailable (%s); using BM25 fallback", type(exc).__name__)
        return None
    results, seen = [], set()
    for doc, score in matches:  # best-first, so the first hit per chunk is its best
        text = doc.metadata.get("chunk", doc.page_content)
        if score < min_score() or text in seen:
            continue
        seen.add(text)
        results.append((text, doc.metadata.get("source", "knowledge-base"), float(score)))
    return results[:limit]
