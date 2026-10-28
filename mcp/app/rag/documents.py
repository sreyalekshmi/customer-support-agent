"""Load policy Markdown into shared chunks for BM25 and semantic retrieval."""

import json
from pathlib import Path

from app.config import REPO_ROOT

KB_DIR = REPO_ROOT / "data" / "knowledge_base"
ALIASES_FILE = KB_DIR / "aliases.json"


def load_aliases() -> dict[str, list[str]]:
    """Customer-style phrasings per FAQ question, embedded to catch vague queries."""
    if not ALIASES_FILE.exists():
        return {}
    return json.loads(ALIASES_FILE.read_text(encoding="utf-8"))


def load_policy_chunks() -> list[tuple[str, str]]:
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )
    except ModuleNotFoundError:
        Document = None
        MarkdownHeaderTextSplitter = RecursiveCharacterTextSplitter = None

    chunks: list[tuple[str, str]] = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if Document and MarkdownHeaderTextSplitter and RecursiveCharacterTextSplitter:
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "title"), ("##", "section")]
            )
            docs = header_splitter.split_text(text)
            for doc in docs:
                doc.metadata["source"] = path.name
            for doc in docs:
                heading = " / ".join(
                    str(doc.metadata[key]) for key in ("title", "section")
                    if doc.metadata.get(key)
                )
                doc.page_content = f"{heading}\n{doc.page_content}".strip()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700, chunk_overlap=100, add_start_index=True
            )
            chunks.extend(
                (doc.page_content, path.name)
                for doc in splitter.split_documents(docs)
                if doc.page_content.strip()
            )
        else:
            for paragraph in text.split("\n\n"):
                paragraph = paragraph.strip()
                if paragraph and paragraph != "---":
                    chunks.append((paragraph, path.name))
    return chunks
