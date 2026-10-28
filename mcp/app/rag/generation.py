"""Grounded answer generation through the project's existing Groq integration."""

import os


def generate_grounded_answer(retrieval) -> str | None:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")
    if not api_key or not retrieval.chunks:
        return None

    from langchain_groq import ChatGroq

    context = "\n\n".join(chunk.text for chunk in retrieval.chunks)
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0,
        api_key=api_key,
    )
    prompt = (
        "You are a customer support agent. Answer using only the supplied policy "
        "context. If it does not answer the question, say you do not know. "
        "Never invent policy details.\n\n"
        f"Policy context:\n{context}\n\nCustomer question: {retrieval.query}"
    )
    response = llm.invoke(prompt, config={
        "run_name": "support-rag-answer",
        "tags": ["customer-support", "rag", "groq"],
        "metadata": {"source_count": len(retrieval.chunks)},
    })
    return str(response.content)
