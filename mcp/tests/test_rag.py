from app.rag.retriever import generate_rag_answer, retrieve_context


def test_rag_retrieves_policy_source_without_optional_dependencies():
    result = retrieve_context("How long does shipping take?")

    assert result.chunks
    assert result.chunks[0].source == "return_and_shipping_policy.md"


def test_rag_does_not_answer_unrelated_questions():
    result = retrieve_context("What is the warranty on laptops? Policy please")

    assert result.chunks == []


def test_groq_is_optional(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API", raising=False)

    assert generate_rag_answer(retrieve_context("How long does shipping take?")) is None


def test_aliases_point_at_real_kb_questions():
    from app.rag.documents import load_aliases, load_policy_chunks

    headings = {text.split("\n", 1)[0].strip() for text, _ in load_policy_chunks()}
    assert set(load_aliases()) <= headings


def test_retrieval_quality_gate():
    """Regression gate on the golden sets; thresholds sit below measured values
    (hybrid MRR 0.87, ~66% of unanswerable queries refused) to absorb noise."""
    from scripts.eval_rag import GOLDEN_FILES, evaluate, load_golden

    rows = load_golden(sorted(GOLDEN_FILES))
    result = evaluate("hybrid", rows, verbose=False)

    assert result["mrr"] >= 0.82
    refused = result["unanswerable_n"] - result["unanswerable_returned_chunks"]
    assert refused / result["unanswerable_n"] >= 0.5
