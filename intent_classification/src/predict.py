"""
Phase 5: Prediction Function
Project: AI-Powered Intelligent Customer Support Agent
Module: Intent Classification (Bitext dataset)

This module exposes a single clean function, predict_intent(), that
teammates working on RAG / agent routing / UI can import and call directly.
"""

import re
import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_DIR = "models/distilbert_intent_classifier"
LABEL_MAP_PATH = "data/label_map.json"
MAX_LENGTH = 32

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load once at import time (not inside the function) so repeated calls are fast
_tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
_model.to(_device)
_model.eval()

with open(LABEL_MAP_PATH) as f:
    _label_map = json.load(f)  # {"0": "cancel_order", "1": "change_order", ...}


def _clean_text(text: str) -> str:
    """Same cleaning applied during training: normalize placeholders + whitespace."""
    text = re.sub(r"\{\{.*?\}\}", "[ENTITY]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_intent(query: str) -> dict:
    """
    Predict the customer support intent for a single query.

    Args:
        query: Raw customer query string.

    Returns:
        {
            "intent": str,        # predicted intent label
            "confidence": float   # softmax probability of the predicted class, rounded to 4 decimals
        }
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    cleaned = _clean_text(query)

    inputs = _tokenizer(
        cleaned,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(_device)

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        confidence, pred_id = torch.max(probs, dim=1)

    intent = _label_map[str(pred_id.item())]
    return {
        "intent": intent,
        "confidence": round(confidence.item(), 4),
    }


if __name__ == "__main__":
    # Quick manual test
    test_queries = [
        "I want to cancel my order",
        "how do i reset my password",
        "can you tell me about your refund policy",
        "I need help talking to a real person",
    ]
    for q in test_queries:
        result = predict_intent(q)
        print(f"{q!r} -> {result}")