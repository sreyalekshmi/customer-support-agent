"""Intent recognition backed by the trained TF-IDF classifier.

The training project exposes 27 Bitext labels while the agent has a smaller,
operation-oriented taxonomy. This module is the single adapter between them.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import joblib

from app.schemas import IntentResult

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = _PROJECT_ROOT / "intent_classification" / "models"
_LABEL_MAP = _PROJECT_ROOT / "intent_classification" / "data" / "label_map.json"
_ENTITY_RE = re.compile(r"\{\{.*?\}\}")

_INTENT_MAP = {
    "track_order": "order_status",
    "check_refund_policy": "policy_faq",
    "delivery_options": "policy_faq",
    "delivery_period": "policy_faq",
    "get_refund": "refund_request",
    "track_refund": "refund_request",
    "check_payment_methods": "payment_status",
    "payment_issue": "payment_status",
    "check_invoice": "payment_status",
    "get_invoice": "payment_status",
    "complaint": "complaint_or_ticket",
    "review": "complaint_or_ticket",
    "contact_human_agent": "human_escalation",
    "contact_customer_service": "human_escalation",
}
_ORDER_WORDS = ("order", "track", "tracking", "parcel", "package", "status")
_POLICY_WORDS = ("shipping", "return", "returns", "policy", "delivery")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", _ENTITY_RE.sub("[ENTITY]", text)).strip()


def _fallback_intent(text: str) -> tuple[str, str] | None:
    """Keep obvious requests routable when the trained model is uncertain."""
    normalized = text.lower()
    if re.search(r"\b(human|agent|representative)\b", normalized):
        return "human_escalation", "fallback_contact_human"
    if re.search(r"\b(refund|money back)\b", normalized):
        if re.search(r"\b(i|my|me|want|need|order)\b", normalized):
            return "refund_request", "fallback_refund"
        return "policy_faq", "fallback_refund_policy"
    if any(re.search(rf"\b{word}\b", normalized) for word in _POLICY_WORDS):
        return "policy_faq", "fallback_policy"
    if re.search(r"\b(payment|paid|charged|charge|invoice)\b", normalized):
        return "payment_status", "fallback_payment"
    if any(re.search(rf"\b{word}\b", normalized) for word in _ORDER_WORDS):
        return "order_status", "fallback_order_status"
    return None


@lru_cache(maxsize=1)
def _load_model():
    vectorizer = joblib.load(_MODEL_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(_MODEL_DIR / "logistic_regression.joblib")
    with _LABEL_MAP.open(encoding="utf-8") as stream:
        labels = json.load(stream)
    return vectorizer, model, labels


def predict_intent(text: str) -> IntentResult:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Message must be a non-empty string.")

    normalized = text.lower()
    if re.search(r"\b(hi|hello|hey|thanks|thank you)\b", normalized):
        return IntentResult(
            intent="greeting_general",
            confidence=1.0,
            raw_label="greeting",
            is_mock=False,
        )

    vectorizer, model, labels = _load_model()
    probabilities = model.predict_proba(vectorizer.transform([_clean_text(text)]))[0]
    class_index = int(model.classes_[probabilities.argmax()])
    raw_label = labels[str(class_index)]
    intent = _INTENT_MAP.get(raw_label, "unknown")
    confidence = float(probabilities.max())
    fallback = _fallback_intent(text)
    if fallback and (confidence < 0.5 or intent != fallback[0]):
        intent, raw_label = fallback
        confidence = max(confidence, 0.5)
    return IntentResult(intent=intent, confidence=confidence, raw_label=raw_label, is_mock=False)
