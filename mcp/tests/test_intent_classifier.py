from types import SimpleNamespace

import pytest

from app.intent import classifier


class _Probabilities:
    def __init__(self, values):
        self.values = values

    def argmax(self):
        return max(range(len(self.values)), key=self.values.__getitem__)

    def max(self):
        return max(self.values)

    def __getitem__(self, index):
        return self.values[index]


class _ProbabilityMatrix:
    def __init__(self, values):
        self.row = _Probabilities(values)

    def __getitem__(self, index):
        assert index == 0
        return self.row


class _Vectorizer:
    def transform(self, values):
        return values


class _Model:
    classes_ = [25]

    def predict_proba(self, values):
        assert values == ["I need to track my order"]
        return _ProbabilityMatrix([0.91])


def test_classifier_maps_trained_track_order_label(monkeypatch):
    monkeypatch.setattr(
        classifier,
        "_load_model",
        lambda: (_Vectorizer(), _Model(), {"25": "track_order"}),
    )

    result = classifier.predict_intent("I need to track my order")

    assert result.intent == "order_status"
    assert result.raw_label == "track_order"
    assert result.confidence == pytest.approx(0.91)
    assert result.is_mock is False


def test_classifier_maps_human_contact_to_escalation(monkeypatch):
    model = SimpleNamespace(
        classes_=[9],
        predict_proba=lambda values: _ProbabilityMatrix([0.88]),
    )
    monkeypatch.setattr(
        classifier,
        "_load_model",
        lambda: (_Vectorizer(), model, {"9": "contact_human_agent"}),
    )

    result = classifier.predict_intent("I need a human agent")

    assert result.intent == "human_escalation"
    assert result.raw_label == "contact_human_agent"


@pytest.mark.parametrize("value", ["", "  ", None, 123])
def test_classifier_rejects_empty_or_non_text(value):
    with pytest.raises(ValueError):
        classifier.predict_intent(value)


def test_greeting_does_not_require_model(monkeypatch):
    def fail():
        raise AssertionError("greeting should be handled before model inference")

    monkeypatch.setattr(classifier, "_load_model", fail)
    result = classifier.predict_intent("Hello there")
    assert result.intent == "greeting_general"
