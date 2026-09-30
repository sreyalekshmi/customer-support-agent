# Intent Classification Module 


**Module:** Customer query intent classification (DistilBERT fine-tuned on Bitext dataset)


---

## How to use it

Import the prediction function directly:

```python
from src.predict import predict_intent

result = predict_intent("I want a refund for my order")
```

## Input

- **Type:** `str`
- **Format:** Raw customer query text, natural language, no special formatting needed.
- Empty strings or non-string input will raise a `ValueError` — validate upstream if you're passing user input directly.

## Output

Returns a `dict` with exactly two keys:

```python
{
    "intent": "get_refund",
    "confidence": 0.9806
}
```

- `intent` (`str`): one of 27 fixed intent labels (see `data/label_map.json` for the full list).
- `confidence` (`float`, 0–1): softmax probability of the predicted class, rounded to 4 decimals.

---

## Important: how to use the confidence score

This is not just a diagnostic number — **please use it in your routing logic.**

Based on generalization testing (see `results/generalization_summary.json` and `results/generalization_with_confidence.csv`), this model was trained on Bitext's synthetic dataset and scores ~99% on that data, but drops to **~73% accuracy on natural, real-world-phrased queries**. Confidence scores meaningfully correlate with correctness:

| | Avg confidence |
|---|---|
| Correct predictions | 0.93 |
| Wrong predictions | 0.66 |

**Recommendation:**
- **confidence ≥ 0.6** → reasonably trustworthy, route normally.
- **confidence < 0.6** → flag as uncertain. Route to a fallback path — human agent handoff, a clarifying follow-up question, or a "did you mean...?" confirmation step — rather than acting on it directly.

This threshold catches roughly a third of misclassifications in testing. It won't catch high-confidence errors between semantically similar intents (e.g., `get_refund` vs `check_refund_policy`, `check_invoice` vs `get_invoice`) — those are a known limitation, not a bug, and worth keeping in mind if your routing logic distinguishes between such pairs.

---

## Full intent list

See `data/label_map.json` for the authoritative list of all 27 intent labels and their IDs.

## Known limitations

- Model is most reliable on queries with vocabulary/structure similar to standard customer-support phrasing (cancel, refund, track, invoice, account, shipping, payment topics).
- Struggles most with: sentiment-based intents (`complaint`, `review`) when phrased indirectly, and fine-grained distinctions between semantically close intents.
- Full error analysis available in `results/generalization_predictions.csv` if you want to see specific failure examples.

## Files you may need

- `models/distilbert_intent_classifier/` — the model + tokenizer (already loaded by `predict.py`, no need to touch directly)
- `data/label_map.json` — intent ID ↔ name mapping
- `src/predict.py` — the function you'll import
