"""Build tests/rag_golden_bitext.jsonl from the Bitext intent data.

    python -m scripts.build_rag_golden

Each Bitext intent is mapped to the knowledge-base headings that answer it.
Intents the KB cannot answer become unanswerable queries (empty `relevant`),
and intents that are only partly covered are skipped rather than guessed at.
Deterministic: fixed seed, so the output only changes if the mapping does.
"""

import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "intent_classification" / "data"
OUT = Path(__file__).resolve().parents[1] / "tests" / "rag_golden_bitext.jsonl"

PER_INTENT = 8
SEED = 7

ANSWERS = {
    "cancel_order": ["Can I cancel my order?"],
    "change_order": ["Can I change or cancel an item in my order?"],
    "change_shipping_address": ["Can I change my shipping address after placing an order?"],
    "check_invoice": ["Can I request an invoice for my order?"],
    "get_invoice": ["Can I request an invoice for my order?"],
    "check_payment_methods": ["What payment methods do you accept?"],
    "check_refund_policy": ["What is your return policy?"],
    "contact_customer_service": ["How can I contact customer support?"],
    "contact_human_agent": ["How can I contact customer support?", "Do you offer live chat support?"],
    "create_account": ["How can I create an account?"],
    "delivery_options": ["Do you offer expedited shipping?", "How long does shipping take?"],
    "delivery_period": ["How long does shipping take?"],
    "newsletter_subscription": ["What is your email newsletter about?"],
    "review": ["How can I leave a product review?"],
    "track_order": ["How can I track my order?"],
}
# Not covered by the KB at all. Ambiguous intents (get_refund, place_order,
# payment_issue, complaint, set_up_shipping_address) are deliberately left out.
UNANSWERABLE = {
    "recover_password", "delete_account", "edit_account", "switch_account",
    "registration_problems", "check_cancellation_fee", "track_refund",
}


def usable(text: str) -> bool:
    return bool(text.strip()) and not re.search(r"\{\{|\}\}", text)


def main() -> None:
    pools: dict[str, list[str]] = {}
    for filename, column, intent_col in (
        ("generalization_test.csv", "text", "true_intent"),
        ("test.csv", "clean_text", "intent"),
    ):
        with open(DATA / filename, newline="") as f:
            for row in csv.DictReader(f):
                text = row[column].replace("[ENTITY]", "48213")
                if usable(text):
                    pools.setdefault(row[intent_col], []).append(text.strip())

    rng = random.Random(SEED)
    rows, seen = [], set()
    for intent in sorted(ANSWERS.keys() | UNANSWERABLE):
        candidates = sorted(set(pools.get(intent, [])))
        rng.shuffle(candidates)
        for query in candidates[:PER_INTENT]:
            if query.lower() in seen:
                continue
            seen.add(query.lower())
            rows.append({
                "query": query,
                "relevant": ANSWERS.get(intent, []),
                "intent": intent,
                **({} if intent in ANSWERS else {"answerable": False}),
            })

    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    answerable = sum(bool(r["relevant"]) for r in rows)
    print(f"wrote {len(rows)} queries ({answerable} answerable) to {OUT}")


if __name__ == "__main__":
    main()
