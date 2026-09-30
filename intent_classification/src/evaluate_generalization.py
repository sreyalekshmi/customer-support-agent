"""
Phase 7: Generalization Evaluation
Compares baseline (TF-IDF+LogReg) and DistilBERT on the hand-written
generalization test set, and against their original Bitext test scores.
"""

import json
import re
import joblib
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

MAX_LENGTH = 32
MODEL_DIR = "models/distilbert_intent_classifier"


def clean_text(text):
    text = re.sub(r"\{\{.*?\}\}", "[ENTITY]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_generalization_set(path="data/generalization_test.csv"):
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def load_label_map(path="data/label_map.json"):
    with open(path) as f:
        label_map = json.load(f)
    # invert: intent name -> id
    intent_to_id = {v: int(k) for k, v in label_map.items()}
    return label_map, intent_to_id


def evaluate_baseline(df, intent_to_id):
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    clf = joblib.load("models/logistic_regression.joblib")

    X = vectorizer.transform(df["clean_text"])
    preds = clf.predict(X)

    true_ids = df["true_intent"].map(intent_to_id).values
    acc = accuracy_score(true_ids, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_ids, preds, average="macro")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}, preds


def evaluate_distilbert(df, label_map, intent_to_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    texts = df["clean_text"].tolist()
    inputs = tokenizer(
        texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    true_ids = df["true_intent"].map(intent_to_id).values
    acc = accuracy_score(true_ids, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_ids, preds, average="macro")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}, preds


def main():
    df = load_generalization_set()
    label_map, intent_to_id = load_label_map()

    print(f"Generalization set: {len(df)} rows, {df['true_intent'].nunique()} intents\n")

    baseline_metrics, baseline_preds = evaluate_baseline(df, intent_to_id)
    distilbert_metrics, distilbert_preds = evaluate_distilbert(df, label_map, intent_to_id)

    print("=== Baseline (TF-IDF + Logistic Regression) on generalization set ===")
    for k, v in baseline_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== DistilBERT on generalization set ===")
    for k, v in distilbert_metrics.items():
        print(f"{k}: {v:.4f}")

    # Load original Bitext test scores for comparison
    with open("results/baseline_metrics.json") as f:
        baseline_original = json.load(f)
    with open("results/distilbert_test_metrics.json") as f:
        distilbert_original = json.load(f)

    print("\n=== Generalization Drop ===")
    print(f"Baseline accuracy:   {baseline_original['accuracy']:.4f} -> {baseline_metrics['accuracy']:.4f} "
          f"(drop: {baseline_original['accuracy'] - baseline_metrics['accuracy']:.4f})")
    print(f"DistilBERT accuracy: {distilbert_original['eval_accuracy']:.4f} -> {distilbert_metrics['accuracy']:.4f} "
          f"(drop: {distilbert_original['eval_accuracy'] - distilbert_metrics['accuracy']:.4f})")

    print(f"\nBaseline macro-F1:   {baseline_original['f1']:.4f} -> {baseline_metrics['f1']:.4f} "
          f"(drop: {baseline_original['f1'] - baseline_metrics['f1']:.4f})")
    print(f"DistilBERT macro-F1: {distilbert_original['eval_macro_f1']:.4f} -> {distilbert_metrics['f1']:.4f} "
          f"(drop: {distilbert_original['eval_macro_f1'] - distilbert_metrics['f1']:.4f})")

    # Save results
    id_to_intent = {int(k): v for k, v in label_map.items()}
    results_df = df.copy()
    results_df["baseline_pred"] = [id_to_intent[p] for p in baseline_preds]
    results_df["distilbert_pred"] = [id_to_intent[p] for p in distilbert_preds]
    results_df.to_csv("results/generalization_predictions.csv", index=False)

    summary = {
        "baseline_on_generalization": baseline_metrics,
        "distilbert_on_generalization": distilbert_metrics,
        "baseline_on_bitext_test": {"accuracy": baseline_original["accuracy"], "f1": baseline_original["f1"]},
        "distilbert_on_bitext_test": {
            "accuracy": distilbert_original["eval_accuracy"],
            "f1": distilbert_original["eval_macro_f1"],
        },
    }
    with open("results/generalization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved predictions to results/generalization_predictions.csv")
    print("Saved summary to results/generalization_summary.json")


if __name__ == "__main__":
    main()