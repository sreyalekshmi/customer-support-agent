"""
Phase 3: Baseline Model — TF-IDF + Logistic Regression
Project: AI-Powered Intelligent Customer Support Agent
Module: Intent Classification (Bitext dataset)
"""

import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

RANDOM_SEED = 42


def load_splits(data_dir="data"):
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/val.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    with open(f"{data_dir}/label_map.json") as f:
        label_map = json.load(f)
    return train_df, val_df, test_df, label_map


def build_pipeline():
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    clf = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    return vectorizer, clf


def train_baseline(train_df, val_df):
    vectorizer, clf = build_pipeline()

    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_val = vectorizer.transform(val_df["clean_text"])

    clf.fit(X_train, train_df["label"])

    val_preds = clf.predict(X_val)
    acc = accuracy_score(val_df["label"], val_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_df["label"], val_preds, average="macro"
    )
    print("\n=== Validation results ===")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")

    return vectorizer, clf


def evaluate_on_test(vectorizer, clf, test_df, label_map, out_dir="results"):
    X_test = vectorizer.transform(test_df["clean_text"])
    test_preds = clf.predict(X_test)

    acc = accuracy_score(test_df["label"], test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df["label"], test_preds, average="macro"
    )
    print("\n=== Test results (Baseline: TF-IDF + Logistic Regression) ===")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")

    target_names = [label_map[str(i)] for i in sorted(map(int, label_map.keys()))]
    report = classification_report(
        test_df["label"], test_preds, target_names=target_names
    )
    print("\n" + report)

    with open(f"{out_dir}/baseline_classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(test_df["label"], test_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Baseline Confusion Matrix (TF-IDF + Logistic Regression)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/baseline_confusion_matrix.png", dpi=150)
    print(f"\nSaved confusion matrix to {out_dir}/baseline_confusion_matrix.png")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def save_baseline_metrics(metrics, out_dir="results"):
    with open(f"{out_dir}/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_dir}/baseline_metrics.json")


def save_artifacts(vectorizer, clf, out_dir="models"):
    joblib.dump(vectorizer, f"{out_dir}/tfidf_vectorizer.joblib")
    joblib.dump(clf, f"{out_dir}/logistic_regression.joblib")
    print(f"Saved vectorizer and model to {out_dir}/")


def main():
    train_df, val_df, test_df, label_map = load_splits()
    vectorizer, clf = train_baseline(train_df, val_df)
    metrics = evaluate_on_test(vectorizer, clf, test_df, label_map)
    save_baseline_metrics(metrics)
    save_artifacts(vectorizer, clf)


if __name__ == "__main__":
    main()