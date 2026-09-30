import re
import json
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MAX_LENGTH = 32
MODEL_DIR = "models/distilbert_intent_classifier"


def clean_text(text):
    text = re.sub(r"\{\{.*?\}\}", "[ENTITY]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv("results/generalization_predictions.csv")
    df["clean_text"] = df["text"].apply(clean_text)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    inputs = tokenizer(
        df["clean_text"].tolist(), return_tensors="pt",
        padding="max_length", truncation=True, max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)

    df["confidence"] = confidences.cpu().numpy().round(4)
    df["correct"] = df["true_intent"] == df["distilbert_pred"]

    print("Avg confidence when CORRECT:", df[df["correct"]]["confidence"].mean().round(4))
    print("Avg confidence when WRONG:  ", df[~df["correct"]]["confidence"].mean().round(4))

    print("\n=== Wrong predictions, sorted by confidence (high = model was confidently wrong) ===")
    wrong = df[~df["correct"]].sort_values("confidence", ascending=False)
    print(wrong[["text", "true_intent", "distilbert_pred", "confidence"]].to_string(index=False))

    df.to_csv("results/generalization_with_confidence.csv", index=False)


if __name__ == "__main__":
    main()