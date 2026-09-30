"""
Phase 2: Preprocessing + Train/Val/Test Split
Project: AI-Powered Intelligent Customer Support Agent
Module: Intent Classification (Bitext dataset)
"""

import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED = 42


def load_raw_data(path="data/bitext_raw.csv"):
    df = pd.read_csv(path)
    print("Loaded:", df.shape)
    return df


def clean_text(text):
    """Replace {{Placeholder}} tokens with a generic [ENTITY] tag, normalize whitespace."""
    text = re.sub(r"\{\{.*?\}\}", "[ENTITY]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df):
    df = df.copy()

    # Clean text
    df["clean_text"] = df["instruction"].apply(clean_text)

    # Drop duplicates based on the CLEANED text (two raw queries that only
    # differ in an entity value, e.g. different order numbers, are otherwise
    # identical -> should count as one duplicate too)
    before = len(df)
    df = df.drop_duplicates(subset=["clean_text", "intent"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate rows (post-cleaning). Remaining: {len(df)}")

    # Drop empty strings after cleaning, just in case
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    return df


def encode_labels(df):
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["intent"])
    print("Num classes:", len(le.classes_))
    return df, le


def split_data(df):
    """70% train, 15% val, 15% test, stratified by label."""
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_SEED
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, label_encoder, out_dir="data"):
    cols = ["clean_text", "intent", "label"]
    train_df[cols].to_csv(f"{out_dir}/train.csv", index=False)
    val_df[cols].to_csv(f"{out_dir}/val.csv", index=False)
    test_df[cols].to_csv(f"{out_dir}/test.csv", index=False)

    # Save label mapping so DistilBERT / inference code can decode predictions later
    label_map = {int(i): label for i, label in enumerate(label_encoder.classes_)}
    with open(f"{out_dir}/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Saved train.csv, val.csv, test.csv, label_map.json to {out_dir}/")


def sanity_check(train_df, val_df, test_df):
    """Confirm no exact text leakage across splits."""
    train_texts = set(train_df["clean_text"])
    val_texts = set(val_df["clean_text"])
    test_texts = set(test_df["clean_text"])

    print("\n=== Leakage check ===")
    print("Train-Val overlap:", len(train_texts & val_texts))
    print("Train-Test overlap:", len(train_texts & test_texts))
    print("Val-Test overlap:", len(val_texts & test_texts))


def main():
    df = load_raw_data()
    df = preprocess(df)
    df, label_encoder = encode_labels(df)

    train_df, val_df, test_df = split_data(df)
    sanity_check(train_df, val_df, test_df)
    save_splits(train_df, val_df, test_df, label_encoder)


if __name__ == "__main__":
    main()