import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 32
RANDOM_SEED = 42

DATA_DIR = "/kaggle/input/datasets/sreyalekshmi/intent-classification-data"
OUTPUT_DIR = "/kaggle/working"


def load_splits(data_dir=DATA_DIR):
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/val.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    with open(f"{data_dir}/label_map.json") as f:
        label_map = json.load(f)
    return train_df, val_df, test_df, label_map


def to_hf_dataset(df):
    return Dataset.from_pandas(
        df[["clean_text", "label"]].rename(columns={"clean_text": "text"}),
        preserve_index=False,
    )


def tokenize_datasets(tokenizer, train_ds, val_ds, test_ds):
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    return train_ds, val_ds, test_ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def main():
    train_df, val_df, test_df, label_map = load_splits()
    num_labels = len(label_map)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )

    train_ds = to_hf_dataset(train_df)
    val_ds = to_hf_dataset(val_df)
    test_ds = to_hf_dataset(test_df)

    train_ds, val_ds, test_ds = tokenize_datasets(tokenizer, train_ds, val_ds, test_ds)

    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/models/distilbert_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_dir=f"{OUTPUT_DIR}/results/logs",
        logging_steps=50,
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n=== Test set evaluation ===")
    test_results = trainer.evaluate(test_ds)
    print(test_results)

    with open(f"{OUTPUT_DIR}/results/distilbert_test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=2)

    model.save_pretrained(f"{OUTPUT_DIR}/models/distilbert_intent_classifier")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/models/distilbert_intent_classifier")
    print(f"\nSaved model to {OUTPUT_DIR}/models/distilbert_intent_classifier/")


if __name__ == "__main__":
    main()