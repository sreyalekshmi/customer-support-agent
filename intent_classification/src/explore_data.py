from datasets import load_dataset
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)


def load_data():
    """Load the Bitext customer support dataset from Hugging Face."""
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    print("=== Dataset structure ===")
    print(ds)
    return ds


def to_dataframe(ds):
    """Convert the train split to a pandas DataFrame."""
    df = ds["train"].to_pandas()
    print("\n=== Columns ===")
    print(df.columns.tolist())
    print("\n=== Sample rows ===")
    print(df[["instruction", "intent", "category"]].head())
    return df


def check_missing_and_duplicates(df):
    print("\n=== Missing values ===")
    print(df[["instruction", "intent"]].isna().sum())

    print("\n=== Exact duplicate queries ===")
    print("Duplicate instructions:", df["instruction"].duplicated().sum())


def check_classes(df):
    print("\n=== Intent classes ===")
    print("Number of unique intents:", df["intent"].nunique())
    print(df["intent"].value_counts())

    print("\n=== Category classes (coarse grouping) ===")
    print("Number of unique categories:", df["category"].nunique())
    print(df["category"].value_counts())


def check_query_length(df):
    lengths = df["instruction"].str.split().str.len()
    print("\n=== Query length (in words) ===")
    print(lengths.describe())


def check_label_conflicts(df):
    """Find identical queries mapped to more than one intent."""
    conflicts = df.groupby("instruction")["intent"].nunique()
    n_conflicts = (conflicts > 1).sum()
    print("\n=== Label conflicts ===")
    print("Queries with more than one distinct intent label:", n_conflicts)


def check_placeholders(df, n=5):
    """Preview queries containing template placeholders like {{Order Number}}."""
    sample = df[df["instruction"].str.contains(r"\{\{.*?\}\}", regex=True)]
    print("\n=== Sample queries with placeholders ===")
    print("Total queries with placeholders:", len(sample))
    print(sample["instruction"].head(n).tolist())


def main():
    ds = load_data()
    df = to_dataframe(ds)

    check_missing_and_duplicates(df)
    check_classes(df)
    check_query_length(df)
    check_label_conflicts(df)
    check_placeholders(df)

    # Save a local copy for the next phase
    df.to_csv("data/bitext_raw.csv", index=False)
    print("\nSaved raw data to data/bitext_raw.csv")


if __name__ == "__main__":
    main()