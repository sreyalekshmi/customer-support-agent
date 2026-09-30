import pandas as pd

pd.set_option("display.max_colwidth", None)

df = pd.read_csv("results/generalization_predictions.csv")

# Where DistilBERT got it wrong
errors = df[df["true_intent"] != df["distilbert_pred"]]
print(f"DistilBERT errors: {len(errors)}/{len(df)}\n")
print(errors[["text", "true_intent", "distilbert_pred"]].to_string(index=False))