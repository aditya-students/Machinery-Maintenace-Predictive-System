import pandas as pd

tr = pd.read_parquet("data/interim/train_clean.parquet")
te = pd.read_parquet("data/interim/test_clean.parquet")
print("only in train:", sorted(set(tr.columns) - set(te.columns)))
print("only in test:", sorted(set(te.columns) - set(tr.columns)))
print("train columns:", tr.columns.tolist())