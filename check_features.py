import pandas as pd

tr = pd.read_parquet("data/processed/train_features.parquet")
te = pd.read_parquet("data/processed/test_features.parquet")

NON_FEATURES = {"unit", "cycle", "rul", "pseudo_timestamp", "is_imputed"}
features = [c for c in tr.columns if c not in NON_FEATURES]

missing = [c for c in features if c not in te.columns]
print("n features:", len(features))
print("train features missing from test:", missing)

for name, df in (("train", tr), ("test", te)):
    for H in (24, 30):
        rate = (df["rul"] <= H).mean()
        print(f"{name} positive rate H={H}: {rate:.3f}")

print("train engines:", tr["unit"].nunique(), "| test engines:", te["unit"].nunique())
print("NaN share in features (train):", round(tr[features].isna().mean().mean(), 4))
print("any feature equal to rul:", [c for c in features if tr[c].equals(tr["rul"])])