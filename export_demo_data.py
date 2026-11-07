import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
art = joblib.load(ROOT / "models/lgbm_h30.joblib")
te = pd.read_parquet(ROOT / "data/processed/test_features.parquet")
te = te.sort_values(["unit", "cycle"]).reset_index(drop=True)
te["proba"] = art["model"].predict_proba(te[art["features"]])[:, 1]

keep = ["unit", "cycle", "rul", "proba", "sensor_11", "sensor_4", "sensor_15"]
df = te[keep].copy()
df["unit"] = df["unit"].astype(int)
df["cycle"] = df["cycle"].astype(int)
df["proba"] = df["proba"].round(4)
for c in ("sensor_11", "sensor_4", "sensor_15"):
    df[c] = df[c].round(3)

out = ROOT / "demo"
out.mkdir(exist_ok=True)
df.to_parquet(out / "scored_test.parquet", index=False)
with open(out / "meta.json", "w") as f:
    json.dump({"H": art["H"], "threshold": art["threshold"]}, f)
print("Saved", df.shape)