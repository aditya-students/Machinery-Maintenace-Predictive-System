"""Plot failure probability and alert status for one test engine."""
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
engine = int(sys.argv[1]) if len(sys.argv) > 1 else 34

art = joblib.load(ROOT / "models/lgbm_h30.joblib")
model, features, thr, H = art["model"], art["features"], art["threshold"], art["H"]

te = pd.read_parquet(ROOT / "data/processed/test_features.parquet")
e = te[te["unit"] == engine].sort_values("cycle").reset_index(drop=True)
e["proba"] = model.predict_proba(e[features])[:, 1]
e["status"] = pd.cut(e["proba"], [-1, thr * 0.5, thr, 2], labels=["OK", "WATCH", "ALERT"])

failure_cycle = int(e["cycle"].iloc[-1] + e["rul"].iloc[-1])
alerts = e[e["proba"] >= thr]
first_alert = int(alerts["cycle"].iloc[0]) if len(alerts) else None

print(f"Engine {engine}: {len(e)} cycles observed, true failure at cycle {failure_cycle}")
print(f"Threshold: {thr:.3f}")
if first_alert:
    print(f"First alert at cycle {first_alert} -> {failure_cycle - first_alert} cycles before failure")
else:
    print("No alert fired in the observed cycles")
print(e[["cycle", "rul", "proba", "status"]].tail(15).to_string(index=False))

fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax[0].plot(e["cycle"], e["proba"], label="P(failure within 30 cycles)")
ax[0].axhline(thr, color="red", linestyle="--", label=f"alert threshold {thr:.2f}")
if first_alert:
    ax[0].axvline(first_alert, color="orange", label=f"first alert (cycle {first_alert})")
ax[0].axvline(failure_cycle - H, color="gray", linestyle=":", label="start of 30-cycle warning window")
ax[0].set_ylabel("probability")
ax[0].set_title(f"Engine {engine}: failure probability (true failure at cycle {failure_cycle})")
ax[0].legend(loc="upper left", fontsize=8)

for col in ["sensor_11", "sensor_4"]:
    if col in e.columns:
        ax[1].plot(e["cycle"], e[col], label=col)
ax[1].set_xlabel("cycle")
ax[1].set_ylabel("sensor value")
ax[1].legend(fontsize=8)

out = ROOT / "reports" / "figures"
out.mkdir(parents=True, exist_ok=True)
path = out / f"engine_{engine}.png"
fig.tight_layout()
fig.savefig(path, dpi=120)
print("Saved", path)
if os.name == "nt":
    os.startfile(path)