"""Predictive maintenance dashboard (C-MAPSS FD001). Run: streamlit run app.py"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Engine Predictive Maintenance", layout="wide")


@st.cache_resource
def load_artifacts():
    return joblib.load(ROOT / "models/lgbm_h30.joblib")


@st.cache_data
def score_test():
    art = load_artifacts()
    te = pd.read_parquet(ROOT / "data/processed/test_features.parquet")
    te = te.sort_values(["unit", "cycle"]).reset_index(drop=True)
    te["proba"] = art["model"].predict_proba(te[art["features"]])[:, 1]
    return te


def add_alerts(df, thr, k):
    """ALERT only when probability stays >= thr for k consecutive cycles."""
    above = (df["proba"] >= thr).astype(int)
    run = above.groupby(df["unit"]).transform(lambda s: s.rolling(k, min_periods=k).sum())
    out = df.copy()
    out["alert"] = run >= k
    out["status"] = "OK"
    out.loc[out["proba"] >= thr * 0.5, "status"] = "WATCH"
    out.loc[out["alert"], "status"] = "ALERT"
    return out


art = load_artifacts()
H = art["H"]
scored = score_test()

st.title("Engine Predictive Maintenance")
st.caption(f"NASA C-MAPSS FD001 | LightGBM | alert = failure expected within {H} cycles")

with st.sidebar:
    st.header("Alert settings")
    thr = st.slider("Probability threshold", 0.05, 0.99, float(round(art["threshold"], 2)), 0.01)
    k = st.slider("Consecutive cycles required (k)", 1, 10, 3)
    st.caption("Higher k = fewer false alarms, but later alerts.")
    engine = st.selectbox("Engine", sorted(scored["unit"].unique()), index=33)

data = add_alerts(scored, thr, k)

# ---- Fleet overview (latest observed cycle per engine) ----
last = data.groupby("unit").tail(1)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Engines monitored", len(last))
c2.metric("ALERT", int((last["status"] == "ALERT").sum()))
c3.metric("WATCH", int((last["status"] == "WATCH").sum()))
c4.metric("OK", int((last["status"] == "OK").sum()))

fleet = (last[["unit", "cycle", "proba", "status"]]
         .rename(columns={"unit": "engine", "cycle": "latest cycle", "proba": "failure prob."})
         .sort_values("failure prob.", ascending=False))
order = {"ALERT": 0, "WATCH": 1, "OK": 2}
fleet = fleet.sort_values(["status", "failure prob."], key=lambda s: s.map(order) if s.name == "status" else -s)

left, right = st.columns([1, 2])
with left:
    st.subheader("Fleet status")
    st.dataframe(fleet.style.format({"failure prob.": "{:.2f}"}), height=520, hide_index=True)

# ---- Single-engine replay ----
with right:
    e = data[data["unit"] == engine].reset_index(drop=True)
    failure_cycle = int(e["cycle"].iloc[-1] + e["rul"].iloc[-1])
    st.subheader(f"Engine {engine}")
    upto = st.slider("Replay: cycles seen so far", int(e["cycle"].min()), int(e["cycle"].max()),
                     int(e["cycle"].max()))
    v = e[e["cycle"] <= upto]
    now = v.iloc[-1]

    a, b, c = st.columns(3)
    a.metric("Current status", now["status"])
    b.metric("Failure probability", f"{now['proba']:.2f}")
    first = v[v["alert"]]
    if len(first):
        lead = failure_cycle - int(first["cycle"].iloc[0])
        c.metric("First alert", f"cycle {int(first['cycle'].iloc[0])}", f"{lead} cycles before failure")
    else:
        c.metric("First alert", "none yet")

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(v["cycle"], v["proba"], label="P(failure within 30 cycles)")
    ax[0].axhline(thr, color="red", linestyle="--", label="threshold")
    ax[0].axvline(failure_cycle - H, color="gray", linestyle=":", label="30-cycle warning window starts")
    if len(first):
        ax[0].axvline(int(first["cycle"].iloc[0]), color="orange", label="first alert")
    ax[0].set_ylim(0, 1.02)
    ax[0].set_xlim(e["cycle"].min(), e["cycle"].max())
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_ylabel("probability")
    for col in ("sensor_11", "sensor_4", "sensor_15"):
        if col in v.columns:
            ax[1].plot(v["cycle"], v[col], label=col)
    ax[1].set_xlabel("cycle")
    ax[1].set_ylabel("sensor value")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

st.caption("The failure-window marker uses the dataset's true RUL, for evaluation only. "
           "A live system would not know it. Test trajectories are cut off before failure, "
           "so many engines legitimately show no alert.")