"""Predictive maintenance dashboard (NASA C-MAPSS FD001). Run: streamlit run app.py"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Engine Predictive Maintenance", layout="wide")


@st.cache_data
def load_demo():
    df = pd.read_parquet(ROOT / "demo" / "scored_test.parquet")
    with open(ROOT / "demo" / "meta.json") as f:
        meta = json.load(f)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True), meta


def add_alerts(df, thr, k):
    """ALERT when proba >= thr for k consecutive cycles within one engine."""
    out = df.copy()
    above = (out["proba"] >= thr).astype(int)
    run = above.groupby(out["unit"]).transform(lambda s: s.rolling(k, min_periods=k).sum())
    out["alert"] = run >= k
    out["status"] = "OK"
    out.loc[out["proba"] >= thr * 0.5, "status"] = "WATCH"
    out.loc[out["alert"], "status"] = "ALERT"
    return out


scored, meta = load_demo()
H = meta["H"]

st.title("Engine Predictive Maintenance")
st.caption(f"NASA C-MAPSS FD001 | LightGBM | alert = failure expected within {H} cycles")

with st.sidebar:
    st.header("Alert settings")
    thr = st.slider("Probability threshold", 0.05, 0.99, float(round(meta["threshold"], 2)), 0.01)
    k = st.slider("Consecutive cycles required (k)", 1, 10, 3)
    st.caption("Higher k means fewer false alarms but later alerts.")
    engines = sorted(scored["unit"].unique())
    engine = st.selectbox("Engine", engines, index=engines.index(34) if 34 in engines else 0)

data = add_alerts(scored, thr, k)

last = data.groupby("unit").tail(1).copy()
last["rank"] = last["status"].map({"ALERT": 0, "WATCH": 1, "OK": 2})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Engines monitored", len(last))
c2.metric("ALERT", int((last["status"] == "ALERT").sum()))
c3.metric("WATCH", int((last["status"] == "WATCH").sum()))
c4.metric("OK", int((last["status"] == "OK").sum()))

fleet = last.sort_values(["rank", "proba"], ascending=[True, False])[["unit", "cycle", "proba", "status"]]
fleet.columns = ["engine", "latest cycle", "failure prob.", "status"]

left, right = st.columns([1, 2])
with left:
    st.subheader("Fleet status")
    st.dataframe(fleet.style.format({"failure prob.": "{:.2f}"}), height=520, hide_index=True)

with right:
    e = data[data["unit"] == engine].reset_index(drop=True)
    failure_cycle = int(e["cycle"].iloc[-1] + e["rul"].iloc[-1])
    st.subheader(f"Engine {engine}")
    upto = st.slider("Replay: cycles seen so far", int(e["cycle"].min()),
                     int(e["cycle"].max()), int(e["cycle"].max()))
    v = e[e["cycle"] <= upto]
    now = v.iloc[-1]
    alerts = v[v["alert"]]

    a, b, c = st.columns(3)
    a.metric("Current status", now["status"])
    b.metric("Failure probability", f"{now['proba']:.2f}")
    if len(alerts):
        fc = int(alerts["cycle"].iloc[0])
        c.metric("First alert", f"cycle {fc}", f"{failure_cycle - fc} cycles before failure")
    else:
        c.metric("First alert", "none yet")

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(v["cycle"], v["proba"], label=f"P(failure within {H} cycles)")
    ax[0].axhline(thr, color="red", linestyle="--", label="threshold")
    ax[0].axvline(failure_cycle - H, color="gray", linestyle=":", label="warning window starts")
    if len(alerts):
        ax[0].axvline(int(alerts["cycle"].iloc[0]), color="orange", label="first alert")
    ax[0].set_ylim(0, 1.02)
    ax[0].set_xlim(e["cycle"].min(), e["cycle"].max())
    ax[0].set_ylabel("probability")
    ax[0].legend(fontsize=8, loc="upper left")
    for col in ("sensor_11", "sensor_4", "sensor_15"):
        ax[1].plot(v["cycle"], v[col], label=col)
    ax[1].set_xlabel("cycle")
    ax[1].set_ylabel("sensor value")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.caption("The failure window uses the dataset’s true RUL, for evaluation only. Test trajectories "
           "are cut off before failure, so many engines show no alert. Threshold and k are demo "
           "defaults, not tuned values. Data: NASA C-MAPSS FD001.")