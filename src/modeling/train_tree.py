"""LightGBM failure-within-H classifier for C-MAPSS FD001.

Engine-level train/validation split, threshold chosen on validation only,
test set touched once at the end.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
H = 30
SEED = 42
NON_FEATURES = {"unit", "cycle", "rul", "pseudo_timestamp", "is_imputed"}


def engine_split(units, val_frac=0.2, seed=SEED):
    """Split engine IDs into disjoint train and validation lists."""
    ids = np.array(sorted({int(u) for u in units}))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_frac))
    return sorted(ids[n_val:].tolist()), sorted(ids[:n_val].tolist())


def row_metrics(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if len(np.unique(y)) == 2:
        out["pr_auc"] = float(average_precision_score(y, proba))
        out["roc_auc"] = float(roc_auc_score(y, proba))
    return out


def engine_alerts(df, proba, thr):
    """Per-engine first alert: RUL remaining when the first alert fired."""
    d = df[["unit", "cycle", "rul"]].copy()
    d["p"] = proba
    d = d.sort_values(["unit", "cycle"])
    first = d[d["p"] >= thr].groupby("unit").first()["rul"]
    n = d["unit"].nunique()
    return {
        "engines": int(n),
        "never_alerted": int(n - len(first)),
        "alerted_too_early_(rul>H)": int((first > H).sum()),
        "alerted_in_window_(rul<=H)": int((first <= H).sum()),
        "median_rul_at_first_alert": float(first.median()) if len(first) else None,
    }


def pick_threshold(y, proba):
    """Threshold maximizing F1 on validation (cost-based tuning comes later)."""
    p, r, t = precision_recall_curve(y, proba)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[int(np.argmax(f1))])


def main():
    import lightgbm as lgb

    tr_all = pd.read_parquet(ROOT / "data/processed/train_features.parquet")
    te = pd.read_parquet(ROOT / "data/processed/test_features.parquet")
    tr_all = tr_all.sort_values(["unit", "cycle"]).reset_index(drop=True)
    te = te.sort_values(["unit", "cycle"]).reset_index(drop=True)

    features = [c for c in tr_all.columns if c not in NON_FEATURES]
    train_ids, val_ids = engine_split(tr_all["unit"].unique())
    assert not set(train_ids) & set(val_ids)

    tr = tr_all[tr_all["unit"].isin(train_ids)]
    va = tr_all[tr_all["unit"].isin(val_ids)]
    y_tr = (tr["rul"] <= H).astype(int)
    y_va = (va["rul"] <= H).astype(int)
    y_te = (te["rul"] <= H).astype(int)

    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        scale_pos_weight=spw,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(
        tr[features], y_tr,
        eval_set=[(va[features], y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    p_va = model.predict_proba(va[features])[:, 1]
    thr = pick_threshold(y_va, p_va)

    p_te = model.predict_proba(te[features])[:, 1]
    last = te.groupby("unit").tail(1)
    p_last = model.predict_proba(last[features])[:, 1]
    y_last = (last["rul"] <= H).astype(int)

    results = {
        "H": H, "seed": SEED, "n_features": len(features),
        "best_iteration": int(model.best_iteration_ or 0),
        "scale_pos_weight": spw, "threshold_chosen_on_validation": thr,
        "validation_rows": row_metrics(y_va, p_va, thr),
        "validation_engines": engine_alerts(va, p_va, thr),
        "test_all_rows": row_metrics(y_te, p_te, thr),
        "test_engines": engine_alerts(te, p_te, thr),
        "test_last_cycle_only": row_metrics(y_last, p_last, thr),
    }

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": features, "threshold": thr, "H": H},
                models_dir / "lgbm_h30.joblib")
    with open(models_dir / "lgbm_h30_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()