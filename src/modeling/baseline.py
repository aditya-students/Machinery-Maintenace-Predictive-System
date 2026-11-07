"""Baseline models for C-MAPSS FD001 predictive maintenance."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.dummy import DummyClassifier

import config


def majority_class_baseline(y_train: pd.Series) -> DummyClassifier:
    """Majority class baseline predictor."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(pd.DataFrame(index=range(len(y_train))), y_train)
    return model


def logistic_regression_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> dict:
    """Logistic regression baseline with class weighting."""
    model = LogisticRegression(
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        max_iter=1000,
    )
    model.fit(X_train, y_train)

    # Predictions
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average="binary")
    cm = confusion_matrix(y_val, y_val_pred)
    try:
        roc_auc = roc_auc_score(y_val, y_val_proba)
    except ValueError:
        roc_auc = float("nan")

    pr_auc = np.trapz(*precision_recall_curve(y_val, y_val_proba)[:2])

    return {
        "model": model,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }


def rule_based_alert(
    df: pd.DataFrame, sensor_col: str = "sensor_5", threshold: float = None
) -> pd.DataFrame:
    """Simple rule-based alert using a single sensor threshold.

    Flags failure when sensor exceeds threshold.
    """
    if threshold is None:
        # Use median as default threshold (computed on train data only in practice)
        threshold = df[sensor_col].median()

    df = df.copy()
    df["alert"] = (df[sensor_col] > threshold).astype(int)

    return df