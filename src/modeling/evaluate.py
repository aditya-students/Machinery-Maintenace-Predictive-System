"""Evaluation module for C-MAPSS predictive maintenance models.
Shared evaluation for all models on the SAME validation and test sets.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score


def row_level_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """Compute row-level metrics.
    
    Returns Precision, Recall, F1, PR-AUC, ROC-AUC, confusion matrix, false-positive rate.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    
    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "false_positive_rate": confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[0, 1]) 
            if (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[0, 1]) > 0 else float("nan"),
    }
    
    if y_proba is not None:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            results["roc_auc"] = float("nan")
        try:
            results["pr_auc"] = average_precision_score(y_true, y_proba)
        except ValueError:
            results["pr_auc"] = float("nan")
    else:
        results["roc_auc"] = float("nan")
        results["pr_auc"] = float("nan")
    
    return results


def engine_level_metrics(
    df: pd.DataFrame,
    alert_col: str = "alert",
    true_failure_col: str = "failure_within_H",
    cycle_col: str = "cycle",
    unit_col: str = "unit",
) -> dict:
    """Compute engine-level metrics.
    
    Returns:
    - first alert cycle per engine
    - lead time (failure cycle minus alert cycle)
    - share of engines warned at least H cycles ahead
    - false-alarm episodes per engine
    - missed failures
    """
    engines = df[unit_col].unique()
    
    first_alert_cycle = {}
    lead_times = {}
    engines_warned_H = 0
    false_alarm_episodes = {}
    missed_failures = 0
    
    for eng in engines:
        eng_df = df[df[unit_col] == eng].sort_values(cycle_col)
        
        # Find first alert cycle
        alert_rows = eng_df[eng_df[alert_col] == 1]
        if len(alert_rows) > 0:
            first_alert_cycle[eng] = int(alert_rows.iloc[0][cycle_col])
            # Lead time: failure cycle - alert cycle
            # A failure is indicated by failure_within_H = 1 at the last cycle
            failure_rows = eng_df[eng_df[true_failure_col] == 1]
            if len(failure_rows) > 0:
                failure_cycle = int(failure_rows.iloc[-1][cycle_col])
                lead_times[eng] = failure_cycle - first_alert_cycle[eng]
            else:
                lead_times[eng] = None
        else:
            first_alert_cycle[eng] = None
            lead_times[eng] = None
        
        # Check if engine was warned at least H cycles ahead
        if first_alert_cycle[eng] is not None:
            # Find the last failure cycle for this engine
            failure_cycle = eng_df[eng_df[true_failure_col] == 1][cycle_col].max() if len(eng_df[eng_df[true_failure_col] == 1]) > 0 else None
            if failure_cycle is not None and first_alert_cycle[eng] <= failure_cycle - config.TARGET_H:
                engines_warned_H += 1
        
        # False alarm episodes: cycles above threshold without actual failure
        # Count consecutive cycles above threshold as episodes
        alert_series = eng_df[alert_col].values
        if len(alert_series) > 0:
            # Count episodes (consecutive groups of 1s)
            diff = np.diff(np.where(np.concatenate(([0], alert_series, [0])) > 0)[0])
            episodes = int(np.sum(diff > 1))  # Each time we have a group ending
            false_alarm_episodes[eng] = episodes
        else:
            false_alarm_episodes[eng] = 0
        
        # Check for missed failure (never warned)
        if first_alert_cycle[eng] is None:
            # Get the last cycle of the engine
            last_cycle = int(eng_df[cycle_col].max())
            # Check if there was a failure in this engine
            if len(eng_df[eng_df[true_failure_col] == 1]) > 0:
                missed_failures += 1
    
    return {
        "first_alert_cycle": first_alert_cycle,
        "lead_times": lead_times,
        "engines_warned_at_least_H_cycles": engines_warned_H,
        "false_alarm_episodes_per_engine": false_alarm_episodes,
        "missed_failures": missed_failures,
        "n_engines": len(engines),
    }


def evaluate_model(
    model_name: str,
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_val_proba: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_proba: np.ndarray,
    test_df: pd.DataFrame = None,
) -> dict:
    """Shared evaluation for all models on SAME validation and test sets.
    
    Returns metrics for both validation and test sets.
    """
    # Validation metrics
    val_metrics = row_level_metrics(y_val_true, y_val_pred, y_val_proba)
    
    # Test metrics
    test_metrics = row_level_metrics(y_test_true, y_test_pred, y_test_proba)
    
    # Engine-level evaluation on test set (if test_df provided)
    if test_df is not None:
        # Need to have prediction columns in test_df
        # This is a simplified engine-level analysis
        eng_metrics = engine_level_metrics(test_df)
    else:
        eng_metrics = {
            "first_alert_cycle": {},
            "lead_times": {},
            "engines_warned_at_least_H_cycles": 0,
            "false_alarm_episodes_per_engine": {},
            "missed_failures": 0,
            "n_engines": 0,
        }
    
    return {
        "model_name": model_name,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "engine_level": eng_metrics,
    }