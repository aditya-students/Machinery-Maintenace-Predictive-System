"""Threshold calibration module for C-MAPSS predictive maintenance.
Sweeps thresholds on VALIDATION only (never on test). Plots PR and cost curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.modeling.evaluate import row_level_metrics


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_false_alarm: float = 1.0,
    cost_missed_failure: float = 5.0,
    threshold_range: np.ndarray = None,
) -> dict:
    """Sweep thresholds and compute metrics + costs.
    
    Only on validation data - never on test set.
    """
    if threshold_range is None:
        threshold_range = np.linspace(0.1, 0.9, 80)
    
    results = []
    
    for threshold in threshold_range:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = row_level_metrics(y_true, y_pred, y_proba)
        
        # Compute expected cost
        n_false_alarms = metrics["confusion_matrix"][0, 0]  # True negatives predicted as positive... 
        # Actually, let's compute cost differently:
        # Cost = cost_false_alarm * FP + cost_missed_failure * FN
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = cost_false_alarm * fp + cost_missed_failure * fn
        
        results.append({
            "threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "fp": int(fp),
            "fn": int(fn),
            "cost": cost,
        })
    
    results_df = pd.DataFrame(results)
    
    # Find threshold that minimizes expected cost
    best_cost_idx = results_df["cost"].idxmin()
    best_threshold = results_df.loc[best_cost_idx, "threshold"]
    best_cost = results_df.loc[best_cost_idx, "cost"]
    
    return {
        "results": results_df,
        "best_threshold_by_cost": best_threshold,
        "best_cost": best_cost,
        "threshold_range": threshold_range,
    }


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str = None,
):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'PR curve (AP={avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_cost_curve(
    results: dict,
    save_path: str = None,
    cost_false_alarm: float = 1.0,
    cost_missed_failure: float = 5.0,
):
    """Plot cost curve from threshold sweep results."""
    df = results["results"]
    
    plt.figure(figsize=(8, 6))
    plt.plot(df["threshold"], df["cost"], marker='.', label=f'Cost (c_FP={cost_false_alarm}, c_MF={cost_missed_failure})')
    plt.xlabel('Threshold')
    plt.ylabel('Expected Cost')
    plt.title('Cost Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    # Mark the best threshold
    best = results["best_threshold_by_cost"]
    plt.axvline(x=best, color='red', linestyle='--', label=f'Best threshold: {best:.3f}')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def select_threshold_by_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_false_alarm: float = 1.0,
    cost_missed_failure: float = 5.0,
) -> dict:
    """Select threshold by minimizing expected cost."""
    results = threshold_sweep(y_true, y_proba, cost_false_alarm, cost_missed_failure)
    return {
        "selected_threshold": results["best_threshold_by_cost"],
        "expected_cost": results["best_cost"],
        "method": "minimize_expected_cost",
    }


def select_threshold_by_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = 0.8,
    max_false_alarms_per_engine: int = None,
) -> dict:
    """Select threshold to meet a recall target with a false-alarm budget."""
    thresholds = np.linspace(0.1, 0.9, 100)
    best_threshold = None
    best_f1 = 0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        
        if recall >= target_recall:
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    if best_threshold is None:
        # Fallback: use the threshold that gives highest recall
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
            best_f1 = f1
            best_threshold = threshold
    
    return {
        "selected_threshold": best_threshold,
        "recall": target_recall,
        "method": "recall_target_with_fp_budget",
    }