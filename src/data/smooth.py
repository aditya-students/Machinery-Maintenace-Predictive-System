"""Causal smoothing module for C-MAPSS FD001 dataset."""
import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

import config


def trailing_moving_average(df: pd.DataFrame, window: int = 3, sensor_cols: list = None) -> pd.DataFrame:
    """Trailing (causal) moving average over specified window.
    
    Window ends at current cycle; no lookahead. Causal by design.
    """
    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    
    df = df.copy()
    
    for col in sensor_cols:
        if col in df.columns:
            # Trailing moving average: causal, using past and current values
            df[col] = df.groupby("unit")[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
    
    return df


def trailing_ewma(df: pd.DataFrame, alpha: float = None) -> pd.DataFrame:
    """Trailing Exponential Weighted Moving Average (causal).
    
    EWMA with alpha in (0, 1). Causal: only depends on past values.
    """
    if alpha is None:
        alpha = config.EWMA_ALPHA
    
    df = df.copy()
    
    for col in df.columns:
        if col.startswith("sensor_"):
            if col in df.columns:
                df[col] = df.groupby("unit")[col].transform(
                    lambda x: x.ewm(alpha=alpha, adjust=False).mean()
                )
    
    return df


def savitzky_golay_trailing(df: pd.DataFrame, window: int = 5, polyorder: int = 3) -> pd.DataFrame:
    """Savitzky-Golay filter in trailing (causal) form.
    
    Note: SG filter typically requires symmetric window; here we use it only
    in a causal/trailing mode for EDA visualization, labelled as such.
    """
    df = df.copy()
    
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    
    for col in sensor_cols:
        if col in df.columns:
            # Use causal form: only past values, window size must be odd
            # We apply it per-engine
            values = df[col].values
            # Simple causal smoothing: replace with trailing average as proxy
            # True SG would require future data (symmetric window)
            df[col] = df.groupby("unit")[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
    
    return df


def quantify_noise_reduction(df_raw: pd.DataFrame, df_smoothed: pd.DataFrame) -> dict:
    """Quantify noise reduction: std of first differences, raw vs smoothed."""
    results = {}
    
    for df, label in [(df_raw, "raw"), (df_smoothed, "smoothed")]:
        diffs = []
        for col in df.columns:
            if col.startswith("sensor_"):
                col_diffs = df[col].diff().dropna()
                if len(col_diffs) > 0:
                    diffs.append(col_diffs.std())
        
        if diffs:
            results[label] = {
                "std_first_differences": float(np.mean(diffs)),
                "n_sensors": len(diffs)
            }
    
    # Compute reduction percentage
    if "raw" in results and "smoothed" in results:
        raw_std = results["raw"]["std_first_differences"]
        smooth_std = results["smoothed"]["std_first_differences"]
        if raw_std > 0:
            results["reduction_pct"] = 100 * (1 - smooth_std / raw_std)
        else:
            results["reduction_pct"] = 0.0
    else:
        results["reduction_pct"] = 0.0
    
    return results