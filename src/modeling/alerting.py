"""Alerting module for C-MAPSS predictive maintenance.
Alert logic beyond a raw threshold: require k consecutive cycles above threshold
or use EWMA of probabilities, plus a latch/cool-down.
"""

import numpy as np
import pandas as pd


def consecutive_cycle_alert(
    probabilities: pd.Series,
    threshold: float,
    k: int = 3,
) -> pd.Series:
    """Require k consecutive cycles above threshold to raise an alert.
    
    Args:
        probabilities: Series of prediction probabilities (indexed by row, ordered by cycle within engine)
        threshold: Probability threshold
        k: Number of consecutive cycles above threshold required to raise alert
    
    Returns:
        Binary series (1 = alert raised, 0 = no alert)
    """
    # Compute whether each cycle is above threshold
    above_threshold = (probabilities >= threshold).astype(int)
    
    # Count consecutive cycles above threshold
    # Use a running count that resets when below threshold
    consecutive = np.zeros(len(above_threshold), dtype=int)
    running_count = 0
    
    for i in range(len(above_threshold)):
        if above_threshold.iloc[i] == 1:
            running_count += 1
            consecutive[i] = running_count
        else:
            running_count = 0
            consecutive[i] = 0
    
    # Alert if consecutive >= k
    # We mark the k-th consecutive cycle as the alert
    alert = (consecutive >= k).astype(int)
    
    return alert


def ewma_alert(
    probabilities: pd.Series,
    threshold: float,
    alpha: float = 0.3,
) -> pd.Series:
    """Use EWMA of probabilities, alert when EWMA exceeds threshold.
    
    Args:
        probabilities: Series of prediction probabilities
        threshold: EWMA threshold
        alpha: EW smoothing factor
    
    Returns:
        Binary series (1 = alert raised)
    """
    # Compute EWMA
    ewma = probabilities.ewm(alpha=alpha, adjust=False).mean()
    alert = (ewma > threshold).astype(int)
    return alert


def latch_alert(
    probabilities: pd.Series,
    threshold: float,
    k: int = 3,
    cool_down: int = 5,
) -> dict:
    """Latch/cool-down alert: one alert per degradation episode.
    
    Args:
        probabilities: Series of prediction probabilities
        threshold: Probability threshold
        k: Minimum consecutive cycles above threshold to trigger alert
        cool_down: Cycles to wait before another alert can be raised
    
    Returns:
        dict with 'alert_series' and 'latch_info'
    """
    # First, apply consecutive cycle rule
    alert_series = consecutive_cycle_alert(probabilities, threshold, k)
    
    # Apply latch/cool-down: once an alert is raised, require cool_down cycles
    # before another alert can be raised
    latched = alert_series.copy()
    alert_count = 0
    last_alert_idx = -cool_down  # Start in cool-down state
    
    for i in range(len(latched)):
        if latched[i] == 1 and i - last_alert_idx >= cool_down:
            alert_count += 1
            latched[i] = 1  # Keep this as the alert
            last_alert_idx = i
        elif latched[i] == 1:
            latched[i] = 0  # Suppress due to cool-down
    
    latch_info = {
        "n_alerts": int(latched.sum()),
        "n_raw_alerts": int(alert_series.sum()),
        "cool_down": cool_down,
        "alert_count": alert_count,
    }
    
    return {
        "alert_series": latched,
        "latch_info": latch_info,
    }