"""Target definition and label construction for C-MAPSS FD001."""
import pandas as pd
import numpy as np

import config


def compute_rul(df: pd.DataFrame, max_cycle: bool = True, rul_series: pd.Series = None) -> pd.DataFrame:
    """Compute Remaining Useful Life (RUL) per row.

    max_cycle=True (train): RUL = last cycle of the engine - current cycle.
    max_cycle=False (test): RUL = rul_series[unit] + (last cycle seen - current cycle),
    where rul_series is indexed by unit id (1..100).
    """
    df = df.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    if max_cycle:
        df["rul"] = last_cycle - df["cycle"]
    else:
        if rul_series is None:
            raise ValueError("rul_series is required when max_cycle=False")
        final_rul = df["unit"].map(rul_series)
        df["rul"] = final_rul + (last_cycle - df["cycle"])
    return df

def failure_within_H(rul: pd.Series, H: int = config.TARGET_H) -> pd.Series:
    """Binary target: 1 if RUL <= H cycles.
    
    Args:
        rul: Series of RUL values
        H: Cycles ahead to flag failure (default: config.TARGET_H = 30)
    
    Returns:
        Binary series where 1 indicates failure within H cycles
    """
    return (rul <= H).astype(int)


def compute_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RUL for training data.
    
    Train RUL = max_cycle(unit) - cycle
    """
    return compute_rul(df, max_cycle=True)


def compute_test_rul(df: pd.DataFrame, rul_file) -> pd.DataFrame:
    """Attach RUL to test rows using RUL_FD001 (value at each engine's last cycle)."""
    if isinstance(rul_file, pd.DataFrame):
        rul_series = rul_file["RUL"] if "RUL" in rul_file.columns else rul_file.iloc[:, 0]
    else:
        rul_series = pd.Series(np.asarray(rul_file).ravel())
    rul_series = rul_series.reset_index(drop=True)
    rul_series.index = range(1, len(rul_series) + 1)  # engine ids 1..100
    return compute_rul(df, max_cycle=False, rul_series=rul_series)


def binary_target(rul, horizon=None, H=None):
    """1 if RUL <= horizon, else 0. Accepts list, ndarray or Series."""
    h = H if H is not None else horizon
    if h is None:
        h = config.TARGET_H
    arr = np.asarray(rul)
    out = (arr <= h).astype(int)
    if isinstance(rul, pd.Series):
        return pd.Series(out, index=rul.index, name="failure_within_H")
    return out

def test_boundary_cases() -> dict:
    """Test boundary cases for label construction.
    
    Tests: RUL exactly H, last cycle, RUL file alignment.
    """
    results = {}
    
    # Test 1: RUL exactly H should be classified as failure within H
    rul_test = pd.Series([config.TARGET_H, config.TARGET_H - 1, config.TARGET_H + 1])
    target = binary_target(rul_test)
    results["rul_exact_H"] = {
        "input": rul_test.tolist(),
        "output": target.tolist(),
        "expected": [1, 1, 0],  # H is inclusive: RUL <= H -> 1
    }
    
    # Test 2: RUL at last cycle should be 0
    # (Engine at failure point has RUL = 0)
    rul_last = pd.Series([0, 1, 5, 10])
    target_last = binary_target(rul_last)
    results["rul_last_cycle"] = {
        "input": rul_last.tolist(),
        "output": target_last.tolist(),
        "expected": [1, 1, 1, 1],  # All <= H (H=30)
    }
    
    # Test 3: H = 24 vs H = 30
    rul_H24 = pd.Series([24, 25, 30, 31])
    target_H24 = binary_target(rul_H24, H=24)
    target_H30 = binary_target(rul_H24, H=30)
    results["H_comparison"] = {
        "input": rul_H24.tolist(),
        "target_H24": target_H24.tolist(),
        "target_H30": target_H30.tolist(),
        "expected_diff": "H=24 should have fewer positives than H=30",
    }
    
    return results