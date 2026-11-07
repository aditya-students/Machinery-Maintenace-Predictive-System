"""Cycle-grid verification and causal missing-value routine for C-MAPSS FD001."""
import os
import pandas as pd
import numpy as np

import config


def verify_cycle_grid(df: pd.DataFrame, name: str = "dataset") -> dict:
    """Verify cycle grid: strictly increasing, starts at 1, no gaps.
    
    FD001 has no gaps, but this function verifies the grid and provides
    a report. 'Resample to hourly' becomes this grid verification.
    """
    errors = []
    warnings = []
    
    if "unit" not in df.columns or "cycle" not in df.columns:
        errors.append(f"{name}: Missing 'unit' or 'cycle' columns")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    for unit_id, group in df.groupby("unit"):
        cycles = group["cycle"].values
        
        # Check strictly increasing
        if not np.all(np.diff(cycles) > 0):
            non_increasing = np.where(np.diff(cycles) <= 0)[0]
            errors.append(f"{name} Unit {unit_id}: cycles not strictly increasing at indices {non_increasing}")
        
        # Check starts at 1
        if cycles[0] != 1:
            errors.append(f"{name} Unit {unit_id}: cycle starts at {cycles[0]}, expected 1")
        
        # Check no gaps (increment by exactly 1)
        if not np.all(np.diff(cycles) == 1):
            gaps = np.where(np.diff(cycles) != 1)[0]
            warnings.append(f"{name} Unit {unit_id}: cycle gaps detected at indices {gaps} (FD001 should have no gaps)")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def add_synthetic_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Add a synthetic pseudo_timestamp column (clearly labelled as synthetic).
    
    Since FD001 has no absolute timestamps, this creates a synthetic timestamp
    based on cycle number, clearly marked as synthetic for compatibility with
    models that expect timestamps.
    """
    df = df.copy()
    df["pseudo_timestamp"] = df["cycle"].astype(float)  # Synthetic, clearly labelled
    return df


def causal_forward_fill(df: pd.DataFrame, gap_limit: int = None) -> pd.DataFrame:
    """Causal forward-fill only (no backward fill, no centered interpolation).
    
    FD001 has no gaps, but this routine is tested and ready for datasets with gaps.
    """
    df = df.copy()
    
    # Forward-fill within each engine (causal: only look backward)
    for col in df.columns:
        if col not in ["unit", "cycle", "is_imputed"]:
            df[col] = df.groupby("unit")[col].transform(
                lambda x: x.ffill(limit=gap_limit)
            )
    
    return df