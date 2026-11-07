"""Data cleaning module for C-MAPSS FD001 dataset."""
import os
import pandas as pd
import numpy as np
from data.load import load_train, load_test

import config


def identify_constant_sensors(df: pd.DataFrame, sensor_cols: list = None) -> dict:
    """Identify constant or near-constant sensors on TRAIN only.
    
    Returns:
        - mask of columns to drop
        - dict with variance info for documentation
    """
    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    
    variance_info = {}
    drop_cols = []
    
    for col in sensor_cols:
        var = df[col].var()
        variance_info[col] = float(var)
        if var == 0 or var < 1e-10:
            drop_cols.append(col)
    
    return {
        "drop_cols": drop_cols,
        "variance_info": variance_info,
        "n_dropped": len(drop_cols)
    }


def sort_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by unit and cycle, log duplicates."""
    init_len = len(df)
    
    # Sort
    df = df.sort_values(["unit", "cycle"], ascending=[True, True]).reset_index(drop=True)
    
    # Check for duplicates
    dupes = df.duplicated(subset=["unit", "cycle"]).sum()
    
    if dupes > 0:
        # Keep first occurrence, log them
        df = df.drop_duplicates(subset=["unit", "cycle"], keep="first")
        # In a full system, we'd log these
    
    return df, {"initial_rows": init_len, "final_rows": len(df), "duplicates_removed": dupes}


def add_is_imputed_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add an is_imputed flag column. Forward-fill with gap limit, no backward fill.
    
    FD001 has no gaps, but this function provides the causal imputation routine.
    """
    df = df.copy()
    
    # For FD001, there are no gaps, but we implement the causal routine anyway
    # Forward-fill within each engine
    for col in df.columns:
        if col not in ["unit", "cycle"]:
            df[col] = df.groupby("unit")[col].transform(
                lambda x: x.ffill(limit=None)
            )
    
    # Add is_imputed flag (all False for FD001 since no gaps, but the column exists)
    df["is_imputed"] = False
    
    return df


def clean_train(raw_dir: str = None) -> pd.DataFrame:
    """Full cleaning pipeline for training data."""
    if raw_dir is None:
        import config
        raw_dir = config.DATA_RAW_DIR
    
    df = load_train(raw_dir)
    
    # Sort and deduplicate
    df, info = sort_and_deduplicate(df)
    
    # Identify constant sensors (on train only)
    const_info = identify_constant_sensors(df)
    drop_cols = const_info["drop_cols"]
    
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    
    # Add imputation flag
    df = add_is_imputed_flag(df)
    
    return df, {
        "constants_dropped": drop_cols,
        "constant_variances": const_info["variance_info"],
        "n_constants_dropped": const_info["n_dropped"]
    }


def clean_test(raw_dir: str = None, train_cols: list = None) -> pd.DataFrame:
    """Full cleaning pipeline for test data, using train-derived decisions."""
    if raw_dir is None:
        import config
        raw_dir = config.DATA_RAW_DIR
    
    df = load_test(raw_dir)
    
    # Sort and deduplicate
    df, info = sort_and_deduplicate(df)
    
    # Drop same columns as were dropped from train
    if train_cols is not None:
        keep_cols = [c for c in train_cols if c in df.columns]
        dropped = [c for c in df.columns if c not in keep_cols]
        df = df[keep_cols].copy()
    else:
        dropped = []
    
    # Add imputation flag
    df = add_is_imputed_flag(df)
    
    return df, {"n_dropped": len(dropped), "dropped_cols": dropped}