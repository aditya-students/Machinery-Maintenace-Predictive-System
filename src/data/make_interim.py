"""Create interim parquet files from raw C-MAPSS data."""
import os
import pandas as pd
import numpy as np

import config
from src.data.load import load_train, load_test, load_rul
from src.data.clean import clean_train, clean_test
from src.data.grid import add_synthetic_timestamp, verify_cycle_grid
from labels import compute_train_rul, compute_test_rul


def make_interim():
    """Full pipeline: load -> clean -> attach RUL -> write parquet."""
    
    # Create output directory
    os.makedirs(config.DATA_INTERIM_DIR, exist_ok=True)
    
    # Load raw data
    print("Loading raw data...")
    train_raw = load_train(config.DATA_RAW_DIR)
    test_raw = load_test(config.DATA_RAW_DIR)
    rul_raw = load_rul(config.DATA_RAW_DIR)
    
    # Verify cycle grids
    print("Verifying cycle grids...")
    train_grid = verify_cycle_grid(train_raw, "train")
    test_grid = verify_cycle_grid(test_raw, "test")
    print(f"  Train grid valid: {train_grid['valid']}")
    print(f"  Test grid valid: {test_grid['valid']}")
    
    # Clean training data
    print("Cleaning training data...")
    train_clean, clean_info = clean_train(config.DATA_RAW_DIR)
    print(f"  Train: {len(train_clean)} rows after cleaning")
    print(f"  Constants dropped: {clean_info['constants_dropped']}")
    
    # Clean test data
    print("Cleaning test data...")
    test_clean, _ = clean_test(config.DATA_RAW_DIR, train_cols=train_clean.columns.tolist())
    print(f"  Test: {len(test_clean)} rows after cleaning")
    
    # Attach RUL to training data
    print("Attaching RUL to training data...")
    train_with_rul = compute_train_rul(train_clean)
    
    # Attach RUL to test data
    print("Attaching RUL to test data...")
    test_with_rul = compute_test_rul(test_raw, rul_raw)
    
    # Add synthetic timestamps (clearly labelled)
    print("Adding synthetic timestamps...")
    train_with_ts = add_synthetic_timestamp(train_with_rul)
    test_with_ts = add_synthetic_timestamp(test_with_rul)
    
    # Write parquet files
    train_path = os.path.join(config.DATA_INTERIM_DIR, "train_clean.parquet")
    test_path = os.path.join(config.DATA_INTERIM_DIR, "test_clean.parquet")
    
    train_with_ts.to_parquet(train_path, index=False)
    test_with_ts.to_parquet(test_path, index=False)
    
    print(f"Wrote {train_path} ({len(train_with_ts)} rows)")
    print(f"Wrote {test_path} ({len(test_with_ts)} rows)")
    
    # Set read-only on raw files (as per spec)
    import stat
    raw_dir = config.DATA_RAW_DIR
    for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"):
        for folder in (raw_dir, os.path.join(raw_dir, "CMAPSSData")):
            f = os.path.join(folder, name)
            if os.path.exists(f):
                mode = os.stat(f).st_mode
                os.chmod(f, mode & ~(stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH))
    
    print("Raw files set to read-only.")
    
    return {
        "train_path": train_path,
        "test_path": test_path,
        "train_rows": len(train_with_ts),
        "test_rows": len(test_with_ts),
    }


if __name__ == "__main__":
    make_interim()