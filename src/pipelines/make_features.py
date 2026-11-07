"""Engineer and save features from interim data."""
import os
import json
import pandas as pd

import config
from src.features import create_all_features
from src.config import load_config


def make_features():
    """Engineer all features and save processed parquet files."""
    
    # Create output directory
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    
    # Load interim data
    train_path = os.path.join(config.DATA_INTERIM_DIR, "train_clean.parquet")
    test_path = os.path.join(config.DATA_INTERIM_DIR, "test_clean.parquet")
    
    print(f"Loading interim data from {train_path} and {test_path}...")
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"Train: {len(train_df)} rows, columns: {len(train_df.columns)}")
    print(f"Test: {len(test_df)} rows, columns: {len(test_df.columns)}")
    
    # Engine-level feature engineering (ALL computations grouped by unit)
    print("Engineering features (config-driven, grouped by unit)...")
    train_features = create_all_features(train_df)
    test_features = create_all_features(test_df)
    
    print(f"Train features: {len(train_features)} rows, {len(train_features.columns)} columns")
    print(f"Test features: {len(test_features)} rows, {len(test_features.columns)} columns")
    
    # Save feature manifest
    feature_cols = [c for c in train_features.columns if c not in ["unit", "cycle", "rul", "true_rul", "test_rul", "failure_within_H", "pseudo_timestamp"]]
    
    manifest = {
        "feature_columns": feature_cols,
        "n_train_rows": len(train_features),
        "n_test_rows": len(test_features),
        "n_features": len(feature_cols),
        "lags": config.LAGS,
        "roll_windows": config.ROLL_WINDOWS,
        "ewma_alpha": config.EWMA_ALPHA,
        "target_H": config.TARGET_H,
        "random_seed": config.RANDOM_SEED,
    }
    
    manifest_path = os.path.join(config.MODELS_DIR, "feature_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Feature manifest saved to {manifest_path}")
    
    # Write parquet files
    train_out_path = os.path.join(config.DATA_PROCESSED_DIR, "train_features.parquet")
    test_out_path = os.path.join(config.DATA_PROCESSED_DIR, "test_features.parquet")
    
    train_features.to_parquet(train_out_path, index=False)
    test_features.to_parquet(test_out_path, index=False)
    
    print(f"Wrote {train_out_path}")
    print(f"Wrote {test_out_path}")
    
    return {
        "train_path": train_out_path,
        "test_path": test_out_path,
        "n_train_rows": len(train_features),
        "n_test_rows": len(test_features),
        "n_features": len(feature_cols),
    }


if __name__ == "__main__":
    make_features()