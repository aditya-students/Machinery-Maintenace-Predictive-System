"""Load saved model artifacts for C-MAPSS predictive maintenance."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def load_artifacts(artifact_path: str = None) -> dict:
    """Load model artifacts from disk.
    
    Returns dict with:
    - model: trained model object
    - scaler: fitted scaler (StandardScaler)
    - feature_manifest: dict of feature column names and metadata
    - threshold: optimal threshold value
    - config: config namespace
    """
    if artifact_path is None:
        artifact_path = os.path.join(config.MODELS_DIR, "artifacts.json")
    
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Artifacts file not found at {artifact_path}")
    
    with open(artifact_path, 'r') as f:
        artifacts = json.load(f)
    
    # Load scaler if present
    scaler_path = artifacts.get("scaler_path", None)
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, 'r') as f:
            scaler_data = json.load(f)
        # Reconstruct StandardScaler from saved parameters
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_data.get("mean", []))
        scaler.scale_ = np.array(scaler_data.get("scale", []))
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler_data.get("mean", []))
    
    # Feature manifest
    feature_manifest = artifacts.get("feature_manifest", {})
    
    # Threshold
    threshold = artifacts.get("threshold", 0.5)
    
    # Config
    config_obj = type('Config', (), {})()  # Simple namespace
    config_obj.RANDOM_SEED = config.RANDOM_SEED
    config_obj.TARGET_H = config.TARGET_H
    config_obj.LAGS = config.LAGS
    config_obj.ROLL_WINDOWS = config.ROLL_WINDOWS
    config_obj.EWMA_ALPHA = config.EWMA_ALPHA
    
    return {
        "model": artifacts["model"],
        "scaler": scaler,
        "feature_manifest": feature_manifest,
        "threshold": threshold,
        "config": config_obj
    }


def save_artifacts(artifacts: dict, path: str = None) -> str:
    """Save model artifacts to disk."""
    if path is None:
        path = os.path.join(config.MODELS_DIR, "artifacts.json")
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Save scaler parameters if present
    scaler = artifacts.get("scaler")
    scaler_data = None
    if scaler is not None:
        scaler_data = {
            "mean": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else [],
            "scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else [],
        }
        scaler_path = os.path.join(config.MODELS_DIR, "scaler.json")
        with open(scaler_path, 'w') as f:
            json.dump(scaler_data, f, indent=2)
    
    # Save artifacts JSON (without the scaler object, since we save it separately)
    save_dict = {
        "model": artifacts["model"],
        "feature_manifest": artifacts["feature_manifest"],
        "threshold": artifacts["threshold"],
    }
    if scaler_data is not None:
        save_dict["scaler_path"] = scaler_path
    
    with open(path, 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    return path