"""Tests for C-MAPSS predictive maintenance pipeline."""
import pytest
import os
import sys

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that all modules can be imported."""
    from src.config import load_config
    from src.data.load import load_train, load_test, load_rul
    from src.data.validate import run_all_validation
    from src.labels import compute_train_rul, compute_test_rul, binary_target
    from src.features import fft_features, create_lag_features, create_all_features
    from src.split import engine_level_split, verify_no_leakage
    print("All imports successful")


def test_fft_features():
    """Test FFT features on a known sine wave."""
    from src.features import fft_features
    import numpy as np

    fs = 100.0
    t = np.linspace(0, 1, int(fs), endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz, well below Nyquist

    features = fft_features(signal, fs=fs)

    assert abs(features["dominant_freq"] - 10.0) < 1.0
    assert features["spectral_energy"] > 0
    assert 0 <= features["spectral_entropy"] <= 1
    assert features["spectral_entropy"] < 0.1  # pure tone is concentrated

def test_lag_features():
    """Lag features are causal and do not bleed across engines."""
    import pandas as pd
    from src.features import create_lag_features

    df = pd.DataFrame({
        "unit": [1, 1, 1, 2, 2, 2],
        "cycle": [1, 2, 3, 1, 2, 3],
        "sensor_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "sensor_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    result = create_lag_features(df, lag_cols=["sensor_1", "sensor_2"], lags=[1, 3])

    # Engine 1: cycle 1 -> NaN, cycle 2 -> 1.0, cycle 3 -> 2.0
    assert pd.isna(result.loc[0, "sensor_1_lag1"])
    assert result.loc[1, "sensor_1_lag1"] == 1.0
    assert result.loc[2, "sensor_1_lag1"] == 2.0
    # Engine 2 must NOT see engine 1's values
    assert pd.isna(result.loc[3, "sensor_1_lag1"])
    assert result.loc[4, "sensor_1_lag1"] == 4.0
    # lag 3 is NaN everywhere here (engines only have 3 cycles)
    assert result["sensor_1_lag3"].isna().all()


def test_binary_target():
    """Test binary target computation."""
    from src.labels import binary_target
    
    # RUL exactly H should be 1 (inclusive: RUL <= H)
    rul = [30, 31, 29]  # H = 30
    target = binary_target(rul)
    assert target.tolist() == [1, 0, 1], f"Expected [1, 0, 1], got {target.tolist()}"
    
    print("Binary target test passed")


def test_label_boundary_cases():
    """Test boundary cases for label construction."""
    from src.labels import test_boundary_cases
    results = test_boundary_cases()
    
    # Check that the test structure is correct
    assert "rul_exact_H" in results
    assert "rul_last_cycle" in results
    assert "H_comparison" in results
    
    print("Label boundary cases test passed")


if __name__ == "__main__":
    test_imports()
    test_fft_features()
    test_lag_features()
    test_binary_target()
    test_label_boundary_cases()
    print("\nAll tests passed!")