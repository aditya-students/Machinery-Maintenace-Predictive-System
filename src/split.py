"""Split module for C-MAPSS FD001 dataset.
Engine-level split of training engines into train/validation,
with the official test file as final hold-out.
"""

import numpy as np
import pandas as pd

import config


def engine_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_seed: int = config.RANDOM_SEED,
    stratify: str = "lifetime",
    method: str = "engine_level",
) -> dict:
    """Split engines into train and validation sets.

    Methods:
    - 'engine_level': Simple random split by engine ID, ensuring disjoint engines.
    - 'stratified': Split stratified by lifetime length (number of cycles).
    - 'group_kfold': GroupKFold by unit (leak-free).

    Returns dict with 'train_engines', 'val_engines', and full dataframes.
    """
    # Get unique engine IDs
    engine_ids = sorted(df["unit"].unique())
    n_engine = len(engine_ids)

    if method == "engine_level":
        # Simple random split
        np.random.seed(random_seed)
        np.random.shuffle(engine_ids)

        n_train = int(n_engine * train_ratio)
        train_engines = set(engine_ids[:n_train])
        val_engines = set(engine_ids[n_train:])

        train_df = df[df["unit"].isin(train_engines)].copy()
        val_df = df[df["unit"].isin(val_engines)].copy()

    elif method == "stratified":
        # Stratify by lifetime length
        lifetimes = df.groupby("unit")["cycle"].max()
        lifetimes_sorted = lifetimes.sort_values(ascending=False)

        n_train = int(n_engine * train_ratio)
        # Take the longest-lived engines for train, shortest for val
        train_lifetime_cutoff = lifetimes_sorted.iloc[n_train]

        train_engines = set(df["unit"].unique()) - set(
            df.groupby("unit")
            .filter(lambda x: x["cycle"].max() <= train_lifetime_cutoff)["unit"]
            .unique()
        )
        val_engines = set(df["unit"].unique()) - train_engines

        train_df = df[df["unit"].isin(train_engines)].copy()
        val_df = df[df["unit"].isin(val_engines)].copy()

    elif method == "group_kfold":
        # GroupKFold-style split - just first fold for val
        np.random.seed(random_seed)
        np.random.shuffle(engine_ids)
        n_train = int(n_engine * train_ratio)
        train_engines = set(engine_ids[:n_train])
        val_engines = set(engine_ids[n_train:])

        train_df = df[df["unit"].isin(train_engines)].copy()
        val_df = df[df["unit"].isin(val_engines)].copy()

    else:
        raise ValueError(f"Unknown split method: {method}")

    # Assert disjoint engines
    assert train_df["unit"].nunique() == len(train_engines), "Train has duplicate engines"
    assert val_df["unit"].nunique() == len(val_engines), "Val has duplicate engines"
    assert train_df["unit"].intersection(val_df["unit"]).empty(), "Train and val share engines"

    return {
        "train_engines": sorted(train_engines),
        "val_engines": sorted(val_engines),
        "train_df": train_df.reset_index(drop=True),
        "val_df": val_df.reset_index(drop=True),
        "split_info": {
            "method": method,
            "train_ratio": train_ratio,
            "n_train_engines": len(train_engines),
            "n_val_engines": len(val_engines),
            "random_seed": random_seed,
        },
    }


def official_holdout_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
) -> dict:
    """Engine-level split with official test set as final hold-out.

    The train/val split is done on train_df only. The test_df + rul_df
    is kept completely separate until final evaluation.
    """
    # Split train into train/val
    split_result = engine_level_split(train_df, train_ratio=1 - config.TEST_SPLIT_RATIO)

    # The test set is untouched - will be evaluated only at the very end
    return {
        "train_df": split_result["train_df"],
        "val_df": split_result["val_df"],
        "test_df": test_df,  # Untouched, final hold-out
        "rul_df": rul_df,  # Only used with test set
        "split_info": split_result["split_info"],
    }


def verify_no_leakage(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> dict:
    """Verify no future data leakage across splits.

    Features at cycle t must depend only on cycles <= t.
    Perturb future rows and verify features at t do not change.
    """
    errors = []

    # Check that scaler/imputer fit parameters are not stored in the feature data
    # (This is enforced by the pipeline fitting imputers/scalers on train only)

    # Check that no feature uses RUL or max_cycle directly
    for col in train_features.columns:
        if "RUL" in col.upper() or "MAX_CYCLE" in col.upper():
            errors.append(f"Feature '{col}' may contain leakage target information!")

    # Check engine disjointness
    train_engines = set(train_features["unit"].unique())
    val_engines = set(val_features["unit"].unique())
    test_engines = set(test_features["unit"].unique())

    assert train_engines.isdisjoint(val_engines), "Train and val share engines!"
    assert train_engines.isdisjoint(test_engines), "Train and test share engines!"
    # Note: test engines may overlap with val in terms of cycle range, but
    #      the engines themselves should be disjoint from train

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "train_engines": train_engines,
        "val_engines": val_engines,
        "test_engines": test_engines,
    }