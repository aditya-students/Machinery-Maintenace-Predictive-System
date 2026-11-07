import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.features import create_all_features
from src.modeling.train_tree import engine_split


def make_df(n_units=3, n_cycles=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            rows.append({
                "unit": u,
                "cycle": c,
                "sensor_2": rng.normal(),
                "sensor_3": rng.normal(),
            })
    return pd.DataFrame(rows)


def test_future_perturbation_does_not_change_past_features():
    df = make_df()
    base = create_all_features(df)

    t = 20
    altered = df.copy()
    mask = (altered["unit"] == 1) & (altered["cycle"] > t)
    altered.loc[mask, ["sensor_2", "sensor_3"]] += 100.0
    new = create_all_features(altered)

    past = df["cycle"] <= t
    assert_frame_equal(
        base[past].reset_index(drop=True),
        new[past].reset_index(drop=True),
    )


def test_no_cross_engine_leakage_in_lags():
    feats = create_all_features(make_df())
    first_rows = feats[feats["cycle"] == 1]
    assert first_rows["sensor_2_lag1"].isna().all()
    assert first_rows["sensor_2_lag24"].isna().all()
    second = feats[feats["cycle"] == 2]
    assert second["sensor_2_lag1"].notna().all()


def test_engine_split_is_disjoint_and_complete():
    train_ids, val_ids = engine_split(range(1, 101))
    assert not set(train_ids) & set(val_ids)
    assert len(train_ids) + len(val_ids) == 100
    assert len(val_ids) == 20