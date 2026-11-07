"""Load raw C-MAPSS FD001 files. Raw files are never modified."""
import os

import pandas as pd

import config

TRAIN_NAME = "train_FD001.txt"
TEST_NAME = "test_FD001.txt"
RUL_NAME = "RUL_FD001.txt"

COLUMN_NAMES = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def _default_dir():
    return getattr(config, "DATA_RAW_DIR", os.path.join("data", "raw"))


def _find_file(raw_dir, filename):
    """Look in raw_dir, then in raw_dir/CMAPSSData."""
    candidates = [
        os.path.join(raw_dir, filename),
        os.path.join(raw_dir, "CMAPSSData", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{filename} not found. Looked in: {candidates}")


def _read_txt(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")  # drop trailing empty columns
    df.columns = COLUMN_NAMES
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_train(raw_dir=None):
    raw_dir = raw_dir or _default_dir()
    return _read_txt(_find_file(raw_dir, TRAIN_NAME))


def load_test(raw_dir=None):
    raw_dir = raw_dir or _default_dir()
    return _read_txt(_find_file(raw_dir, TEST_NAME))


def load_rul(raw_dir=None):
    """Return a DataFrame with one column 'RUL' (one row per test engine, index 1..100)."""
    raw_dir = raw_dir or _default_dir()
    path = _find_file(raw_dir, RUL_NAME)
    rul = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    rul = rul.dropna(axis=1, how="all")
    rul.columns = ["RUL"]
    rul.index = range(1, len(rul) + 1)
    return rul