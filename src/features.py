"""Feature engineering module for C-MAPSS FD001 dataset.
Pure, tested, config-driven functions. ALL computations grouped by `unit`.
"""

import re
import numpy as np 
import pandas as pd

import config


def fft_features(signal, fs):
    """One-sided FFT features for a 1-D signal sampled at fs Hz.

    Returns dominant frequency (Hz), total spectral energy, normalized
    spectral entropy in [0, 1], and energy in three equal frequency bands.
    """
    x = np.asarray(signal, dtype=float)
    n = len(x)

    empty = {
        "dominant_freq": 0.0,
        "spectral_energy": 0.0,
        "spectral_entropy": 0.0,
        "band_energy_low": 0.0,
        "band_energy_mid": 0.0,
        "band_energy_high": 0.0,
    }
    if n < 2:
        return empty

    x = x - x.mean()  # remove DC component
    power = np.abs(np.fft.rfft(x)) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    total = float(power.sum())

    if total <= 0.0:
        return empty

    p = power / total
    nonzero = p[p > 0]
    entropy = -(nonzero * np.log2(nonzero)).sum()
    max_entropy = np.log2(len(p)) if len(p) > 1 else 1.0
    spectral_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))

    nyquist = fs / 2.0
    low = power[freqs < nyquist / 3].sum()
    mid = power[(freqs >= nyquist / 3) & (freqs < 2 * nyquist / 3)].sum()
    high = power[freqs >= 2 * nyquist / 3].sum()

    return {
        "dominant_freq": float(freqs[np.argmax(power)]),
        "spectral_energy": total,
        "spectral_entropy": spectral_entropy,
        "band_energy_low": float(low),
        "band_energy_mid": float(mid),
        "band_energy_high": float(high),
    }


def _sensor_cols(df):
    return [c for c in df.columns if re.fullmatch(r"sensor_\d+", c)]


def create_lag_features(df, lag_cols=None, lags=(1, 3, 6, 12, 24)):
    """Lagged sensor values per engine (positive shifts only, no cross-engine bleed)."""
    cols = lag_cols or _sensor_cols(df)
    g = df.groupby("unit")[cols]
    parts = []
    for k in lags:
        if k <= 0:
            raise ValueError("lags must be positive")
        s = g.shift(k)
        s.columns = [f"{c}_lag{k}" for c in cols]
        parts.append(s)
    return pd.concat(parts, axis=1)


def create_rolling_features(df, windows=(3, 12, 24), stats=("mean", "std", "min", "max")):
    """Trailing rolling statistics per engine; each window ends at the current cycle."""
    cols = _sensor_cols(df)
    g = df.groupby("unit")[cols]
    parts = []
    for w in windows:
        r = g.rolling(window=w, min_periods=1)
        for s in stats:
            out = getattr(r, s)().reset_index(level=0, drop=True)
            if s == "std":
                out = out.fillna(0.0)
            out.columns = [f"{c}_roll{w}_{s}" for c in cols]
            parts.append(out)
    return pd.concat(parts, axis=1)


def create_ewma_features(df, alphas=(0.1, 0.3)):
    """Exponentially weighted means per engine (causal)."""
    cols = _sensor_cols(df)
    parts = []
    for a in alphas:
        out = df.groupby("unit")[cols].ewm(alpha=a, adjust=False).mean()
        out = out.reset_index(level=0, drop=True)
        out.columns = [f"{c}_ewma{a}" for c in cols]
        parts.append(out)
    return pd.concat(parts, axis=1)


def create_baseline_deviation(df, baseline_cycles=10):
    """Deviation from the engine's own early-life baseline (causal).

    Rows in the first N cycles use the expanding mean so far; later rows use
    the frozen mean of the first N cycles.
    """
    cols = _sensor_cols(df)
    g = df.groupby("unit")
    exp = g[cols].expanding().mean().reset_index(level=0, drop=True)
    early = (g.cumcount() <= baseline_cycles - 1).to_numpy()
    mask = np.repeat(early[:, None], len(cols), axis=1)
    mask = pd.DataFrame(mask, index=exp.index, columns=cols)
    base = exp.where(mask)
    base = base.groupby(df["unit"]).ffill()
    dev = df[cols] - base
    dev.columns = [f"{c}_dev_baseline" for c in cols]
    return dev

def create_all_features(df):
    """All engineered features. Input sorted by unit, cycle, with a unique index."""
    if not df.index.is_unique:
        raise ValueError("create_all_features needs a unique index")
    out = pd.concat(
        [
            df,
            create_lag_features(df),
            create_rolling_features(df),
            create_ewma_features(df),
            create_baseline_deviation(df),
        ],
        axis=1,
    )
    return out.copy()