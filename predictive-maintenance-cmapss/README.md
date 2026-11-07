# C-MAPSS Predictive Maintenance

## Problem
Predict engine failure BEFORE it happens from multivariate sensor time series, moving from reactive to predictive maintenance. Using the NASA C-MAPSS FD001 dataset, the goal is to flag failure at least H cycles ahead (default H = 30) with a low false-positive rate.

## Dataset
- **Source**: NASA Prognostics Center of Excellence, Saxena & Goebel 2008
- **License**: Free for academic research (verify on NASA TI Arc source page)
- **Files**: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- **Format**: 26 space-separated columns, no header: `unit, cycle, op_setting_1..3, sensor_1..sensor_21`
- **FD001**: One operating condition, one fault mode (HPC degradation)
- **Cycles**: Integer cycle numbers per engine (no timestamps, no sampling-rate variation)

## Project Structure

```
predictive-maintenance-cmapss/
├── README.md (this file)
├── .gitignore
├── requirements.txt
├── Makefile
├── pyproject.toml
├── configs/config.yaml
├── data/{raw,interim,processed}/
├── notebooks/
│   ├── 01_eda_sensor_variance.ipynb
│   ├── 02_feature_correlation_quality.ipynb
│   ├── 03_model_error_analysis.ipynb
│   └── 04_fft_demo_synthetic.ipynb
├── src/
│   ├── config.py
│   ├── data/ (load.py, validate.py, clean.py, grid.py, smooth.py)
│   ├── features.py
│   ├── labels.py
│   ├── split.py
│   ├── pipelines/ (make_interim.py, make_features.py)
│   └── modeling/ (baseline.py, train_tree.py, train_lstm.py, evaluate.py,
                     threshold.py, alerting.py, predict.py)
├── scripts/ (check_data.py, run_all.sh)
├── models/ (versioned metadata JSON)
├── reports/ (figures/, model_comparison.md, threshold_calibration.md, final_report.md)
├── docs/ (DATASET.md, CLEANING_DECISIONS.md, LEAKAGE_PREVENTION.md, DEMO_CHECKLIST.md)
├── tests/
└── .github/workflows/ci.yml
```

## Key Caveats (Fit-to-Brief)

- **No timestamps**: Time is the integer `cycle` per engine. "Resample to hourly" becomes a cycle-grid verification (no gaps, strictly increasing, starts at 1). "24 hours" becomes "H cycles". A synthetic `pseudo_timestamp` may be added clearly as such.
- **No vibration**: The 21 sensors are steady-state operating parameters, not vibration waveforms. FFT is not physically meaningful on raw features; `fft_features()` is implemented and unit-tested on a synthetic sine wave, with an optional ablation on trailing windows.
- **All trajectories end in failure**: A within-engine chronological split would put all positives at the end. Leak-free evaluation is done by: (a) causal features only, (b) engine-level train/validation split, (c) official test file as final hold-out.
- **Single operating condition**: FD001 has one operating condition; models may not generalize to varying conditions without domain adaptation.

## Setup

```bash
# Clone and navigate
git clone <repo-url>
cd predictive-maintenance-cmapss

# Install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Verify data files are in data/raw/CMAPSSData/
# Required: train_FD001.txt, test_FD001.txt, RUL_FD001.txt

# Run full pipeline
make all
```

## Pipeline Overview

1. `make check-data` - Verify data files exist and are valid
2. `make interim` - Load raw data, clean, sort, attach RUL, write parquet
3. `make features` - Engineer lag, rolling, EWMA, FFT features
4. `make train` - Engine-level split, train LightGBM model
5. `make evaluate` - Evaluate on validation and test sets
6. `make threshold` - Calibrate decision threshold on validation set
7. `make stream-demo` - Run streaming prediction simulator

## Results

Expected metrics table (populated after running pipeline):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|----|---------|--------|
| Majority Baseline | ... | ... | ... | ... | ... |
| Logistic Regression | ... | ... | ... | ... | ... |
| LightGBM | ... | ... | ... | ... | ... |
| LSTM (if applicable) | ... | ... | ... | ... | ... |

## Threshold Choice

Default threshold selected by minimizing expected cost with:
- `cost_false_alarm` = 1.0
- `cost_missed_failure` = 5.0

Sensitivity table over ratios 1:5, 1:10, 1:50 available in `reports/threshold_calibration.md`.

## Limitations

- Single operating condition (FD001 only)
- One fault mode (HPC degradation)
- Cycles instead of hours (no real timestamps)
- No real vibration data; FFT is an ablation only
- Synthetic targets and labels

## Future Work

- FD002-FD004 (multiple operating conditions)
- Real vibration data with meaningful FFT
- Drift monitoring and online adaptation
- Hyperparameter optimization with Optuna