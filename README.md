# Machinery Predictive Maintenance: NASA C-MAPSS FD001

Predicts jet-engine failure from multivariate sensor time series, so maintenance can be planned
before a breakdown instead of after it.

**Live dashboard:** https://machinery-maintenace-predictive-system-vprxrtkgmmaiy4jxx4qnzu.streamlit.app/

**Task:** binary classification. "Will this engine fail within the next H = 30 cycles?"
The label is 1 when remaining useful life (RUL) is 30 cycles or fewer.

**Data:** NASA C-MAPSS FD001. 100 training engines run to failure, and 100 test engines whose
trajectories are cut off before failure. One operating condition, one fault mode.

## Dashboard

The dashboard replays each test engine cycle by cycle and shows the failure probability. It
applies an alert rule: an engine is in **ALERT** when the probability stays at or above the
threshold for k consecutive cycles. **WATCH** means the probability is at or above half the
threshold. Threshold and k can be changed in the sidebar.

The dashboard shows results that were scored in advance. It does not score new data.

## Pipeline

1. **Load and validate** the raw files: schema, cycle continuity, duplicates, RUL file alignment.
2. **Clean:** sort by engine and cycle, and drop sensors that are constant in the training data
   (sensors 1, 5, 10, 16, 18, 19).
3. **Labels:**
   - Training RUL = last cycle of the engine minus the current cycle.
   - Test RUL = value from the RUL file plus (last observed cycle minus the current cycle).
4. **Features (318):** computed per engine from the current and earlier cycles only.
   - Lags of 1, 3, 6, 12 and 24 cycles
   - Trailing rolling mean, std, min and max over 3, 12 and 24 cycles
   - Exponentially weighted means
   - Deviation from each engine's early-life baseline
5. **Model:** LightGBM with class weighting and early stopping.
   - Engines are split 80/20 into train and validation, so no engine appears on both sides.
   - The alert threshold is chosen on validation only (maximum F1).
   - The test set is used once.

## Results (seed 42, H = 30)

| Set | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Validation (20 engines, all rows) | 0.945 | 0.921 | 0.933 | 0.984 | 0.996 |
| Test, all rows | 0.849 | 0.744 | 0.793 | 0.895 | 0.997 |
| Test, last cycle of each engine (100 rows) | 0.955 | 0.840 | 0.894 | 0.973 | 0.991 |

Alert threshold (maximum F1 on validation): **0.698**.

**Engine-level behaviour (first alert per engine, no consecutive-cycle rule)**

| Set | Engines | Never alerted | Alerted too early (RUL > 30) | Alerted inside window (RUL ≤ 30) | Median RUL at first alert |
|---|---|---|---|---|---|
| Validation | 20 | 0 | 8 | 12 | 29.5 |
| Test | 100 | 78 | 9 | 13 | 28.5 |

**How to read this**
- Only 2.5% of test rows are positive, because test trajectories stop before failure. Row-level
  test precision is therefore dominated by negatives.
- The last-cycle result uses 100 rows only and is noisy. It flagged 21 of 25 failing engines
  and raised 1 false alarm.
- Most of the 78 engines that never alerted are not misses. They were still far from failure
  when their data ended.

## Tests

`python -m pytest tests/ -v` runs 8 tests, all passing. They cover:
- Future-data leakage: changing rows after cycle t does not change any feature at or before t.
- Lag features do not mix values across engines.
- Train and validation engines are disjoint.
- Label thresholds, including the boundary at exactly H cycles.
- The FFT utility on a known sine wave.

## Limitations

- **Thin lead time.** Alerts fire close to the edge of the 30-cycle window, with a median of
  about 29 cycles remaining.
- **Early alerts.** A share of engines alert before RUL ≤ 30 (8 of 20 in validation, 9 of 22
  alerted engines in test). The consecutive-cycle rule in the dashboard reduces this, but the
  default k = 3 is a demo value and has **not been tuned**.
- **Optimistic validation.** There are only 20 validation engines, one seed and one split, and
  the threshold maximizes row-level F1. There is no cross-validation yet.
- **No cost-based threshold.** F1 treats a false alarm and a missed failure as equally costly,
  which is rarely true in maintenance.
- **Narrow dataset.** FD001 has one operating condition and one fault mode, so these results say
  nothing about other conditions.
- **Simulated data in cycles, not hours.** There is no vibration signal, so FFT features are not
  used in the model. An FFT utility exists in `src/features.py` and is unit-tested.

## Not implemented yet

GRU/LSTM model, cost-based threshold calibration, tuned consecutive-cycle rule, streaming
`predict.py`, MLflow tracking, executed notebooks, grouped cross-validation.

## Run it locally

```bash
pip install -r requirements-dev.txt
# Put train_FD001.txt, test_FD001.txt and RUL_FD001.txt in data/raw/CMAPSSData/
python -m src.data.make_interim
python -m src.pipelines.make_features
python -m src.modeling.train_tree
python -m pytest tests/ -v
python export_demo_data.py
streamlit run app.py
```

On Windows PowerShell, run `$env:PYTHONPATH = "src"` first, in the same window.

## Repository layout

```
src/              pipeline code (data loading, cleaning, labels, features, modeling)
tests/            unit and leakage tests
reports/          metrics and figures
demo/             small pre-scored file used by the dashboard
app.py            Streamlit dashboard (reads demo/ only)
app_local.py      dashboard variant that loads the trained model
docs/             dataset notes
requirements.txt          dashboard dependencies (used by Streamlit Cloud)
requirements-dev.txt      full pipeline dependencies
```

## Data and credits

NASA C-MAPSS turbofan engine degradation simulation data, by A. Saxena and K. Goebel (2008),
from the NASA Prognostics Center of Excellence data repository. The raw files are not included
in this repository. Download them from NASA and check the terms of use there.
