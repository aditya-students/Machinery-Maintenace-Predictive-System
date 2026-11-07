"""Validation checks for the C-MAPSS FD001 data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports"

COLUMN_NAMES = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def _result(errors):
    return {"valid": len(errors) == 0, "errors": errors}


def validate_schema(df, name="dataset"):
    """Check 26 expected columns, integer unit/cycle and no NaN."""
    errors = []
    if df is None or df.empty:
        return _result([f"{name}: DataFrame is empty"])
    if list(df.columns) != COLUMN_NAMES:
        errors.append(f"{name}: columns do not match the expected 26 names")
    for col in ("unit", "cycle"):
        if col in df.columns and not pd.api.types.is_integer_dtype(df[col]):
            errors.append(f"{name}: '{col}' should be an integer type")
    nan_count = int(df.isna().sum().sum())
    if nan_count > 0:
        errors.append(f"{name}: {nan_count} NaN values found")
    return _result(errors)


def validate_cycles(df, name="dataset"):
    """Check cycle starts at 1 and increases by exactly 1 within each unit."""
    errors = []
    if "unit" not in df.columns or "cycle" not in df.columns:
        return _result([f"{name}: missing 'unit' or 'cycle' column"])
    ordered = df.sort_values(["unit", "cycle"])
    for unit_id, group in ordered.groupby("unit"):
        cycles = group["cycle"].to_numpy()
        if cycles[0] != 1:
            errors.append(f"{name}: unit {unit_id} starts at cycle {cycles[0]}, not 1")
        if not np.all(np.diff(cycles) == 1):
            errors.append(f"{name}: unit {unit_id} has gaps or repeated cycles")
    return _result(errors)


def validate_duplicates(df, name="dataset"):
    """Check for duplicate (unit, cycle) pairs."""
    dupes = int(df.duplicated(subset=["unit", "cycle"]).sum())
    errors = []
    if dupes > 0:
        errors.append(f"{name} has {dupes} duplicate (unit, cycle) pairs")
    return _result(errors)


def validate_units(df, name="dataset", expected=100):
    """Check unit IDs are exactly 1..expected."""
    units = sorted(int(u) for u in df["unit"].unique())
    errors = []
    if units != list(range(1, expected + 1)):
        errors.append(f"{name}: unit IDs are not exactly 1..{expected}")
    return _result(errors)


def validate_rul(rul, test, expected=100):
    """Check the RUL file has one value per test engine and no negatives."""
    values = np.asarray(rul).ravel()
    errors = []
    if len(values) != expected:
        errors.append(f"RUL file has {len(values)} values, expected {expected}")
    if len(values) != test["unit"].nunique():
        errors.append("RUL length does not match the number of test engines")
    if np.any(values < 0):
        errors.append("RUL file contains negative values")
    return _result(errors)


def validate_all(train, test, rul, write_reports=True):
    """Run all checks; optionally write JSON and markdown reports to reports/."""
    results = {
        "schema_train": validate_schema(train, "train"),
        "schema_test": validate_schema(test, "test"),
        "cycles_train": validate_cycles(train, "train"),
        "cycles_test": validate_cycles(test, "test"),
        "duplicates_train": validate_duplicates(train, "train"),
        "duplicates_test": validate_duplicates(test, "test"),
        "units_train": validate_units(train, "train"),
        "units_test": validate_units(test, "test"),
        "rul": validate_rul(rul, test),
    }
    if write_reports:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORT_DIR / "validation_report.json", "w") as f:
            json.dump(results, f, indent=2)
        lines = ["# C-MAPSS Validation Report", ""]
        for key, res in results.items():
            status = "PASS" if res["valid"] else "FAIL"
            lines.append(f"- **{status}** `{key}`")
            for err in res["errors"]:
                lines.append(f"  - {err}")
        with open(REPORT_DIR / "validation_report.md", "w") as f:
            f.write("\n".join(lines) + "\n")
    return results

run_all_validation = validate_all