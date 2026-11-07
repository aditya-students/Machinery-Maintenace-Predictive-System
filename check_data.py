#!/usr/bin/env python3
"""Check C-MAPSS data files and report status."""
import os
import sys

BASE = r"D:\Projects\Machinery Maintainence Predictive System (demo)"
DATA_RAW = os.path.join(BASE, "data", "raw", "CMAPSSData")

print("=" * 60)
print("C-MAPSS Data File Check")
print("=" * 60)

# Check training data
train_path = os.path.join(DATA_RAW, "train_FD001.txt")
if os.path.exists(train_path):
    with open(train_path, 'r') as f:
        lines = f.readlines()
    print(f"\ntrain_FD001.txt: EXISTS ({len(lines)} lines)")
    # Parse first few valid lines
    for i, line in enumerate(lines[:3]):
        parts = line.strip().split()
        if len(parts) >= 2:
            print(f"  Line {i+1}: unit={parts[0]}, cycle={parts[1]}, cols={len(parts)}")
else:
    print(f"\ntrain_FD001.txt: MISSING at {train_path}")

# Check test data
test_path = os.path.join(DATA_RAW, "test_FD001.txt")
if os.path.exists(test_path):
    with open(test_path, 'r') as f:
        lines = f.readlines()
    print(f"\ntest_FD001.txt: EXISTS ({len(lines)} lines)")
    for i, line in enumerate(lines[:3]):
        parts = line.strip().split()
        if len(parts) >= 2:
            print(f"  Line {i+1}: unit={parts[0]}, cycle={parts[1]}, cols={len(parts)}")
else:
    print(f"\ntest_FD001.txt: MISSING at {test_path}")

# Check RUL data
rul_path = os.path.join(DATA_RAW, "RUL_FD001.txt")
if os.path.exists(rul_path):
    with open(rul_path, 'r') as f:
        lines = f.readlines()
    print(f"\nRUL_FD001.txt: EXISTS ({len(lines)} lines)")
    for i, line in enumerate(lines[:5]):
        print(f"  Line {i+1}: {line.strip()}")
else:
    print(f"\nRUL_FD001.txt: MISSING at {rul_path}")

# Check column count consistency
print("\n" + "=" * 60)
print("Column analysis (first 3 lines of train):")
print("=" * 60)
for path in [train_path, test_path]:
    if os.path.exists(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines[:3]):
            parts = line.strip().split()
            if len(parts) >= 2:
                print(f"  {os.path.basename(path)} line {i+1}: {len(parts)} columns, unit={parts[0]}, cycle={parts[1]}")

print("\n" + "=" * 60)
print("Check complete")
print("=" * 60)