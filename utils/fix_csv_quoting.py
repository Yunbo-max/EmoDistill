"""
One-time fixer: re-quote text fields in malformed CSVs so pandas can parse them.

For each CSV:
  1. Read raw lines
  2. Determine expected column count from header
  3. For each data row with > N fields, identify the "overflow" and merge
     overflow fields into the longest-currently text field (commonly the one
     describing reason/situation/notes which contains commas).
  4. Write back with proper quoting (csv.QUOTE_ALL).

Usage:
  python utils/fix_csv_quoting.py data/hospital_surgery_scenarios.csv
  python utils/fix_csv_quoting.py data/education_sleep_scenarios.csv
  python utils/fix_csv_quoting.py data/disaster_survivor_scenarios.csv
"""

import csv
import os
import sys
import shutil
from typing import List


def _infer_overflow_target(header: List[str]) -> int:
    """Heuristic: pick the column most likely to contain commas (longest text-y name)."""
    # Strong candidates by keyword
    KEYWORDS = ["reason", "notes", "description", "feel", "thought", "key_", "argument",
                "needs", "situation", "endurance", "argument", "background", "annoyance",
                "risk", "decision", "suggestion"]
    best_idx = None
    best_score = 0
    for i, name in enumerate(header):
        n = name.lower()
        score = 0
        for kw in KEYWORDS:
            if kw in n:
                score += 10
        score += min(len(name), 30) // 5  # prefer longer column names
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx if best_idx is not None else max(0, len(header) - 2)


def fix_csv(path: str) -> int:
    """Rewrite path in-place (with .bak backup). Returns number of rows fixed."""
    with open(path) as f:
        raw = f.read().splitlines()
    if not raw:
        return 0

    header_line = raw[0]
    header = next(csv.reader([header_line]))
    expected = len(header)
    overflow_target = _infer_overflow_target(header)
    print(f"  {path}")
    print(f"    expected cols: {expected}")
    print(f"    overflow target (will absorb extras): '{header[overflow_target]}' (col {overflow_target})")

    rows_out: List[List[str]] = [header]
    fixed = 0
    skipped = 0
    for i, line in enumerate(raw[1:], start=2):
        if not line.strip():
            continue
        # Try strict parse first
        try:
            fields = next(csv.reader([line]))
        except Exception:
            skipped += 1
            continue
        n = len(fields)
        if n == expected:
            rows_out.append(fields)
        elif n > expected:
            # Merge fields [overflow_target : overflow_target + (n - expected) + 1]
            extra = n - expected
            merged = ", ".join(fields[overflow_target : overflow_target + extra + 1])
            new_row = fields[:overflow_target] + [merged] + fields[overflow_target + extra + 1:]
            assert len(new_row) == expected, (i, n, expected, len(new_row))
            rows_out.append(new_row)
            fixed += 1
        else:
            # Fewer fields → pad with empty strings
            rows_out.append(fields + [""] * (expected - n))
            fixed += 1

    if fixed == 0 and skipped == 0:
        print(f"    nothing to fix.")
        return 0

    backup = path + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"    backup → {backup}")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for row in rows_out:
            writer.writerow(row)
    print(f"    fixed {fixed} rows, skipped {skipped}, wrote {len(rows_out)-1} data rows.")
    return fixed


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        "data/hospital_surgery_scenarios.csv",
        "data/education_sleep_scenarios.csv",
        "data/disaster_survivor_scenarios.csv",
    ]
    print("Fixing CSV quoting (preserving .bak backups):")
    total = 0
    for f in files:
        if os.path.exists(f):
            total += fix_csv(f)
        else:
            print(f"  {f} NOT FOUND")
    print(f"\nTotal rows fixed: {total}")


if __name__ == "__main__":
    main()
