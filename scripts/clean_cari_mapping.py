"""
clean_cari_mapping.py — Strip PHI from cari_mapping.csv.

Input:  cari_mapping.csv (full mapping with MRNs, paths, dates)
Output: data/cari_cohort.csv (only: new_name, label, age, gender, race, ethnicity)

Usage:
    python scripts/clean_cari_mapping.py \
        --input /path/to/cari_mapping.csv \
        --output data/cari_cohort.csv
"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to cari_mapping.csv")
    parser.add_argument("--output", default="data/cari_cohort.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Columns: {list(df.columns)}")

    # Keep only the columns needed — no MRNs, paths, or dates
    keep = {
        "new_name":             "new_name",
        "hATTR vs. wtATTR":     "label",
        "age":                  "age",
        "gender_source_value":  "gender",
        "race_source_value":    "race",
        "ethnicity_source_value": "ethnicity",
    }

    missing = [c for c in keep if c not in df.columns]
    if missing:
        print(f"WARNING: missing columns: {missing}")

    out = pd.DataFrame()
    for src, dst in keep.items():
        if src in df.columns:
            out[dst] = df[src]
        else:
            out[dst] = None

    # Filter to labeled rows only
    out = out[out["label"].isin(["hATTR", "wtATTR"])].reset_index(drop=True)

    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out)} rows → {args.output}")
    print(f"hATTR: {(out['label']=='hATTR').sum()} | wtATTR: {(out['label']=='wtATTR').sum()}")
    print(f"\nFirst few rows:")
    print(out.head())

if __name__ == "__main__":
    main()
