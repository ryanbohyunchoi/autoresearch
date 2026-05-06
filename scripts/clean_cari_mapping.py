"""
clean_cari_mapping.py — Strip PHI from cari_mapping.csv.

Input:  cari_mapping.csv (full mapping with MRNs, paths, dates)
Output: data/cari_cohort.csv with columns:
          new_name, label, age, gender, race, ethnicity, patient_group

patient_group: deidentified sequential integer per unique patient (0, 1, 2, ...).
  Allows MRN-level grouped CV in train.py without storing actual MRNs.
  Multiple ECGs from the same patient share the same patient_group value.

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

    # MRN column — try PAT_MRN_ID first, fall back to MRN
    mrn_col = "PAT_MRN_ID" if "PAT_MRN_ID" in df.columns else "MRN"
    if mrn_col not in df.columns:
        print("WARNING: No MRN column found — patient_group will be row index (no grouping)")
        df["_mrn_tmp"] = range(len(df))
        mrn_col = "_mrn_tmp"

    # Deidentified patient group: sequential integer per unique MRN (MRN never stored)
    df["patient_group"] = pd.factorize(df[mrn_col])[0]
    n_patients = df["patient_group"].nunique()

    # Keep only safe columns
    keep = {
        "new_name":               "new_name",
        "hATTR vs. wtATTR":       "label",
        "age":                    "age",
        "gender_source_value":    "gender",
        "race_source_value":      "race",
        "ethnicity_source_value": "ethnicity",
        "patient_group":          "patient_group",
    }

    missing = [c for c in keep if c not in df.columns and c != "patient_group"]
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
    print(f"Unique patients: {n_patients} | ECGs per patient: {len(out)/n_patients:.1f} avg")
    print(f"hATTR: {(out['label']=='hATTR').sum()} | wtATTR: {(out['label']=='wtATTR').sum()}")
    print(f"\nFirst few rows:")
    print(out.head())

if __name__ == "__main__":
    main()
