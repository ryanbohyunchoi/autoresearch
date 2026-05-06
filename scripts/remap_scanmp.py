"""
remap_scanmp.py — Copy SCAN MP ECG images to scanmp_renamed/ with anonymous names,
                  and produce two CSVs:
                    1. scanmp_mapping.csv   — full mapping with PHI (keep secure)
                    2. data/scanmp_cohort.csv — deidentified, safe to put in container

Source images:  /mnt/raid0/rbc58/variant/scanmp_images_cropped/{StudyID}.png
Metadata CSV:   /home/rbc58/mnt/mm_vhd/variant/variant_scan_mp.csv

Usage:
    python scripts/remap_scanmp.py \
        --images-dir /mnt/raid0/rbc58/variant/scanmp_images_cropped \
        --metadata   /home/rbc58/mnt/mm_vhd/variant/variant_scan_mp.csv \
        --out-images scanmp_renamed \
        --out-mapping scanmp_mapping.csv \
        --out-deident data/scanmp_cohort.csv
"""

import argparse
import os
import shutil

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir",  default="/mnt/raid0/rbc58/variant/scanmp_images_cropped")
    parser.add_argument("--metadata",    default="/home/rbc58/mnt/mm_vhd/variant/variant_scan_mp.csv")
    parser.add_argument("--out-images",  default="scanmp_renamed")
    parser.add_argument("--out-mapping", default="scanmp_mapping.csv")
    parser.add_argument("--out-deident", default="data/scanmp_cohort.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    print(f"Loaded metadata: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Filter to labeled patients only (PYP+GEN+ = hATTR, PYP+GEN- = wtATTR)
    labeled = df[(df["PYP+GEN+"] == 1) | (df["PYP+GEN-"] == 1)].copy()
    labeled["label"] = labeled["PYP+GEN+"].astype(int)
    print(f"Labeled rows: {len(labeled)} | hATTR={labeled['label'].sum()} wtATTR={(labeled['label']==0).sum()}")

    os.makedirs(args.out_images, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_deident) or ".", exist_ok=True)

    mapping_rows = []
    skipped = 0

    for i, (_, row) in enumerate(labeled.iterrows(), start=1):
        study_id = str(row["StudyID"])
        src_path = os.path.join(args.images_dir, f"{study_id}.png")
        new_name = f"smp{i}.png"
        dst_path = os.path.join(args.out_images, new_name)

        if not os.path.exists(src_path):
            print(f"  MISSING: {src_path}")
            skipped += 1
            continue

        shutil.copy2(src_path, dst_path)
        mapping_rows.append({
            "new_name":     new_name,
            "StudyID":      study_id,              # PHI — only in mapping CSV
            "original_path": src_path,             # PHI — only in mapping CSV
            "label":        int(row["label"]),
            "Age":          row.get("Age", None),
            "Gender":       row.get("Gender", None),
            "Black_race":   row.get("Black_race", None),
            "Hispanic_ethnicity": row.get("Hispanic_ethnicity", None),
        })

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(args.out_mapping, index=False)
    print(f"\nFull mapping (with PHI) → {args.out_mapping}  ({len(mapping_df)} rows)")

    # Deidentified: drop StudyID and original_path
    deident = mapping_df.drop(columns=["StudyID", "original_path"])
    deident.to_csv(args.out_deident, index=False)
    print(f"Deidentified CSV → {args.out_deident}  ({len(deident)} rows)")
    print(f"Skipped (image not found): {skipped}")
    print(f"\nFirst few rows of deidentified CSV:")
    print(deident.head())


if __name__ == "__main__":
    main()
