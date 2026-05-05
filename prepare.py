"""
prepare.py — Fixed infrastructure. Agents do NOT modify this file.

Contains: constants, embedding/label loading, and the fixed evaluation
metric (evaluate_auroc). The only things train.py uses from here:
  CARI_EMB_PATH, CARI_CSV, SCANMP_EMB_PATH, SCANMP_CSV,
  load_cari_data, load_scanmp_data, evaluate_auroc, make_synthetic_data

To prepare data locally: python prepare.py
  - If data/ is empty, generates synthetic embeddings for pipeline testing.
  - Prints dataset statistics.

Real data (cluster):
  Set env vars CARI_EMB_PATH, CARI_CSV, SCANMP_EMB_PATH, SCANMP_CSV
  Or symlink: ln -s /mnt/.../variant_files ./data
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FOLDS     = 5
RANDOM_SEED = 42
EMB_DIM     = 1536   # EfficientNet-B3 embedding dimension

# Data paths — override via env vars when running on cluster
CARI_EMB_PATH   = os.environ.get("CARI_EMB_PATH",   "./data/cari_ecg_embeddings_pretrained_1m.npy")
CARI_CSV        = os.environ.get("CARI_CSV",         "./data/variant_cari_cohort.csv")
SCANMP_EMB_PATH = os.environ.get("SCANMP_EMB_PATH",  "./data/scanmp_ecg_embeddings_pretrained_1m.npy")
SCANMP_CSV      = os.environ.get("SCANMP_CSV",       "./data/variant_scan_mp.csv")

# CARI column names
CARI_ID_COL    = "fileID"
CARI_LABEL_COL = "hATTR vs. wtATTR"    # "hATTR" -> 1, "wtATTR" -> 0

# SCAN MP column names
SCANMP_ID_COL  = "StudyID"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cari_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load CARI embeddings and binary labels.
    Returns X (n, 1536) float32, y (n,) int (1=hATTR, 0=wtATTR).
    Raises FileNotFoundError if files are missing — run prepare.py first.
    """
    _check_exists(CARI_EMB_PATH, "CARI embeddings")
    _check_exists(CARI_CSV, "CARI CSV")

    emb_dict = np.load(CARI_EMB_PATH, allow_pickle=True).item()
    df = pd.read_csv(CARI_CSV)
    df = df[df[CARI_LABEL_COL].isin(["hATTR", "wtATTR"])].copy()
    df["label"] = (df[CARI_LABEL_COL] == "hATTR").astype(int)

    ids = [str(i) for i in df[CARI_ID_COL]]
    valid = [(i, int(l)) for i, l in zip(ids, df["label"]) if i in emb_dict]
    if not valid:
        raise ValueError("No ID overlap between CARI embeddings and CSV. Check file paths.")

    ids_valid, labels = zip(*valid)
    X = np.stack([emb_dict[i] for i in ids_valid]).astype(np.float32)
    y = np.array(labels, dtype=int)
    return X, y


def load_scanmp_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load SCAN MP embeddings and binary labels.
    Returns X (n, 1536) float32, y (n,) int (1=hATTR/PYP+GEN+, 0=wtATTR/PYP+GEN-).
    Excludes rows that are neither PYP+GEN+ nor PYP+GEN-.
    """
    _check_exists(SCANMP_EMB_PATH, "SCAN MP embeddings")
    _check_exists(SCANMP_CSV, "SCAN MP CSV")

    emb_dict = np.load(SCANMP_EMB_PATH, allow_pickle=True).item()
    df = pd.read_csv(SCANMP_CSV)
    df = df[(df["PYP+GEN+"] == 1) | (df["PYP+GEN-"] == 1)].copy()
    df["label"] = df["PYP+GEN+"].astype(int)

    ids = [str(i) for i in df[SCANMP_ID_COL]]
    valid = [(i, int(l)) for i, l in zip(ids, df["label"]) if i in emb_dict]
    if not valid:
        raise ValueError("No ID overlap between SCAN MP embeddings and CSV.")

    ids_valid, labels = zip(*valid)
    X = np.stack([emb_dict[i] for i in ids_valid]).astype(np.float32)
    y = np.array(labels, dtype=int)
    return X, y


# ---------------------------------------------------------------------------
# Fixed evaluation metric — DO NOT MODIFY
# ---------------------------------------------------------------------------

def evaluate_auroc(clf_factory, X: np.ndarray, y: np.ndarray) -> float:
    """
    5-fold stratified CV AUROC on X, y.
    clf_factory() must return a fresh sklearn-compatible classifier.
    Returns mean AUROC across folds. Higher is better.

    This is the immutable ground-truth metric. Agents must not modify it.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    aurocs = []
    for train_idx, val_idx in skf.split(X, y):
        clf = clf_factory()
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        aurocs.append(roc_auc_score(y[val_idx], probs))
    return float(np.mean(aurocs))


# ---------------------------------------------------------------------------
# Synthetic data generator (for local pipeline testing without cluster data)
# ---------------------------------------------------------------------------

def make_synthetic_data(n_cari: int = 150, n_scanmp: int = 46, seed: int = 0):
    """
    Generates synthetic 1536-dim embeddings and binary labels.
    hATTR embeddings have a slight positive shift in the first 10 dims
    to give the classifier a learnable signal.
    """
    rng = np.random.default_rng(seed)
    os.makedirs("./data", exist_ok=True)

    def _make_cohort(n, n_pos, id_prefix):
        y = np.array([1] * n_pos + [0] * (n - n_pos))
        rng.shuffle(y)
        X = rng.standard_normal((n, EMB_DIM)).astype(np.float32)
        X[y == 1, :10] += 0.5   # learnable signal
        ids = [f"{id_prefix}_{i:04d}" for i in range(n)]
        emb_dict = {i: X[j] for j, i in enumerate(ids)}
        return emb_dict, ids, y

    # CARI: ~33% hATTR (similar to real cohort)
    cari_emb, cari_ids, cari_y = _make_cohort(n_cari, n_pos=n_cari // 3, id_prefix="cari")
    np.save("./data/cari_ecg_embeddings_pretrained_1m.npy", cari_emb)
    pd.DataFrame({
        "fileID": cari_ids,
        "hATTR vs. wtATTR": ["hATTR" if l == 1 else "wtATTR" for l in cari_y],
        "age": rng.integers(40, 80, size=n_cari),
        "gender_source_value": rng.choice(["Male", "Female"], size=n_cari),
    }).to_csv("./data/variant_cari_cohort.csv", index=False)

    # SCAN MP: 19 hATTR, 27 wtATTR (matches real cohort size)
    scanmp_emb, scanmp_ids, scanmp_y = _make_cohort(n_scanmp, n_pos=19, id_prefix="scanmp")
    np.save("./data/scanmp_ecg_embeddings_pretrained_1m.npy", scanmp_emb)
    pd.DataFrame({
        "StudyID": scanmp_ids,
        "PYP+GEN+": scanmp_y,
        "PYP+GEN-": 1 - scanmp_y,
        "Age": rng.integers(50, 85, size=n_scanmp),
        "Gender": rng.choice(["M", "F"], size=n_scanmp),
    }).to_csv("./data/variant_scan_mp.csv", index=False)

    print(f"Synthetic data written to ./data/")
    print(f"  CARI:    {n_cari} patients | hATTR={cari_y.sum()} wtATTR={(cari_y==0).sum()}")
    print(f"  SCAN MP: {n_scanmp} patients | hATTR={scanmp_y.sum()} wtATTR={(scanmp_y==0).sum()}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_exists(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found at: {path}\n"
            "Run: python prepare.py   (generates synthetic data for local testing)\n"
            "Or set CARI_EMB_PATH / CARI_CSV / SCANMP_EMB_PATH / SCANMP_CSV env vars."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cari_missing = not os.path.exists(CARI_EMB_PATH) or not os.path.exists(CARI_CSV)
    if cari_missing:
        print("Data not found. Generating synthetic embeddings for pipeline testing...")
        make_synthetic_data()
    else:
        print("Data already exists. Skipping generation.")

    # Print stats
    try:
        X, y = load_cari_data()
        print(f"CARI:    {len(y)} patients | hATTR={y.sum()} wtATTR={(y==0).sum()} | emb_dim={X.shape[1]}")
    except Exception as e:
        print(f"CARI load error: {e}")

    try:
        X, y = load_scanmp_data()
        print(f"SCAN MP: {len(y)} patients | hATTR={y.sum()} wtATTR={(y==0).sum()} | emb_dim={X.shape[1]}")
    except Exception as e:
        print(f"SCAN MP load error: {e}")
