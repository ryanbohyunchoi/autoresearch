"""
train.py — The ONE file agents edit.

Usage: python train.py > run.log 2>&1

Trains a classifier on ECG embeddings for hATTR vs wtATTR classification.
Primary metric: val_auroc (5-fold CARI CV, higher is better).
Secondary metric: scanmp_auroc (external SCAN MP hold-out — true generalization).

Modify anything: ENCODER, classifier, preprocessing, feature engineering.
Do NOT modify prepare.py.

The goal: maximize val_auroc without sacrificing scanmp_auroc.
Best result so far: scanmp_auroc = 0.79 (pretrained_1m + LR + demographics).
"""

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from prepare import (
    load_cari_data, load_scanmp_data, evaluate_auroc, N_FOLDS, RANDOM_SEED,
    extract_embeddings, ENCODER_CONFIGS, _emb_cache_path
)

# ---------------------------------------------------------------------------
# Hyperparameters — edit these directly, no CLI flags
# ---------------------------------------------------------------------------

ENCODER          = "pretrained_1m"    # onnx | pretrained_1m | biocontrastive | biocontrastive_pretrained
USE_DEMOGRAPHICS = False              # WARNING: demographics hurt SCAN MP (0.59 vs 0.645 ECG-only)
C_VALUES         = [0.001, 0.01, 0.1]  # ensemble of LR models with different regularization
MAX_ITER         = 1000
CLASS_WEIGHT     = "balanced"

# ---------------------------------------------------------------------------
# Ensure embeddings are cached (extracts if missing)
# ---------------------------------------------------------------------------

t_start = time.time()
np.random.seed(RANDOM_SEED)

import os
for cohort in ["cari", "scanmp"]:
    cache = _emb_cache_path(cohort, ENCODER)
    if not os.path.exists(cache):
        print(f"Cache missing for {cohort}/{ENCODER} — extracting now...")
        extract_embeddings(ENCODER)
        break

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

X_cari, y_cari, demo_cari = load_cari_data(ENCODER)
print(f"CARI: {len(y_cari)} patients | hATTR={y_cari.sum()} wtATTR={(y_cari==0).sum()} | emb_dim={X_cari.shape[1]}")

# ---------------------------------------------------------------------------
# Demographics preprocessing
# Normalizes CARI and SCAN MP to 4 consistent binary features:
#   age (scaled), is_male, is_black, is_hispanic
# This ensures identical feature space across both cohorts.
# ---------------------------------------------------------------------------

def normalize_demo(demo: pd.DataFrame) -> pd.DataFrame:
    """Convert CARI or SCAN MP demographics to a consistent 4-column format."""
    n = len(demo)
    out = pd.DataFrame(index=demo.index)

    # Age
    age_col = "age" if "age" in demo.columns else "Age"
    out["age"] = pd.to_numeric(demo[age_col], errors="coerce").fillna(demo[age_col].median() if age_col in demo.columns else 65)

    # is_male
    if "gender" in demo.columns:
        out["is_male"] = (demo["gender"].str.upper().str.strip() == "MALE").astype(float)
    elif "Gender" in demo.columns:
        out["is_male"] = (demo["Gender"].str.strip().str.upper() == "MALE").astype(float)
    else:
        out["is_male"] = 0.0

    # is_black
    if "race" in demo.columns:
        out["is_black"] = demo["race"].str.lower().str.contains("black|african", na=False).astype(float)
    elif "Black_race" in demo.columns:
        br = demo["Black_race"]
        if pd.api.types.is_numeric_dtype(br):
            out["is_black"] = br.fillna(0).astype(float)
        else:
            out["is_black"] = br.str.lower().str.contains("black|african", na=False).astype(float)
    else:
        out["is_black"] = 0.0

    # is_hispanic
    if "ethnicity" in demo.columns:
        out["is_hispanic"] = demo["ethnicity"].str.lower().str.contains("hispanic", na=False).astype(float)
    elif "Hispanic_ethnicity" in demo.columns:
        he = demo["Hispanic_ethnicity"]
        if pd.api.types.is_numeric_dtype(he):
            out["is_hispanic"] = he.fillna(0).astype(float)
        else:
            out["is_hispanic"] = he.str.lower().str.contains("yes|hispanic", na=False).astype(float)
    else:
        out["is_hispanic"] = 0.0

    return out


def build_demo_features(demo: pd.DataFrame, fit_scaler=None):
    """Normalize + standardize age; return (n, 4) feature array."""
    norm = normalize_demo(demo)
    age = norm[["age"]].values.astype(np.float32)
    binary = norm[["is_male", "is_black", "is_hispanic"]].values.astype(np.float32)
    if fit_scaler is None:
        scaler = StandardScaler()
        age = scaler.fit_transform(age)
    else:
        scaler = fit_scaler
        age = scaler.transform(age)
    return np.hstack([age, binary]), scaler


# ---------------------------------------------------------------------------
# 5-fold CV — grouped by patient if patient_group column is available
# ---------------------------------------------------------------------------

if "patient_group" in demo_cari.columns:
    groups = demo_cari["patient_group"].values
    cv = StratifiedGroupKFold(n_splits=N_FOLDS)
    splits = list(cv.split(X_cari, y_cari, groups=groups))
    n_patients = len(np.unique(groups))
    print(f"Using StratifiedGroupKFold: {n_patients} unique patients, {len(y_cari)} ECGs")
else:
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(cv.split(X_cari, y_cari))
    print("WARNING: no patient_group column — CV splits by row (patient leakage possible)")

fold_aurocs = []

for fold, (train_idx, val_idx) in enumerate(splits):
    X_tr, X_val = X_cari[train_idx], X_cari[val_idx]

    # Scale embedding
    emb_scaler = StandardScaler()
    X_tr  = emb_scaler.fit_transform(X_tr)
    X_val = emb_scaler.transform(X_val)

    # Concatenate demographics
    if USE_DEMOGRAPHICS:
        demo_tr, demo_scaler = build_demo_features(demo_cari.iloc[train_idx])
        demo_val, _          = build_demo_features(demo_cari.iloc[val_idx], fit_scaler=demo_scaler)
        X_tr  = np.hstack([X_tr,  demo_tr])
        X_val = np.hstack([X_val, demo_val])

    fold_probs = []
    for c in C_VALUES:
        clf = LogisticRegression(C=c, max_iter=MAX_ITER, class_weight=CLASS_WEIGHT,
                                 solver="lbfgs", random_state=RANDOM_SEED)
        clf.fit(X_tr, y_cari[train_idx])
        fold_probs.append(clf.predict_proba(X_val)[:, 1])
    # L1 models (liblinear, fast feature selection)
    for c in [0.001, 0.01, 0.1]:
        clf_l1 = LogisticRegression(C=c, penalty="l1", max_iter=MAX_ITER,
                                    class_weight=CLASS_WEIGHT, solver="liblinear",
                                    random_state=RANDOM_SEED)
        clf_l1.fit(X_tr, y_cari[train_idx])
        fold_probs.append(clf_l1.predict_proba(X_val)[:, 1])
    probs = np.mean(fold_probs, axis=0)
    fold_aurocs.append(roc_auc_score(y_cari[val_idx], probs))
    print(f"  fold {fold+1}/{N_FOLDS}: AUROC={fold_aurocs[-1]:.4f}")

val_auroc = float(np.mean(fold_aurocs))

# ---------------------------------------------------------------------------
# SCAN MP evaluation (fit on all CARI, evaluate on held-out SCAN MP)
# ---------------------------------------------------------------------------

scanmp_auroc = float("nan")
try:
    X_scanmp, y_scanmp, demo_scanmp = load_scanmp_data(ENCODER)

    # Fit on full CARI
    emb_scaler_full = StandardScaler()
    X_cari_scaled   = emb_scaler_full.fit_transform(X_cari)
    X_scanmp_scaled = emb_scaler_full.transform(X_scanmp)

    if USE_DEMOGRAPHICS:
        demo_cari_feat, demo_scaler_full = build_demo_features(demo_cari)
        demo_scanmp_feat, _              = build_demo_features(demo_scanmp, fit_scaler=demo_scaler_full)
        X_cari_full   = np.hstack([X_cari_scaled,   demo_cari_feat])
        X_scanmp_full = np.hstack([X_scanmp_scaled, demo_scanmp_feat])
    else:
        X_cari_full   = X_cari_scaled
        X_scanmp_full = X_scanmp_scaled

    scanmp_probs = []
    for c in C_VALUES:
        clf_full = LogisticRegression(C=c, max_iter=MAX_ITER, class_weight=CLASS_WEIGHT,
                                      solver="lbfgs", random_state=RANDOM_SEED)
        clf_full.fit(X_cari_full, y_cari)
        scanmp_probs.append(clf_full.predict_proba(X_scanmp_full)[:, 1])
    for c in [0.001, 0.01, 0.1]:
        clf_l1 = LogisticRegression(C=c, penalty="l1", max_iter=MAX_ITER,
                                    class_weight=CLASS_WEIGHT, solver="liblinear",
                                    random_state=RANDOM_SEED)
        clf_l1.fit(X_cari_full, y_cari)
        scanmp_probs.append(clf_l1.predict_proba(X_scanmp_full)[:, 1])
    clf_full = None  # ensemble; no single model to save
    probs_scanmp = np.mean(scanmp_probs, axis=0)
    scanmp_auroc = float(roc_auc_score(y_scanmp, probs_scanmp))
    print(f"SCAN MP: {len(y_scanmp)} patients | AUROC={scanmp_auroc:.4f}")

except FileNotFoundError as e:
    print(f"SCAN MP skipped: {e}")

t_end = time.time()

# ---------------------------------------------------------------------------
# Summary — grep-able output for the agent loop
# ---------------------------------------------------------------------------

print("---")
print(f"val_auroc:     {val_auroc:.6f}")
print(f"scanmp_auroc:  {scanmp_auroc:.6f}")
print(f"total_seconds: {t_end - t_start:.1f}")
print(f"encoder:       {ENCODER}")
print(f"n_cari:        {len(y_cari)}")
print(f"n_folds:       {N_FOLDS}")

# Save model
with open("model_latest.pkl", "wb") as f:
    pickle.dump({"clf": clf_full if "clf_full" in dir() else None,
                 "emb_scaler": emb_scaler_full if "emb_scaler_full" in dir() else None,
                 "encoder": ENCODER,
                 "val_auroc": val_auroc}, f)
