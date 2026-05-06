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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from prepare import (
    load_cari_data, load_scanmp_data, evaluate_auroc, N_FOLDS, RANDOM_SEED,
    extract_embeddings, ENCODER_CONFIGS, _emb_cache_path
)

# ---------------------------------------------------------------------------
# Hyperparameters — edit these directly, no CLI flags
# ---------------------------------------------------------------------------

ENCODER          = "pretrained_1m"    # onnx | pretrained_1m | biocontrastive | biocontrastive_pretrained
USE_DEMOGRAPHICS = True               # concatenate age + gender + race + ethnicity with embedding
C                = 1.0                # LR regularization (try: 0.01, 0.1, 1.0, 10.0)
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
# ---------------------------------------------------------------------------

def build_demo_features(demo: pd.DataFrame, fit_scaler=None, fit_ohe_cols=None):
    """
    Encode demographics: standardize age, one-hot encode categoricals.
    Returns (features_array, scaler, ohe_cols).
    Pass fit_scaler/fit_ohe_cols from training fold to apply same transform to val.
    """
    # Detect column names (CARI uses lowercase, SCAN MP uses Title case)
    age_col     = "age"     if "age"     in demo.columns else "Age"
    gender_col  = "gender"  if "gender"  in demo.columns else "Gender"
    race_col    = "race"    if "race"    in demo.columns else "Black_race"
    eth_col     = "ethnicity" if "ethnicity" in demo.columns else "Hispanic_ethnicity"

    num = demo[[age_col]].fillna(demo[age_col].median()).values.astype(np.float32)
    cat_cols = [c for c in [gender_col, race_col, eth_col] if c in demo.columns]
    cat = pd.get_dummies(demo[cat_cols].fillna("Unknown"), drop_first=False).values.astype(np.float32)

    if fit_scaler is None:
        scaler = StandardScaler()
        num = scaler.fit_transform(num)
    else:
        scaler = fit_scaler
        num = scaler.transform(num)

    features = np.hstack([num, cat]) if cat.shape[1] > 0 else num
    return features, scaler, cat_cols


# ---------------------------------------------------------------------------
# 5-fold CV with per-fold preprocessing (no leakage)
# ---------------------------------------------------------------------------

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_aurocs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_cari, y_cari)):
    X_tr, X_val = X_cari[train_idx], X_cari[val_idx]

    # Scale embedding
    emb_scaler = StandardScaler()
    X_tr  = emb_scaler.fit_transform(X_tr)
    X_val = emb_scaler.transform(X_val)

    # Concatenate demographics
    if USE_DEMOGRAPHICS:
        demo_tr, demo_scaler, ohe_cols = build_demo_features(demo_cari.iloc[train_idx])
        demo_val, _, _                 = build_demo_features(demo_cari.iloc[val_idx],
                                                              fit_scaler=demo_scaler)
        X_tr  = np.hstack([X_tr,  demo_tr])
        X_val = np.hstack([X_val, demo_val])

    clf = LogisticRegression(C=C, max_iter=MAX_ITER, class_weight=CLASS_WEIGHT,
                             solver="lbfgs", random_state=RANDOM_SEED)
    clf.fit(X_tr, y_cari[train_idx])
    probs = clf.predict_proba(X_val)[:, 1]
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
        demo_cari_feat, demo_scaler_full, _ = build_demo_features(demo_cari)
        demo_scanmp_feat, _, _              = build_demo_features(demo_scanmp,
                                                                   fit_scaler=demo_scaler_full)
        # Align columns (SCAN MP may have different demo cols than CARI)
        n_cari_demo = demo_cari_feat.shape[1]
        n_smp_demo  = demo_scanmp_feat.shape[1]
        if n_smp_demo < n_cari_demo:
            pad = np.zeros((len(demo_scanmp_feat), n_cari_demo - n_smp_demo), dtype=np.float32)
            demo_scanmp_feat = np.hstack([demo_scanmp_feat, pad])
        elif n_smp_demo > n_cari_demo:
            demo_scanmp_feat = demo_scanmp_feat[:, :n_cari_demo]

        X_cari_full   = np.hstack([X_cari_scaled,   demo_cari_feat])
        X_scanmp_full = np.hstack([X_scanmp_scaled, demo_scanmp_feat])
    else:
        X_cari_full   = X_cari_scaled
        X_scanmp_full = X_scanmp_scaled

    clf_full = LogisticRegression(C=C, max_iter=MAX_ITER, class_weight=CLASS_WEIGHT,
                                  solver="lbfgs", random_state=RANDOM_SEED)
    clf_full.fit(X_cari_full, y_cari)
    probs_scanmp = clf_full.predict_proba(X_scanmp_full)[:, 1]
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
