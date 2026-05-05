"""
train.py — The ONE file agents edit.

Usage: python train.py > run.log 2>&1

Trains a classifier on CARI ECG embeddings (hATTR vs wtATTR binary).
Evaluates via 5-fold stratified CV AUROC on CARI (primary metric)
and hold-out AUROC on SCAN MP (secondary — external generalization).

Modify anything in this file: classifier, preprocessing, feature engineering,
hyperparameters. Do NOT modify prepare.py.

The goal: maximize val_auroc (5-fold CARI CV AUROC, higher is better).
"""

import time
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from prepare import (
    load_cari_data, load_scanmp_data, evaluate_auroc, N_FOLDS, RANDOM_SEED
)

# ---------------------------------------------------------------------------
# Hyperparameters — edit these directly, no CLI flags
# ---------------------------------------------------------------------------

C            = 1.0      # LR regularization (inverse strength; try 0.01–10)
MAX_ITER     = 1000
CLASS_WEIGHT = "balanced"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
np.random.seed(RANDOM_SEED)

X_cari, y_cari = load_cari_data()
print(f"CARI: {len(y_cari)} patients | hATTR={y_cari.sum()} wtATTR={(y_cari==0).sum()}")

# ---------------------------------------------------------------------------
# Preprocessing (applied inside each fold to avoid leakage)
# ---------------------------------------------------------------------------

def preprocess_factory(X_tr, X_val):
    """Fit scaler on train, apply to val. Returns (X_tr_scaled, X_val_scaled)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_val)


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

def clf_factory():
    return LogisticRegression(
        C=C,
        max_iter=MAX_ITER,
        class_weight=CLASS_WEIGHT,
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )


# ---------------------------------------------------------------------------
# CV evaluation with preprocessing inside folds
# ---------------------------------------------------------------------------

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_aurocs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_cari, y_cari)):
    X_tr, X_val = preprocess_factory(X_cari[train_idx], X_cari[val_idx])
    clf = clf_factory()
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
    X_scanmp, y_scanmp = load_scanmp_data()
    scaler_full = StandardScaler()
    X_cari_scaled = scaler_full.fit_transform(X_cari)
    X_scanmp_scaled = scaler_full.transform(X_scanmp)
    clf_full = clf_factory()
    clf_full.fit(X_cari_scaled, y_cari)
    probs_scanmp = clf_full.predict_proba(X_scanmp_scaled)[:, 1]
    scanmp_auroc = float(roc_auc_score(y_scanmp, probs_scanmp))
    print(f"SCAN MP: {len(y_scanmp)} patients | AUROC={scanmp_auroc:.4f}")
except FileNotFoundError:
    print("SCAN MP data not available — skipping external eval")

t_end = time.time()

# ---------------------------------------------------------------------------
# Summary — grep-able output for the agent loop
# ---------------------------------------------------------------------------

print("---")
print(f"val_auroc:     {val_auroc:.6f}")
print(f"scanmp_auroc:  {scanmp_auroc:.6f}")
print(f"total_seconds: {t_end - t_start:.1f}")
print(f"n_cari:        {len(y_cari)}")
print(f"n_folds:       {N_FOLDS}")

# Save fitted model for downstream use
scaler_save = StandardScaler()
X_cari_scaled_save = scaler_save.fit_transform(X_cari)
clf_save = clf_factory()
clf_save.fit(X_cari_scaled_save, y_cari)
with open("model_latest.pkl", "wb") as f:
    pickle.dump({"clf": clf_save, "scaler": scaler_save, "val_auroc": val_auroc}, f)
