"""
Baseline train.py — agent will modify this to improve AUROC.
Writes results.json with {"auroc": <float>} upon completion.
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# --- Paths injected via environment ---
EMB_PATH = os.environ["EMB_PATH"]        # .npy embedding dict {id: vector}
LABELS_PATH = os.environ["LABELS_PATH"]  # .npy label dict {id: 0/1}
OUT_DIR = Path(os.environ["EXP_DIR"])

# --- Load data ---
embeddings = np.load(EMB_PATH, allow_pickle=True).item()
labels_dict = np.load(LABELS_PATH, allow_pickle=True).item()

ids = sorted(set(embeddings) & set(labels_dict))
X = np.stack([embeddings[i] for i in ids])
y = np.array([labels_dict[i] for i in ids])

# --- 5-fold CV ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aurocs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    probs = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, probs)
    fold_aurocs.append(auroc)
    print(f"Fold {fold+1}: AUROC={auroc:.4f}")

mean_auroc = float(np.mean(fold_aurocs))
print(f"Mean AUROC: {mean_auroc:.4f}")

(OUT_DIR / "results.json").write_text(json.dumps({"auroc": mean_auroc}))
