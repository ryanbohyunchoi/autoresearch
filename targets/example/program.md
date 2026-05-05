# Experiment Instructions

## Goal
Maximize CARI cross-validation AUROC for hATTR vs wtATTR binary classification using ECG embeddings.

## What you can modify in train.py
- Classifier architecture (LR, MLP depth/width, dropout)
- Optimizer and learning rate schedule
- Data augmentation or embedding normalization
- Feature engineering (e.g., PCA dimensionality, concatenation strategies)
- Class weighting strategy

## What you must NOT change
- The embedding loading logic (`np.load(...)`)
- The cross-validation split structure (StratifiedKFold, 5 folds)
- The output format: `results.json` must contain `{"auroc": <float>}`
- Data paths (they are injected via environment variables)

## Evaluation
- Primary metric: mean 5-fold CV AUROC (higher is better)
- Secondary: SCAN MP external AUROC (reported but not optimized)

## Constraints
- Max runtime: 30 minutes per experiment
- Single GPU only
- Do not overfit to CARI demographics — generalization to SCAN MP matters
