# autoresearch — ATTR-CM hATTR vs wtATTR

This is an autonomous research experiment. You classify hereditary vs wild-type
ATTR cardiomyopathy from ECG embeddings. Your job: iterate on `train.py` to
maximize **val_auroc** (5-fold stratified CV AUROC on CARI, higher is better).

## Setup

Work with the user to set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may05`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Create checkpoints directory**: `mkdir -p checkpoints`
4. **Read the in-scope files** for full context:
   - `CLAUDE.md` — project overview, data format, model contract, known results, gotchas.
   - `prepare.py` — fixed constants, data loading, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Classifier, preprocessing, feature engineering.
5. **Verify data exists**: Run `python prepare.py`.
   If data is missing, it generates synthetic embeddings for pipeline testing.
6. **Initialize results.tsv**: Create with just the header row (see format below).
7. **Push the branch**: `git push -u origin autoresearch/<tag>`
8. **Confirm and go.**

Once confirmed, begin the experiment loop immediately.

## Experimentation

Run an experiment with:
```
python train.py > run.log 2>&1
```

**What you CAN do (modify `train.py` only):**
- Classifier: LR, MLP, XGBoost, SVM, ensemble — anything sklearn-compatible
- Preprocessing: StandardScaler, PCA, UMAP, feature selection
- Hyperparameters: C, regularization, kernel, depth, width, etc.
- Feature engineering: concatenate demographics with embeddings, PCA dim reduction
- Class weighting strategy

**What you CANNOT do:**
- Modify `prepare.py` — it is read-only fixed infrastructure.
- Modify `evaluate_auroc()` — it is the immutable ground-truth metric.
- Install packages not in `requirements.txt`.
- Touch the data files or random seed in prepare.py.

**The goal:** highest `val_auroc` (5-fold CARI CV AUROC). Watch `scanmp_auroc`
too — a CARI improvement that tanks SCAN MP is demographic overfitting, not
a real win. The best result so far on SCAN MP is 0.79 (pretrained_1m LR baseline).

**Simplicity criterion:** A simpler change with equal val_auroc beats a complex
change with small improvement. Deleting code and getting the same result is great.

**The first run:** Always establish the baseline first by running `train.py` as-is.

## Output format

The script prints a summary at the end:

```
---
val_auroc:     0.921000
scanmp_auroc:  0.790000
total_seconds: 4.2
n_cari:        142
n_folds:       5
```

Extract the key metrics:
```
grep "^val_auroc\|^scanmp_auroc" run.log
```

If `grep` returns nothing, the run crashed. Inspect with:
```
tail -n 50 run.log
```

## Logging results

Log each experiment to `results.tsv` (tab-separated). Do not commit this file.

Header and columns:
```
commit	val_auroc	scanmp_auroc	seconds	status	description
```

1. Git commit hash (short, 7 chars)
2. `val_auroc` — 5-fold CARI CV AUROC — use `0.000000` for crashes
3. `scanmp_auroc` — external SCAN MP AUROC — use `nan` if unavailable
4. `seconds` — total_seconds from run.log — use `0.0` for crashes
5. Status: `keep`, `discard`, or `crash`
6. Short description of what this experiment tried

Example:
```
commit	val_auroc	scanmp_auroc	seconds	status	description
a1b2c3d	0.921000	0.790000	4.2	keep	baseline LR C=1.0
b2c3d4e	0.934000	0.761000	3.8	discard	PCA 64 dims (SCAN MP dropped)
c3d4e5f	0.928000	0.795000	5.1	keep	LR C=0.1 + StandardScaler
d4e5f6g	0.000000	nan	0.0	crash	XGB import error
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/may05`).

LOOP FOREVER:

1. Check git state: what branch and commit are you on?
2. Modify `train.py` with an experimental idea.
3. `git commit -m "short description of change"`
4. Run: `python train.py > run.log 2>&1`
5. Extract results: `grep "^val_auroc\|^scanmp_auroc" run.log`
6. If grep is empty → crashed. Run `tail -n 50 run.log` for the traceback.
   Fix trivial issues. Otherwise log as `crash` and move on.
7. Record in `results.tsv`.
8. If `val_auroc` improved (higher) AND `scanmp_auroc` did not drop by >0.05 → **keep**:
   ```
   cp model_latest.pkl checkpoints/model_$(git rev-parse --short HEAD).pkl
   git push origin autoresearch/<tag>
   ```
9. If `val_auroc` equal/worse, OR `scanmp_auroc` dropped by >0.05 → **discard**:
   ```
   git reset --hard HEAD~1
   ```

**NEVER STOP.** Once the experiment loop begins, do not pause to ask the human
if you should continue. The loop runs until the human manually interrupts you.

## Promising directions to explore

- **Regularization:** LR C value (0.001, 0.01, 0.1, 1, 10), L1 vs L2 penalty
- **Dimensionality reduction:** PCA (16, 32, 64, 128 dims), UMAP, feature selection
- **Classifiers:** MLP (sklearn), SVM (RBF kernel), XGBoost, Random Forest, ensemble
- **Demographics:** concatenate age + gender + race with ECG embedding
- **Normalization:** StandardScaler, RobustScaler, L2 normalize embeddings
- **Class weighting:** balanced, manual inverse-frequency, no weighting
- **Calibration:** Platt scaling, isotonic regression on top of classifier
- **Feature selection:** SelectKBest, variance threshold, recursive elimination
- **Ensembling:** average predictions from multiple classifiers or embeddings
