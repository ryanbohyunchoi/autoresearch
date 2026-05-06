# CLAUDE.md — autoresearch (ATTR-CM)

Autonomous AI research for ATTR cardiomyopathy classification. We apply
Karpathy's autoresearch framework to optimize a classifier for hereditary (hATTR)
vs wild-type (wtATTR) transthyretin amyloid cardiomyopathy from ECG embeddings.

## Commands

```bash
# Generate synthetic images + CSVs for local pipeline testing
python prepare.py --synthetic

# Extract embeddings for a specific encoder (run once per encoder)
python prepare.py --encoder pretrained_1m
python prepare.py --encoder onnx
python prepare.py --encoder biocontrastive
python prepare.py --encoder biocontrastive_pretrained

# Run one experiment
python train.py > run.log 2>&1

# Extract metric
grep "^val_auroc\|^scanmp_auroc" run.log

# Agent keep/discard loop (see program.md for full protocol)
git add train.py && git commit -m "experiment: <description>"   # keep improvement
git reset --hard HEAD~1                                          # discard regression
```

## Project Overview

**Task:** Binary classification — hATTR (hereditary) vs wtATTR (wild-type) ATTR-CM.

**Input:** Pre-computed 1536-dim ECG embeddings from EfficientNet-B3 (pretrained_1m).
Each embedding represents one ECG image processed through the AI-ECG encoder.

**Metric:** val_auroc (5-fold stratified CV AUROC on CARI cohort, higher is better).
Secondary: scanmp_auroc (external hold-out AUROC on SCAN MP, 46 patients — true
generalization metric; values near 0.80 are clinically meaningful).

**The workflow mirrors autoresearch:**
- Agent modifies only `train.py`
- Each experiment is a deterministic run (fixed seed, fixed 5-fold splits)
- Improvements are kept (git commit advanced), regressions discarded (git reset)
- Results tracked in `results.tsv` (untracked, local only)

See `program.md` for the full experiment loop protocol.

## File Roles

| File | Role | Who edits |
|---|---|---|
| `prepare.py` | Fixed infrastructure: data loading, `evaluate_auroc()`, constants | **Nobody** (read-only) |
| `train.py` | Classifier, preprocessing, hyperparameters | **AI agent only** |
| `program.md` | Experiment loop instructions for the agent | Human (rarely) |
| `CLAUDE.md` | This file — project context for AI assistants | Human |
| `Singularity.def` | Container definition for SLURM cluster | Human |
| `.devcontainer/` | Docker setup for local VS Code development | Human |
| `data/` | Embeddings + CSVs. **Never commit.** | Human (places data here) |
| `checkpoints/` | Saved models from kept experiments. **Never commit.** | Agent (auto-saves) |
| `results.tsv` | Experiment history. **Never commit** (untracked). | Agent (writes each run) |

## Data Format

```
data/
  cari_images/                  ← deidentified CARI ECG PNGs (cari1.png, cari2.png, ...)
  scanmp_images/                ← deidentified SCAN MP ECG PNGs (smp1.png, smp2.png, ...)
  cari_cohort.csv               ← new_name, label ("hATTR"/"wtATTR"), age, gender, race, ethnicity
  scanmp_cohort.csv             ← new_name, label (1=hATTR/0=wtATTR), Age, Gender, Black_race, Hispanic_ethnicity
  encoders/
    converted_model.onnx                  ← amyloid ONNX frozen baseline
    demo_pretrained_encoder_1m.pt         ← best result so far (0.79 SCAN MP)
    biocontrastive_encoder.onnx           ← biocontrastive frozen
    biocontrastive_pretrained_encoder.pt  ← biocontrastive + demo pretraining
  cari_embeddings_{encoder}.npy     ← cached after: python prepare.py --encoder <name>
  scanmp_embeddings_{encoder}.npy   ← same
```

**Image format:** PNG, 300×300 RGB, raw pixel values [0,255]. ONNX model has built-in normalization.

**Encoder pipeline:** `prepare.py` loads images → resizes to 300×300 → runs ONNX/PyTorch encoder → caches 1536-dim embeddings as `.npy` dict `{new_name: np.ndarray(1536,)}`. `train.py` always loads from cache.

**CARI cohort** (training data):
- n ≈ 142 patients | split: 5-fold stratified CV (never a fixed test split on CARI)

**SCAN MP cohort** (external test — NEVER train on this):
- n = 46 patients (hATTR=19, wtATTR=27) — small, results are noisy

## Model Contract

`train.py` must:
1. Load CARI data via `load_cari_data()` from prepare.py
2. Run 5-fold CV and report `val_auroc`
3. Optionally load SCAN MP via `load_scanmp_data()` and report `scanmp_auroc`
4. Print summary block starting with `---` followed by `val_auroc:` on its own line
5. Save `model_latest.pkl` (dict with `clf`, `scaler`, `val_auroc` keys)

Agents can replace the entire classifier pipeline as long as this I/O is preserved.

## Known Results

### Original variant pipeline (~142 patients, one ECG per patient)
| Encoder | CARI CV AUROC | SCAN MP AUROC |
|---------|--------------|---------------|
| onnx frozen | 0.61 | 0.71 |
| pretrained_1m ECG-only LR | 0.93 | 0.79 |
| biocontrastive_pretrained | 0.92 | 0.69 |

### Current autoresearch baseline (297 patients, 1308 ECGs, grouped CV)
| Configuration | CARI CV | SCAN MP |
|---|---|---|
| pretrained_1m ECG-only LR | 0.64 | **0.645** ← start here |
| pretrained_1m + demographics LR | 0.91 | 0.59 ← demographics HURT SCAN MP |

Key insight: **demographics overfit to CARI** with 297 patients. Age/race predict
hATTR in CARI (0.91 CV) but don't transfer to SCAN MP. ECG-only (0.645) is the
honest baseline. Agent should explore whether demographics can be used carefully.

**Target:** beat 0.79 SCAN MP (original pipeline best).
**Clinical utility threshold:** AUROC ≥ 0.80 on SCAN MP.

## Gotchas

**CARI demographics confound the metric.**
Age and race are strong predictors of hATTR vs wtATTR in CARI. A classifier
that learns demographics can hit 0.93 CV AUROC while generalizing poorly.
Watch scanmp_auroc — it does not have the same demographic distribution.

**SCAN MP has only 46 patients.**
SCAN MP AUROC is noisy. A 0.05 swing can be sampling noise. Don't over-interpret
small differences; focus on consistent trends across multiple experiments.

**evaluate_auroc() is the immutable ground truth.**
Never modify it in prepare.py. It uses fixed seed=42 and N_FOLDS=5.
All experiments are comparable because the splits are deterministic.

**results.tsv is never committed.**
Each machine maintains its own experiment history. Use `[test]` prefix for
debugging runs (small data, quick checks).

**SCAN MP is external eval only.**
Never include SCAN MP samples in training or cross-validation.
`load_scanmp_data()` is for evaluation only.

## Environment

**Local dev:** `.devcontainer/` — Python 3.11 slim Docker image, no GPU needed
(sklearn classifiers on 1536-dim embeddings run in <10s on CPU).

**Cluster:** `Singularity.def` — PyTorch + CUDA base image for the SLURM cluster.
Set env vars for data paths:
```bash
export CARI_EMB_PATH=/home/rbc58/mnt/mm_vhd/variant/cari_ecg_embeddings_pretrained_1m.npy
export CARI_CSV=/home/rbc58/mnt/mm_vhd/variant/variant_cari_cohort.csv
export SCANMP_EMB_PATH=/home/rbc58/mnt/mm_vhd/variant/scanmp_ecg_embeddings_pretrained_1m.npy
export SCANMP_CSV=/home/rbc58/mnt/mm_vhd/variant/variant_scan_mp.csv
```

## Best Practices

- **Only modify `train.py`** — `prepare.py` is read-only infrastructure.
- **Flat constants, no CLI args** — hyperparameters are Python constants at the top. No argparse.
- **Simplicity criterion** — equal val_auroc with less code beats complex changes with tiny gains.
- **Watch both metrics** — val_auroc (CARI CV) and scanmp_auroc (external). Optimize for both.
- **Demographics are a double-edged sword** — adding them boosts CARI CV but may hurt SCAN MP.
