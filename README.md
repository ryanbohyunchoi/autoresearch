# autoresearch — ATTR-CM

Autonomous AI research for ATTR cardiomyopathy (hATTR vs wtATTR) classification
from ECG embeddings. Applies Karpathy's autoresearch framework: an AI agent
iterates on `train.py`, evaluates via 5-fold CV AUROC, and keeps or discards
each change using git.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
and [ai-ai-ecg](https://github.com/LovedeepDhingra/ai-ai-ecg).

## Structure

```
autoresearch/
├── .devcontainer/
│   ├── Dockerfile           # Python 3.11 slim — no GPU for local dev
│   └── devcontainer.json    # VS Code Dev Container config
├── Singularity.def          # Container for SLURM cluster
├── prepare.py               # Fixed infrastructure: data loading, evaluate_auroc (IMMUTABLE)
├── train.py                 # The ONE file the agent modifies
├── program.md               # Agent experiment loop instructions
├── CLAUDE.md                # Project context for AI assistants
└── requirements.txt
```

## How it works

1. Agent reads `CLAUDE.md` + `prepare.py` + `train.py` for context
2. Modifies `train.py` (classifier, preprocessing, hyperparameters)
3. `git commit -m "description"`
4. `python train.py > run.log 2>&1`
5. `grep "^val_auroc" run.log` — extracts the metric
6. If improved → keep the commit + `git push`
7. If worse → `git reset --hard HEAD~1`
8. Repeat forever

## Local setup (Mac — for development without cluster data)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- VS Code + [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Steps
1. Open this repo in VS Code
2. `Ctrl+Shift+P` → **Dev Containers: Reopen in Container**
3. Wait for build (~2 min first time)
4. `python prepare.py` — generates synthetic embeddings if no real data

### Running an experiment locally
```bash
python train.py > run.log 2>&1
grep "^val_auroc" run.log
```

## Cluster setup (SLURM + Apptainer)

```bash
# Build container (on a machine with build rights, or use --remote)
apptainer build autoresearch.sif Singularity.def

# Transfer to cluster
scp autoresearch.sif user@cluster:/path/to/autoresearch/

# Set data paths and run
export CARI_EMB_PATH=/home/rbc58/mnt/mm_vhd/variant/cari_ecg_embeddings_pretrained_1m.npy
export CARI_CSV=/home/rbc58/mnt/mm_vhd/variant/variant_cari_cohort.csv
export SCANMP_EMB_PATH=/home/rbc58/mnt/mm_vhd/variant/scanmp_ecg_embeddings_pretrained_1m.npy
export SCANMP_CSV=/home/rbc58/mnt/mm_vhd/variant/variant_scan_mp.csv

apptainer exec autoresearch.sif python train.py > run.log 2>&1
```

## Starting an agent run

Open this repo in Claude Code and type:
```
/autoresearch
```

Or just describe what you want: *"Start an autoresearch experiment run to optimize
hATTR vs wtATTR classification AUROC"* — Claude will read `program.md` and begin.
