# autoresearch

Autonomous ML experiment loop for HPC/SLURM clusters. An LLM agent proposes modifications to a training script, submits SLURM jobs, reads results, and iterates toward a target metric.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Structure

```
autoresearch/
├── Singularity.def          # Container definition (Apptainer/Singularity)
├── agent/
│   ├── loop.py              # Main agent loop
│   ├── llm.py               # Claude API (propose + describe changes)
│   ├── runner.py            # SLURM job submission + polling
│   └── evaluator.py         # Read results.json, reconstruct history
├── targets/
│   └── example/
│       ├── train.py         # Baseline training script (agent modifies this)
│       └── program.md       # Instructions to the agent
├── slurm/
│   ├── agent.sh             # SLURM script to run the agent loop itself
│   └── train_job.sh         # Template SLURM script for each experiment
└── results/                 # gitignored — experiment outputs live here
```

## Setup

### 1. Build the container

On a machine with Apptainer (or use the free Sylabs remote builder):

```bash
apptainer build autoresearch.sif Singularity.def
# or without root:
apptainer build --remote autoresearch.sif Singularity.def
```

Transfer to your cluster:
```bash
scp autoresearch.sif user@cluster:/path/to/autoresearch/
```

### 2. Configure SLURM templates

Edit `slurm/train_job.sh`:
- Set `EMB_PATH` and `LABELS_PATH` to your data paths
- Set the path to `autoresearch.sif`
- Adjust `--partition`, `--mem`, `--time` for your cluster

Edit `slurm/agent.sh`:
- Set `--partition` for a CPU node (agent loop is lightweight)

### 3. Store your API key

```bash
echo "sk-ant-..." > ~/.anthropic_key
chmod 600 ~/.anthropic_key
```

### 4. Run

```bash
sbatch slurm/agent.sh
```

Or locally (if you have `sbatch` access from your login node):

```bash
export ANTHROPIC_API_KEY=$(cat ~/.anthropic_key)
python -m agent.loop \
    --target targets/example \
    --results results/run_001 \
    --slurm slurm/train_job.sh \
    --metric auroc \
    --n-experiments 20
```

## How it works

1. Agent reads `program.md` (instructions) + current `train.py`
2. Sends to Claude with experiment history → gets modified `train.py`
3. Saves modified script to `results/exp_NNN/train.py`
4. Submits `train_job.sh` via `sbatch` with `EXP_DIR` set
5. Polls `squeue` until job finishes
6. Reads `results/exp_NNN/results.json` for the metric
7. If improved: uses new script as base for next iteration
8. Repeats for `--n-experiments` iterations

## Adding a new target

1. Copy `targets/example/` to `targets/my_experiment/`
2. Edit `train.py` to be your baseline training script
   - Must read `EXP_DIR`, `EMB_PATH`, `LABELS_PATH` from env
   - Must write `{"auroc": <float>}` (or your metric key) to `$EXP_DIR/results.json`
3. Edit `program.md` to describe what the agent should/shouldn't touch
4. Point `--target targets/my_experiment` when running the loop
