#!/bin/bash
#SBATCH --job-name=automl_agent
#SBATCH --output=logs/agent_%j.out
#SBATCH --error=logs/agent_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpu

# The agent loop itself is lightweight (just LLM calls + squeue polling).
# It submits GPU jobs via sbatch — those run independently.

export ANTHROPIC_API_KEY=$(cat ~/.anthropic_key)

python -m agent.loop \
    --target targets/example \
    --results results/run_001 \
    --slurm slurm/train_job.sh \
    --metric auroc \
    --n-experiments 20 \
    --model claude-opus-4-7 \
    --poll-interval 60 \
    --job-timeout 3600
