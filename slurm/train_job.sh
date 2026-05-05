#!/bin/bash
#SBATCH --job-name=automl_exp
#SBATCH --output={{EXP_DIR}}/slurm_%j.out
#SBATCH --error={{EXP_DIR}}/slurm_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# --- Environment ---
export EXP_DIR={{EXP_DIR}}
export EMB_PATH=/home/rbc58/mnt/mm_vhd/variant/cari_ecg_embeddings_pretrained_1m.npy
export LABELS_PATH=/home/rbc58/mnt/mm_vhd/variant/cari_labels.npy
export ANTHROPIC_API_KEY=""  # not needed inside train.py

# --- Run ---
apptainer exec --nv \
    --bind /home/rbc58/mnt:/home/rbc58/mnt \
    --bind /mnt/raid0/rbc58:/mnt/raid0/rbc58 \
    /path/to/autoresearch.sif \
    python {{EXP_DIR}}/train.py
