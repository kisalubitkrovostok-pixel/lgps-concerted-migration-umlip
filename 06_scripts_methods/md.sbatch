#!/bin/bash
#SBATCH --job-name=umlip-md
#SBATCH --array=0-23
#SBATCH --cpus-per-task=8
#SBATCH --mem=8500M
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/md_%A_%a.out

# 4 potentials x 2 materials x 3 temperatures = 24 jobs.
# Task 0 = MACE / LGPS / 700 K  -> the course trajectory, starts first.
POTS=(mace sevennet chgnet m3gnet)
MATS=(LGPS LLZO)
TS=(700 900 1100)
i=$SLURM_ARRAY_TASK_ID
POT=${POTS[$((i % 4))]}
MAT=${MATS[$(((i / 4) % 2))]}
T=${TS[$((i / 8))]}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# one conda env per potential (matgl/DGL often conflicts with the others)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umlip-$POT

# rerunning this array after the 3-day limit continues every unfinished trajectory
python run_md.py --potential $POT --material $MAT --T $T
