#!/bin/bash
#SBATCH --job-name=NEB-MD-sevennet
#SBATCH --partition=hpc4-el7-3d
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00

set -euo pipefail
cd /s/ls4/users/archbtwww/umlip-benchmark

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$NEB_RESULTS/.matplotlib_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

hostname
./envs/sevennet/bin/python -u "$NEB_RESULTS/neb_from_md.py" \
    "$PWD/runs/LGPS_sevennet_700K" \
    --potential sevennet \
    --n "$SLURM_ARRAY_TASK_ID" \
    --per_n 4 \
    --max_tries 15 \
    --nimages 7
