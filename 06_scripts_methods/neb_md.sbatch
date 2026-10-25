#!/bin/bash
#SBATCH --job-name=neb-md
#SBATCH --array=1-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=8500M
#SBATCH --time=24:00:00
#SBATCH --output=logs/neb_md_%A_n%a.out
# one job per event size n = 1..4; 4 NEBs each (SevenNet, LGPS 700 K)
cd /s/ls4/users/archbtwww/umlip-benchmark
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
./envs/mace/bin/python -u neb_from_md.py runs/LGPS_sevennet_700K --potential sevennet \
    --n $SLURM_ARRAY_TASK_ID --per_n 4 --nimages 7
