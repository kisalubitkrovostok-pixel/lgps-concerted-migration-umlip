#!/bin/bash
#SBATCH --job-name=scan-sevennet-700
#SBATCH --partition=hpc4-el7-3d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00

set -euo pipefail
cd /s/ls4/users/archbtwww/umlip-benchmark

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$RESULT_DIR/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

date
hostname
./envs/mace/bin/python -u "$RESULT_DIR/concerted_scan.py" \
    "$PWD/runs/LGPS_sevennet_700K" \
    --d_hop 1.8 1.4 2.2 \
    --tau 2.35 1.5 4 \
    --rcut 2.0 1.6 2.4
date
echo "SCAN COMPLETED"
