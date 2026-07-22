#!/bin/bash
# ── SBATCH directives MUST sit above `set -euo pipefail`; if they land below a
#    shell statement Slurm stops parsing them and the job silently inherits the
#    dev-partition 15-min walltime cap (see clv-sbatch-directive-trap).
#SBATCH --job-name=lorenz_onset
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-36
#SBATCH --output=logs/lorenz_onset_%A_%a.out

set -euo pipefail

# One K value per array task -> fans across the general partition (47 nodes).
# Ladder: K = 0.000 .. 0.900 step 0.025  (37 values, indices 0..36).
K=$(python3 -c "print(f'{0.025*${SLURM_ARRAY_TASK_ID}:.4f}')")

# NOTE: this pipeline is torch-based; ACRES has no system torch. Point this at a
# venv/conda env that provides torch (CPU build) before relying on the cluster.
# module load python/3.11  # adjust to the Lmod name available on ACRES
# source ~/venvs/pythongpu/bin/activate

echo "[task ${SLURM_ARRAY_TASK_ID}] K=${K}"
python3 -m pythongpu.pipeline.lorenz_fine_coupling_sweep \
    --node-x 73 --node-y 81 \
    --k-start "${K}" --k-stop "${K}" --k-step 0.025 \
    --grid-n 256 --k-clusters 2 \
    --outdir data/derivatives
