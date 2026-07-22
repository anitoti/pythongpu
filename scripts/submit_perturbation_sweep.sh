#!/bin/bash
# ── SBATCH directives MUST sit above `set -euo pipefail`; if they land below a
#    shell statement Slurm stops parsing them and the job silently inherits the
#    dev-partition 15-min walltime cap (see clv-sbatch-directive-trap).
#SBATCH --job-name=perturb_sweep
#SBATCH --partition=general
# No --qos: `sacctmgr show user "$USER" withassoc` (2026-07-21) shows an empty
# QOS column for this account — there is no named QOS (e.g. 'normal') to pass,
# and general's MaxTime=7-00:00:00 is enforced by the partition directly.
# Passing a nonexistent --qos value makes sbatch reject the job outright.
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --output=logs/perturb_sweep_%A_%a.out

set -euo pipefail

# One coupling per array task -> fans across the general partition (47 nodes).
# Ladder matches perturbation_sensitivity.py's own DEFAULT_COUPLINGS:
#   0 = uncoupled control, 0.1 = onset region, 0.5 = measured riddled regime.
COUPLINGS=(0.0 0.1 0.5)
K="${COUPLINGS[${SLURM_ARRAY_TASK_ID}]}"

# NOTE: this pipeline is torch-based; ACRES has no system torch/CuPy (CPU-only
# array jobs). Point this at a venv/conda env that provides a CPU torch build
# before relying on the cluster.
# module load python/3.11  # adjust to the Lmod name available on ACRES
# source ~/venvs/pythongpu/bin/activate

OUTDIR="data/derivatives/perturbation_K${K}"
mkdir -p "$OUTDIR"

echo "[task ${SLURM_ARRAY_TASK_ID}] K=${K} -> ${OUTDIR}"

# Every task independently locates its boundary/interior base ICs from the
# K=0.5 lobe-sign field (--boundary-coupling), so the base points are the same
# across all three tasks (same seed) even though each only tests its own K's
# flip sensitivity from those points. That's deliberate: it is the whole
# comparison — same starting points, does P_flip(delta) differ by coupling.
# It does mean each task repeats that one slice integration; at grid-n=128
# that is a small fraction of the per-K perturbation cost below, so it is not
# worth caching across tasks for a 3-array job.
python3 -m pythongpu.pipeline.perturbation_sensitivity \
    --couplings "${K}" \
    --base-ic-mode boundary \
    --boundary-coupling 0.5 \
    --slice-grid-n 128 \
    --node-x 28 --node-y 79 \
    --n-points 8 --n-directions 16 --n-delta 20 \
    --delta-min 1e-8 --delta-max 1e-1 \
    --t-transient 100 --tmax 500 \
    --dti-path data/DTI-og.mat \
    --outdir "$OUTDIR"
