#!/bin/bash
# ── SBATCH directives MUST sit above `set -euo pipefail`; if they land below a
#    shell statement Slurm stops parsing them and the job silently inherits the
#    dev-partition 15-min walltime cap (see clv-sbatch-directive-trap).
#SBATCH --job-name=highres_slices
#SBATCH --partition=general
# No --qos: same reasoning as submit_perturbation_sweep.sh -- no named QOS on
# this account, general's MaxTime is enforced by the partition directly.
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-3
#SBATCH --output=logs/highres_slices_%A_%a.out
#
# Higher-resolution / additional 2D basin slices than the original paper's
# 750x750: one array task per (node_x, node_y) slice, all at the same fixed
# K=0.5 (the measured riddled regime) and the same GRID_N resolution.
#
# MEMORY: run_sweep_streaming streams over time (integration length is free)
# but still holds one (B, 2*C(83,2)) = (B, 6806) float32 VPS array in memory
# for the whole batch. B = GRID_N^2, so bytes = GRID_N^2 * 6806 * 4. At the
# paper's own 750x750 that's already ~15GB; GRID_N=900 (below) is ~22GB.
# --mem=48G leaves real headroom for k-means/box-counting temporaries on top.
# If you push GRID_N higher, recompute this and raise --mem to match --
# there is no other guard against an OOM kill at higher resolution.

set -euo pipefail

module load Python/3.10.4-GCCcore-11.3.0 || true
module load libjpeg-turbo || true
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "Using active virtualenv"
elif [ -f venv/bin/activate ]; then
  echo "Activating venv/bin/activate"
  # shellcheck source=/dev/null
  source venv/bin/activate
fi
python3 --version

# Four slices: the two already covered by the onset/basin sweeps, run here at
# higher resolution, plus two new node pairs for broader connectome coverage
# (arbitrary spread, not claiming anatomical significance -- just "more
# slices than the paper mapped," per its own stated limitation).
NODE_PAIRS=(
  "28 79"
  "73 81"
  "5 60"
  "15 45"
)
read -r NODE_X NODE_Y <<< "${NODE_PAIRS[${SLURM_ARRAY_TASK_ID}]}"

K=${K:-0.5}
GRID_N=${GRID_N:-900}

OUTDIR="data/derivatives/highres_n${NODE_X}_n${NODE_Y}"
mkdir -p "$OUTDIR"

echo "[task ${SLURM_ARRAY_TASK_ID}] node_x=${NODE_X} node_y=${NODE_Y} " \
     "K=${K} grid_n=${GRID_N} -> ${OUTDIR}"

python3 -m pythongpu.pipeline.lorenz_fine_coupling_sweep \
    --dti-path data/DTI-og.mat \
    --node-x "$NODE_X" --node-y "$NODE_Y" \
    --k-start "$K" --k-stop "$K" --k-step 0.025 \
    --grid-n "$GRID_N" --k-clusters 2 \
    --outdir "$OUTDIR"
