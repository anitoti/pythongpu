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
#SBATCH --mem=96G
#SBATCH --exclusive
#SBATCH --exclude=compute-11-3,compute-11-4,compute-11-5,compute-11-34,compute-11-35,compute-11-36,compute-11-37,compute-11-38,compute-11-39,compute-11-40,compute-21-1,compute-21-3,compute-21-4,compute-21-5,compute-21-6,compute-21-8,compute-21-9,compute-21-11,compute-21-31,compute-21-32,compute-21-33,compute-21-38,compute-21-39
#SBATCH --array=0-3
#SBATCH --output=logs/highres_slices_%A_%a.out
#
# TEMPORARY tonight-only fix (2026-07-22): both prior attempts (4556090,
# 4556109) OOM'd/FAILED with a genuine allocation error (RuntimeError: tried
# to allocate 33GB) despite the --mem=96G / GRID_N=800 formula below
# estimating a 78GB peak with real margin against a 116GB node -- the
# formula's own math checks out (9 * GRID_N^2 * 3403 * 4 bytes = ~78GB), so
# the actual OOM cause was node-sharing: SLURM co-located this job's tasks
# with each other AND with another user's (watermlj) 2+ day job, none of
# which get the full 116GB they were individually promised. --exclude
# blacklists every node shown busy in `sinfo -p general` just before this
# submission; --exclusive forces this array onto genuinely free whole
# nodes. (Tried --nodelist with multiple hosts first -- this ACRES SLURM
# version misparses it against --array, throwing "invalid number of
# nodes"; --exclude doesn't have that bug.) This list goes stale as
# cluster load changes -- re-check before reuse.
#
# Higher-resolution / additional 2D basin slices than the original paper's
# 750x750: one array task per (node_x, node_y) slice, all at the same fixed
# K=0.5 (the measured riddled regime) and the same GRID_N resolution.
#
# MEMORY: task 3 of the GRID_N=900/--mem=48G run OOM'd (0:125), and the other
# 3 tasks were very likely doomed the same way, just slower to hit it. The
# --mem=48G estimate only counted one (B, 2*C(83,2)) = (B, 6806) float32
# array -- run_sweep_streaming's inner loop actually holds ~9 arrays of that
# per-pair size alive simultaneously (diff is (B,C,3) = 3 units; dx_abs/
# L_val/delta are 1 unit each per step; mean_dx/M2_dx/mean_L are 3 more,
# persistent), i.e. real peak =~ 9 * GRID_N^2 * 3403 * 4 bytes, not 1x.
#
# ACRES `general` nodes have 116GB total (`sinfo -o "%n %m %e" -p general`,
# 2026-07-22) -- GRID_N=900's real peak (~99GB) left only ~17GB margin
# against a formula that had already been wrong once. Dropped to GRID_N=800
# (~78GB peak, ~38GB margin) -- still legitimately above the paper's
# 750x750, with real headroom against estimation error this time.
# --mem=96G gives ~18GB of buffer above the ~78GB estimate itself.
# If you push GRID_N higher, recompute against the 9x formula (not the old
# 1x one) and check it against the node ceiling before trusting it.

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
GRID_N=${GRID_N:-800}

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
