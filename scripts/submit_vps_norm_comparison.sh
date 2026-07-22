#!/bin/bash
# ── SBATCH directives MUST sit above `set -euo pipefail`; if they land below a
#    shell statement Slurm stops parsing them and the job silently inherits the
#    dev-partition 15-min walltime cap (see clv-sbatch-directive-trap).
#SBATCH --job-name=vps_norm_cmp
#SBATCH --partition=general
# No --qos: same reasoning as submit_perturbation_sweep.sh -- no named QOS on
# this account, general's MaxTime is enforced by the partition directly.
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-6
#SBATCH --output=logs/vps_norm_cmp_%A_%a.out
#
# The production VPS-norm comparison: one array task per norm, same fixed
# K=0.5 (measured riddled regime), same node pair (28, 79), same grid
# resolution -- everything held fixed except --vps-norm, per the paper's own
# flagged "L2 chosen by fiat, alternatives unexplored" gap (math_textbook.md
# Sec I.6b).
#
# GRID_N=256 matches the largest resolution already timed in
# submit_benchmark_scaling.sh (serial integration ~635s there), so the
# --time budget below is grounded in a real measurement, not a guess --
# clustering/box-counting adds some overhead on top, hence the 2h margin
# instead of just the ~11 min the integration itself took.
#
# MEMORY: all 7 array tasks OOM'd at the previous --mem=8G. That estimate
# (GRID_N^2 * 2*C(83,2) * 4 =~ 1.8GB) only counted the final (B,C) feature
# array -- run_sweep_streaming's inner loop actually holds ~9 arrays of that
# size alive simultaneously (diff is (B,C,3) = 3 units; dx_abs/L_val/delta
# are 1 unit each per step; mean_dx/M2_dx/mean_L are 3 more, persistent).
# Real peak at GRID_N=256: 9 * 256^2 * 6806 * 4 =~ 8.0GB -- exactly why an
# 8G request had zero headroom and guaranteed the OOM. --mem=16G leaves
# real margin for k-means/box-counting temporaries and allocator overhead.
# (see submit_highres_slices.sh for the same formula at higher GRID_N).
#
# --k-clusters 2 (fixed k, not 'auto') routes through kmeans_gpu, which is
# now seeded (--kmeans-seed, default 42) -- required for this comparison to
# mean anything: an earlier unseeded run made two different norms look like
# they disagreed for no reason other than which local optimum k-means
# happened to land in that particular process (see the kmeans_gpu seeding
# fix commits and math_textbook.md Sec I.6b).

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

# Seven norms: the two magnitude norms the paper could plausibly have used
# instead of L2 (l1, and the l2 baseline itself), the coordinatewise
# extremes (inf, -inf), two intermediate p values for a continuous sweep
# (0.5, 3), and cosine -- the one qualitatively different measure (direction
# instead of magnitude), not reachable by any p.
NORMS=(l2 l1 inf -inf 0.5 3 cosine)
NORM="${NORMS[${SLURM_ARRAY_TASK_ID}]}"
# Filesystem-safe tag for the output dir -- "-inf" and "0.5" are both fine
# as directory names on any POSIX filesystem, but spell out the sign/dot
# explicitly so a directory listing reads unambiguously at a glance.
NORM_TAG="${NORM//./_}"
NORM_TAG="${NORM_TAG/-inf/neg_inf}"

K=${K:-0.5}
GRID_N=${GRID_N:-256}
NODE_X=${NODE_X:-28}
NODE_Y=${NODE_Y:-79}

OUTDIR="data/derivatives/vps_norm_${NORM_TAG}"
mkdir -p "$OUTDIR"

echo "[task ${SLURM_ARRAY_TASK_ID}] vps_norm=${NORM} K=${K} grid_n=${GRID_N} " \
     "nodes=${NODE_X},${NODE_Y} -> ${OUTDIR}"

# Equals form required for negative values (e.g. -inf) -- argparse otherwise
# reads a leading '-' as the start of a new flag, not this flag's value.
python3 -m pythongpu.pipeline.lorenz_fine_coupling_sweep \
    --dti-path data/DTI-og.mat \
    --node-x "$NODE_X" --node-y "$NODE_Y" \
    --k-start "$K" --k-stop "$K" --k-step 0.025 \
    --grid-n "$GRID_N" --k-clusters 2 --kmeans-seed 42 \
    --vps-norm="${NORM}" \
    --outdir "$OUTDIR"
