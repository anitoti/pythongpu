#!/bin/bash
# ── SBATCH directives MUST sit above `set -euo pipefail`; if they land below a
#    shell statement Slurm stops parsing them and the job silently inherits the
#    dev-partition 15-min walltime cap (see clv-sbatch-directive-trap).
#SBATCH --job-name=benchmark_scaling
#SBATCH --partition=general
# No --qos: same reasoning as submit_perturbation_sweep.sh -- this account has
# no named QOS (sacctmgr shows an empty column), general's MaxTime is enforced
# by the partition directly, and a nonexistent --qos makes sbatch reject the job.
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-35
#SBATCH --output=logs/benchmark_scaling_%A_%a.out
#
# 0-35 = 4 grid sizes x (1 serial + 8 chunk) tasks, matching the GRID_SIZES/
# N_CHUNKS defaults below. If you override either via env var, this directive
# does NOT update itself (SBATCH directives are static) -- pass a matching
# --array on the sbatch command line instead, e.g. for 3 grid sizes x 4 chunks
# ((1+4)*3=15 tasks): GRID_SIZES="64 128 256" N_CHUNKS=4 sbatch --array=0-14 \
# scripts/submit_benchmark_scaling.sh

set -euo pipefail

# ACRES's bare python3 is 3.6 -- too old for this codebase's
# `from __future__ import annotations` usage (see submit_perturbation_sweep.sh
# for the job that first hit this). Same known-working recipe as run_clv_sweep.sh.
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

# ── grid layout ───────────────────────────────────────────────────────────
# Per grid size: 1 serial task (does the whole grid_n^2 batch alone) +
# N_CHUNKS distributed tasks (each does 1/N_CHUNKS of the same batch, landing
# on a different node -- the array fans them out for us). Distributed
# wall-clock for that grid size is the MAX over its N_CHUNKS tasks, computed
# later by plot_scaling_benchmark.py, since they run concurrently.
#
# Integration length here is intentionally short (T_TRANSIENT/TMAX far below
# the real basin-sweep values in submit_perturbation_sweep.sh /
# submit_lorenz_onset_sweep.sh): every RK4 step costs the same regardless of
# how many steps there are, so shortening the run doesn't change the
# serial-vs-distributed SHAPE we're measuring, it just makes 36 tasks at up
# to grid_n=256 (65536 ICs) affordable on a CPU partition instead of hours
# each. Override via env vars if you want the genuine-length numbers instead.
GRID_SIZES=(${GRID_SIZES:-32 64 128 256})
N_CHUNKS=${N_CHUNKS:-8}
T_TRANSIENT=${T_TRANSIENT:-20}
TMAX=${TMAX:-50}

TASKS_PER_GRID=$((1 + N_CHUNKS))
grid_idx=$((SLURM_ARRAY_TASK_ID / TASKS_PER_GRID))
local_idx=$((SLURM_ARRAY_TASK_ID % TASKS_PER_GRID))
GRID_N=${GRID_SIZES[$grid_idx]}

if [ -z "${GRID_N:-}" ]; then
  echo "task ${SLURM_ARRAY_TASK_ID} has no matching grid size (only ${#GRID_SIZES[@]} " \
       "grid sizes x ${TASKS_PER_GRID} tasks-each = $((${#GRID_SIZES[@]} * TASKS_PER_GRID)) " \
       "tasks defined -- check --array matches)" >&2
  exit 1
fi

if [ "$local_idx" -eq 0 ]; then
  MODE=serial
  CHUNKS=1
  CHUNK_IDX=0
else
  MODE=chunk
  CHUNKS=$N_CHUNKS
  CHUNK_IDX=$((local_idx - 1))
fi

echo "[task ${SLURM_ARRAY_TASK_ID}] grid_n=${GRID_N} mode=${MODE} " \
     "n_chunks=${CHUNKS} chunk_index=${CHUNK_IDX}"

python3 scripts/benchmark_scaling.py \
    --dti-path data/DTI-og.mat \
    --grid-n "$GRID_N" \
    --mode "$MODE" \
    --n-chunks "$CHUNKS" \
    --chunk-index "$CHUNK_IDX" \
    --t-transient "$T_TRANSIENT" \
    --tmax "$TMAX" \
    --outdir data/derivatives/scaling_benchmark
