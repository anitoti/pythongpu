#!/usr/bin/env bash
#SBATCH --job-name=clv-sweep
#SBATCH --partition=gpu
# GPU count (--gres) is managed by scripts/submit_clv_sweep.sh so CPU-partition
# runs can omit it. If you sbatch THIS file directly and need a GPU, add --gres=gpu:1.
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-25
#
# ---------------------------------------------------------------------------
# CLV diagnostics sweep on GPU (ACRES cluster).
#
# The #SBATCH block above MUST stay in this contiguous comment block at the very
# top of the file, before the first executable line. sbatch stops scanning for
# #SBATCH directives at the first non-comment line (`set -euo pipefail` below);
# any directive placed after it is silently treated as an ordinary comment and
# ignored. That is exactly what dropped earlier jobs onto the cluster's default
# partition + its 15-minute walltime cap despite the 8-hour request here.
#
# Preferred submission path: scripts/submit_clv_sweep.sh, which ALSO passes
# -p gpu / -t 08:00:00 / --qos=<qos> as sbatch command-line flags. CLI flags
# override the directives above and cannot be nullified by placement mistakes.
#
# Loads modules: Python/3.10.4-GCCcore-11.3.0, libjpeg-turbo
# ---------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

# Logging
LOGDIR=${LOGDIR:-output/clv_results}
mkdir -p "$LOGDIR"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
LOGFILE="$LOGDIR/run_clv_sweep_${TIMESTAMP}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Starting CLV sweep: $(date -u)"

# NOTE: SBATCH directives are declared at the very top of this file (above
# `set -euo pipefail`). They must NOT be repeated here — below the first
# executable line sbatch ignores #SBATCH lines entirely. Slurm's own stdout goes
# to the path given by the wrapper's --output (or the default slurm-%j.out); the
# tee'd $LOGFILE above is the rich, timestamped run log.

# Load environment modules (ACRES example)
module load Python/3.10.4-GCCcore-11.3.0 || true
module load libjpeg-turbo || true

# Activate virtualenv if available
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "Using active virtualenv"
else
  if [ -f venv/bin/activate ]; then
    echo "Activating venv/bin/activate"
    # shellcheck source=/dev/null
    source venv/bin/activate
  fi
fi

# Ensure CLI is available (prefer installed editable install)
if ! command -v pythongpu-clv &>/dev/null; then
  echo "pythongpu-clv not found on PATH. Attempting to install editable package into venv..."
  python3 -m pip install -e . || true
fi

# Target DTI file
DTI_PATH=${DTI_PATH:-data/DTI-og.mat}
OUTDIR=${OUTDIR:-output/clv_results}
mkdir -p "$OUTDIR"

# Coupling sweep: generate couplings from 0.25 .. 1.50 in steps of 0.05
# sequence length: inclusive 0.25..1.50 -> 26 values; sbatch array indices are 0..25
COUPLINGS=()
for i in $(seq 0 25); do
  # compute 0.25 + i*0.05 using awk to avoid relying on bc
  c=$(awk -v i="$i" 'BEGIN{printf "%.2f", 0.25 + i*0.05}')
  COUPLINGS+=("$c")
done

# If running as a Slurm array task, select the coupling by SLURM_ARRAY_TASK_ID
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  idx=${SLURM_ARRAY_TASK_ID}
  if [ -z "${COUPLINGS[$idx]:-}" ]; then
    echo "ERROR: Invalid SLURM_ARRAY_TASK_ID=$idx" >&2
    exit 1
  fi
  c=${COUPLINGS[$idx]}
  echo "Array task id=${idx} selecting coupling=${c}"
  echo "Running pythongpu-clv with coupling=${c} at $(date -u)"
  pythongpu-clv --mat "$DTI_PATH" --outdir "$OUTDIR/clv_c${c//./_}" --coupling "$c" --steps 1000 --m 10 --K 10 --coupling-mode x || {
    echo "pythongpu-clv failed for coupling=${c}. Exiting with non-zero status.";
    exit 1
  }
  echo "Completed coupling=${c}"
else
  # not running as an array -- run sequentially for convenience
  for c in "${COUPLINGS[@]}"; do
    echo "Running pythongpu-clv with coupling=${c} at $(date -u)"
    pythongpu-clv --mat "$DTI_PATH" --outdir "$OUTDIR/clv_c${c//./_}" --coupling "$c" --steps 1000 --m 10 --K 10 --coupling-mode x || {
      echo "pythongpu-clv failed for coupling=${c}. Continuing to next value.";
      continue
    }
    echo "Completed coupling=${c}"
  done
fi

echo "CLV sweep finished: $(date -u)"
