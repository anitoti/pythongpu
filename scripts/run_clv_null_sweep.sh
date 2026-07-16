#!/usr/bin/env bash
# SBATCH script for ACRES cluster: run CLV diagnostics null-model sweep on GPU
# Requests: --gres=gpu:1, memory 12GB, walltime 6 hours
# Loads modules: Python/3.10.4-GCCcore-11.3.0, libjpeg-turbo
# Robust shell options and logging

set -euo pipefail
IFS=$'\n\t'

# Logging
LOGDIR=${LOGDIR:-output/clv_null_results}
mkdir -p "$LOGDIR"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
LOGFILE="$LOGDIR/run_clv_null_sweep_${TIMESTAMP}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Starting CLV null-model sweep: $(date -u)"

# Slurm SBATCH directives
#SBATCH --job-name=clv-null-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=${LOGDIR}/slurm-%j.out

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

OUTDIR=${OUTDIR:-output/clv_null_results}
mkdir -p "$OUTDIR"

# Coupling sweep: 0.05..0.20 step 0.05
COUPLINGS=(0.05 0.10 0.15 0.20)

for c in "${COUPLINGS[@]}"; do
  echo "Running pythongpu-clv (null-model) with coupling=${c} at $(date -u)"
  pythongpu-clv --null-model --outdir "$OUTDIR/clv_c${c//./_}" --coupling "$c" --steps 1000 --m 10 --K 10 || {
    echo "pythongpu-clv failed for coupling=${c}. Continuing to next value.";
    continue
  }
  echo "Completed coupling=${c}"
done

echo "CLV null-model sweep finished: $(date -u)"
