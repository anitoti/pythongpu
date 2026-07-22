#!/bin/bash
# ── SBATCH directives MUST sit above `set -euo pipefail`; if they land below a
#    shell statement Slurm stops parsing them and the job silently inherits the
#    dev-partition 15-min walltime cap (see clv-sbatch-directive-trap).
#SBATCH --job-name=perturb_cmp
#SBATCH --partition=general
# No --qos: `sacctmgr show user "$USER" withassoc` (2026-07-21) shows an empty
# QOS column for this account — there is no named QOS (e.g. 'normal') to pass,
# and general's MaxTime=7-00:00:00 is enforced by the partition directly.
# Passing a nonexistent --qos value makes sbatch reject the job outright.
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/perturb_cmp_%j.out
#
# The fixed perturbation experiment: --compare-boundary-interior samples both
# boundary and interior ICs from the SAME K=0.5 reference slice and tests both
# pools at the SAME coupling, producing p_flip_boundary/p_flip_interior side
# by side. Replaces submit_perturbation_sweep.sh's K=0.0/0.1/0.5 array design,
# whose K=0.0 "control" was invalid (0.0% locked, so the lobe-sign label was
# already a coin flip before any perturbation -- see perturbation_sensitivity.py's
# module docstring). No array needed: one process runs both pools sequentially.

set -euo pipefail

# ACRES's bare `python3` is 3.6 -- too old for `from __future__ import
# annotations` used throughout this pipeline. Same recipe as run_clv_sweep.sh,
# the one known to work on this cluster.
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
python3 -c "import torch; print('torch', torch.__version__)"

OUTDIR="data/derivatives/perturbation_boundary_interior"
mkdir -p "$OUTDIR"

echo "[perturb_cmp] compare-boundary-interior @ K=0.5, grid=128 -> ${OUTDIR}"

python3 -m pythongpu.pipeline.perturbation_sensitivity \
    --compare-boundary-interior \
    --boundary-coupling 0.5 --couplings 0.5 \
    --slice-grid-n 128 \
    --node-x 28 --node-y 79 \
    --n-points 8 --n-directions 16 --n-delta 20 \
    --delta-min 1e-8 --delta-max 1e-1 \
    --t-transient 100 --tmax 500 \
    --dti-path data/DTI-og.mat \
    --outdir "$OUTDIR"
