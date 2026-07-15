#!/bin/bash
# ============================================================================
#  diagnose.slurm — overnight structure gate + attractor count on finished sweeps
#  Project : Nimble Brain — Clarkson University REU (Dr. Jeremie Fish)
#
#  Purpose
#  -------
#  Answer, unattended, the question every downstream number depends on:
#  WHICH coupling values actually have separable basin structure, and how many
#  attractors do they support? Runs entirely on the SAVED VPS `vectors`, so it
#  re-uses completed integrations and never re-integrates a trajectory.
#
#  For every basin_data.npz under each root it runs
#    1. vps_diagnostics — column-shuffle null test + spectral eigengap; prints a
#       STRUCTURED / NO SEPARABLE STRUCTURE verdict per K,
#    2. attractor_id    — emergent (DBSCAN, non-imposed) attractor count that is
#       allowed to be 1, with radius-sensitivity reported.
#
#  CPU-only (sklearn/numpy): no GPU is requested, so it queues fast and does not
#  hold a scarce GPU overnight. --no-plot avoids matplotlib's imaging stack
#  entirely (a login node without libjpeg-turbo makes `import matplotlib` throw).
#
#  Submission
#  ----------
#    sbatch pythongpu/hpc/diagnose.slurm
#    sbatch --export=ALL,ROOTS="/mnt/data/tmp/$USER/coupling_sweep_lowK" \
#           pythongpu/hpc/diagnose.slurm
#
#  Results: <root>/vps_diagnostics.csv  and the job log under logs/.
#SBATCH --job-name=vps-diagnose
# 'general': 7-day limit, many CPU nodes. Deliberately NOT 'gpu' — this job is
# CPU-only (sklearn/numpy) and the cluster has a single GPU node, which must stay
# free for the integration sweeps. ('dev' caps at 15 minutes: validation only.)
#SBATCH --partition=general
#SBATCH --output=logs/diagnose_%j.out
#SBATCH --error=logs/diagnose_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -uo pipefail          # NOT -e: one bad root must not kill the whole night

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load Python/3.10.4-GCCcore-11.3.0
# libjpeg-turbo is what matplotlib's PIL backend needs; harmless with --no-plot
# but loaded so any later plotting step in this job would work.
module load libjpeg-turbo/2.1.3-GCCcore-11.3.0 2>/dev/null || \
    echo "[warn] libjpeg-turbo module unavailable; running --no-plot anyway"

: "${VENV_PATH:=fmri_env}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi
export PYTHONPATH=$SLURM_SUBMIT_DIR

# Roots to scan (space-separated; override with --export=ALL,ROOTS="...").
: "${ROOTS:=/mnt/data/tmp/${USER}/coupling_sweep_lowK /mnt/data/tmp/${USER}/coupling_sweep}"
: "${NULL_REPS:=20}"
: "${N_SUB:=6000}"

echo "==========================================================="
echo " VPS structure gate + attractor count"
echo "   roots      : ${ROOTS}"
echo "   null reps  : ${NULL_REPS}    subsample: ${N_SUB}"
echo "   started    : $(date)"
echo "==========================================================="

for ROOT in ${ROOTS}; do
    if [ ! -d "$ROOT" ]; then
        echo "[skip] no such directory: $ROOT"
        continue
    fi
    n=$(find "$ROOT" -name 'basin_data.npz' | wc -l)
    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "ROOT: $ROOT   ($n basin_data.npz)"
    echo "───────────────────────────────────────────────────────────"
    [ "$n" -eq 0 ] && { echo "[skip] nothing to analyse"; continue; }

    echo "--- [1/2] structure gate -----------------------------------"
    python3 -m pythongpu.pipeline.vps_diagnostics \
        --root "$ROOT" --null-reps "$NULL_REPS" --n-sub "$N_SUB" --no-plot \
        || echo "[warn] vps_diagnostics failed on $ROOT (continuing)"

    echo "--- [2/2] emergent attractor count -------------------------"
    python3 -m pythongpu.pipeline.attractor_id \
        --root "$ROOT" --n-sub "$N_SUB" \
        || echo "[warn] attractor_id failed on $ROOT (continuing)"
done

echo ""
echo "==========================================================="
echo " finished : $(date)"
echo " Read the verdicts above, and <root>/vps_diagnostics.csv."
echo " Only quote D_f / S_bb / Wada for K values reported STRUCTURED."
echo "==========================================================="
