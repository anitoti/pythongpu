#!/bin/bash
# ============================================================================
#  attractor_integrate.slurm — independent, VPS-free attractor count
#  Project : Nimble Brain — Clarkson University REU (Dr. Jeremie Fish)
#
#  Purpose
#  -------
#  Cross-check the basin count using machinery that shares NOTHING with the VPS
#  clustering pipeline. Each array task integrates the real DTI-coupled Lorenz
#  network at one coupling K, builds a compact final-state descriptor per
#  initial condition (per-node long-time mean X, std X, mean |X| — all averages
#  that are constant on a single attractor), and groups them with DBSCAN at an
#  auto-selected radius, so the count EMERGES and may legitimately be 1.
#
#  Why this run matters
#  --------------------
#  `attractor_id --npz` already reports 1 attractor at every K, but it reads the
#  saved VPS `vectors`. This path never touches VPS, never touches k-means, and
#  never touches the silhouette/null statistic (which needed three separate
#  fixes). If it independently also returns 1, the conclusion rests on two
#  methods with no shared machinery and cannot be dismissed as a clustering
#  artifact.
#
#  Known caveat (read before believing a low count)
#  ------------------------------------------------
#  The descriptor is per-node X statistics. It may UNDER-separate attractors that
#  differ by *which nodes synchronise with which* rather than by amplitude — the
#  VPS pairwise coherence captures that better. A count of 1 here is therefore
#  consistent with, but not proof of, a single attractor. Disagreement with the
#  --npz path would be the interesting outcome.
#
#  Submission (7 tasks: the QOS caps submitted jobs per user)
#  ---------------------------------------------------------
#    sbatch --array=0-6 pythongpu/hpc/attractor_integrate.slurm
#    sbatch --array=0-6 --export=ALL,GRID_N=181,TMAX=200 \
#           pythongpu/hpc/attractor_integrate.slurm
#
#  Results: $RESULTS_ROOT/attr_K<K>.csv per task, plus logs/attr_%A_%a.out
#SBATCH --job-name=attr-integrate
#SBATCH --partition=gpu
#SBATCH --output=logs/attr_%A_%a.out
#SBATCH --error=logs/attr_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -uo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load Python/3.10.4-GCCcore-11.3.0
module load libjpeg-turbo/2.1.3-GCCcore-11.3.0 2>/dev/null || true

: "${VENV_PATH:=fmri_env}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$SLURM_SUBMIT_DIR

# ── Tunables ────────────────────────────────────────────────────────────────
: "${GRID_N:=361}"                    # match the sweeps' IC resolution
: "${TMAX:=500}"                      # long-time averaging window
: "${T_TRANSIENT:=100}"               # burn-in, discarded
: "${DTI_PATH:=data/DTI_A.mat}"
: "${RESULTS_ROOT:=/mnt/data/tmp/${USER}/attractor_integrate}"
# Coupling ladder spanning BOTH the low-K window (where structure is most
# plausible) and the 0.45-0.65 window the earlier sweeps covered.
: "${K_LIST:=0.00 0.10 0.20 0.30 0.45 0.55 0.65}"

read -r -a KS <<< "$K_LIST"
IDX="${SLURM_ARRAY_TASK_ID}"
if [ "$IDX" -ge "${#KS[@]}" ]; then
    echo "ERROR: task $IDX exceeds K_LIST length ${#KS[@]}; use --array=0-$(( ${#KS[@]} - 1 ))" >&2
    exit 1
fi
COUPLING="${KS[$IDX]}"

mkdir -p "$RESULTS_ROOT"
CSV="${RESULTS_ROOT}/attr_K${COUPLING}.csv"

echo "==========================================================="
echo " VPS-free attractor count (final-state descriptor + DBSCAN)"
echo "   array/task : ${SLURM_ARRAY_JOB_ID}/${IDX}"
echo "   coupling K : ${COUPLING}   (ladder: ${K_LIST})"
echo "   grid       : ${GRID_N}^2 ICs   tmax=${TMAX}s  transient=${T_TRANSIENT}s"
echo "   csv        : ${CSV}"
echo "   started    : $(date)"
echo "==========================================================="

python3 -m pythongpu.pipeline.attractor_id \
    --integrate \
    --coupling     "$COUPLING" \
    --grid-n       "$GRID_N" \
    --tmax         "$TMAX" \
    --t-transient  "$T_TRANSIENT" \
    --dti-path     "$DTI_PATH" \
    --csv          "$CSV"

echo "finished : $(date)"
echo "Compare n_attractors against the --npz path (which reported 1 at every K)."
