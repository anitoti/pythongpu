#!/bin/bash
# ============================================================================
#  sweep_array.slurm — ACRES cluster coupling-coefficient job array
#  Project : Nimble Brain — Clarkson University REU (Dr. Jeremie Fish)
#
#  Purpose
#  -------
#  Distribute a coupling-coefficient sweep of the DTI-coupled Lorenz-63
#  network across independent GPU nodes. Each array task is assigned a
#  single coupling coefficient K drawn from the closed interval
#  [K_MIN, K_MAX] (default [0.45, 0.65]) via an affine map on
#  SLURM_ARRAY_TASK_ID, integrates the dynamical system on one GPU, and
#  writes its per-node basin morphometry to an isolated output directory.
#  The resulting per-K basin frames are subsequently compiled into an
#  animation by compile_animation.slurm.
#
#  Submission
#  ----------
#    # 21 tasks spanning K in [0.45, 0.65] at delta-K = 0.01:
#    sbatch --array=0-20 pythongpu/hpc/sweep_array.slurm
#
#    # Override the coupling interval and grid resolution at submit time:
#    sbatch --array=0-40 \
#           --export=ALL,K_MIN=0.45,K_MAX=0.65,GRID_N=361 \
#           pythongpu/hpc/sweep_array.slurm
#SBATCH --job-name=coupling-sweep
#SBATCH --partition=gpu               # GPU partition (gpu-21-28, Tesla V100 x4);
                                      # substitute 'dev' for brief validation.
#SBATCH --output=logs/sweep_%A_%a.out # %A = array job id, %a = task index
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --ntasks=1                    # single task per array element
#SBATCH --gres=gpu:1                  # one GPU device per array element
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                     # host memory per array element
#SBATCH --time=02:00:00

set -euo pipefail

# ── Tunable parameters (override via --export at submit time) ───────────────
# These defaults define a delta-K = 0.01 lattice over [0.45, 0.65] when the
# array is submitted as --array=0-20.
: "${K_MIN:=0.45}"                    # lower coupling coefficient
: "${K_MAX:=0.65}"                    # upper coupling coefficient
: "${GRID_N:=361}"                    # initial-condition grid points per axis
: "${K_CLUSTERS:=auto}"              # basin count; 'auto' = VPS consensus
: "${DTI_PATH:=data/DTI_A.mat}"      # structural connectivity matrix
: "${RESULTS_ROOT:=data/coupling_sweep}"  # per-task output root

# ── Working directory and module environment ────────────────────────────────
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load Python/3.10.4-GCCcore-11.3.0
module load libjpeg-turbo/2.1.3-GCCcore-11.3.0

# Activate a virtualenv only if one is present. VENV_PATH may be overridden at
# submit time (--export=ALL,VENV_PATH=/path/to/venv); when it is absent the job
# falls back to the module Python plus any user-site (~/.local) packages, which
# is how the ACRES environment is provisioned.
: "${VENV_PATH:=fmri_env}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated virtualenv: ${VENV_PATH}"
else
    echo "No virtualenv at '${VENV_PATH}'; using module Python + user-site packages."
fi

# Defragment GPU memory segments to avoid allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Required for the `from pythongpu...` absolute-import convention — without
# this only the invoked module's directory is placed on sys.path, not the
# repository root.
export PYTHONPATH=$SLURM_SUBMIT_DIR

# ── Affine map: SLURM_ARRAY_TASK_ID -> coupling coefficient K ────────────────
# For a task count M and task index a in [0, M-1]:
#     K(a) = K_MIN + a * (K_MAX - K_MIN) / (M - 1)
# so a = 0 yields K_MIN and a = M-1 yields K_MAX (endpoints inclusive).
# A degenerate single-element array (M = 1) collapses to K = K_MIN.
TASK_ID="${SLURM_ARRAY_TASK_ID}"
TASK_COUNT="${SLURM_ARRAY_TASK_COUNT}"

COUPLING=$(awk -v a="$TASK_ID" -v lo="$K_MIN" -v hi="$K_MAX" -v n="$TASK_COUNT" \
    'BEGIN { if (n <= 1) printf "%.4f", lo;
             else         printf "%.4f", lo + a * (hi - lo) / (n - 1) }')

# Zero-padded frame index preserves K-ordering for the animation compiler.
FRAME_IDX=$(printf "%04d" "$TASK_ID")
TASK_OUTDIR="${RESULTS_ROOT}/task_${FRAME_IDX}_K${COUPLING}"
mkdir -p "$TASK_OUTDIR"

echo "==========================================================="
echo " ACRES coupling-sweep array element"
echo "   array job id      : ${SLURM_ARRAY_JOB_ID}"
echo "   task index        : ${TASK_ID} of ${TASK_COUNT}"
echo "   coupling K         : ${COUPLING}  (interval [${K_MIN}, ${K_MAX}])"
echo "   grid resolution   : ${GRID_N}^2 initial conditions"
echo "   output directory  : ${TASK_OUTDIR}"
echo "==========================================================="

# ── Integrate the dynamical system for this coupling coefficient ────────────
python3 -m pythongpu.pipeline.lorenz_sweep \
    --grid-n     "$GRID_N" \
    --k-clusters "$K_CLUSTERS" \
    --coupling   "$COUPLING" \
    --dti-path   "$DTI_PATH" \
    --outdir     "$TASK_OUTDIR"

# ── Register the per-node basin frame in a flat, K-ordered gallery ──────────
# compile_animation.slurm consumes FRAME_GALLERY as a contiguous, zero-padded
# image sequence. The basin map is copied (not moved) so the complete
# per-task morphometry — basin_data.npz, boundary, box-counting fit — remains
# archived under TASK_OUTDIR.  
FRAME_GALLERY="${RESULTS_ROOT}/frames"
mkdir -p "$FRAME_GALLERY"

# lorenz_sweep writes plots into <outdir>/derivatives/ with a module-name
# prefix. Point at the explicit expected filename first, then fall back to a
# glob (in case the module prefix ever changes) so frame collection is robust.
BASIN_PNG="${TASK_OUTDIR}/derivatives/lorenz_vps_clustering__basin_map_kmeans.png"
if [ ! -f "$BASIN_PNG" ]; then
    BASIN_PNG=$(ls "${TASK_OUTDIR}"/derivatives/*basin_map_kmeans.png 2>/dev/null | head -1)
fi
if [ -n "$BASIN_PNG" ] && [ -f "$BASIN_PNG" ]; then
    cp "$BASIN_PNG" "${FRAME_GALLERY}/frame_${FRAME_IDX}.png"
    echo "Array element ${TASK_ID} complete: K=${COUPLING} -> frame_${FRAME_IDX}.png"
else
    echo "ERROR: no basin_map_kmeans.png under ${TASK_OUTDIR}/derivatives" >&2
    exit 1
fi
