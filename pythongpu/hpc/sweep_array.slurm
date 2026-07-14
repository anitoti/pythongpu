#!/bin/bash
# ============================================================================
#  sweep_array.slurm — ACRES cluster coupling-coefficient job array
#  Project : Nimble Brain — Clarkson University REU (Dr. Jeremie Fish)
#
#  Purpose
#  -------
#  Distribute a coupling-coefficient sweep of a *selectable* DTI-coupled
#  oscillator — Lorenz-63, Rössler, or Hindmarsh–Rose — across independent
#  GPU nodes. The oscillator is chosen with a positional argument (or the
#  OSC environment variable). Each array task is assigned a single coupling
#  coefficient K drawn from the closed interval [K_MIN, K_MAX] via an affine
#  map on SLURM_ARRAY_TASK_ID, integrates that system on one GPU, and writes
#  its per-node basin morphometry to data/task_<idx>_<osc>_K<coupling>/,
#  with all figures under that task's derivatives/ subdirectory. The
#  resulting per-K basin frames are compiled into an animation downstream.
#
#  When K_MIN/K_MAX are not overridden they default to the chosen
#  oscillator's stable, basin-structured coupling window (see below):
#  Lorenz is bounded at all K, Hindmarsh–Rose syncs by ~0.3, and the
#  X-coupled Rössler destabilises above ~0.15 so its window is narrow.
#
#  Submission
#  ----------
#    # Lorenz (default), 21 tasks spanning its default K window:
#    sbatch --array=0-20 pythongpu/hpc/sweep_array.slurm
#
#    # Rössler / Hindmarsh–Rose via positional argument:
#    sbatch --array=0-20 pythongpu/hpc/sweep_array.slurm rossler
#    sbatch --array=0-20 pythongpu/hpc/sweep_array.slurm hr
#
#    # Override the oscillator, coupling interval and grid resolution:
#    sbatch --array=0-40 \
#           --export=ALL,OSC=hr,K_MIN=0.05,K_MAX=0.30,GRID_N=361 \
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
# K_MIN/K_MAX are intentionally left unset here — they default to the
# selected oscillator's stable coupling window in the routing block below,
# unless the submitter overrides them via --export.
: "${GRID_N:=361}"                    # initial-condition grid points per axis
: "${K_CLUSTERS:=auto}"              # basin count; 'auto' = VPS consensus
: "${DTI_PATH:=data/DTI_A.mat}"      # structural connectivity matrix
: "${RESULTS_ROOT:=data}"            # per-task output root -> data/task_*/

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

# ── Oscillator selection: positional arg, else $OSC, else lorenz ────────────
# Selects the integrator module, the get_plot_path() filename prefix used to
# locate the basin frame, and the default coupling window when K_MIN/K_MAX
# were not overridden. ${1:-...} is safe under `set -u`.
OSC_SEL="$(echo "${1:-${OSC:-lorenz}}" | tr '[:upper:]' '[:lower:]')"
case "$OSC_SEL" in
    lorenz)
        OSC_MODULE="pythongpu.pipeline.lorenz_sweep"
        OSC_PREFIX="lorenz_vps_clustering"
        : "${K_MIN:=0.45}" ; : "${K_MAX:=0.65}"   # bounded at all K
        ;;
    rossler|roessler)
        OSC_SEL="rossler"
        OSC_MODULE="pythongpu.pipeline.rossler_sweep"
        OSC_PREFIX="rossler_vps_clustering"
        : "${K_MIN:=0.02}" ; : "${K_MAX:=0.10}"   # narrow: destabilises >~0.15
        ;;
    hr|hindmarsh_rose|hindmarsh-rose)
        OSC_SEL="hr"
        OSC_MODULE="pythongpu.pipeline.hr_sweep"
        OSC_PREFIX="hr_vps_clustering"
        : "${K_MIN:=0.05}" ; : "${K_MAX:=0.30}"   # sync window ≲0.3
        ;;
    *)
        echo "ERROR: unknown OSC '$OSC_SEL' (choose: lorenz | rossler | hr)" >&2
        exit 1
        ;;
esac

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
# The task directory lives directly under data/ (matches data/task_*/), and
# the sweep routes every figure into its derivatives/ subdirectory.
FRAME_IDX=$(printf "%04d" "$TASK_ID")
TASK_OUTDIR="${RESULTS_ROOT}/task_${FRAME_IDX}_${OSC_SEL}_K${COUPLING}"
mkdir -p "$TASK_OUTDIR/derivatives"

echo "==========================================================="
echo " ACRES coupling-sweep array element"
echo "   array job id      : ${SLURM_ARRAY_JOB_ID}"
echo "   task index        : ${TASK_ID} of ${TASK_COUNT}"
echo "   oscillator        : ${OSC_SEL}  (${OSC_MODULE})"
echo "   coupling K         : ${COUPLING}  (interval [${K_MIN}, ${K_MAX}])"
echo "   grid resolution   : ${GRID_N}^2 initial conditions"
echo "   output directory  : ${TASK_OUTDIR}"
echo "==========================================================="

# ── Integrate the dynamical system for this coupling coefficient ────────────
python3 -m "$OSC_MODULE" \
    --grid-n     "$GRID_N" \
    --k-clusters "$K_CLUSTERS" \
    --coupling   "$COUPLING" \
    --dti-path   "$DTI_PATH" \
    --outdir     "$TASK_OUTDIR"

# ── Register the per-node basin frame in a flat, K-ordered gallery ──────────
# compile_animation.slurm consumes FRAME_GALLERY as a contiguous, zero-padded
# image sequence. A per-oscillator gallery keeps concurrent oscillator sweeps
# from colliding. The basin map is copied (not moved) so the complete per-task
# morphometry — basin_data.npz, boundary, box-counting fit — remains archived
# under TASK_OUTDIR.
FRAME_GALLERY="${RESULTS_ROOT}/frames_${OSC_SEL}"
mkdir -p "$FRAME_GALLERY"

# Each sweep writes plots into <outdir>/derivatives/ with a module-name prefix
# (OSC_PREFIX). Point at the explicit expected filename first, then fall back
# to a glob (in case the module prefix ever changes) so frame collection is
# robust.
BASIN_PNG="${TASK_OUTDIR}/derivatives/${OSC_PREFIX}__basin_map_kmeans.png"
if [ ! -f "$BASIN_PNG" ]; then
    BASIN_PNG=$(ls "${TASK_OUTDIR}"/derivatives/*basin_map_kmeans.png 2>/dev/null | head -1)
fi
if [ -n "$BASIN_PNG" ] && [ -f "$BASIN_PNG" ]; then
    cp "$BASIN_PNG" "${FRAME_GALLERY}/frame_${FRAME_IDX}.png"
    echo "Array element ${TASK_ID} complete: ${OSC_SEL} K=${COUPLING} -> frame_${FRAME_IDX}.png"
else
    echo "ERROR: no basin_map_kmeans.png under ${TASK_OUTDIR}/derivatives" >&2
    exit 1
fi
