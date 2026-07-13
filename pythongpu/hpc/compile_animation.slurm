#!/bin/bash
# ============================================================================
#  compile_animation.slurm — ACRES post-processing / frame muxing
#  Project : Nimble Brain — Clarkson University REU (Dr. Jeremie Fish)
#
#  Purpose
#  -------
#  Compile the per-coupling basin frames emitted by sweep_array.slurm into a
#  single MP4 rendered at 12 frames per second. This is a lightweight,
#  CPU-only post-processing stage: it performs no numerical integration and
#  therefore requests no GPU device.
#
#  Ordering guarantee
#  ------------------
#  This job must execute only after every array element has terminated
#  successfully. Submit it with an afterok dependency on the array job so it
#  is released solely when all tasks exit cleanly (non-zero exit in any array
#  element cancels the compilation):
#
#    ARRAY_ID=$(sbatch --parsable --array=0-20 pythongpu/hpc/sweep_array.slurm)
#    sbatch --dependency=afterok:${ARRAY_ID} pythongpu/hpc/compile_animation.slurm
#
#  The zero-padded frame_%04d.png naming imposed by the array script yields a
#  contiguous, K-monotone sequence, so ffmpeg concatenates the frames in
#  ascending coupling-coefficient order without an explicit frame list.
# ============================================================================

#SBATCH --job-name=lorenz_compile_mp4
#SBATCH --partition=general           # CPU-only muxing; no GPU requested.
#SBATCH --output=logs/compile_%j.out
#SBATCH --error=logs/compile_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G                      # frame decode/encode buffers only
#SBATCH --time=00:20:00

set -euo pipefail

# ── Tunable parameters (override via --export at submit time) ───────────────
: "${RESULTS_ROOT:=data/coupling_sweep}"          # matches sweep_array.slurm
: "${FRAME_GALLERY:=${RESULTS_ROOT}/frames}"       # zero-padded frame sequence
: "${FRAME_RATE:=12}"                              # output rate, frames/second
: "${OUTPUT_MP4:=${RESULTS_ROOT}/coupling_sweep.mp4}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ── Provision an ffmpeg encoder ─────────────────────────────────────────────
# ffmpeg is not present in the base environment on this host; load the
# cluster-provided module. If the ACRES ffmpeg module carries a different
# version string, adjust the identifier below (query with `module avail ffmpeg`).
module load FFmpeg/4.4.2-GCCcore-11.3.0 2>/dev/null \
    || module load FFmpeg 2>/dev/null \
    || echo "[warn] no FFmpeg module resolved; relying on PATH-resident ffmpeg"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[fatal] ffmpeg not found. Load a cluster FFmpeg module, or fall back"
    echo "        to the cv2.VideoWriter(mp4v) muxer in"
    echo "        pythongpu/pipeline/animate_coupling.py."
    exit 1
fi

# ── Validate that the array stage produced a complete frame sequence ────────
FRAME_COUNT=$(find "$FRAME_GALLERY" -maxdepth 1 -name 'frame_*.png' | wc -l)
if [ "$FRAME_COUNT" -eq 0 ]; then
    echo "[fatal] no frames found in ${FRAME_GALLERY}; did sweep_array.slurm run?"
    exit 1
fi
echo "Compiling ${FRAME_COUNT} basin frames -> ${OUTPUT_MP4} at ${FRAME_RATE} fps"

# ── Mux the zero-padded frame sequence into an MP4 ──────────────────────────
# -framerate governs the input timing so each basin frame occupies exactly
# 1/FRAME_RATE seconds; yuv420p and the even-dimension pad maximise
# compatibility with standard players (H.264 requires even width/height).
ffmpeg -y \
    -framerate "$FRAME_RATE" \
    -i "${FRAME_GALLERY}/frame_%04d.png" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    -r "$FRAME_RATE" \
    "$OUTPUT_MP4"

echo "Animation compilation complete: ${OUTPUT_MP4}"
