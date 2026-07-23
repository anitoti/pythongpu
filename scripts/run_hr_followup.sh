#!/bin/bash
# Sequential (not concurrent -- single local GPU) follow-up to tonight's HR
# streaming ladder: waits for that job's PID to exit, then runs
#   (1) HR CLV/Kaplan-Yorke sweep, same 9-point coupling ladder as the basin
#       sweep, matching this project's own production CLV convention
#       (scripts/run_clv_sweep.sh: --steps 1000 --m 83 --K 83)
#   (2) a resolution-matched true-VPS-vs-streaming-surrogate comparison for
#       HR at grid_n=48 (smaller than the 96 used for the main basin ladder,
#       to keep true-VPS runtime bounded -- see hr_sweep.py's steps_record
#       being 2x Lorenz's, measured at 32.8s/chunk locally) across 4 of the
#       9 couplings, both methods rerun fresh at the SAME grid so the
#       comparison isn't confounded by resolution.
#
# Usage: nohup scripts/run_hr_followup.sh <streaming_pid> > /tmp/hr_followup.log 2>&1 &

set -euo pipefail
WAIT_PID="${1:?usage: run_hr_followup.sh <pid-to-wait-for>}"

echo "[followup] waiting for pid ${WAIT_PID} (HR streaming ladder) to finish..."
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 10
done
echo "[followup] pid ${WAIT_PID} finished at $(date -u). Starting CLV sweep."

# ── Phase 1: HR CLV/Kaplan-Yorke sweep ──────────────────────────────────
mkdir -p output/hr_clv_results
for c in 0.45 0.475 0.50 0.525 0.55 0.575 0.60 0.625 0.65; do
  echo "[followup] HR CLV coupling=${c} at $(date -u)"
  python3 -m pythongpu.pipeline.hr_clv_cli \
    --mat data/DTI-og.mat --outdir "output/hr_clv_results/clv_c${c//./_}" \
    --coupling "$c" --steps 1000 --m 83 --K 83 --qr-interval 10 || {
      echo "[followup] HR CLV FAILED for coupling=${c}, continuing"; continue; }
  echo "[followup] HR CLV done coupling=${c}"
done

# ── Phase 2: resolution-matched true-VPS-vs-surrogate for HR ───────────
echo "[followup] Starting matched true-vs-surrogate HR comparison at grid_n=48"
mkdir -p data/derivatives/hr_matched48
for c in 0.45 0.50 0.55 0.60; do
  echo "[followup] HR streaming (grid_n=48) K=${c} at $(date -u)"
  python3 -m pythongpu.pipeline.hr_fine_coupling_sweep \
    --node-x 73 --node-y 81 --k-start "$c" --k-stop "$c" --k-step 0.025 \
    --grid-n 48 --k-clusters 2 --kmeans-seed 42 \
    --outdir data/derivatives/hr_matched48 || {
      echo "[followup] HR streaming FAILED for K=${c}"; continue; }

  echo "[followup] HR true-VPS (grid_n=48) K=${c} at $(date -u)"
  python3 -m pythongpu.pipeline.hr_fine_coupling_sweep \
    --node-x 73 --node-y 81 --k-start "$c" --k-stop "$c" --k-step 0.025 \
    --grid-n 48 --k-clusters 2 --kmeans-seed 42 \
    --vps-method true --true-vps-chunk-size 64 --true-vps-pair-chunk-size 100 \
    --outdir "data/derivatives/hr_true_vps_matched48_c${c//./_}" || {
      echo "[followup] HR true-VPS FAILED for K=${c}"; continue; }
  echo "[followup] HR matched pair done K=${c}"
done

echo "[followup] ALL DONE at $(date -u)"
