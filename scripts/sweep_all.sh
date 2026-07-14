#!/usr/bin/env bash
# ============================================================================
#  sweep_all.sh — sequential Lorenz -> Rössler -> Hindmarsh–Rose coupling sweep
#  Project : Nimble Brain — Clarkson University REU (Dr. Jeremie Fish)
#
#  Runs each oscillator's DTI-coupled basin sweep on a single GPU, one coupling
#  per invocation, writing every run to its own per-K directory
#  (<OUT_ROOT>/<osc>/K<coupling>/) so each frame's basin_data.npz and figures
#  are preserved rather than overwritten. Runs are strictly sequential, so they
#  never contend for GPU memory — no scheduler required. Start it once and walk
#  away.
#
#  A failed coupling (e.g. a fully-synchronised slice with no basin boundary,
#  which makes box-counting raise) is logged and skipped; it does NOT abort the
#  rest of the sweep.
#
#  Usage
#  -----
#    scripts/sweep_all.sh                          # all three, default windows
#    OSCILLATORS="rossler hr" scripts/sweep_all.sh # a subset
#    GRID_N=128 K_CLUSTERS=auto scripts/sweep_all.sh
#    WAIT_PID=12345 scripts/sweep_all.sh           # start only after PID exits
#
#  Env overrides
#  -------------
#    PYTHON (python3), GRID_N (361), K_CLUSTERS (4), DTI_PATH (data/DTI_A.mat),
#    OUT_ROOT (data), OSCILLATORS ("lorenz rossler hr"), WAIT_PID (none),
#    and per-oscillator windows <OSC>_KMIN / <OSC>_KMAX / <OSC>_KSTEPS.
#
#  Rössler's window is deliberately narrow (≲0.1): the X-coupled Rössler
#  destabilises above ~0.15, unlike the globally-bounded Lorenz and HR.
# ============================================================================

# NOT `set -e`: a single bad coupling must not kill the whole sweep.
set -uo pipefail

# ── repo root + import/runtime environment ──────────────────────────────────
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"

# NOTE: `expandable_segments:True` reduces GPU fragmentation OOM at large grids
# but is rejected by older PyTorch builds (crashes at CUDA init). It is left
# OFF by default so the driver runs everywhere; enable it only if your torch
# supports it:  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF || true

PYTHON="${PYTHON:-python3}"
GRID_N="${GRID_N:-361}"
K_CLUSTERS="${K_CLUSTERS:-4}"
DTI_PATH="${DTI_PATH:-data/DTI_A.mat}"
OUT_ROOT="${OUT_ROOT:-data}"
OSCILLATORS="${OSCILLATORS:-lorenz rossler hr}"

# Per-oscillator coupling windows: min max steps.
LORENZ_KMIN="${LORENZ_KMIN:-0.00}";   LORENZ_KMAX="${LORENZ_KMAX:-1.00}";   LORENZ_KSTEPS="${LORENZ_KSTEPS:-21}"
ROSSLER_KMIN="${ROSSLER_KMIN:-0.02}"; ROSSLER_KMAX="${ROSSLER_KMAX:-0.10}"; ROSSLER_KSTEPS="${ROSSLER_KSTEPS:-9}"
HR_KMIN="${HR_KMIN:-0.05}";           HR_KMAX="${HR_KMAX:-0.30}";           HR_KSTEPS="${HR_KSTEPS:-11}"

LOG_DIR="${OUT_ROOT}/sweep_logs"
mkdir -p "$LOG_DIR"

module_for() {
    case "$1" in
        lorenz)  echo "pythongpu.pipeline.lorenz_sweep"  ;;
        rossler) echo "pythongpu.pipeline.rossler_sweep" ;;
        hr)      echo "pythongpu.pipeline.hr_sweep"       ;;
        *)       echo "" ;;
    esac
}

# ── optionally hold until an in-flight run finishes ─────────────────────────
if [ -n "${WAIT_PID:-}" ]; then
    echo "[wait]  holding until PID ${WAIT_PID} exits ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[wait]  PID ${WAIT_PID} gone — starting."
fi

run_one() {
    local osc="$1" coupling="$2" module outdir log rc df
    module="$(module_for "$osc")"
    outdir="${OUT_ROOT}/${osc}/K${coupling}"
    log="${LOG_DIR}/${osc}_K${coupling}.log"
    mkdir -p "$outdir"
    echo "  -> ${osc} K=${coupling}  (log: ${log})"
    if "$PYTHON" -m "$module" \
            --grid-n     "$GRID_N" \
            --k-clusters "$K_CLUSTERS" \
            --coupling   "$coupling" \
            --dti-path   "$DTI_PATH" \
            --outdir     "$outdir" >"$log" 2>&1; then
        df="$(grep -oE 'D_f = [0-9.]+' "$log" | tail -1)"
        echo "     ok   ${df:-done}"
    else
        rc=$?
        echo "     FAIL (exit ${rc}) — see ${log}"
        echo "${osc} K=${coupling} exit=${rc}" >> "${LOG_DIR}/failures.txt"
    fi
}

# ── main: each oscillator across its coupling window, sequentially ──────────
started="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[start] ${started}  grid=${GRID_N}  k=${K_CLUSTERS}  python=${PYTHON}"
echo "[start] oscillators: ${OSCILLATORS}  ->  ${OUT_ROOT}/<osc>/K<coupling>/"

for osc in $OSCILLATORS; do
    module="$(module_for "$osc")"
    if [ -z "$module" ]; then
        echo "skip: unknown oscillator '${osc}'" >&2
        continue
    fi
    up="$(echo "$osc" | tr '[:lower:]' '[:upper:]')"
    eval "kmin=\${${up}_KMIN}; kmax=\${${up}_KMAX}; ksteps=\${${up}_KSTEPS}"
    echo "==================================================================="
    echo " ${osc}: K in [${kmin}, ${kmax}], ${ksteps} steps"
    echo "==================================================================="
    couplings="$(awk -v lo="$kmin" -v hi="$kmax" -v n="$ksteps" \
        'BEGIN { for (i = 0; i < n; i++)
                     printf "%.3f\n", (n <= 1 ? lo : lo + i * (hi - lo) / (n - 1)) }')"
    for c in $couplings; do
        run_one "$osc" "$c"
    done
done

echo "==================================================================="
echo "[done]  started ${started}  finished $(date '+%Y-%m-%d %H:%M:%S')"
if [ -s "${LOG_DIR}/failures.txt" ]; then
    echo "[done]  some runs failed — see ${LOG_DIR}/failures.txt:"
    sed 's/^/          /' "${LOG_DIR}/failures.txt"
else
    echo "[done]  all runs succeeded. Per-run logs in ${LOG_DIR}/"
fi
