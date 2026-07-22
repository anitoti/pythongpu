#!/bin/bash
# ── SBATCH directives MUST sit above `set -euo pipefail`; if they land below a
#    shell statement Slurm stops parsing them and the job silently inherits the
#    dev-partition 15-min walltime cap (see clv-sbatch-directive-trap).
#SBATCH --job-name=true_vps_prod
#SBATCH --partition=general
# No --qos: no named QOS on this account, general's MaxTime is enforced by
# the partition directly (see submit_perturbation_sweep.sh).
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-3
#SBATCH --output=logs/true_vps_prod_%A_%a.out
#
# First production-scale run of the paper's ACTUAL VPS (Definition A: FFT
# cross-correlation + lag alignment) on real integrated Lorenz-DTI data,
# instead of only the static Example_A_3.mat test matrix it was validated
# against so far (talk/figs/alignment_bias.png). Uses run_sweep_true_vps
# (pythongpu/pipeline/lorenz_sweep.py), which chunks over the IC batch and
# auto-halves chunk size on OOM -- the true VPS needs the WHOLE recorded
# trajectory per pair for the lag search, so memory is O(chunk*T*N) instead
# of the O(B*C) the streaming surrogate gets away with (see
# run_sweep_true_vps's docstring and vps-code-purpose memo).
#
# Same K ladder, node pair, grid_n, and tmax as the EXISTING streaming-
# surrogate files this compares against (data/derivatives/
# lorenz_basins_n73_n81_K{0.05,0.10,0.15,0.20}.npz, used by
# plot_vps_clv_comparison.py) -- confirmed via each file's saved config:
# node_x=73, node_y=81, grid_n=96, tmax=500 (steps_record=10000, dt=0.05),
# k_clusters=2. Only --vps-method differs (true vs streaming), so any
# difference in the result is attributable to the VPS definition itself,
# not a confound from resolution or coupling.
#
# MEMORY: true VPS's FFT/lag-search buffers dominate over the raw
# trajectory -- roughly 550 MB per IC at T=10000, C(83,2)=3403 pairs
# (T * C * 8 bytes * ~2 for the rfft/irfft working buffers). At grid_n=96,
# B=9216 ICs; --true-vps-chunk-size=64 keeps one chunk's peak at
# ~64 * 550MB =~ 35GB, well under --mem=48G's 13GB margin for the RK4
# trajectory buffer (~640MB/chunk) and k-means/box-counting temporaries.
# The auto-halving in run_sweep_true_vps is still live as a fallback if
# this estimate is wrong, same safety net as the smoke test that already
# demonstrated it (chunk_size=512 -> OOM -> halved to 256 -> completed).
#
# RUNTIME: total RK4 integration work (B * steps_record node-updates) is
# identical to what the streaming-surrogate run already completed for
# these same files, so integration cost should be comparable; the FFT lag
# search adds overhead on top, chunked over ~144 chunks (9216/64) instead
# of one shot. 12h budget is a margin, not a measured number -- this is the
# FIRST production run of the true VPS, unlike the benchmark-grounded
# --time in submit_vps_norm_comparison.sh.

set -euo pipefail

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

# Same four couplings as the existing surrogate/CLV comparison
# (data/derivatives/vps_surrogate_vs_clv_comparison.png), so this drops
# straight into a true-VPS-vs-surrogate-vs-CLV three-way comparison once done.
COUPLINGS=(0.05 0.10 0.15 0.20)
K="${COUPLINGS[${SLURM_ARRAY_TASK_ID}]}"

NODE_X=${NODE_X:-73}
NODE_Y=${NODE_Y:-81}
GRID_N=${GRID_N:-96}
CHUNK=${CHUNK:-64}
ALIGNMENT=${ALIGNMENT:-corrected}

OUTDIR="data/derivatives/true_vps_c${K//./_}"
mkdir -p "$OUTDIR"

echo "[task ${SLURM_ARRAY_TASK_ID}] true VPS  K=${K}  grid_n=${GRID_N}  " \
     "nodes=${NODE_X},${NODE_Y}  chunk=${CHUNK}  alignment=${ALIGNMENT} -> ${OUTDIR}"

python3 -m pythongpu.pipeline.lorenz_fine_coupling_sweep \
    --dti-path data/DTI-og.mat \
    --node-x "$NODE_X" --node-y "$NODE_Y" \
    --k-start "$K" --k-stop "$K" --k-step 0.025 \
    --grid-n "$GRID_N" --k-clusters 2 --kmeans-seed 42 \
    --vps-method true --true-vps-alignment "$ALIGNMENT" \
    --true-vps-chunk-size "$CHUNK" \
    --outdir "$OUTDIR"
