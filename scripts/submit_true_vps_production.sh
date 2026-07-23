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
#SBATCH --mem=24G
#SBATCH --array=0-3
#SBATCH --output=logs/true_vps_prod_%A_%a.out
#
# First production-scale run of the paper's ACTUAL VPS (Definition A: FFT
# cross-correlation + lag alignment) on real integrated Lorenz-DTI data,
# instead of only the static Example_A_3.mat test matrix it was validated
# against so far (talk/figs/alignment_bias.png). Uses run_sweep_true_vps
# (pythongpu/pipeline/lorenz_sweep.py), which chunks over BOTH the IC batch
# and the node-pair axis and auto-halves either on OOM -- the true VPS
# needs the WHOLE recorded trajectory per pair for the lag search, so
# memory is O(chunk_size * T * pair_chunk_size), not O(B * C(N,2)) like
# the streaming surrogate gets away with.
#
# ATTEMPT #1 (job 4556141) FAILED: --true-vps-chunk-size=64 with NO pair
# chunking (i.e. pair_chunk_size=C=3403) OOM-killed all 4 tasks under a
# 48GB cgroup limit, on the FIRST chunk. The original ~550MB/IC estimate
# only counted 1-2 of the SIX (chunk, T, C) tensors alive at once during
# the lag search (corr, ti, tj, valid, xi, xj) -- the real number was
# several times higher. Fix: added pair_chunk_size chunking inside
# vector_pattern_state_batched (pythongpu/processing/feature_extraction.py)
# so those six tensors are bounded by pair_chunk_size, not the full C=3403.
#
# MEMORY (attempt #2, measured not estimated): one chunk at
# chunk_size=64, pair_chunk_size=400, T=10000 (production tmax=500,
# dt=0.05) peaked at 9.9GB RSS, measured directly via `/usr/bin/time -v`
# on this machine, CPU-only (torch.device("cpu")) to match ACRES's
# general partition having no GPU. --mem=24G leaves ~14GB margin over
# that measurement for the main process, k-means, and box-counting.
# run_sweep_true_vps's two-level auto-halving (pair_chunk_size first --
# cheap, reuses the already-integrated trajectory; chunk_size only if
# that's not enough -- expensive, re-integrates) is still live as a
# fallback if this real-but-single-run measurement doesn't generalize.
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
# RUNTIME: the same memory probe measured ~100s wall-clock for one 64-IC
# chunk at T=10000. At grid_n=96 (B=9216), that's 9216/64=144 chunks per
# task, ~4h per task -- all 4 array tasks run in parallel, so ~4h to
# completion, well inside the 12h budget. Grounded in one measured chunk,
# not a full production run, so treat as an estimate with real headroom,
# not a guarantee.

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
PAIR_CHUNK=${PAIR_CHUNK:-400}
ALIGNMENT=${ALIGNMENT:-corrected}

# OUTDIR includes ALIGNMENT so a "matlab"-mode run (exact reproduction of the
# paper's reference off-by-one, for matching their published numbers) never
# collides with a "corrected"-mode run (the fixed lag alignment) at the same
# coupling -- these are two deliberately different results, not retries of
# the same one.
OUTDIR="data/derivatives/true_vps_${ALIGNMENT}_c${K//./_}"
mkdir -p "$OUTDIR"

echo "[task ${SLURM_ARRAY_TASK_ID}] true VPS  K=${K}  grid_n=${GRID_N}  " \
     "nodes=${NODE_X},${NODE_Y}  chunk=${CHUNK}  pair_chunk=${PAIR_CHUNK}  " \
     "alignment=${ALIGNMENT} -> ${OUTDIR}"

python3 -m pythongpu.pipeline.lorenz_fine_coupling_sweep \
    --dti-path data/DTI-og.mat \
    --node-x "$NODE_X" --node-y "$NODE_Y" \
    --k-start "$K" --k-stop "$K" --k-step 0.025 \
    --grid-n "$GRID_N" --k-clusters 2 --kmeans-seed 42 \
    --vps-method true --true-vps-alignment "$ALIGNMENT" \
    --true-vps-chunk-size "$CHUNK" --true-vps-pair-chunk-size "$PAIR_CHUNK" \
    --outdir "$OUTDIR"
