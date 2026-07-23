#!/bin/bash
# ── SBATCH directives MUST sit above `set -euo pipefail`; if they land below a
#    shell statement Slurm stops parsing them and the job silently inherits the
#    dev-partition 15-min walltime cap (see clv-sbatch-directive-trap).
#SBATCH --job-name=true_vps_prod_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# No --qos: no named QOS on this account (see submit_perturbation_sweep.sh).
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=logs/true_vps_prod_gpu_%A_%a.out
#
# GPU version of submit_true_vps_production.sh. The CPU run (job 4556148,
# ACRES general partition) took ~5h/task at chunk_size=64, pair_chunk_size=400.
# Measured on this project's own dev GPU (a 12GB Titan Xp -- old, no tensor
# cores) at the SAME production T=10000:
#
#   Bc=64  pair_chunk=100   13.1s/chunk   peak 5.3GB
#   Bc=64  pair_chunk=200   12.8s/chunk   peak 9.9GB
#   Bc=128 pair_chunk=100   16.9s/chunk   peak 10.5GB  <- best throughput found
#   Bc=256+  any pair_chunk  OOM on a 12GB card
#
# Bc=128/pair_chunk=100 gives 9216/128=72 chunks * 16.9s =~ 20 min/task --
# roughly 15x faster than the CPU run, on a GPU that is NOT particularly
# fast by current standards.
#
# IMPORTANT CAVEAT (this is why the defaults below are conservative, not the
# most aggressive config found): ACRES's gpu partition (`sinfo -p gpu`, one
# node, gpu-21-28, gres=gpu:Tesla:4, 2026-07-22) was not directly reachable
# to query its GPU model/memory before this script was written -- the actual
# Tesla card here is UNKNOWN and could have more or less memory than the
# 12GB dev card these numbers came from. Also: GPU memory for this workload
# is NOT a simple rescale of the CPU numbers -- cuFFT's internal plan/work
# buffers add real overhead beyond raw tensor bytes, which is why
# pair_chunk_size had to be ~4x SMALLER on a 12GB GPU than the 400 that fit
# comfortably in system RAM on CPU. Starting values here (64/100) are the
# safely-verified-on-12GB config, not the fastest one (128/100) -- bump to
# 128/100 only after confirming this partition's actual GPU has >=12GB free,
# e.g. via a squeue/nvidia-smi check once a job lands on gpu-21-28.
#
# run_sweep_true_vps's two-level auto-halving (pair_chunk_size first --
# cheap; chunk_size second -- expensive, re-integrates) is still live as a
# safety net regardless of the starting guess.
#
# Same K ladder / node pair / grid_n / tmax as submit_true_vps_production.sh,
# so results are directly comparable regardless of which partition ran them.

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
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

COUPLINGS=(0.05 0.10 0.15 0.20)
K="${COUPLINGS[${SLURM_ARRAY_TASK_ID}]}"

NODE_X=${NODE_X:-73}
NODE_Y=${NODE_Y:-81}
GRID_N=${GRID_N:-96}
CHUNK=${CHUNK:-64}
PAIR_CHUNK=${PAIR_CHUNK:-100}
ALIGNMENT=${ALIGNMENT:-corrected}

OUTDIR="data/derivatives/true_vps_gpu_c${K//./_}"
mkdir -p "$OUTDIR"

echo "[task ${SLURM_ARRAY_TASK_ID}] true VPS (GPU)  K=${K}  grid_n=${GRID_N}  " \
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
