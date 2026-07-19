#!/bin/bash
# ============================================================================
#  bootstrap_venv.sh — provision the ACRES virtualenv this repo's jobs expect
#  Project : Nimble Brain — Clarkson University REU (Dr. Jeremie Fish)
#
#  Purpose
#  -------
#  sweep_array.slurm and diagnose.slurm both activate ${VENV_PATH:-fmri_env}
#  if it exists and silently fall back to the module Python otherwise. That
#  fallback cannot work for the sweeps: every oscillator imports torch, and
#  ACRES has no PyTorch module. This script builds the venv that makes the
#  fallback unnecessary.
#
#  The module stack alone is not an option, and not only because of torch.
#  The newest SciPy-bundle is foss-2022a while scikit-learn (0.24.2) and
#  networkx (2.5.1) are built only for foss-2021a; and even a consistent
#  foss-2021a set would put sklearn a major version behind the 1.3.2 this
#  code is developed against, which is precisely where the clustering
#  behaviour has moved. Pinning to the dev box (requirements-acres.txt) keeps
#  the two environments honest.
#
#  Run on the HEAD NODE (it needs outbound network for pip; compute nodes may
#  not have it). It is idempotent: an existing, healthy venv is verified and
#  left alone unless --force is given.
#
#  Usage
#  -----
#    bash pythongpu/hpc/bootstrap_venv.sh              # build ./fmri_env
#    bash pythongpu/hpc/bootstrap_venv.sh --force      # rebuild from scratch
#    VENV_PATH=/mnt/data/$USER/fmri_env \
#        bash pythongpu/hpc/bootstrap_venv.sh          # build elsewhere
#    TORCH_VARIANT=cpu bash pythongpu/hpc/bootstrap_venv.sh
#
#  Then pass the same path to the jobs if it is not the default:
#    sbatch --export=ALL,VENV_PATH=/mnt/data/$USER/fmri_env ... sweep_array.slurm
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQUIREMENTS="${REPO_ROOT}/pythongpu/hpc/requirements-acres.txt"

: "${VENV_PATH:=${REPO_ROOT}/fmri_env}"
: "${PYTHON_MODULE:=Python/3.10.4-GCCcore-11.3.0}"

# cu118 wheels run on any CUDA 11.x driver >= 450.80.02 (CUDA minor version
# compatibility), which covers the driver behind the CUDA/11.1.1 module. If
# nvidia-smi is reachable the driver is checked below and this is downgraded
# to 'cpu' when it is too old to load the CUDA runtime.
: "${TORCH_VERSION:=2.0.1}"
: "${TORCH_VARIANT:=cu118}"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# ── Preconditions ───────────────────────────────────────────────────────────
if [ ! -f "$REQUIREMENTS" ]; then
    echo "ERROR: missing $REQUIREMENTS" >&2
    exit 1
fi

say "[1/5] Loading module environment"
# Lmod is not initialised in a non-interactive shell on every node; source the
# profile hook if 'module' is not already a function.
if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
fi
module purge 2>/dev/null || true
module load "$PYTHON_MODULE"
# matplotlib's PIL backend links against libjpeg-turbo; loading it here means a
# venv built now can plot later without a surprise ImportError mid-job.
module load libjpeg-turbo/2.1.3-GCCcore-11.3.0 2>/dev/null || \
    echo "  [warn] libjpeg-turbo module unavailable; plotting may need --no-plot"
echo "  python: $(command -v python3)  ($(python3 --version 2>&1))"

# ── Decide the torch wheel before building anything ─────────────────────────
say "[2/5] Selecting torch wheel"
if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    if [ -n "$DRIVER" ]; then
        echo "  GPU visible here: ${GPU_NAME:-unknown}  (driver ${DRIVER})"
        # CUDA 11.x needs >= 450.80.02. Compare the full version string: any
        # truncation makes every 450.x driver look older than the threshold.
        if [ "$(printf '%s\n450.80.02\n' "$DRIVER" | sort -V | head -1)" != "450.80.02" ]; then
            echo "  [warn] driver ${DRIVER} predates 450.80.02; falling back to CPU torch."
            TORCH_VARIANT=cpu
        fi
    fi
else
    echo "  No nvidia-smi here (expected on the head node) — assuming the GPU"
    echo "  node's driver supports ${TORCH_VARIANT}. Override with TORCH_VARIANT=cpu"
    echo "  if the verification step below reports a CUDA load failure on gpu-21-28."
fi
echo "  installing torch==${TORCH_VERSION} (${TORCH_VARIANT})"

# ── Create the venv ─────────────────────────────────────────────────────────
say "[3/5] Creating virtualenv at ${VENV_PATH}"
if [ -d "$VENV_PATH" ] && [ "$FORCE" -eq 1 ]; then
    echo "  --force: removing existing ${VENV_PATH}"
    rm -rf "$VENV_PATH"
fi
if [ -f "${VENV_PATH}/bin/activate" ]; then
    echo "  Existing venv found; reusing it (pass --force to rebuild)."
else
    # --system-site-packages is deliberately OFF: inheriting ~/.local packages
    # is exactly the silent-drift problem this venv exists to eliminate.
    python3 -m venv "$VENV_PATH"
fi
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
python3 -m pip install --quiet --upgrade pip setuptools wheel

# ── Install ─────────────────────────────────────────────────────────────────
say "[4/5] Installing packages (several minutes; torch is a large download)"
if [ "$TORCH_VARIANT" = "cpu" ]; then
    python3 -m pip install "torch==${TORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu
else
    python3 -m pip install "torch==${TORCH_VERSION}+${TORCH_VARIANT}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}"
fi
python3 -m pip install -r "$REQUIREMENTS"

# ── Verify ──────────────────────────────────────────────────────────────────
# Import every top-level dependency the pipeline actually uses. A venv that
# builds but cannot import is worse than no venv, because the job scripts will
# happily activate it and fail two hours into an array element.
say "[5/5] Verifying imports"
python3 - <<'PY'
import importlib, sys

expected = {
    "numpy": "1.24.4", "scipy": "1.10.1", "sklearn": "1.3.2",
    "matplotlib": "3.7.5", "networkx": "3.1", "pandas": "2.0.3",
}
failed = []
for name in ["torch", "numpy", "scipy", "sklearn", "matplotlib", "networkx",
             "pandas", "tqdm", "nilearn", "nibabel", "cv2"]:
    try:
        mod = importlib.import_module(name)
        got = getattr(mod, "__version__", "?")
        want = expected.get(name)
        flag = "  <-- EXPECTED %s" % want if want and not got.startswith(want) else ""
        print("  ok   %-12s %s%s" % (name, got, flag))
    except Exception as exc:
        print("  FAIL %-12s %s" % (name, exc))
        failed.append(name)

import torch
print("\n  torch.cuda.is_available(): %s" % torch.cuda.is_available())
if not torch.cuda.is_available():
    print("  (expected on the head node — the GPU check that matters runs on gpu-21-28)")

if failed:
    print("\nFAILED to import: %s" % ", ".join(failed))
    sys.exit(1)
PY

say "Done. Venv ready at ${VENV_PATH}"
cat <<EOF

Next:
  1. Confirm CUDA actually initialises on the GPU node (the head node cannot
     tell you this):

       srun -p gpu --gres=gpu:1 -t 5 --pty bash -c \\
         'module load ${PYTHON_MODULE}; source ${VENV_PATH}/bin/activate; \\
          python3 -c "import torch; print(torch.cuda.get_device_name(0))"'

     If that errors on the CUDA runtime, rebuild with TORCH_VARIANT=cpu and
     run the sweeps on the 'general' partition instead.

  2. Smoke-test one array element on 'dev' (15-minute cap) before committing
     to an overnight array:

       sbatch --partition=dev --array=0-0 --time=00:10:00 \\
              --export=ALL,GRID_N=41,VENV_PATH=${VENV_PATH} \\
              pythongpu/hpc/sweep_array.slurm lorenz

EOF
