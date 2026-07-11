#!/bin/bash
#SBATCH --job-name=lorenz_vps_highres
#SBATCH --partition=gpu
#SBATCH --output=lorenz_gpu_%j.out
#SBATCH --error=lorenz_gpu_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Go to the directory where sbatch was executed
cd $SLURM_SUBMIT_DIR

# Load cluster modules
module load Python/3.10.4-GCCcore-11.3.0

# Activate environment using a local relative path
source fmri_env/bin/activate

# 🌟 CRITICAL FIX: Defragment GPU memory segments dynamically
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Required for the `from pythongpu...` absolute-import convention —
# without this, python3 only puts pipeline/ (the script's own dir) on
# sys.path, not the repo root.
export PYTHONPATH=$SLURM_SUBMIT_DIR

echo "Launching High-Resolution Lorenz VPS Sweep with Defrag..."
python3 -m pythongpu.pipeline.lorenz_sweep \
    --grid-n 361 \
    --k-clusters 4 \
    --coupling 0.5 \
    --dti-path data/DTI_A.mat \
    --outdir data/

echo "Lorenz Simulation Run Complete!"
