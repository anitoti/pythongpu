#!/bin/bash
#SBATCH --job-name=vdp_coupling_sweep
#SBATCH --partition=gpu
#SBATCH --output=vdp_sweep_%j.out
#SBATCH --error=vdp_sweep_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# NOTE: --partition=gpu matches the partition name already used by
# pipeline/slurm_submit.sh on this cluster. If the acres GPU partition
# has a different name (e.g. acres-gpu), update this line accordingly —
# no acres-specific partition name was available to confirm at write time.

# Go to the directory where sbatch was executed
cd $SLURM_SUBMIT_DIR

# Load cluster modules
module load Python/3.10.4-GCCcore-11.3.0

# Activate environment using a local relative path
source fmri_env/bin/activate

# Defragment GPU memory segments dynamically
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Required for the `from pythongpu...` absolute-import convention —
# without this, python3 only puts pipeline/ (the script's own dir) on
# sys.path, not the repo root.
export PYTHONPATH=$SLURM_SUBMIT_DIR

echo "Launching Van der Pol coupling sweep on subject 01 structural connectome..."
python3 -m pythongpu.pipeline.vdp_sweep \
    --dti-path data/DTI-og.mat \
    --mu 1.5 \
    --coupling-min 0.0 \
    --coupling-max 1.0 \
    --coupling-steps 21 \
    --outdir data/

echo "Van der Pol Coupling Sweep Complete!"
