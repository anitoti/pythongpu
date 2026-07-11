#!/bin/bash
# sweep_coupling.sh

# Memory configuration to prevent out-of-memory fragmentation errors
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Required for the `from pythongpu...` absolute-import convention —
# run this script from the repo root.
export PYTHONPATH=.

GRID_N=361
K_CLUSTERS=4
DTI_PATH="data/DTI_A.mat"
OUT_DIR="data/"

# Loop from 0 to 100 to scale coupling strength from 0.00 to 1.00
for i in $(seq -w 0 100); do
    COUPLING=$(echo "scale=2; $i / 100" | bc)

    echo "=========================================================="
    echo "Processing Coupling Strength: $COUPLING ($i/100)"
    echo "=========================================================="

    python -m pythongpu.pipeline.lorenz_sweep \
        --grid-n $GRID_N \
        --k-clusters $K_CLUSTERS \
        --coupling $COUPLING \
        --dti-path $DTI_PATH \
        --outdir $OUT_DIR
done

echo "All 101 frames have completed successfully!"
