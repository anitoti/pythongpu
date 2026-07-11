import argparse
import logging
import os
import time
from pathlib import Path

import nibabel as nib
import pandas as pd
from nilearn.regions import Parcellations

# =============================================================================
# PIPELINE STAGE: Parcellated Signal Extraction
# =============================================================================
# PURPOSE:
#   Reduce the full fMRI volume (~200k+ voxels) to N parcellated ROI timeseries
#   for downstream oCSE causal inference and basin stability mapping.
#
# PARCELLATION METHOD: Ward Hierarchical Clustering (Nilearn)
#   - Groups spatially contiguous voxels with similar BOLD timeseries
#   - Purely data-driven — no anatomical atlas required
#   - n_parcels is set dynamically via arguments
#
# PREPROCESSING APPLIED (inside Parcellations.fit_transform):
#   - standardize=True  : z-score each voxel timeseries (zero mean, unit var)
#   - detrend=True      : remove linear drift (scanner warmup artifact)
#   - low_pass=0.08 Hz  : remove high-freq noise (respiration ~0.3Hz, cardiac ~1Hz)
#   - high_pass=0.009 Hz: remove very slow drift (below BOLD signal band)
#   - t_r=0.72 s        : HCP 7T repetition time — MUST match your acquisition
#                         Change if using 3T data (TR typically 0.72–3.0s)
#
# OUTPUT:
#   parcellated_timeseries.csv — shape (T timepoints × N ROI columns)
#   This is the direct input to discover_network() in the causal inference step.
# =============================================================================

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    parser = argparse.ArgumentParser(description="Extract parcellated timeseries from fMRI.")
    parser.add_argument("--input", required=True, help="Path to input raw NIfTI file (.nii.gz)")
    parser.add_argument("--output", required=True, help="Path to output timeseries CSV")
    parser.add_argument("--atlas_output", default=None,
                         help="Path to save the parcel label volume (NIfTI). Required for "
                              "downstream tools like tck2connectome that need voxel labels, "
                              "not the timeseries CSV. Default: --output with .nii.gz extension.")
    parser.add_argument("--n_parcels", type=int, default=240, help="Number of parcels/ROIs (default: 240)")
    parser.add_argument("--tr", type=float, default=0.72, help="Repetition Time (TR) in seconds (default: 0.72)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of CPU cores to use. -1 uses all available.")
    args = parser.parse_args()

    # Pre-flight checks
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}\nConfirm data is staged to the correct directory.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    logging.info(f"Initializing Ward parcellation with n_parcels={args.n_parcels}, TR={args.tr}")
    
    start_time = time.time()
    
    # Expected shape: (x, y, z, T) — 4th dim is timepoints
    func_img = nib.load(args.input)
    logging.info(f"Loaded fMRI image | shape: {func_img.shape}")

    # ── Build Ward parcellator ───────────────────────────────────────────────────
    ward = Parcellations(
        method="ward",
        n_parcels=args.n_parcels,
        standardize=True,   # z-score: required for oCSE entropy estimation
        detrend=True,       # remove linear scanner drift
        low_pass=0.08,      # Hz — BOLD signal upper bound
        high_pass=0.009,    # Hz — removes very slow drift; keeps resting-state band
        t_r=args.tr,        # seconds — TR matching the scanner acquisition
        memory="nilearn_cache",
        memory_level=1,     # cache intermediate results — speeds up reruns
        n_jobs=args.n_jobs, # Multiprocessing magic for HPC speedup
        verbose=1
    )

    # ── Fit + extract ────────────────────────────────────────────────────────────
    # fit_transform:
    #   1. Applies preprocessing (detrend, bandpass, standardize)
    #   2. Fits Ward clustering on voxel timeseries
    #   3. Averages within each parcel → returns (T, N) array
    logging.info("Fitting Ward parcellation and extracting signals (this may take a moment)...")
    signals = ward.fit_transform(func_img)

    logging.info(f"Extracted signals shape: {signals.shape} (timepoints × ROIs)")

    # ── Save ─────────────────────────────────────────────────────────────────────
    df = pd.DataFrame(signals)
    # Columns = ROI 0..N-1, Rows = TR 0..T-1
    # Column names are integer indices — downstream causal script uses these as node IDs
    df.to_csv(args.output, index=False)

    # Persist the parcel label volume itself (not just the extracted signal
    # table) -- tck2connectome and any other voxel-space consumer needs the
    # NIfTI labels, which fit_transform() computes internally but never
    # wrote to disk on its own.
    atlas_output = args.atlas_output or str(Path(args.output).with_suffix("").with_suffix(".nii.gz"))
    os.makedirs(os.path.dirname(os.path.abspath(atlas_output)), exist_ok=True)
    ward.labels_img_.to_filename(atlas_output)
    logging.info(f"Atlas label volume saved to {atlas_output}")

    elapsed = (time.time() - start_time) / 60
    logging.info(f"Success! Parcellated timeseries saved to {args.output}")
    logging.info(f"Total time elapsed: {elapsed:.2f} minutes.")

if __name__ == "__main__":
    main()