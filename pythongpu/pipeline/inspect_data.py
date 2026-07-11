import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a NIfTI file and print its basic shape.")
    ap.add_argument("--input", type=Path, default=DATA_DIR / "rfMRI_REST1_LR_hp2000_clean.nii.gz", help="Path to the input NIfTI file.")
    args = ap.parse_args()

    img = nib.load(str(args.input))
    data = img.get_fdata()
    print("full data shape:", data.shape)

    flat_data = data.reshape(-1, data.shape[-1])
    subset_flat = flat_data[:100, :]

    adj_matrix = np.corrcoef(subset_flat)
    print("safe adjacency matrix shape:", adj_matrix.shape)


if __name__ == "__main__":
    main()
