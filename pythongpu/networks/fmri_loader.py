import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute a voxel correlation matrix from fMRI data.")
    ap.add_argument("--input", type=Path, default=DATA_DIR / "rfMRI_REST1_LR_hp2000_clean.nii.gz", help="Input NIfTI file.")
    ap.add_argument("--output", type=Path, default=DATA_DIR / "adjacency_matrix.npy", help="Output adjacency matrix file.")
    ap.add_argument("--chunk-size", type=int, default=5000, help="Number of rows to process per GPU chunk.")
    args = ap.parse_args()

    img = nib.load(str(args.input))
    data = img.get_fdata()
    print("data shape:", data.shape)

    flat_data = data.reshape(-1, data.shape[-1])
    variance = np.var(flat_data, axis=1)
    mask = variance > 0
    masked_data = flat_data[mask].astype(np.float32)
    N = masked_data.shape[0]

    print("masked voxels:", N)
    print("moving data to GPU...")

    X = torch.from_numpy(masked_data).cuda()
    X = X - X.mean(dim=1, keepdim=True)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-8)

    print("creating memory-mapped array on disk...")
    adj_mmap = np.memmap(str(args.output), dtype='float32', mode='w+', shape=(N, N))

    print("computing on GPU in chunks...")
    for i in range(0, N, args.chunk_size):
        end_i = min(i + args.chunk_size, N)
        X_chunk = X[i:end_i]
        corr_chunk = torch.matmul(X_chunk, X.t())
        adj_mmap[i:end_i, :] = corr_chunk.cpu().numpy()
        print(f"processed rows {i} to {end_i}")
    adj_mmap.flush()
    print("done!")


if __name__ == "__main__":
    main()

