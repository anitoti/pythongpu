import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot an adjacency matrix subset.")
    ap.add_argument("--matrix", type=Path, default=DATA_DIR / "adjacency_matrix.npy", help="Input adjacency matrix file.")
    ap.add_argument("--output", type=Path, default=DATA_DIR / "brain_connectivity.png", help="Output figure file.")
    ap.add_argument("--subset-size", type=int, default=500, help="Size of the square subset to plot.")
    args = ap.parse_args()

    matrix = np.load(str(args.matrix), mmap_mode='r')
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D adjacency matrix")

    subset = matrix[: args.subset_size, : args.subset_size]

    plt.figure(figsize=(10, 8))
    plt.imshow(subset, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Functional Connectivity Correlation')
    plt.title(f'Brain Connectivity Adjacency Matrix Subset ({args.subset_size}x{args.subset_size})')
    plt.savefig(str(args.output), dpi=300)
    print(f"saved plot as {args.output}!")


if __name__ == "__main__":
    main()
