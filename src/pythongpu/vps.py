#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import scipy.io
import torch
from sklearn.cluster import KMeans

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def vector_pattern_state(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    device = x.device
    T_len, n = x.shape
    if n < 2:
        raise ValueError("Expected at least two oscillators")

    indices = torch.triu_indices(n, n, offset=1, device=device)
    pad_len = 2 * T_len - 1

    x_fft = torch.fft.rfft(x, n=pad_len, dim=0)
    pair_1 = x_fft[:, indices[0]]
    pair_2 = x_fft[:, indices[1]]
    corr_fft = pair_1 * torch.conj(pair_2)
    corr = torch.fft.irfft(corr_fft, n=pad_len, dim=0)

    lags_indices = torch.argmax(corr, dim=0)
    tau_x = torch.where(lags_indices >= T_len, lags_indices - pad_len, lags_indices)

    L = torch.zeros(indices.shape[1], device=device)
    for lag_val in range(-T_len + 1, T_len):
        mask = tau_x == lag_val
        if not mask.any():
            continue

        p1 = indices[0, mask]
        p2 = indices[1, mask]
        if lag_val > 0:
            diff = x[lag_val:, p1] - x[:-lag_val, p2]
        elif lag_val < 0:
            abs_lag = abs(lag_val)
            diff = x[:-abs_lag, p1] - x[abs_lag:, p2]
        else:
            diff = x[:, p1] - x[:, p2]

        L[mask] = torch.linalg.norm(diff, dim=0)

    return torch.cat([tau_x, L * alpha], dim=0)


def KmeansBIC(ClusterNums, SumD, N, d):
    _, counts = np.unique(ClusterNums, return_counts=True)
    C = counts
    K = len(C)
    Sig_sqrd = np.sum(SumD) / (N - K)
    log_likli = np.sum(C * np.log(C)) - N * np.log(N) - (N * d / 2) * np.log(2 * np.pi * Sig_sqrd) - (d / 2) * (N - K)
    return log_likli - ((K + K * d) / 2) * np.log(N)


def run_kmeans(x: np.ndarray, max_k: int = 5):
    bic_values = []
    sum_distances = []
    N, d = x.shape

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(x)
        bic_values.append(KmeansBIC(kmeans.labels_, kmeans.inertia_, N, d))
        sum_distances.append(kmeans.inertia_)

    optimal_k = int(np.argmax(bic_values) + 1)
    return optimal_k, bic_values, sum_distances


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute vector pattern state from .mat oscillator data.")
    ap.add_argument("--mat-file", type=Path, default=DATA_DIR / "Example_A_3.mat", help="Path to the MATLAB .mat file containing the data.")
    ap.add_argument("--variable", type=str, default="A", help="Name of the variable inside the .mat file.")
    ap.add_argument("--alpha", type=float, default=1.5, help="Scaling factor applied to the norm component.")
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device to run on.")
    ap.add_argument("--run-kmeans", action="store_true", help="Run k-means clustering and BIC analysis on the loaded data.")
    ap.add_argument("--kmax", type=int, default=5, help="Maximum number of clusters to test for BIC.")
    args = ap.parse_args()

    mat_path = args.mat_file
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    x_dict = scipy.io.loadmat(str(mat_path))
    if args.variable not in x_dict:
        raise KeyError(f"Variable '{args.variable}' not found in {mat_path}")

    x = torch.tensor(x_dict[args.variable], dtype=torch.float32)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)

    print("Running vector pattern state computation on:", device)
    vps_result = vector_pattern_state(x, alpha=args.alpha)
    print("VPS result shape:", tuple(vps_result.shape))

    if args.run_kmeans:
        x_np = x.cpu().numpy()
        optimal_k, bic_values, _ = run_kmeans(x_np, max_k=args.kmax)
        print("Optimal number of clusters based on BIC:", optimal_k)
        print("BIC values:", bic_values)


if __name__ == "__main__":
    main()
