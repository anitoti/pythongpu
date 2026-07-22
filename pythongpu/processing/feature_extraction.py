#!/usr/bin/env python3
from __future__ import annotations

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

    L = torch.zeros(indices.shape[1], device=device, dtype=x.dtype)
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


def vector_pattern_state_fast(x: torch.Tensor, alpha: float = 1.0,
                              alignment: str = "corrected") -> torch.Tensor:
    """
    Vectorised VPS: same mathematics as vector_pattern_state, no lag loop.

    WHY: the original computes L with `for lag_val in range(-T+1, T)` -- 2T-1
    Python iterations (19,999 at the production T=10,000). That loop dominates
    runtime and throws away everything the batched FFT buys: measured against a
    serial CPU reference at N=83 (3403 pairs), the looped GPU version reaches only
    3.0x (T=256) to 6.5x (T=2048), and is actually SLOWER than serial at T=128.
    Replacing the loop with a single gather gives 233x-375x over the same
    reference -- a further ~57x over the looped version -- and matches it to
    float32 precision (max |dL| = 1.9e-06).

    HOW: every pair has its own lag, so instead of looping over candidate lags we
    build per-pair aligned index grids and gather once.

    alignment:
      'corrected' -- shift by |tau|. What the measured lag actually means.
      'matlab'    -- shift by |tau|-1, reproducing VectorPatternState.m.

    The two differ because the MATLAB indexes with the PHYSICAL lag from xcorr's
    `lagsx` as though it were a 1-based index: `x(lagsx(f):end)` starts at sample
    lagsx(f), i.e. a shift of lag-1. At lag=1 it applies no shift at all. The
    under-shift mis-aligns the pair, which inflates L on every pair (measured:
    L_matlab >= L_corrected on 36/36 pairs of Example_A_3, mean ratio 1.96).

    CAVEAT worth carrying: L is norm() over T-|tau| samples and is NOT length-
    normalised, so it shrinks mechanically as |tau| grows -- L is confounded with
    tau, and both are fed to k-means as if independent.
    """
    if alignment not in ("corrected", "matlab"):
        raise ValueError(f"alignment must be 'corrected' or 'matlab', got {alignment!r}")
    T, n = x.shape
    if n < 2:
        raise ValueError("Expected at least two oscillators")
    idx = torch.triu_indices(n, n, offset=1, device=x.device)
    i_p, j_p = idx[0], idx[1]

    pad = 2 * T - 1
    xf = torch.fft.rfft(x, n=pad, dim=0)
    corr = torch.fft.irfft(xf[:, i_p] * torch.conj(xf[:, j_p]), n=pad, dim=0)
    li = torch.argmax(corr, dim=0)
    tau = torch.where(li >= T, li - pad, li)

    shift = tau.abs()
    if alignment == "matlab":
        shift = torch.clamp(shift - 1, min=0)

    # Residual x_i(t + tau) - x_j(t): node i carries the lag for tau > 0, node j
    # for tau < 0 (equivalently x_i(t) - x_j(t + |tau|)). Each node is shifted by
    # its OWN lag-dependent amount; there is no a/b swap. (A prior version applied
    # both the shift and an index swap, which left node i shifted in both signs
    # and inflated L for every tau < 0 pair.)
    shift_i = torch.where(tau > 0, shift, torch.zeros_like(shift))[None, :]
    shift_j = torch.where(tau < 0, shift, torch.zeros_like(shift))[None, :]
    t = torch.arange(T, device=x.device)[:, None]
    ti, tj = t + shift_i, t + shift_j
    valid = (ti < T) & (tj < T)

    xi = x[ti.clamp(max=T - 1), i_p[None, :].expand(T, -1)]
    xj = x[tj.clamp(max=T - 1), j_p[None, :].expand(T, -1)]
    L = torch.linalg.norm((xi - xj) * valid, dim=0)
    return torch.cat([tau.to(L.dtype), L * alpha])


def vector_pattern_state_batched(x: torch.Tensor, alpha: float = 1.0,
                                 alignment: str = "corrected",
                                 pair_chunk_size: int | None = None) -> torch.Tensor:
    """
    Same statistic as vector_pattern_state_fast, with a leading batch dim B
    (one initial condition per row) so the true (lag-based) VPS can run on a
    whole grid sweep instead of only the paper's static test matrix.

    x : (B, T, n)  -- B initial conditions, T timesteps, n oscillators
    returns : (B, 2*C(n,2))  -- [tau_1..tau_C, L_1..L_C] per row, same layout
    as run_sweep_streaming's vps output so it drops into the same downstream
    k-means/box-counting code.

    pair_chunk_size: process this many of the C(n,2) node pairs at a time
    instead of all of them at once. NEEDED because six separate (B, T, C)
    tensors are alive simultaneously per pair batch (corr, ti, tj, valid,
    xi, xj) -- a first production run at B=64, T=10000, C=3403 with no pair
    chunking (i.e. pair_chunk_size=C) was OOM-killed under a 48GB cgroup
    limit: the true peak is ~6x a single (B,T,C) tensor's size, not 1x or
    2x, an estimate this project got wrong once already (see the
    IC-batch-only chunking in run_sweep_true_vps, which alone was NOT
    enough). `x`'s own FFT (xf) is computed once, outside the pair loop --
    it's tiny (B, T, n), not (B, T, C) -- and reused across pair chunks.
    Default: all C pairs at once (matches the pre-chunking behaviour);
    callers doing a full production sweep should pass an explicit,
    memory-budgeted value (see run_sweep_true_vps).
    """
    if alignment not in ("corrected", "matlab"):
        raise ValueError(f"alignment must be 'corrected' or 'matlab', got {alignment!r}")
    B, T, n = x.shape
    if n < 2:
        raise ValueError("Expected at least two oscillators")
    idx = torch.triu_indices(n, n, offset=1, device=x.device)
    i_p, j_p = idx[0], idx[1]
    C = i_p.shape[0]
    pc = pair_chunk_size or C

    pad = 2 * T - 1
    xf = torch.fft.rfft(x, n=pad, dim=1)                                    # (B, F, n) -- shared

    t = torch.arange(T, device=x.device)[None, :, None]                    # (1, T, 1)
    tau_parts, L_parts = [], []
    for s in range(0, C, pc):
        e = min(s + pc, C)
        ip, jp = i_p[s:e], j_p[s:e]

        corr = torch.fft.irfft(xf[..., ip] * torch.conj(xf[..., jp]), n=pad, dim=1)  # (B, pad, c)
        li = torch.argmax(corr, dim=1)                                      # (B, c)
        tau = torch.where(li >= T, li - pad, li)
        del corr, li

        shift = tau.abs()
        if alignment == "matlab":
            shift = torch.clamp(shift - 1, min=0)

        shift_i = torch.where(tau > 0, shift, torch.zeros_like(shift))      # (B, c)
        shift_j = torch.where(tau < 0, shift, torch.zeros_like(shift))
        ti = (t + shift_i[:, None, :]).clamp(max=T - 1)                    # (B, T, c)
        tj = (t + shift_j[:, None, :]).clamp(max=T - 1)
        valid = (t + shift_i[:, None, :] < T) & (t + shift_j[:, None, :] < T)

        xi = torch.gather(x[..., ip], dim=1, index=ti)                     # (B, T, c)
        xj = torch.gather(x[..., jp], dim=1, index=tj)
        L = torch.linalg.norm((xi - xj) * valid, dim=1)                    # (B, c)
        del shift, shift_i, shift_j, ti, tj, valid, xi, xj

        tau_parts.append(tau.to(L.dtype))
        L_parts.append(L * alpha)

    tau_all = torch.cat(tau_parts, dim=-1)
    L_all = torch.cat(L_parts, dim=-1)
    return torch.cat([tau_all, L_all], dim=-1)                             # (B, 2C)


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
