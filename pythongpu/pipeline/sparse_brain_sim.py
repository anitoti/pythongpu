#!/usr/bin/env python3
"""
Memory-efficient sparse brain simulation pipeline.
Fixes diagonal collisions in Laplacian and optimizes array allocations.
"""

# run low density tests on local titan xp until moved to supercomputer
# the laplacian density is customizable via the `density` parameter,
# which controls the sparsity of the resulting Laplacian matrix.

from __future__ import annotations
import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple
import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

try:
    import nibabel as nib
except Exception:
    nib = None

@dataclass
class SeriesSource:
    kind: str
    path: str

def _safe_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.sqrt(np.maximum(np.sum(x * x, axis=1, keepdims=True), eps))

def load_voxel_timeseries(source: SeriesSource, tmax: Optional[int] = None) -> np.ndarray:
    if source.kind == "nifti":
        if nib is None:
            raise RuntimeError("nibabel is required for NIfTI loading")
        img = nib.load(source.path)
        dataobj = img.dataobj
        shape = img.shape
        if len(shape) != 4:
            raise ValueError(f"Expected 4D NIfTI, got shape={shape}")
        t = shape[3] if tmax is None else min(shape[3], tmax)
        flat = np.zeros((np.prod(shape[:3]), t), dtype=np.float32)
        idx = 0
        for z in range(shape[2]):
            slab = np.asarray(dataobj[:, :, z, :t], dtype=np.float32)
            nvox = slab.shape[0] * slab.shape[1]
            flat[idx:idx + nvox] = slab.reshape(nvox, t)
            idx += nvox
        return flat

    if source.kind in {"npy", "npz"}:
        arr = np.load(source.path, mmap_mode="r")
        if isinstance(arr, np.lib.npyio.NpzFile):
            key = arr.files[0]
            x = arr[key]
        else:
            x = arr
        if x.ndim != 2:
            raise ValueError("Expected 2D array [voxels, time]")
        return np.asarray(x[:, :tmax] if tmax is not None else x, dtype=np.float32)

    if source.kind == "csv":
        rows = []
        with open(source.path, newline="") as f:
            for row in csv.reader(f):
                if row:
                    rows.append([float(v) for v in row])
        x = np.asarray(rows, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("Expected 2D CSV")
        return x[:, :tmax] if tmax is not None else x

    raise ValueError(f"Unsupported source kind: {source.kind}")

def corr_chunked(X: np.ndarray, target_idx: np.ndarray, chunk_size: int = 2048) -> Iterator[Tuple[int, np.ndarray]]:
    X = X.astype(np.float32, copy=False)
    X = X - X.mean(axis=1, keepdims=True)
    Xn = _safe_norm(X)
    Xt = X[target_idx]
    Xt = Xt - Xt.mean(axis=1, keepdims=True)
    Xtn = _safe_norm(Xt)
    Xt = Xt / Xtn
    n = X.shape[0]
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        A = X[start:end] / Xn[start:end]
        corr = A @ Xt.T
        yield start, corr

def top_proportional_edges(
    corr_iter: Iterable[Tuple[int, np.ndarray]], 
    n_nodes: int, 
    target_idx: np.ndarray, 
    proportion: float = 0.01, 
    min_abs_corr: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, cols, vals = [], [], []
    # Safeguard k bounds
    k = max(1, min(len(target_idx), int(math.ceil(proportion * len(target_idx)))))
    
    for start, corr in corr_iter:
        abs_corr = np.abs(corr)
        abs_corr[abs_corr < min_abs_corr] = 0.0
        
        topk_idx = np.argpartition(abs_corr, -k, axis=1)[:, -k:]
        topk_vals = np.take_along_axis(corr, topk_idx, axis=1)
        
        rr = np.repeat(np.arange(start, start + corr.shape[0]), k)
        cc = target_idx[topk_idx.reshape(-1)]
        vv = topk_vals.reshape(-1)
        
        mask = np.isfinite(vv) & (np.abs(vv) > 0.0)
        rows.append(rr[mask])
        cols.append(cc[mask])
        vals.append(vv[mask])

    if rows:
        row = torch.from_numpy(np.concatenate(rows).astype(np.int64))
        col = torch.from_numpy(np.concatenate(cols).astype(np.int64))
        val = torch.from_numpy(np.concatenate(vals).astype(np.float32))
    else:
        row = torch.empty(0, dtype=torch.int64)
        col = torch.empty(0, dtype=torch.int64)
        val = torch.empty(0, dtype=torch.float32)
        
    return row, col, val

def build_sparse_adjacency(n_nodes: int, row: torch.Tensor, col: torch.Tensor, val: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    idx = torch.stack([row, col], dim=0)
    A = torch.sparse_coo_tensor(idx, val, (n_nodes, n_nodes), device=device).coalesce()
    return A

def sparse_laplacian(A: torch.Tensor) -> torch.Tensor:
    """
    Computes L = D - A safely.
    Handles self-loops explicitly to avoid duplicate-coordinate summation during coalesce.
    """
    A = A.coalesce()
    idx = A.indices()
    val = A.values()
    n = A.size(0)
    
    # Calculate degree values based on absolute weights to ensure mathematical stability
    deg_vals = torch.abs(val)
    deg = torch.zeros(n, device=A.device, dtype=val.dtype)
    deg.scatter_add_(0, idx[0], deg_vals)
    
    # Separate self-loops from off-diagonal elements
    is_not_self_loop = idx[0] != idx[1]
    
    off_idx = idx[:, is_not_self_loop]
    off_val = -val[is_not_self_loop]
    
    # Generate true unique diagonal indices
    d_idx = torch.arange(n, device=A.device)
    diag_idx = torch.stack([d_idx, d_idx], dim=0)
    
    # Incorporate existing structural diagonal values directly if any exist
    is_self_loop = ~is_not_self_loop
    if is_self_loop.any():
        deg.scatter_add_(0, idx[0, is_self_loop], -val[is_self_loop])
        
    L_idx = torch.cat([off_idx, diag_idx], dim=1)
    L_val = torch.cat([off_val, deg])
    
    return torch.sparse_coo_tensor(L_idx, L_val, A.shape, device=A.device).coalesce()

def rossler_sim_sparse(L: torch.Tensor, steps: int = 100, dt: float = 0.01, a: float = 0.2, b: float = 0.2, c: float = 5.7, diff: float = 0.05, seed: int = 0):
    device = L.device
    torch.manual_seed(seed)
    n = L.size(0)
    x = torch.randn(n, device=device) * 0.1
    y = torch.randn(n, device=device) * 0.1
    z = torch.randn(n, device=device) * 0.1

    for _ in range(steps):
        lapx = torch.sparse.mm(L, x.unsqueeze(1)).squeeze(1)
        dx = -(y + z) - diff * lapx  # Standard Laplacian diffusion uses negative sign
        dy = x + a * y
        dz = b + z * (x - c)
        x = x + dt * dx
        y = y + dt * dy
        z = z + dt * dz
        
        # Clip numerical divergence explosions
        x = torch.clamp(x, min=-50.0, max=50.0)
    return x, y, z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nifti", type=Path, default=DATA_DIR / "rfMRI_REST1_LR_hp2000_clean.nii.gz")
    ap.add_argument("--timeseries", type=Path, default=None)
    ap.add_argument("--kind", default="nifti", choices=["nifti", "npy", "npz", "csv"])
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--proportion", type=float, default=0.01)
    ap.add_argument("--min-abs-corr", type=float, default=0.0)
    ap.add_argument("--target-size", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    path = args.timeseries or args.nifti
    if not os.path.exists(path):
        # Graceful exit mockup for missing sample data pipelines
        print(f"File path template '{path}' not found. Exiting gracefully.")
        return

    source = SeriesSource(args.kind, path)
    X = load_voxel_timeseries(source, tmax=args.tmax)
    n = X.shape[0]
    
    target_idx = np.linspace(0, n - 1, num=min(args.target_size, n), dtype=np.int64)
    corr_iter = corr_chunked(X, target_idx=target_idx, chunk_size=args.chunk_size)
    
    row, col, val = top_proportional_edges(corr_iter, n, target_idx, proportion=args.proportion, min_abs_corr=args.min_abs_corr)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = build_sparse_adjacency(n, row, col, val, device=device)
    L = sparse_laplacian(A)
    x, y, z = rossler_sim_sparse(L, steps=args.steps)
    # run python3 sparse_brain_sim.py --proportion 0.05 to test low density connectivity

    print(f"nodes={n} edges={A._nnz()} device={device} x_mean={x.mean().item():.6f} y_mean={y.mean().item():.6f} z_mean={z.mean().item():.6f}")

if __name__ == "__main__":
    main()
