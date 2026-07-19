"""
Shared network-loading utilities deduped from the four Lorenz/Rössler
VPS-clustering pipeline scripts (byte-identical across all of them,
confirmed via AST diff before extraction).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.io import loadmat


def load_dti_laplacian(
    mat_path : str,
    device   : torch.device,
) -> tuple[torch.Tensor, int]:
    """
    Load DTI-og.mat (professor original), build the weighted graph Laplacian, return (L, n).

    Replicates the MATLAB preamble exactly:
        load('DTI-og.mat')
        A = double(A)
        n = size(A, 2)
        L = diag(sum(A, 2)) - A
        gel   = 0.5
        H     = [0 0 0; 0 1 0; 0 0 0]
        gelLH = gel * kron(L, H)
    [Page 28, full_.m_script.pdf]

    kron(L, H) is NOT formed explicitly — the H selection (coupling only
    through X) is applied directly in lorenz_rhs_batched by subtracting
    the Laplacian term from dX only.

    Processing steps:
        1. loadmat       — reads the raw .mat struct
        2. float64 cast  — MATLAB "A = double(A)"
        3. symmetrise    — 0.5*(A + A^T), guards against tractography asymmetry
        4. zero diagonal — removes self-loops (spurious Laplacian diagonal bias)
        5. L = D - A     — standard graph Laplacian
    [Page 28, full_.m_script.pdf: "L = diag(sum(A,2)) - A"]

    Args
    ----
    mat_path : path to DTI-og.mat  (variable named 'A')  # original professor-provided DTI matrix
    device   : torch.device

    Returns
    -------
    L_gpu : (n, n) float32 Laplacian tensor on device
    n     : number of nodes  (= size(A,2) in MATLAB)
    """
    mat = loadmat(mat_path)

    # [Page 28, full_.m_script.pdf: "load('DTI-og.mat') A = double(A)"]
    A = mat["A"].astype(np.float64)
    n = A.shape[1]                      # "n = size(A,2)"
    print(f"[dti]      loaded {mat_path}  raw shape={A.shape}  n={n}")

    # Symmetrise — DTI fibre counts can have tiny float asymmetries
    A = 0.5 * (A + A.T)

    # Zero diagonal — self-loops have no meaning in diffusive coupling
    np.fill_diagonal(A, 0.0)

    # [Page 28, full_.m_script.pdf: "L = diag(sum(A,2)) - A"]
    D = np.diag(A.sum(axis=1))
    L = (D - A).astype(np.float32)

    edge_count = int((A > 0).sum() // 2)
    print(f"[dti]      n={n} nodes  {edge_count} edges  "
          f"max_weight={A.max():.4f}")

    return torch.tensor(L, dtype=torch.float32, device=device), n


def rewire_edges(A: np.ndarray, num_edges: int) -> np.ndarray:
    """
    Randomly rewire num_edges edges of adjacency matrix A.
    [Page 41, full_.m_script.pdf:
     "R1 = randperm(n); R1 = R1(1); F = find(A(R1,:));
      A(R1,R2) = 0; A(R2,R1) = 0;
      A(R1,R3) = 1; A(R3,R1) = 1;"]
    """
    A = A.copy()
    n = A.shape[0]
    for _ in range(num_edges):
        R1 = np.random.permutation(n)[0]
        F  = np.where(A[R1, :] > 0)[0]
        if len(F) == 0:
            continue
        R2 = F[np.random.permutation(len(F))[0]]
        A[R1, R2] = 0
        A[R2, R1] = 0
        F_new   = np.where(A[R1, :] > 0)[0]
        setdiff = np.setdiff1d(np.setdiff1d(np.arange(n), F_new), [R1])
        if len(setdiff) == 0:
            A[R1, R2] = 1
            A[R2, R1] = 1
            continue
        R3        = setdiff[np.random.permutation(len(setdiff))[0]]
        A[R1, R3] = 1
        A[R3, R1] = 1
    return A
