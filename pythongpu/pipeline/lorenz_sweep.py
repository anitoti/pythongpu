#!/usr/bin/env python3
"""
lorenz_vps_clustering.py
------------------------
Author : anitoti
Date   : summer 2026
Project: Nimble Brain — Clarkson University REU (Dr. Jeremie Fish)

Pipeline:
  1. Load DTI-og.mat structural connectivity matrix via scipy.io.loadmat.
     Symmetrise, zero diagonal, build weighted graph Laplacian.
     [Page 28, full_.m_script.pdf:
      "load('DTI-og.mat') A = double(A); n = size(A,2);
       L = diag(sum(A,2)) - A; gel = 0.5;
       H = [0 0 0; 0 1 0; 0 0 0]; gelLH = gel*kron(L,H)"]

  2. Sweep a 2-D affine slice of the high-dimensional IC space
     grid range [-9, 9] x [-9, 9], matching MATLAB source exactly.
     [Page 37, full_.m_script.pdf:
      "xgmin = -9 ... xgmax = 9 ... ygmin = -9 ... ygmax = 9"]

  3. Integrate each IC forward with RK4 (explicit transient burn-in).
     [Page 37, full_.m_script.pdf:
      "Hardcode the 'after transient' time ... Tminus = Parameters.Tminus"]

  4. Stream VPS features via Welford online mean/variance — no OOM.
     [Page 37, full_.m_script.pdf:
      "KmeansMat = zeros(NumCond, 2*size(Vec,1))"]

  5. GPU Lloyd's k-means on independently standardised VPS blocks.
     [fractal_basins_as_mechanism_for_the_nimble_brain_tex_pdf.pdf:
      "k=8 was found to be optimal using the version of the elbow method"]

  6. GPU box-counting via max_pool2d -> fractal dimension D_f.
     [Page 37, full_.m_script.pdf:
      "f2 = fit(log(BxR2'),log(BxN2'),'poly1')"
      "p = log(width)/log(2); % nbre of generations"]

  7. Recursive boxdiv2 fractal boundary synthesis.
     [Page 37, full_.m_script.pdf: boxdiv2 recursive 2D subdivision]

  8. Edge-rewiring for random graph perturbation.
     [Page 41, full_.m_script.pdf:
      "R1 = randperm(n) ... A(R1,R3) = 1; A(R3,R1) = 1;"]

Run:
    python3 pipeline/lorenz_sweep.py
    python3 pipeline/lorenz_sweep.py \
        --grid-n 64 --coupling 0.5 \
        --dti-path data/DTI-og.mat \
        --outdir /home/atotilca/pythongpu/data/
"""

# ── from __future__ MUST be the absolute first statement in the file ────────
# Fixes: TypeError: 'type' object is not subscriptable  (Python 3.8)
# If ANY import appears before this line the backport silently fails.
from __future__ import annotations

# ── non-interactive backend — fixes headless HPC Gdk-CRITICAL errors ───────
# Must come before any other matplotlib import.
# "Unable to init server: Could not connect" = GTK trying to open a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── stdlib ──────────────────────────────────────────────────────────────────
import argparse
from dataclasses import dataclass, field
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import torch
from tqdm import tqdm

from pythongpu.utils import get_plot_path
from pythongpu.networks.static_adjacency import load_dti_laplacian, rewire_edges
from pythongpu.networks.random_graphs import (
    generate_ba_graph, generate_gnm_laplacian, generate_ws_graph,
)
from pythongpu.processing.box_counting import boxcount_2d_gpu, fractal_dimension, boxdiv2, extract_boundary
from pythongpu.processing.basin_clustering import select_optimal_clusters, plot_selection


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DTI LOADER
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class LorenzParams:
    """
    Parameters matched to MATLAB DTI script exactly.
    [Page 28, full_.m_script.pdf:
     "gel = 0.5 ... T = 2000 ... T1 = 10000 ... dt = 0.05"]

    n_osc is set from the DTI matrix shape in main() — do not set by hand.
    steps_transient and steps_record are derived from float times in
    __post_init__ so they stay consistent if dt is changed.
    """
    sigma           : float = 10.0
    rho             : float = 28.0
    beta            : float = 8.0 / 3.0
    coupling        : float = 0.5       # gel = 0.5 [Page 28, full_.m_script.pdf]
    dt              : float = 0.05      # matches MATLAB ode45 step
    t_transient     : float = 100.0     # burn-in seconds — discarded
    tmax            : float = 500.0     # T1=10000 steps * dt=0.05 = 500 s
    slice_node_x    : int   = 28        # node varied along grid x-axis
    slice_node_y    : int   = 79        # node varied along grid y-axis
    n_osc           : int   = 0         # filled from DTI matrix in main()
    steps_transient : int   = field(init=False, repr=False)
    steps_record    : int   = field(init=False, repr=False)

    def __post_init__(self):
        self.steps_transient = int(self.t_transient / self.dt)
        self.steps_record    = int(self.tmax        / self.dt)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PHYSICS — batched coupled Lorenz RHS with DTI Laplacian
# ═══════════════════════════════════════════════════════════════════════════
def lorenz_rhs_batched(
    x     : torch.Tensor,
    L_gpu : torch.Tensor,
    p     : LorenzParams,
) -> torch.Tensor:
    """
    Batched RHS of the DTI-coupled Lorenz-63 network.

    Args
    ----
    x     : (B, N, 3)  — state; B=batch(ICs), N=nodes, 3=[X,Y,Z]
    L_gpu : (N, N)     — graph Laplacian on GPU
    p     : LorenzParams

    Returns
    -------
    f : (B, N, 3)  — dx/dt for every IC and node simultaneously

    Physics
    -------
    Intrinsic Lorenz-63:
        dX = sigma * (Y - X)
        dY = X * (rho - Z) - Y
        dZ = X * Y - beta * Z

    DTI Laplacian coupling on X only:
        dX_i -= gel * (L @ X)_i
    Equivalent to MATLAB:
        gelLH = gel * kron(L, H)
    where H = [0 0 0; 0 1 0; 0 0 0] selects the X component only.
    [Page 28, full_.m_script.pdf:
     "gel = 0.5; H = [0 0 0; 0 1 0; 0 0 0]; gelLH = gel*kron(L,H)"]

    Vectorisation:
        x[..., 0] @ L_gpu.T  is (B,N) @ (N,N) = (B,N).
        All B ICs and N nodes are processed in one CUDA matmul kernel.
        L_gpu.T == L_gpu for symmetric (undirected) DTI graphs;
        written as .T for correctness in the directed case.
    """
    dX = p.sigma * (x[..., 1] - x[..., 0])             # (B, N)
    dY = x[..., 0] * (p.rho - x[..., 2]) - x[..., 1]  # (B, N)
    dZ = x[..., 0] * x[..., 1] - p.beta * x[..., 2]   # (B, N)

    f = torch.stack([dX, dY, dZ], dim=-1)               # (B, N, 3)

    # [Page 28, full_.m_script.pdf: "gelLH = gel*kron(L,H)"]
    # (B, N) @ (N, N) — one kernel, all ICs and nodes simultaneously
    f[..., 0] -= p.coupling * (x[..., 0] @ L_gpu.T)

    return f                                            # (B, N, 3)


@torch.no_grad()
def rk4_step_batched(
    x     : torch.Tensor,
    L_gpu : torch.Tensor,
    p     : LorenzParams,
) -> torch.Tensor:
    """
    Single RK4 step over the full IC batch.

    x     : (B, N, 3)  →  returns (B, N, 3)
    L_gpu : (N, N) Laplacian — threaded through every sub-step.

    x_{n+1} = x_n + (dt/6)(k1 + 2*k2 + 2*k3 + k4)

    @torch.no_grad() disables autograd — saves ~30-40% memory.
    """
    k1 = lorenz_rhs_batched(x,                    L_gpu, p)
    k2 = lorenz_rhs_batched(x + 0.5 * p.dt * k1, L_gpu, p)
    k3 = lorenz_rhs_batched(x + 0.5 * p.dt * k2, L_gpu, p)
    k4 = lorenz_rhs_batched(x +       p.dt * k3,  L_gpu, p)
    return x + (p.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  STREAMING VPS — Welford + independent block standardisation
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_sweep_streaming(
    x0     : torch.Tensor,
    L_gpu  : torch.Tensor,
    p      : LorenzParams,
    device : torch.device,
    return_mean_x : bool = False,
    norm   : str | float = "l2",
) -> torch.Tensor:
    """
    Integrate all B ICs simultaneously and accumulate VPS features
    on-the-fly using Welford's online algorithm.

    Memory cost: O(B * C(N,2)) instead of O(T * B * N * 3).
    For grid=64, N=83: ~135 MB instead of ~41 GB.

    Welford update (numerically stable, single-pass):
        count  += 1
        delta   = x_new - mean
        mean   += delta / count
        M2     += delta * (x_new - updated_mean)   ← updated_mean, not old mean
        var     = M2 / (count - 1)

    VPS features per pair (i, j), i < j:
        tau_x = mean|dX| / std|dX|    phase-coherence proxy
        L     = mean Euclidean distance in (X,Y,Z)

    SCALING FIX:
        tau_x and mean_L live on completely different scales.
        tau_x is dimensionless and typically O(1).
        mean_L is in state-space units and can be O(10)–O(100).
        If concatenated raw, mean_L dominates all k-means distances
        and tau_x carries no weight.
        Fix: standardise each block independently before concatenation:
            tau_x_std  = (tau_x  - mean(tau_x))  / (std(tau_x)  + 1e-8)
            mean_L_std = (mean_L - mean(mean_L))  / (std(mean_L) + 1e-8)
        This gives both blocks unit variance across the batch before
        they are concatenated into the VPS vector fed to k-means.
    [Page 37, full_.m_script.pdf:
     "KmeansMat = zeros(NumCond, 2*size(Vec,1))"
     where "Vec = nchoosek([1:n], 2)" — all unique oscillator pairs]

    Args
    ----
    x0     : (B, N, 3) post-burn-in ICs on device
    L_gpu  : (N, N) Laplacian on device
    p      : LorenzParams
    device : torch.device

    return_mean_x : bool
        If True, also accumulate the per-node running mean of X over the
        recording window and return it as a second tensor. sign(mean_x) is
        the clustering-free lobe-locking basin label (see the
        lobe-locking-is-the-mechanism memo): each node locks to the Lorenz
        lobe whose X-sign matches, so the exact attractor id is the 83-bit
        sign pattern — no k-means required.

    norm : "cosine", "l1", "l2", "inf", "-inf", or a real number
        Distance used for the L feature (the paper's "L² similarity measure
        ... by fiat", flagged as unexplored future work). All non-"cosine"
        values are the `ord` argument of `torch.linalg.norm` applied to the
        pairwise (X,Y,Z) difference: "l2"/"l1" are shorthand for ord=2/1,
        "inf"/"-inf" select the max/min-coordinate (Chebyshev) norm, and any
        other real p gives torch.linalg.norm(diff, ord=p) -- a continuous
        sweep of norm sensitivity, not just three fixed cases. "cosine" is
        qualitatively different in kind, not a member of the ord family: it
        measures whether two nodes' raw state vectors point the same
        direction in phase space, independent of magnitude, so it reads
        x0 directly rather than diff.

    Returns
    -------
    vps : (B, 2*C(N,2)) independently standardised VPS features on device
    mean_x : (B, N) per-node time-mean of X  — only when return_mean_x=True,
             in which case the return value is the tuple (vps, mean_x).
    """
    _NAMED_ORDS = {"l2": 2.0, "l1": 1.0, "inf": float("inf"), "-inf": float("-inf")}
    is_cosine = (norm == "cosine")
    if not is_cosine:
        if isinstance(norm, str):
            if norm not in _NAMED_ORDS:
                raise ValueError(
                    f"norm must be 'cosine', {sorted(_NAMED_ORDS)}, or a real number, got {norm!r}")
            ord_val = _NAMED_ORDS[norm]
        else:
            ord_val = float(norm)

    B, N, _ = x0.shape
    mean_x = torch.zeros(B, N, device=device) if return_mean_x else None

    # All unique pairs (i < j), upper triangle, no diagonal
    # C(N,2) = N*(N-1)/2  pairs total
    iu   = torch.triu_indices(N, N, offset=1, device=device)
    i, j = iu[0], iu[1]
    C    = i.shape[0]

    # Welford accumulators — (B, C), all on GPU
    count   = 0
    mean_dx = torch.zeros(B, C, device=device)   # running mean of |dX|
    M2_dx   = torch.zeros(B, C, device=device)   # running sum of sq. devs
    mean_L  = torch.zeros(B, C, device=device)   # running mean of ||d_state||

    for step in tqdm(range(p.steps_record), desc="streaming"):
        x0 = rk4_step_batched(x0, L_gpu, p)

        # Pairwise state difference at this timestep: (B, C, 3)
        diff = x0[:, i, :] - x0[:, j, :]

        dx_abs = diff[..., 0].abs()                       # (B, C)

        # L feature: distance/coherence between node pairs i,j. Every ord_val
        # measures separation magnitude of the raw difference; cosine instead
        # measures directional (mis)alignment of the two nodes' full state
        # vectors, so it reads from x0 directly rather than from diff.
        if is_cosine:
            L_val = 1.0 - torch.nn.functional.cosine_similarity(
                x0[:, i, :], x0[:, j, :], dim=-1)                          # (B, C)
        else:
            L_val = torch.linalg.norm(diff, ord=ord_val, dim=-1)          # (B, C)

        # Welford update for |dX| — gives both mean and variance
        count  += 1
        delta   = dx_abs - mean_dx
        mean_dx = mean_dx + delta / count
        M2_dx   = M2_dx   + delta * (dx_abs - mean_dx)   # uses updated mean

        # Simple Welford mean for L (variance not needed for L feature)
        mean_L = mean_L + (L_val - mean_L) / count

        # Optional: running mean of per-node X for the sign-based lobe label
        if mean_x is not None:
            mean_x = mean_x + (x0[:, :, 0] - mean_x) / count

    # ── tau_x: normalised phase-coherence proxy ───────────────────────
    # std|dX| = sqrt(M2 / (count-1)); 1e-8 guards against zero variance
    var_dx = M2_dx / max(count - 1, 1)
    tau_x  = mean_dx / (var_dx.sqrt() + 1e-8)    # (B, C)  dimensionless, O(1)

    # ── independent block standardisation ─────────────────────────────
    # Problem:
    #   tau_x  is dimensionless,  typically  O(1)
    #   mean_L is in state units, typically  O(10)–O(100) for Lorenz
    #   Raw concatenation makes mean_L dominate all k-means distances;
    #   tau_x contributes essentially nothing to cluster separation.
    #
    # Fix: standardise each block to zero mean, unit variance across
    # the batch dimension B independently, BEFORE concatenation.
    # This is equivalent to running StandardScaler separately on each
    # half of the feature matrix before passing to k-means.
    #
    # dim=0 = mean/std over the B ICs (the sample axis).
    # keepdim=True preserves (1, C) shape for broadcasting over (B, C).

    tau_x_std = (
        (tau_x  - tau_x.mean(dim=0,  keepdim=True))
        / (tau_x.std(dim=0,  keepdim=True) + 1e-8)
    )   # (B, C) — zero mean, unit std across batch

    mean_L_std = (
        (mean_L - mean_L.mean(dim=0, keepdim=True))
        / (mean_L.std(dim=0, keepdim=True) + 1e-8)
    )   # (B, C) — zero mean, unit std across batch

    # Concatenate standardised blocks → (B, 2*C)
    # Both halves now contribute equally to k-means distances.
    vps = torch.cat([tau_x_std, mean_L_std], dim=-1)
    if mean_x is not None:
        return vps, mean_x
    return vps


# ═══════════════════════════════════════════════════════════════════════════
# 5.  GPU K-MEANS — Lloyd's algorithm entirely on GPU
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def kmeans_gpu(
    X      : torch.Tensor,
    k      : int,
    n_iter : int   = 300,
    tol    : float = 1e-4,
    seed   : int   = 42,
) -> torch.Tensor:
    """
    Lloyd's k-means entirely on GPU — no sklearn, no CPU round-trips.

    Distance formula avoids materialising the full (B, k, D) tensor:
        ||x - c||² = ||x||² - 2<x,c> + ||c||²

    Input X is already independently standardised per block (done in
    run_sweep_streaming). A second global normalisation is applied here
    so that any residual scale differences across the 2*C features are
    removed before distance calculations.

    [Page 37, full_.m_script.pdf:
     "KmeansMat ... k=8 was found to be optimal using the elbow method"]

    Args
    ----
    X      : (B, D) float32 VPS features on device — already block-standardised
    k      : number of clusters
    n_iter : max Lloyd iterations
    tol    : centroid shift convergence threshold (L2 norm)
    seed   : Forgy-init seed, via a local torch.Generator (does not touch
             global RNG state). Previously unseeded (plain torch.randperm on
             the global generator), so every invocation landed in a
             different local optimum -- confounded a VPS-norm comparison
             that (correctly) held everything else fixed. Default 42
             matches select_optimal_clusters's random_state=42 convention,
             so both clustering paths in this codebase are now equally
             reproducible by default.

    Returns
    -------
    labels : (B,) int64 cluster assignments on device
    """
    B, D = X.shape

    # Global normalisation — zero mean, unit std per feature across batch.
    # Input is already block-standardised; this handles any residual
    # scale differences between the tau_x and mean_L halves.
    # Single clean line — no dead mu/std variables.
    Xn = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-8)

    # Forgy initialisation: pick k random samples as starting centroids,
    # from a seeded local generator so runs are reproducible without
    # mutating torch's global RNG state (which the rest of the pipeline,
    # e.g. the IC jitter elsewhere, may still depend on being unseeded).
    gen = torch.Generator(device=X.device)
    gen.manual_seed(seed)
    idx       = torch.randperm(B, device=X.device, generator=gen)[:k]
    centroids = Xn[idx].clone()                 # (k, D)
    labels    = torch.zeros(B, dtype=torch.long, device=X.device)

    for _ in range(n_iter):
        # Pairwise squared distances (B, k) — pure matrix algebra, one kernel
        dist = (
              Xn.pow(2).sum(dim=1, keepdim=True)        # (B, 1)
            - 2.0 * (Xn @ centroids.T)                  # (B, k)
            + centroids.pow(2).sum(dim=1).unsqueeze(0)  # (1, k)
        )                                                # (B, k)

        new_labels = dist.argmin(dim=1)                  # (B,)

        # Update centroids: mean of all points assigned to each cluster
        new_centroids = torch.zeros_like(centroids)      # (k, D)
        counts        = torch.zeros(k, device=X.device)  # (k,)

        # scatter_add accumulates all points in each cluster simultaneously
        new_centroids.scatter_add_(
            0, new_labels.unsqueeze(1).expand(B, D), Xn
        )
        counts.scatter_add_(
            0, new_labels, torch.ones(B, device=X.device)
        )
        # Guard: empty clusters keep their old centroid
        new_centroids /= counts.clamp(min=1).unsqueeze(1)

        shift     = (new_centroids - centroids).pow(2).sum().sqrt()
        centroids = new_centroids
        labels    = new_labels

        if shift < tol:
            break

    return labels   # (B,) int64


# ═══════════════════════════════════════════════════════════════════════════
# 6.  GPU BOX-COUNTING — fractal dimension via max_pool2d
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()




# ═══════════════════════════════════════════════════════════════════════════
# 7.  BOXDIV2 — recursive fractal boundary synthesis
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 8.  EDGE REWIRING
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 9.  BASIN BOUNDARY EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():

    # ── CLI ─────────────────────────────────────────────────────────────
    ap = argparse.ArgumentParser(
        description="DTI-coupled Lorenz basin map + fractal dimension"
    )
    ap.add_argument("--grid-n",     type=int,   default=128,
        help="Points per axis. 64→4096 ICs. 361 matches MATLAB step=0.05.")
    ap.add_argument("--coupling",   type=float, default=0.5,
        help="gel — Laplacian coupling strength. "
             "[Page 28, full_.m_script.pdf: 'gel = 0.5']")
    ap.add_argument("--k-clusters", default="auto",
        help="Number of basins K. 'auto' (default) selects K dynamically from "
             "the VPS feature geometry via the Elbow+BIC+Silhouette consensus; "
             "pass an integer to force a fixed K instead.")
    ap.add_argument("--cluster-criterion", default="consensus",
        choices=["consensus", "elbow", "bic", "silhouette"],
        help="Model-selection criterion that sets K when --k-clusters=auto.")
    ap.add_argument("--graph", default="dti",
        choices=["dti", "er", "ba", "ws"],
        help="Network substrate: 'dti' (default, the empirical connectome) or a "
             "null model with the same node count — 'er' (Erdos-Renyi G(n,m), "
             "edge count matched to DTI by default), 'ba' (Barabasi-Albert "
             "scale-free), 'ws' (Watts-Strogatz small-world). Use these to test "
             "whether fractal basins come from brain wiring or from any "
             "comparable graph.")
    ap.add_argument("--graph-n", type=int, default=None,
        help="Node count for the null model (default: match the DTI matrix).")
    ap.add_argument("--graph-seed", type=int, default=None,
        help="Seed for null-model graph generation (set it for reproducibility).")
    ap.add_argument("--ba-m", type=int, default=5,
        help="Barabasi-Albert: edges attached per new node (--graph ba).")
    ap.add_argument("--er-m", type=int, default=None,
        help="Erdos-Renyi: total edges (--graph er; default: DTI's edge count).")
    ap.add_argument("--ws-k", type=int, default=10,
        help="Watts-Strogatz: neighbours per node, must be even (--graph ws).")
    ap.add_argument("--ws-p", type=float, default=0.1,
        help="Watts-Strogatz: rewiring probability (--graph ws).")
    ap.add_argument("--null-reps",  type=int,   default=0,
        help="With --k-clusters=auto, run a structure-vs-noise guard with this "
             "many column-shuffle null reps; if the features carry no separable "
             "structure the sweep records a single basin (k=1) instead of "
             "fabricating basins. 0 (default) disables the guard.")
    ap.add_argument("--min-silhouette", type=float, default=0.10,
        help="Silhouette floor the real optimum must exceed to count as "
             "structured under --null-reps.")
    ap.add_argument("--boxdiv-p",   type=float, default=0.7,
        help="Boxdiv2 survival probability.")
    ap.add_argument("--rewire-n",   type=int,   default=5,
        help="Edges to rewire in perturbation demo.")
    ap.add_argument("--dti-path",   type=str,   default="data/DTI-og.mat",
        help="Path to DTI-og.mat (professor original DTI). "
             "[Page 28, full_.m_script.pdf: \"load('DTI-og.mat')\"]")
    ap.add_argument("--outdir",     type=str,   default=".",
        help="Output directory for all files.")
    args = ap.parse_args()

    # ── device ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device]   {device}")

    # ── load DTI-og.mat (professor original) and build Laplacian ───────────────────────────────
    # [Page 28, full_.m_script.pdf:
    #  "load('DTI-og.mat') A = double(A); n = size(A,2);
    #   L = diag(sum(A,2)) - A; gel = 0.5;
    #   H = [0 0 0; 0 1 0; 0 0 0]; gelLH = gel*kron(L,H)"]
    L_gpu, n_dti = load_dti_laplacian(args.dti_path, device)

    # ── null-model substitution ──────────────────────────────────────────
    # Swap the empirical connectome for a synthetic graph of the SAME node
    # count, so any change in basin structure is attributable to topology
    # alone rather than to network size. This is what answers "is fractality
    # a property of brain wiring, or of any comparably dense graph?".
    if args.graph != "dti":
        n_null = args.graph_n if args.graph_n else n_dti
        seed = args.graph_seed
        # Edge count of the empirical connectome (L = D - A, so the off-diagonal
        # non-zeros are the edges, counted twice for an undirected graph).
        _Lnp = L_gpu.detach().cpu().numpy()
        _A = np.diag(np.diag(_Lnp)) - _Lnp
        np.fill_diagonal(_A, 0.0)
        dti_edges = int((np.abs(_A) > 0).sum() // 2)
        if args.graph == "ba":
            # m = edges attached per new node; total edges ~ m*(n-m).
            L_gpu = generate_ba_graph(n_null, args.ba_m, device=device,
                                      plot=False, seed=seed)
            desc = f"Barabasi-Albert  n={n_null}  m={args.ba_m}"
        elif args.graph == "er":
            # Match the empirical edge count by default so density is comparable
            # and only the topology differs.
            m_edges = args.er_m if args.er_m else dti_edges
            L_gpu = generate_gnm_laplacian(n_null, m_edges, device=device,
                                           plot=False, seed=seed)
            desc = f"Erdos-Renyi G(n,m)  n={n_null}  m={m_edges} (DTI has {dti_edges})"
        elif args.graph == "ws":
            L_gpu = generate_ws_graph(n_null, args.ws_k, args.ws_p,
                                      device=device, plot=False, seed=seed)
            desc = f"Watts-Strogatz  n={n_null}  k={args.ws_k}  p={args.ws_p}"
        else:
            raise ValueError(f"unknown --graph {args.graph!r}")
        L_gpu = L_gpu.to(device=device, dtype=torch.float32)
        n_dti = n_null
        print(f"[graph]    NULL MODEL: {desc}  (seed={seed}) — DTI_A replaced")

    # ── params — n_osc set from actual matrix size ───────────────────────
    p = LorenzParams(
        coupling = args.coupling,
        n_osc    = n_dti,            # "n = size(A,2)" [Page 28]
    )
    print(
        f"[config]   grid={args.grid_n}²  N={p.n_osc} (from DTI-og.mat)  "
        f"coupling={p.coupling}  dt={p.dt}\n"
        f"           transient={p.t_transient}s ({p.steps_transient} steps)  "
        f"record={p.tmax}s ({p.steps_record} steps)\n"
        f"           slice: node_x={p.slice_node_x}  node_y={p.slice_node_y}"
    )

    # ── output directory ─────────────────────────────────────────────────
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── memory estimate ──────────────────────────────────────────────────
    m       = args.grid_n
    B       = m * m
    N       = p.n_osc
    C_pairs = N * (N - 1) // 2

    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        stream_gb = 3 * B * C_pairs * 4 / 1e9
        traj_gb   = p.steps_record * B * N * 3 * 4 / 1e9
        print(
            f"[memory]   GPU free={free/1e9:.1f} GB  |  "
            f"streaming ~{stream_gb:.2f} GB  "
            f"(full traj would be ~{traj_gb:.1f} GB)"
        )

    # ── 2-D affine slice grid ────────────────────────────────────────────
    # [Page 37, full_.m_script.pdf:
    #  "xgmin = -9 ... xgmax = 9 ... ygmin = -9 ... ygmax = 9"]
    ax     = np.linspace(-9.0, 9.0, m, dtype=np.float32)
    Xg, Yg = np.meshgrid(ax, ax)

    # ── build IC batch ───────────────────────────────────────────────────
    # [Page 37, full_.m_script.pdf:
    #  "S1 = S0; S1(NodexVals(Randnodei)) = Xg(j);
    #            S1(NodexVals(Randnodej)) = Yg(j)"]
    x0  = torch.ones((B, N, 3), dtype=torch.float32, device=device)
    x0 += 0.05 * torch.randn_like(x0)

    # H = [0 0 0; 0 1 0; 0 0 0] selects X component only
    # [Page 28, full_.m_script.pdf: "H = [0 0 0; 0 1 0; 0 0 0]"]
    x0[:, p.slice_node_x, 0] = torch.tensor(Xg.ravel(), device=device)
    x0[:, p.slice_node_y, 0] = torch.tensor(Yg.ravel(), device=device)

    # ── Phase 1: burn-in ─────────────────────────────────────────────────
    # [Page 37, full_.m_script.pdf:
    #  "Hardcode the 'after transient' time ... Tminus = Parameters.Tminus"]
    print(f"[burn-in]  {p.steps_transient} steps  ({B} ICs in parallel) ...")
    for step in tqdm(range(p.steps_transient), desc="burn-in"):
        x0 = rk4_step_batched(x0, L_gpu, p)

    # ── Phase 2: streaming VPS accumulation ─────────────────────────────
    print(f"[stream]   {p.steps_record} steps  "
          f"(Welford + block standardisation, {C_pairs} pairs) ...")
    vectors_gpu = run_sweep_streaming(x0, L_gpu, p, device)  # (B, 2*C)
    vectors     = vectors_gpu.cpu().numpy()
    print(f"[vps]      shape = {vectors.shape}")

    # ── Basin partition: dynamic K selection or user-forced K ────────────
    # Replaces the previously hard-wired k with a data-driven estimate: the
    # optimal number of basins is read off the VPS feature geometry via the
    # Elbow (Kneedle), Gaussian-mixture BIC, and Silhouette criteria and their
    # consensus, so K is no longer a transcribed constant.
    if str(args.k_clusters).lower() == "auto":
        print("[cluster]  dynamic K selection "
              "(Elbow + BIC + Silhouette), k=2..15 ...")
        sel = select_optimal_clusters(
            vectors, k_min=2, k_max=15, criterion=args.cluster_criterion,
            null_reps=args.null_reps, min_silhouette=args.min_silhouette)
        k_used = sel.best_k
        structured = sel.structured
        labels = sel.labels.reshape(m, m).astype(np.int32)
        if not sel.structured:
            print(f"[cluster]  NULL GUARD: no separable structure "
                  f"(null_p95={sel.null_p95:.3f}) -> single basin (k=1); "
                  f"box-counting skipped, D_f undefined.")
        else:
            print(f"[cluster]  elbow={sel.k_elbow}  BIC={sel.k_bic}  "
                  f"silhouette={sel.k_silhouette}  ->  K={k_used} "
                  f"(criterion={sel.criterion})")
        plot_selection(
            sel,
            get_plot_path("lorenz_vps_clustering", "cluster_selection.png", args.outdir),
            title=f"DTI_A Lorenz basins  coupling={p.coupling}")
        print("[saved]    cluster_selection.png")
    else:
        k_used = int(args.k_clusters)
        structured = True
        print(f"[kmeans]   GPU Lloyd's  k={k_used} (user-forced) ...")
        labels_gpu = kmeans_gpu(vectors_gpu, k=k_used)
        labels     = labels_gpu.cpu().numpy().reshape(m, m)
        print(f"[kmeans]   {k_used} basin labels assigned")

    # ── Basin map ────────────────────────────────────────────────────────
    fig_bm, ax_bm = plt.subplots(figsize=(7, 6))
    im = ax_bm.imshow(
        labels, origin="lower", cmap="tab20",
        extent=[-9.0, 9.0, -9.0, 9.0], interpolation="nearest",
    )
    ax_bm.set_xlabel(f"Node {p.slice_node_x}  X perturbation")
    ax_bm.set_ylabel(f"Node {p.slice_node_y}  X perturbation")
    ax_bm.set_title(
        f"K-Means Basin Map (k={k_used})\n"
        f"N={p.n_osc} coupling={p.coupling} grid={m}²"
    )
    plt.colorbar(im, ax=ax_bm, label="Basin label")
    plt.tight_layout()
    plt.savefig(get_plot_path("lorenz_vps_clustering", "basin_map_kmeans.png", args.outdir), dpi=150)
    plt.close(fig_bm)
    print("[saved]    basin_map_kmeans.png")

    # ── Basin boundary + box-counting ────────────────────────────────────
    boundary  = extract_boundary(labels)
    # A single basin (e.g. the null guard's k=1, or a fully synchronized slice)
    # has no boundary, so the box-counting fractal dimension is undefined rather
    # than 0 — skip the fit instead of feeding empty arrays to polyfit.
    has_boundary = bool(boundary.any())
    if has_boundary:
        r, n      = boxcount_2d_gpu(boundary, device)
        D_f, r_sq = fractal_dimension(r, n)
        print(f"[fractal]  D_f = {D_f:.4f}  (R² = {r_sq:.4f})")
        print(
            "           D_f ≈ 1.0 → smooth curve\n"
            "           D_f ≈ 2.0 → space-filling\n"
            "           Expected chimera range: ~1.2–1.8"
        )
    else:
        r = np.array([], dtype=np.int64)
        n = np.array([], dtype=np.int64)
        D_f, r_sq = float("nan"), float("nan")
        print("[fractal]  single basin — no boundary; D_f undefined (skipped).")

    fig_bd, ax_bd = plt.subplots(figsize=(7, 6))
    ax_bd.imshow(
        boundary, origin="lower", cmap="binary",
        extent=[-9.0, 9.0, -9.0, 9.0], interpolation="nearest",
    )
    ax_bd.set_xlabel(f"Node {p.slice_node_x}  X perturbation")
    ax_bd.set_ylabel(f"Node {p.slice_node_y}  X perturbation")
    ax_bd.set_title(
        f"Basin Boundary — DTI_A\n"
        f"N={p.n_osc}  coupling={p.coupling}  grid={m}²"
    )
    plt.tight_layout()
    plt.savefig(get_plot_path("lorenz_vps_clustering", "basin_boundary.png", args.outdir), dpi=150)
    plt.close(fig_bd)
    print("[saved]    basin_boundary.png")

    if has_boundary:
        mask      = n > 0
        log_r_fit = np.log(r[mask].astype(float))
        log_n_fit = np.polyval(
            np.polyfit(log_r_fit, np.log(n[mask].astype(float)), 1),
            log_r_fit,
        )
        fig_bc, ax_bc = plt.subplots(figsize=(6, 5))
        ax_bc.loglog(r[mask], n[mask], "o-", color="crimson", label="box count")
        ax_bc.loglog(r[mask], np.exp(log_n_fit), "--", color="navy",
                     label=f"fit  D_f={D_f:.3f}  R²={r_sq:.3f}")
        ax_bc.set_xlabel("Box size r")
        ax_bc.set_ylabel("Box count N(r)")
        ax_bc.set_title("Box-Counting — Fractal Dimension")
        ax_bc.legend()
        ax_bc.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(get_plot_path("lorenz_vps_clustering", "boxcount_loglog.png", args.outdir), dpi=150)
        plt.close(fig_bc)
        print("[saved]    boxcount_loglog.png")
    else:
        print("[skip]     boxcount_loglog.png — single basin, no boundary to fit.")

    # ── Boxdiv2 synthetic fractal ────────────────────────────────────────
    # [Page 37, full_.m_script.pdf: boxdiv2 recursive 2D subdivision]
    print(f"[boxdiv2]  p={args.boxdiv_p} ...")
    seed = np.ones((256, 256), dtype=bool)
    frac = boxdiv2(seed, p=args.boxdiv_p)
    fig_bx, ax_bx = plt.subplots(figsize=(6, 6))
    ax_bx.imshow(frac, origin="lower", cmap="gray", interpolation="nearest")
    ax_bx.set_title(f"Synthetic Fractal (boxdiv2, p={args.boxdiv_p})")
    ax_bx.axis("off")
    plt.tight_layout()
    plt.savefig(get_plot_path("lorenz_vps_clustering", "boxdiv2_synthetic.png", args.outdir), dpi=150)
    plt.close(fig_bx)
    print("[saved]    boxdiv2_synthetic.png")

    # ── Edge rewiring demo on DTI adjacency ─────────────────────────────
    # [Page 41, full_.m_script.pdf:
    #  "R1 = randperm(n) ... A(R1,R3) = 1; A(R3,R1) = 1;"]
    print(f"[rewire]   {args.rewire_n} edge rewires on DTI_A ...")
    L_np  = L_gpu.cpu().numpy()
    A_dti = (np.diag(np.diag(L_np)) - L_np).astype(np.float32)
    A_rew = rewire_edges(A_dti, num_edges=args.rewire_n)

    fig_rw, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(A_dti, cmap="hot", interpolation="nearest")
    axes[0].set_title("DTI_A adjacency (original)")
    axes[1].imshow(A_rew, cmap="hot", interpolation="nearest")
    axes[1].set_title(f"After {args.rewire_n} rewires")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(get_plot_path("lorenz_vps_clustering", "rewired_adjacency.png", args.outdir), dpi=150)
    plt.close(fig_rw)
    print("[saved]    rewired_adjacency.png")

    # ── Save ─────────────────────────────────────────────────────────────
    # Embed the LIVE simulation configuration so the replotter renders the
    # actual coupling and swept-node identities rather than stale hard-coded
    # defaults (previously the replotter had no coupling to show → it fell
    # back to an unlabelled / coupling=0 title).
    config = dict(
        coupling     = float(p.coupling),
        slice_node_x = int(p.slice_node_x),
        slice_node_y = int(p.slice_node_y),
        n_osc        = int(p.n_osc),
        grid_n       = int(m),
        k_clusters   = int(k_used),
        structured   = bool(structured),
        graph        = str(args.graph),        # 'dti' | 'er' | 'ba' | 'ws'
        graph_seed   = args.graph_seed,
        grid_lo      = -9.0,
        grid_hi      = 9.0,
        sigma        = float(p.sigma),
        rho          = float(p.rho),
        beta         = float(p.beta),
        dt           = float(p.dt),
        tmax         = float(p.tmax),
    )
    np.savez_compressed(
        out_dir / "basin_data.npz",
        Xg          = Xg,
        Yg          = Yg,
        labels      = labels,
        boundary    = boundary,
        vectors     = vectors,
        boxcount_r  = r,
        boxcount_n  = n,
        fractal_dim = np.array([D_f]),
        r_squared   = np.array([r_sq]),
        A_dti       = A_dti,
        A_rewired   = A_rew,
        config      = np.array(config, dtype=object),
    )
    print(f"[saved]    basin_data.npz  →  {out_dir / 'basin_data.npz'}")
    print(f"\n{'─'*55}")
    print(f"  Fractal dimension  D_f = {D_f:.4f}  (R² = {r_sq:.4f})")
    print(f"{'─'*55}")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()