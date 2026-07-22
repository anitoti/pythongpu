#!/usr/bin/env python3
"""
rossler_vps_clustering.py
-------------------------
Author : anitoti
Date   : summer 2026
Project: Nimble Brain — Clarkson University REU (Dr. Jeremie Fish)

Rössler variant of the DTI-coupled VPS basin-mapping pipeline. Strict
feature parity with lorenz_sweep.py — identical pipeline stages, CLI,
output artefacts and file names — with the Lorenz-63 vector field swapped
for the canonical chaotic Rössler flow.

Pipeline (mirrors lorenz_sweep.py exactly):
  1. Load DTI-og.mat structural connectivity (professor original), build the graph Laplacian.
  2. Sweep a 2-D affine slice of the high-dimensional IC space.
  3. Integrate each IC forward with RK4 (explicit transient burn-in).
  4. Stream VPS features via Welford online mean/variance — no OOM.
  5. Dynamic K basin partition (Elbow + BIC + Silhouette consensus) or a
     GPU Lloyd's k-means with a user-forced K.
  6. GPU box-counting -> fractal dimension D_f.
  7. Recursive boxdiv2 fractal boundary synthesis.
  8. Edge-rewiring perturbation demo on the DTI adjacency.

Physics — canonical chaotic Rössler (a=0.2, b=0.2, c=5.7):
    dX_i = -Y_i - Z_i - gel*(L@X)_i
    dY_i =  X_i + a*Y_i
    dZ_i =  b + Z_i*(X_i - c)

Coupling acts on the X component only — the oscillatory component of the
Rössler flow (Z is a slow spiking variable and is a poor synchronisation
channel), matching the H = diag[1,0,0] projection every oscillator in this
package uses.

Run:
    python3 -m pythongpu.pipeline.rossler_sweep
    python3 -m pythongpu.pipeline.rossler_sweep \
        --grid-n 64 --coupling 0.5 \
        --dti-path data/DTI-og.mat \
        --outdir /home/atotilca/pythongpu/data/rossler/
"""

# ── from __future__ MUST be the absolute first statement in the file ────────
# Fixes: TypeError: 'type' object is not subscriptable  (Python 3.8)
# If ANY import appears before this line the backport silently fails.
from __future__ import annotations

# ── non-interactive backend — fixes headless HPC Gdk-CRITICAL errors ───────
# Must come before any other matplotlib import.
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
from pythongpu.processing.box_counting import boxcount_2d_gpu, fractal_dimension, boxdiv2, extract_boundary
from pythongpu.processing.basin_clustering import select_optimal_clusters, plot_selection


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DTI LOADER  (shared with lorenz_sweep via static_adjacency)
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class RosslerParams:
    """
    Canonical chaotic Rössler parameters (a=0.2, b=0.2, c=5.7).

    n_osc is set from the DTI matrix shape in main() — do not set by hand.
    steps_transient and steps_record are derived from float times in
    __post_init__ so they stay consistent if dt is changed.

    grid_lo / grid_hi bound the 2-D IC slice. Perturbations are kept within
    the Rössler x-attractor (roughly x ∈ [-9, 11]) at [-9, 9] — the same
    slice as Lorenz — so no initial condition starts off-basin and escapes.

    dt is smaller than the Lorenz variant (0.05): the DTI-coupled Rössler is
    stiff (eig_max(L) ≈ 45) and the attractor is not globally attracting, so
    dt=0.05 diverges to Z→∞; dt=0.01 stays bounded for coupling ∈ [0, 1].
    """
    a               : float = 0.2       # canonical chaotic Rössler
    b               : float = 0.2
    c               : float = 5.7
    coupling        : float = 0.05      # gel — Laplacian coupling on X. The
                                        # X-coupled Rössler destabilises above
                                        # ~0.15 (Z→∞), so its useful basin-
                                        # structure window is small (≲0.1),
                                        # unlike the globally-bounded Lorenz/HR.
    dt              : float = 0.01      # small step for stiff coupled Rössler
    t_transient     : float = 100.0     # burn-in seconds — discarded
    tmax            : float = 200.0     # record window (s) — ~33 Rössler cycles
    grid_lo         : float = -9.0      # IC slice lower bound (within attractor)
    grid_hi         : float = 9.0       # IC slice upper bound (within attractor)
    slice_node_x    : int   = 28        # node varied along grid x-axis
    slice_node_y    : int   = 79        # node varied along grid y-axis
    n_osc           : int   = 0         # filled from DTI matrix in main()
    steps_transient : int   = field(init=False, repr=False)
    steps_record    : int   = field(init=False, repr=False)

    def __post_init__(self):
        self.steps_transient = int(self.t_transient / self.dt)
        self.steps_record    = int(self.tmax        / self.dt)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PHYSICS — batched coupled Rössler RHS with DTI Laplacian
# ═══════════════════════════════════════════════════════════════════════════
def rossler_rhs_batched(
    x     : torch.Tensor,
    L_gpu : torch.Tensor,
    p     : RosslerParams,
) -> torch.Tensor:
    """
    Batched RHS of the DTI-coupled Rössler network.

    Args
    ----
    x     : (B, N, 3)  — state; B=batch(ICs), N=nodes, 3=[X,Y,Z]
    L_gpu : (N, N)     — graph Laplacian on GPU
    p     : RosslerParams

    Returns
    -------
    f : (B, N, 3)  — dx/dt for every IC and node simultaneously

    Physics
    -------
    Intrinsic Rössler:
        dX = -Y - Z
        dY =  X + a*Y
        dZ =  b + Z*(X - c)

    DTI Laplacian coupling on X only:
        dX_i -= gel * (L @ X)_i
    which is the H = diag[1,0,0] projection shared by every oscillator in
    this package. The X component is the oscillatory channel of the Rössler
    flow; Z is a slow spiking variable and is a poor synchronisation route.

    Vectorisation:
        x[..., 0] @ L_gpu.T  is (B,N) @ (N,N) = (B,N).
        All B ICs and N nodes are processed in one CUDA matmul kernel.
        L_gpu.T == L_gpu for symmetric (undirected) DTI graphs;
        written as .T for correctness in the directed case.
    """
    dX = -x[..., 1] - x[..., 2]                        # (B, N)
    dY =  x[..., 0] + p.a * x[..., 1]                  # (B, N)
    dZ =  p.b + x[..., 2] * (x[..., 0] - p.c)          # (B, N)

    f = torch.stack([dX, dY, dZ], dim=-1)               # (B, N, 3)

    # (B, N) @ (N, N) — one kernel, all ICs and nodes simultaneously
    f[..., 0] -= p.coupling * (x[..., 0] @ L_gpu.T)

    return f                                            # (B, N, 3)


@torch.no_grad()
def rk4_step_batched(
    x     : torch.Tensor,
    L_gpu : torch.Tensor,
    p     : RosslerParams,
) -> torch.Tensor:
    """
    Single RK4 step over the full IC batch.

    x     : (B, N, 3)  →  returns (B, N, 3)
    L_gpu : (N, N) Laplacian — threaded through every sub-step.

    x_{n+1} = x_n + (dt/6)(k1 + 2*k2 + 2*k3 + k4)

    @torch.no_grad() disables autograd — saves ~30-40% memory.
    """
    k1 = rossler_rhs_batched(x,                    L_gpu, p)
    k2 = rossler_rhs_batched(x + 0.5 * p.dt * k1, L_gpu, p)
    k3 = rossler_rhs_batched(x + 0.5 * p.dt * k2, L_gpu, p)
    k4 = rossler_rhs_batched(x +       p.dt * k3,  L_gpu, p)
    return x + (p.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  STREAMING VPS — Welford + independent block standardisation
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_sweep_streaming(
    x0     : torch.Tensor,
    L_gpu  : torch.Tensor,
    p      : RosslerParams,
    device : torch.device,
) -> torch.Tensor:
    """
    Integrate all B ICs simultaneously and accumulate VPS features
    on-the-fly using Welford's online algorithm.

    Memory cost: O(B * C(N,2)) instead of O(T * B * N * 3).

    VPS features per pair (i, j), i < j:
        tau_x = mean|dX| / std|dX|    phase-coherence proxy (X is the
                                      oscillatory Rössler component)
        L     = mean Euclidean distance in (X,Y,Z)

    tau_x and mean_L live on different scales, so each block is
    standardised independently across the batch before concatenation,
    giving both unit variance in the VPS vector fed to k-means.

    Returns
    -------
    vps : (B, 2*C(N,2)) independently standardised VPS features on device
    """
    B, N, _ = x0.shape

    # All unique pairs (i < j), upper triangle, no diagonal
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

        dx_abs = diff[..., 0].abs()                       # (B, C)  — X only
        L_val  = torch.linalg.norm(diff, dim=-1)          # (B, C)

        # Welford update for |dX| — gives both mean and variance
        count  += 1
        delta   = dx_abs - mean_dx
        mean_dx = mean_dx + delta / count
        M2_dx   = M2_dx   + delta * (dx_abs - mean_dx)   # uses updated mean

        # Simple Welford mean for L (variance not needed for L feature)
        mean_L = mean_L + (L_val - mean_L) / count

    # ── tau_x: normalised phase-coherence proxy ───────────────────────
    var_dx = M2_dx / max(count - 1, 1)
    tau_x  = mean_dx / (var_dx.sqrt() + 1e-8)    # (B, C)  dimensionless, O(1)

    # ── independent block standardisation ─────────────────────────────
    # tau_x is dimensionless O(1); mean_L is in state units and larger.
    # Standardise each block to zero mean, unit variance across the batch
    # dimension B before concatenation so both contribute equally to
    # k-means distances.
    tau_x_std = (
        (tau_x  - tau_x.mean(dim=0,  keepdim=True))
        / (tau_x.std(dim=0,  keepdim=True) + 1e-8)
    )   # (B, C)

    mean_L_std = (
        (mean_L - mean_L.mean(dim=0, keepdim=True))
        / (mean_L.std(dim=0, keepdim=True) + 1e-8)
    )   # (B, C)

    # Concatenate standardised blocks → (B, 2*C)
    return torch.cat([tau_x_std, mean_L_std], dim=-1)


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

    Input X is already independently standardised per block. A second
    global normalisation removes any residual scale differences across
    the 2*C features before distance calculations.

    Args
    ----
    X      : (B, D) float32 VPS features on device — already block-standardised
    k      : number of clusters
    n_iter : max Lloyd iterations
    tol    : centroid shift convergence threshold (L2 norm)
    seed   : Forgy-init seed, via a local torch.Generator (does not touch
             global RNG state). Previously unseeded -- see
             lorenz_sweep.py::kmeans_gpu for the full story.

    Returns
    -------
    labels : (B,) int64 cluster assignments on device
    """
    B, D = X.shape

    # Global normalisation — zero mean, unit std per feature across batch.
    Xn = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-8)

    # Forgy initialisation: pick k random samples as starting centroids,
    # from a seeded local generator (reproducible, no global RNG side effects).
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
# 6.  GPU BOX-COUNTING — imported from processing.box_counting
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 7.  BOXDIV2 — imported from processing.box_counting
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 8.  EDGE REWIRING — imported from networks.static_adjacency
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 9.  BASIN BOUNDARY EXTRACTION — imported from processing.box_counting
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():

    # ── CLI ─────────────────────────────────────────────────────────────
    ap = argparse.ArgumentParser(
        description="DTI-coupled Rössler basin map + fractal dimension"
    )
    ap.add_argument("--grid-n",     type=int,   default=128,
        help="Points per axis. 64→4096 ICs.")
    ap.add_argument("--coupling",   type=float, default=0.05,
        help="gel — Laplacian coupling strength on the X component. "
             "Rössler destabilises above ~0.15; keep in its ≲0.1 window.")
    ap.add_argument("--k-clusters", default="auto",
        help="Number of basins K. 'auto' (default) selects K dynamically from "
             "the VPS feature geometry via the Elbow+BIC+Silhouette consensus; "
             "pass an integer to force a fixed K instead.")
    ap.add_argument("--cluster-criterion", default="consensus",
        choices=["consensus", "elbow", "bic", "silhouette"],
        help="Model-selection criterion that sets K when --k-clusters=auto.")
    ap.add_argument("--boxdiv-p",   type=float, default=0.7,
        help="Boxdiv2 survival probability.")
    ap.add_argument("--rewire-n",   type=int,   default=5,
        help="Edges to rewire in perturbation demo.")
    ap.add_argument("--dti-path",   type=str,   default="data/DTI-og.mat", 
        help="Path to DTI-og.mat structural connectivity matrix.")
    ap.add_argument("--outdir",     type=str,   default=".",
        help="Output directory for all files.")
    args = ap.parse_args()

    # ── device ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device]   {device}")
    print("[variant]  Rössler X-coupled  (canonical chaotic a=0.2 b=0.2 c=5.7)")

    # ── load DTI-og.mat and build Laplacian ───────────────────────────────
    L_gpu, n_dti = load_dti_laplacian(args.dti_path, device)

    # ── params — n_osc set from actual matrix size ───────────────────────
    p = RosslerParams(
        coupling = args.coupling,
        n_osc    = n_dti,
    )
    print(
        f"[config]   grid={args.grid_n}²  N={p.n_osc} (from DTI-og.mat)  "
        f"coupling={p.coupling}  dt={p.dt}\n"
        f"           a={p.a}  b={p.b}  c={p.c}  "
        f"grid=[{p.grid_lo}, {p.grid_hi}]\n"
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
    ax     = np.linspace(p.grid_lo, p.grid_hi, m, dtype=np.float32)
    Xg, Yg = np.meshgrid(ax, ax)

    # ── build IC batch ───────────────────────────────────────────────────
    # The Rössler attractor is NOT globally attracting: an IC off the basin
    # escapes to infinity (Z→∞). Seed every node on the attractor by first
    # integrating one isolated (uncoupled) Rössler forward, then perturb the
    # two slice nodes' X component across the [-9, 9] grid — which stays
    # within the attractor, so no IC starts off-basin.
    seed  = torch.tensor([1.0, 1.0, 1.0], device=device).reshape(1, 1, 3)
    L0    = torch.zeros(1, 1, device=device)
    p_iso = RosslerParams(coupling=0.0, n_osc=1)
    p_iso.dt = p.dt
    for _ in range(p.steps_transient):
        seed = rk4_step_batched(seed, L0, p_iso)
    seed = seed.reshape(3)                      # (x*, y*, z*) on the attractor

    x0  = torch.zeros((B, N, 3), dtype=torch.float32, device=device)
    x0 += 0.05 * torch.randn_like(x0)
    x0 += seed                                   # broadcast (3,) over (B, N, 3)

    x0[:, p.slice_node_x, 0] = torch.tensor(Xg.ravel(), device=device)
    x0[:, p.slice_node_y, 0] = torch.tensor(Yg.ravel(), device=device)

    # ── Phase 1: burn-in ─────────────────────────────────────────────────
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
    if str(args.k_clusters).lower() == "auto":
        print("[cluster]  dynamic K selection "
              "(Elbow + BIC + Silhouette), k=2..15 ...")
        sel = select_optimal_clusters(
            vectors, k_min=2, k_max=15, criterion=args.cluster_criterion)
        k_used = sel.best_k
        labels = sel.labels.reshape(m, m).astype(np.int32)
        print(f"[cluster]  elbow={sel.k_elbow}  BIC={sel.k_bic}  "
              f"silhouette={sel.k_silhouette}  ->  K={k_used} "
              f"(criterion={sel.criterion})")
        plot_selection(
            sel,
            get_plot_path("rossler_vps_clustering", "cluster_selection.png", args.outdir),
            title=f"DTI_A Rössler basins  coupling={p.coupling}")
        print("[saved]    cluster_selection.png")
    else:
        k_used = int(args.k_clusters)
        print(f"[kmeans]   GPU Lloyd's  k={k_used} (user-forced) ...")
        labels_gpu = kmeans_gpu(vectors_gpu, k=k_used)
        labels     = labels_gpu.cpu().numpy().reshape(m, m)
        print(f"[kmeans]   {k_used} basin labels assigned")

    # ── Basin map ────────────────────────────────────────────────────────
    fig_bm, ax_bm = plt.subplots(figsize=(7, 6))
    im = ax_bm.imshow(
        labels, origin="lower", cmap="tab20",
        extent=[p.grid_lo, p.grid_hi, p.grid_lo, p.grid_hi],
        interpolation="nearest",
    )
    ax_bm.set_xlabel(f"Node {p.slice_node_x}  X perturbation")
    ax_bm.set_ylabel(f"Node {p.slice_node_y}  X perturbation")
    ax_bm.set_title(
        f"K-Means Basin Map — Rössler X-coupled (k={k_used})\n"
        f"N={p.n_osc} coupling={p.coupling} grid={m}²"
    )
    plt.colorbar(im, ax=ax_bm, label="Basin label")
    plt.tight_layout()
    plt.savefig(get_plot_path("rossler_vps_clustering", "basin_map_kmeans.png", args.outdir), dpi=150)
    plt.close(fig_bm)
    print("[saved]    basin_map_kmeans.png")

    # ── Basin boundary + box-counting ────────────────────────────────────
    boundary  = extract_boundary(labels)
    r, n      = boxcount_2d_gpu(boundary, device)
    D_f, r_sq = fractal_dimension(r, n)
    print(f"[fractal]  D_f = {D_f:.4f}  (R² = {r_sq:.4f})")
    print(
        "           D_f ≈ 1.0 → smooth curve\n"
        "           D_f ≈ 2.0 → space-filling\n"
        "           Expected chimera range: ~1.2–1.8"
    )

    fig_bd, ax_bd = plt.subplots(figsize=(7, 6))
    ax_bd.imshow(
        boundary, origin="lower", cmap="binary",
        extent=[p.grid_lo, p.grid_hi, p.grid_lo, p.grid_hi],
        interpolation="nearest",
    )
    ax_bd.set_xlabel(f"Node {p.slice_node_x}  X perturbation")
    ax_bd.set_ylabel(f"Node {p.slice_node_y}  X perturbation")
    ax_bd.set_title(
        f"Basin Boundary — Rössler X-coupled\n"
        f"N={p.n_osc}  coupling={p.coupling}  grid={m}²"
    )
    plt.tight_layout()
    plt.savefig(get_plot_path("rossler_vps_clustering", "basin_boundary.png", args.outdir), dpi=150)
    plt.close(fig_bd)
    print("[saved]    basin_boundary.png")

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
    ax_bc.set_title("Box-Counting — Rössler X-coupled")
    ax_bc.legend()
    ax_bc.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_plot_path("rossler_vps_clustering", "boxcount_loglog.png", args.outdir), dpi=150)
    plt.close(fig_bc)
    print("[saved]    boxcount_loglog.png")

    # ── Boxdiv2 synthetic fractal ────────────────────────────────────────
    print(f"[boxdiv2]  p={args.boxdiv_p} ...")
    seed = np.ones((256, 256), dtype=bool)
    frac = boxdiv2(seed, p=args.boxdiv_p)
    fig_bx, ax_bx = plt.subplots(figsize=(6, 6))
    ax_bx.imshow(frac, origin="lower", cmap="gray", interpolation="nearest")
    ax_bx.set_title(f"Synthetic Fractal (boxdiv2, p={args.boxdiv_p})")
    ax_bx.axis("off")
    plt.tight_layout()
    plt.savefig(get_plot_path("rossler_vps_clustering", "boxdiv2_synthetic.png", args.outdir), dpi=150)
    plt.close(fig_bx)
    print("[saved]    boxdiv2_synthetic.png")

    # ── Edge rewiring demo on DTI adjacency ─────────────────────────────
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
    plt.savefig(get_plot_path("rossler_vps_clustering", "rewired_adjacency.png", args.outdir), dpi=150)
    plt.close(fig_rw)
    print("[saved]    rewired_adjacency.png")

    # ── Save ─────────────────────────────────────────────────────────────
    # Embed the LIVE simulation configuration so the replotter renders the
    # actual coupling and swept-node identities rather than stale defaults.
    config = dict(
        oscillator   = "rossler",
        coupling     = float(p.coupling),
        slice_node_x = int(p.slice_node_x),
        slice_node_y = int(p.slice_node_y),
        n_osc        = int(p.n_osc),
        grid_n       = int(m),
        k_clusters   = int(k_used),
        grid_lo      = float(p.grid_lo),
        grid_hi      = float(p.grid_hi),
        a            = float(p.a),
        b            = float(p.b),
        c            = float(p.c),
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
