#!/usr/bin/env python3
"""
filename: rossler_vps_clustering.py
------------------------
Author : anitoti
Date   : summer 2026
Project: Nimble Brain — Clarkson University REU (Dr. Jeremie Fish)

Rössler oscillator variant of the VPS basin-mapping pipeline.

Physics:
    dX_i = -Y_i - Z_i - gel*(L@X)_i
    dY_i =  X_i + a*Y_i
    dZ_i =  b + Z_i*(X_i - c)

Parameters from paper_pounder.pdf:
    a=0.165, f(=b)=0.2, c=10, omega=0.97
    x_rms = 10.7177  [Page 13-14, paper_pounder.pdf]

Coupling component: X only.
    "we will rely on the oscillatory behavior of the x-component
     of the Rössler system ... the z-component, however, is not
     oscillatory, so it would not make a good choice."
    [Page 9-10, paper_pounder.pdf]

VPS feature: tau_x = mean|dX| / std|dX| — phase coherence proxy.
    Same as Lorenz X-coupled variant. Z is explicitly excluded
    per paper_pounder.pdf guidance.

IC slice grid:
    X component of two slice nodes swept over [-20, 20] x [-20, 20].
    Wider than Lorenz [-9,9] because Rössler attractor is larger
    (x_rms ~ 10.7 [Page 13-14, paper_pounder.pdf]).

Run:
    python3 pipeline/rossler_sweep.py
    python3 pipeline/rossler_sweep.py \
        --grid-n 64 --coupling 0.5 \
        --dti-path data/DTI_A.mat \
        --outdir /home/atotilca/pythongpu/data/rossler/
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from tqdm import tqdm

from pythongpu.utils import get_plot_path
from pythongpu.networks.static_adjacency import load_dti_laplacian, rewire_edges
from pythongpu.processing.box_counting import boxcount_2d_gpu, fractal_dimension, extract_boundary


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DTI LOADER  (identical to Lorenz scripts)
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class RosslerParams:
    """
    Rössler parameters from paper_pounder.pdf.

    a, b, c:
        Standard Rössler parameters.
        "a = 0.165, f(=b) = 0.2, c = 10"
        [Page 13-14, paper_pounder.pdf]

    x_rms = 10.7177:
        RMS of X on the attractor — used to set IC grid range.
        Grid swept over [-2*x_rms, 2*x_rms] ~ [-21.4, 21.4].
        [Page 13-14, paper_pounder.pdf]

    coupling:
        gel — DTI Laplacian coupling strength on X component.
        "Two Rösslers phase and fully synchronize monotonically"
        [Page 13-14, paper_pounder.pdf]
        sweep 0.0 -> 1.0 to map basin structure across this transition.

    dt:
        Rössler period ~ 2*pi/omega ~ 6.5s at omega=0.97.
        dt=0.05 gives ~130 steps/cycle — adequate for RK4.

    t_transient:
        200s burn-in — longer than Lorenz (100s) because Rössler
        has slower dynamics (omega ~ 1 rad/s vs Lorenz fast chaos).

    tmax:
        1000s record — ~154 full Rössler cycles at omega=0.97.
        Lorenz used 500s; Rössler needs more cycles for VPS convergence
        because it is slower and more periodic.
    """
    a               : float = 0.165      # [Page 13-14, paper_pounder.pdf]
    b               : float = 0.2        # f in paper notation
    c               : float = 10.0       # [Page 13-14, paper_pounder.pdf]
    x_rms           : float = 10.7177    # [Page 13-14, paper_pounder.pdf]
    coupling        : float = 0.5
    dt              : float = 0.05
    t_transient     : float = 200.0
    tmax            : float = 1000.0
    slice_node_x    : int   = 28
    slice_node_y    : int   = 79
    n_osc           : int   = 0
    steps_transient : int   = field(init=False, repr=False)
    steps_record    : int   = field(init=False, repr=False)

    def __post_init__(self):
        self.steps_transient = int(self.t_transient / self.dt)
        self.steps_record    = int(self.tmax        / self.dt)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PHYSICS — Rössler RHS, X-coupled
# ═══════════════════════════════════════════════════════════════════════════
def rossler_rhs_batched(
    x     : torch.Tensor,
    L_gpu : torch.Tensor,
    p     : RosslerParams,
) -> torch.Tensor:
    """
    Batched RHS of DTI-coupled Rössler network.

    Args
    ----
    x     : (B, N, 3)  — state; B=batch, N=nodes, 3=[X,Y,Z]
    L_gpu : (N, N)     — graph Laplacian
    p     : RosslerParams

    Returns
    -------
    f : (B, N, 3)

    Physics
    -------
    Intrinsic Rössler:
        dX =  -Y - Z
        dY =   X + a*Y
        dZ =   b + Z*(X - c)

    Coupling on X only:
        dX_i -= gel * (L @ X)_i

    Rationale for X-coupling:
        "we will rely on the oscillatory behavior of the x-component
         of the Rössler system ... the z-component, however, is not
         oscillatory, so it would not make a good choice."
        [Page 9-10, paper_pounder.pdf]

    Z is a spiking/slow variable on the Rössler attractor — coupling
    through it would not drive phase synchronisation meaningfully.
    """
    X = x[..., 0]   # (B, N)
    Y = x[..., 1]   # (B, N)
    Z = x[..., 2]   # (B, N)

    dX = -Y - Z                         # (B, N)
    dY =  X + p.a * Y                   # (B, N)
    dZ =  p.b + Z * (X - p.c)          # (B, N)

    f = torch.stack([dX, dY, dZ], dim=-1)   # (B, N, 3)

    # DTI coupling on X only — index 0
    # [Page 9-10, paper_pounder.pdf: x-component is oscillatory]
    f[..., 0] -= p.coupling * (X @ L_gpu.T)

    return f


@torch.no_grad()
def rk4_step_batched(
    x     : torch.Tensor,
    L_gpu : torch.Tensor,
    p     : RosslerParams,
) -> torch.Tensor:
    """RK4 step. @no_grad saves ~35% memory vs autograd."""
    k1 = rossler_rhs_batched(x,                    L_gpu, p)
    k2 = rossler_rhs_batched(x + 0.5 * p.dt * k1, L_gpu, p)
    k3 = rossler_rhs_batched(x + 0.5 * p.dt * k2, L_gpu, p)
    k4 = rossler_rhs_batched(x +       p.dt * k3,  L_gpu, p)
    return x + (p.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  STREAMING VPS — Welford, X coherence feature
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_sweep_streaming(
    x0     : torch.Tensor,
    L_gpu  : torch.Tensor,
    p      : RosslerParams,
    device : torch.device,
) -> torch.Tensor:
    """
    Welford online VPS accumulation.

    Feature choice: tau_x = mean|dX| / std|dX|
        X is the oscillatory component of Rössler.
        [Page 9-10, paper_pounder.pdf]
        Identical formula to Lorenz X-coupled variant.

    Second feature: mean_L = mean pairwise Euclidean distance.
        Uses full (X,Y,Z) state — captures overall separation.

    Both blocks independently standardised before concatenation.
    [Page 45, fractal_basins_as_mechanism_for_the_nimble_brain_tex_pdf.pdf:
     "To construct the VPS, we use β=1 in Equation (12)"]
    """
    B, N, _ = x0.shape

    iu   = torch.triu_indices(N, N, offset=1, device=device)
    i, j = iu[0], iu[1]
    C    = i.shape[0]

    count   = 0
    mean_dx = torch.zeros(B, C, device=device)
    M2_dx   = torch.zeros(B, C, device=device)
    mean_L  = torch.zeros(B, C, device=device)

    for _ in tqdm(range(p.steps_record), desc="streaming"):
        x0 = rk4_step_batched(x0, L_gpu, p)

        diff   = x0[:, i, :] - x0[:, j, :]       # (B, C, 3)
        dx_abs = diff[..., 0].abs()               # (B, C) — X only
        L_val  = torch.linalg.norm(diff, dim=-1)  # (B, C)

        count  += 1
        delta   = dx_abs - mean_dx
        mean_dx = mean_dx + delta / count
        M2_dx   = M2_dx   + delta * (dx_abs - mean_dx)
        mean_L  = mean_L  + (L_val - mean_L) / count

    var_dx = M2_dx / max(count - 1, 1)
    tau_x  = mean_dx / (var_dx.sqrt() + 1e-8)

    tau_x_std = (
        (tau_x  - tau_x.mean(dim=0,  keepdim=True))
        / (tau_x.std(dim=0,  keepdim=True) + 1e-8)
    )
    mean_L_std = (
        (mean_L - mean_L.mean(dim=0, keepdim=True))
        / (mean_L.std(dim=0, keepdim=True) + 1e-8)
    )

    return torch.cat([tau_x_std, mean_L_std], dim=-1)   # (B, 2*C)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  GPU K-MEANS
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def kmeans_gpu(
    X      : torch.Tensor,
    k      : int,
    n_iter : int   = 300,
    tol    : float = 1e-4,
) -> tuple[torch.Tensor, float]:
    """Lloyd's k-means on GPU. Returns (labels, inertia)."""
    B, D = X.shape
    Xn   = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-8)

    idx       = torch.randperm(B, device=X.device)[:k]
    centroids = Xn[idx].clone()
    labels    = torch.zeros(B, dtype=torch.long, device=X.device)
    dist      = torch.zeros(B, k, device=X.device)

    for _ in range(n_iter):
        dist = (
              Xn.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * (Xn @ centroids.T)
            + centroids.pow(2).sum(dim=1).unsqueeze(0)
        )
        new_labels    = dist.argmin(dim=1)
        new_centroids = torch.zeros_like(centroids)
        counts        = torch.zeros(k, device=X.device)

        new_centroids.scatter_add_(
            0, new_labels.unsqueeze(1).expand(B, D), Xn
        )
        counts.scatter_add_(0, new_labels, torch.ones(B, device=X.device))
        new_centroids /= counts.clamp(min=1).unsqueeze(1)

        shift     = (new_centroids - centroids).pow(2).sum().sqrt()
        centroids = new_centroids
        labels    = new_labels

        if shift < tol:
            break

    inertia = dist[torch.arange(B, device=X.device), labels].sum().item()
    return labels, inertia


# ═══════════════════════════════════════════════════════════════════════════
# 6.  BOX-COUNTING
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()




# ═══════════════════════════════════════════════════════════════════════════
# 7.  BOUNDARY EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 8.  EDGE REWIRING
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():

    ap = argparse.ArgumentParser(
        description="DTI-coupled Rössler basin map — X coupling"
    )
    ap.add_argument("--grid-n",     type=int,   default=128,
        help="Points per axis. 64→4096 ICs.")
    ap.add_argument("--coupling",   type=float, default=0.5,
        help="gel — Laplacian coupling strength.")
    ap.add_argument("--k-clusters", type=int,   default=5,
        help="K-means k.")
    ap.add_argument("--rewire-n",   type=int,   default=5,
        help="Edges to rewire.")
    ap.add_argument("--dti-path",   type=str,   default="data/DTI_A.mat")
    ap.add_argument("--outdir",     type=str,   default=".")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[time]     {timestamp}")
    print(f"[variant]  Rössler X-coupled  (f[...,0] -= gel * L @ X)")
    print(f"           a={0.165}  b={0.2}  c={10}  "
          f"[Page 13-14, paper_pounder.pdf]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device]   {device}")

    L_gpu, n_dti = load_dti_laplacian(args.dti_path, device)

    p = RosslerParams(coupling=args.coupling, n_osc=n_dti)
    print(
        f"[config]   grid={args.grid_n}²  N={p.n_osc}  "
        f"coupling={p.coupling}  dt={p.dt}\n"
        f"           transient={p.t_transient}s ({p.steps_transient} steps)  "
        f"record={p.tmax}s ({p.steps_record} steps)\n"
        f"           x_rms={p.x_rms}  grid=[-{2*p.x_rms:.1f}, {2*p.x_rms:.1f}]"
        f"  [Page 13-14, paper_pounder.pdf]\n"
        f"           slice: node_x={p.slice_node_x}  node_y={p.slice_node_y}"
    )

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m       = args.grid_n
    B       = m * m
    N       = p.n_osc
    C_pairs = N * (N - 1) // 2

    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        print(
            f"[memory]   GPU free={free/1e9:.1f} GB  |  "
            f"streaming ~{3*B*C_pairs*4/1e9:.2f} GB"
        )

    # ── IC grid — wider than Lorenz: [-2*x_rms, 2*x_rms] ────────────────
    # x_rms = 10.7177 [Page 13-14, paper_pounder.pdf]
    # Lorenz used [-9, 9]; Rössler attractor is larger so grid is wider.
    grid_range = 2.0 * p.x_rms        # ~ 21.4
    ax_grid    = np.linspace(-grid_range, grid_range, m, dtype=np.float32)
    Xg, Yg     = np.meshgrid(ax_grid, ax_grid)

    # ── build IC batch ────────────────────────────────────────────────────
    # Rössler attractor lives roughly at X~0, Y~0, Z~0 centre.
    # Initialise near (1,0,0) + small noise, then perturb slice nodes.
    x0    = torch.zeros((B, N, 3), dtype=torch.float32, device=device)
    x0   += 0.1 * torch.randn_like(x0)
    x0[..., 0] += 1.0                  # nudge X off origin (avoid Z blow-up)

    # Perturb X component of slice nodes
    # "x-component is oscillatory" [Page 9-10, paper_pounder.pdf]
    x0[:, p.slice_node_x, 0] = torch.tensor(Xg.ravel(), device=device)
    x0[:, p.slice_node_y, 0] = torch.tensor(Yg.ravel(), device=device)

    # ── burn-in ──────────────────────────────────────────────────────────
    print(f"[burn-in]  {p.steps_transient} steps ...")
    for _ in tqdm(range(p.steps_transient), desc="burn-in"):
        x0 = rk4_step_batched(x0, L_gpu, p)

    # ── streaming VPS ────────────────────────────────────────────────────
    print(f"[stream]   {p.steps_record} steps ...")
    vectors_gpu = run_sweep_streaming(x0, L_gpu, p, device)
    vectors     = vectors_gpu.cpu().numpy()
    print(f"[vps]      shape = {vectors.shape}")

    # ── k-means ──────────────────────────────────────────────────────────
    print(f"[kmeans]   k={args.k_clusters} ...")
    labels_gpu, cluster_inertia = kmeans_gpu(vectors_gpu, k=args.k_clusters)
    labels = labels_gpu.cpu().numpy().reshape(m, m)
    print(f"[kmeans]   inertia = {cluster_inertia:.4f}")

    # ── elbow ─────────────────────────────────────────────────────────────
    print("[elbow]    k=2..15 ...")
    inertias = []
    for k in tqdm(range(2, 16), desc="elbow"):
        _, _inertia = kmeans_gpu(vectors_gpu, k=k, n_iter=100)
        inertias.append(_inertia)

    fig_el, ax_el = plt.subplots(figsize=(6, 4))
    ax_el.plot(range(2, 16), inertias, "o-", color="steelblue")
    ax_el.set_xlabel("k")
    ax_el.set_ylabel("Inertia")
    ax_el.set_title("Elbow Method — Rössler X-coupled")
    ax_el.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_plot_path("rossler_vps_clustering", f"elbow_curve_{timestamp}.png", args.outdir), dpi=150)
    plt.close(fig_el)
    print(f"[saved]    elbow_curve_{timestamp}.png")

    # ── basin map ─────────────────────────────────────────────────────────
    fig_bm, ax_bm = plt.subplots(figsize=(7, 6))
    im = ax_bm.imshow(
        labels, origin="lower", cmap="tab20",
        extent=[-grid_range, grid_range, -grid_range, grid_range],
        interpolation="nearest",
    )
    ax_bm.set_xlabel(f"Node {p.slice_node_x}  X perturbation")
    ax_bm.set_ylabel(f"Node {p.slice_node_y}  X perturbation")
    ax_bm.set_title(
        f"K-Means Basin Map — Rössler X-coupled (k={args.k_clusters})\n"
        f"N={p.n_osc}  coupling={p.coupling}  grid={m}²"
    )
    plt.colorbar(im, ax=ax_bm, label="Basin label")
    plt.tight_layout()
    plt.savefig(get_plot_path("rossler_vps_clustering", f"basin_map_kmeans_{timestamp}.png", args.outdir), dpi=150)
    plt.close(fig_bm)
    print(f"[saved]    basin_map_kmeans_{timestamp}.png")

    # ── boundary + box-counting ───────────────────────────────────────────
    boundary  = extract_boundary(labels)
    r, n      = boxcount_2d_gpu(boundary, device)
    D_f, r_sq = fractal_dimension(r, n, r_min=2, r_max=90)
    print(f"[fractal]  D_f = {D_f:.4f}  (R² = {r_sq:.4f})")
    print(
        f"           D_f ≈ 1.0 → smooth  |  "
        f"D_f ≈ 2.0 → space-filling  |  "
        f"chimera range: ~1.2–1.8"
    )

    fig_bd, ax_bd = plt.subplots(figsize=(7, 6))
    ax_bd.imshow(
        boundary, origin="lower", cmap="binary",
        extent=[-grid_range, grid_range, -grid_range, grid_range],
        interpolation="nearest",
    )
    ax_bd.set_xlabel(f"Node {p.slice_node_x}  X perturbation")
    ax_bd.set_ylabel(f"Node {p.slice_node_y}  X perturbation")
    ax_bd.set_title(
        f"Basin Boundary — Rössler X-coupled\n"
        f"N={p.n_osc}  coupling={p.coupling}  grid={m}²"
    )
    plt.tight_layout()
    plt.savefig(get_plot_path("rossler_vps_clustering", f"basin_boundary_{timestamp}.png", args.outdir), dpi=150)
    plt.close(fig_bd)
    print(f"[saved]    basin_boundary_{timestamp}.png")

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
    plt.savefig(get_plot_path("rossler_vps_clustering", f"boxcount_loglog_{timestamp}.png", args.outdir), dpi=150)
    plt.close(fig_bc)
    print(f"[saved]    boxcount_loglog_{timestamp}.png")

    # ── rewiring ─────────────────────────────────────────────────────────
    print(f"[rewire]   {args.rewire_n} rewires ...")
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

    # ── save ─────────────────────────────────────────────────────────────
    out_name = f"basin_data_rossler_c{args.coupling:.2f}_{timestamp}.npz"
    np.savez_compressed(
        out_dir / out_name,
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
        coupling    = np.array([args.coupling]),
    )
    print(f"[saved]    {out_name}  →  {out_dir / out_name}")
    print(f"\n{'─'*55}")
    print(f"  Fractal dimension  D_f = {D_f:.4f}  (R² = {r_sq:.4f})")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()