#!/usr/bin/env python3
"""
filename: lorenz_vps_clustering_zcoupled.py
------------------------
Author : anitoti
Date   : summer 2026
Project: Nimble Brain — Clarkson University REU (Dr. Jeremie Fish)

Identical pipeline to xcoupled variant EXCEPT:

COUPLING: Z-component only.
  H_z = [0 0 0; 0 0 0; 0 0 1] — selects Z (index 2) from each oscillator.
  Contrast with MATLAB X-coupled:
  [Page 28, full_.m_script.pdf: "H = [0 0 0; 0 1 0; 0 0 0]; gelLH = gel*kron(L,H)"]
  Here we use H_z instead: dZ_i -= gel * (L @ Z)_i

  Physical interpretation:
    X-coupling: diffusive coupling through the fast variable (butterfly wings).
    Z-coupling: diffusive coupling through the slow variable (height/energy).
    Z is bounded below (Z >= 0 on attractor), non-negative, lower-frequency.
    Expect qualitatively different basin geometry and D_f values.

IC SLICE: grid perturbs Z component of slice nodes (index 2),
  not X (index 0), so the 2D slice is in Z-Z space.

Run:
    python3 pipeline/lorenz_sweep_z.py
    python3 pipeline/lorenz_sweep_z.py \
        --grid-n 64 --coupling 0.5 \
        --dti-path data/DTI-og.mat \
        --outdir /home/atotilca/pythongpu/data/
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
from tqdm import tqdm

from pythongpu.utils import get_plot_path
from pythongpu.networks.static_adjacency import load_dti_laplacian, rewire_edges
from pythongpu.processing.box_counting import boxcount_2d_gpu, fractal_dimension, boxdiv2, extract_boundary


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DTI LOADER  (identical to X-coupled)
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class LorenzParams:
    """
    [Page 28, full_.m_script.pdf:
     "gel = 0.5 ... T = 2000 ... T1 = 10000 ... dt = 0.05"]
    """
    sigma           : float = 10.0
    rho             : float = 28.0
    beta            : float = 8.0 / 3.0
    coupling        : float = 0.5
    dt              : float = 0.05
    t_transient     : float = 100.0
    tmax            : float = 500.0
    slice_node_x    : int   = 28
    slice_node_y    : int   = 79
    n_osc           : int   = 0
    steps_transient : int   = field(init=False, repr=False)
    steps_record    : int   = field(init=False, repr=False)

    def __post_init__(self):
        self.steps_transient = int(self.t_transient / self.dt)
        self.steps_record    = int(self.tmax        / self.dt)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PHYSICS — Z-coupled
# ═══════════════════════════════════════════════════════════════════════════
def lorenz_rhs_batched(
    x     : torch.Tensor,
    L_gpu : torch.Tensor,
    p     : LorenzParams,
) -> torch.Tensor:
    """
    Batched RHS — DTI Laplacian coupling on Z (index 2) only.

    Intrinsic Lorenz-63:
        dX = sigma*(Y - X)
        dY = X*(rho - Z) - Y
        dZ = X*Y - beta*Z

    Coupling (Z only):
        dZ_i -= gel * (L @ Z)_i

    Contrast with X-coupled MATLAB reference:
    [Page 28, full_.m_script.pdf:
     "H = [0 0 0; 0 1 0; 0 0 0]; gelLH = gel*kron(L,H)"]
    Here H_z = diag(0,0,1) is used instead.

    Note on Z dynamics:
        Z is the slow, non-negative variable (Z >= 0 on attractor).
        Coupling through Z creates fundamentally different synchronisation
        geometry compared to X-coupling — expect different D_f values.
    """
    dX = p.sigma * (x[..., 1] - x[..., 0])
    dY = x[..., 0] * (p.rho - x[..., 2]) - x[..., 1]
    dZ = x[..., 0] * x[..., 1] - p.beta * x[..., 2]

    f = torch.stack([dX, dY, dZ], dim=-1)          # (B, N, 3)

    # Coupling applied to Z component only — index 2
    f[..., 2] -= p.coupling * (x[..., 2] @ L_gpu.T)

    return f


@torch.no_grad()
def rk4_step_batched(
    x     : torch.Tensor,
    L_gpu : torch.Tensor,
    p     : LorenzParams,
) -> torch.Tensor:
    k1 = lorenz_rhs_batched(x,                    L_gpu, p)
    k2 = lorenz_rhs_batched(x + 0.5 * p.dt * k1, L_gpu, p)
    k3 = lorenz_rhs_batched(x + 0.5 * p.dt * k2, L_gpu, p)
    k4 = lorenz_rhs_batched(x +       p.dt * k3,  L_gpu, p)
    return x + (p.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  STREAMING VPS — Z-coherence feature
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_sweep_streaming(
    x0     : torch.Tensor,
    L_gpu  : torch.Tensor,
    p      : LorenzParams,
    device : torch.device,
) -> torch.Tensor:
    """
    Welford VPS accumulation. tau_z replaces tau_x:
        tau_z = mean|dZ| / std|dZ|   — phase-coherence proxy for Z component.

    [Page 37, full_.m_script.pdf:
     "KmeansMat = zeros(NumCond, 2*size(Vec,1))"]
    """
    B, N, _ = x0.shape

    iu   = torch.triu_indices(N, N, offset=1, device=device)
    i, j = iu[0], iu[1]
    C    = i.shape[0]

    count   = 0
    mean_dz = torch.zeros(B, C, device=device)   # Z component (index 2)
    M2_dz   = torch.zeros(B, C, device=device)
    mean_L  = torch.zeros(B, C, device=device)

    for _ in tqdm(range(p.steps_record), desc="streaming"):
        x0 = rk4_step_batched(x0, L_gpu, p)

        diff   = x0[:, i, :] - x0[:, j, :]       # (B, C, 3)
        dz_abs = diff[..., 2].abs()               # (B, C) — Z component
        L_val  = torch.linalg.norm(diff, dim=-1)  # (B, C)

        count  += 1
        delta   = dz_abs - mean_dz
        mean_dz = mean_dz + delta / count
        M2_dz   = M2_dz   + delta * (dz_abs - mean_dz)

        mean_L  = mean_L + (L_val - mean_L) / count

    var_dz = M2_dz / max(count - 1, 1)
    tau_z  = mean_dz / (var_dz.sqrt() + 1e-8)    # Z coherence proxy

    tau_z_std = (
        (tau_z  - tau_z.mean(dim=0,  keepdim=True))
        / (tau_z.std(dim=0,  keepdim=True) + 1e-8)
    )
    mean_L_std = (
        (mean_L - mean_L.mean(dim=0, keepdim=True))
        / (mean_L.std(dim=0, keepdim=True) + 1e-8)
    )

    return torch.cat([tau_z_std, mean_L_std], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  GPU K-MEANS  (identical to X-coupled)
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def kmeans_gpu(
    X      : torch.Tensor,
    k      : int,
    n_iter : int   = 300,
    tol    : float = 1e-4,
) -> tuple[torch.Tensor, float]:
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
# 6.  BOX-COUNTING  (identical to X-coupled)
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()




# ═══════════════════════════════════════════════════════════════════════════
# 7.  BOXDIV2  (identical)
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 8.  EDGE REWIRING  (identical)
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 9.  BOUNDARY  (identical)
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():

    ap = argparse.ArgumentParser(
        description="DTI-coupled Lorenz basin map — Z coupling"
    )
    ap.add_argument("--grid-n",     type=int,   default=128)
    ap.add_argument("--coupling",   type=float, default=0.5)
    ap.add_argument("--k-clusters", type=int,   default=5)
    ap.add_argument("--boxdiv-p",   type=float, default=0.7)
    ap.add_argument("--rewire-n",   type=int,   default=5)
    ap.add_argument("--dti-path",   type=str,   default="data/DTI-og.mat")
    ap.add_argument("--outdir",     type=str,   default=".")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[time]     {timestamp}")
    print("[variant]  Z-coupled  (f[...,2] -= gel * L @ Z)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device]   {device}")

    L_gpu, n_dti = load_dti_laplacian(args.dti_path, device)

    p = LorenzParams(coupling=args.coupling, n_osc=n_dti)
    print(
        f"[config]   grid={args.grid_n}²  N={p.n_osc}  "
        f"coupling={p.coupling}  dt={p.dt}\n"
        f"           transient={p.t_transient}s ({p.steps_transient} steps)  "
        f"record={p.tmax}s ({p.steps_record} steps)\n"
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

    ax_grid = np.linspace(-9.0, 9.0, m, dtype=np.float32)
    Xg, Yg  = np.meshgrid(ax_grid, ax_grid)

    x0  = torch.ones((B, N, 3), dtype=torch.float32, device=device)
    x0 += 0.05 * torch.randn_like(x0)

    # ── Z-coupled: perturb Z component (index 2) of slice nodes ─────────
    # X-coupled used index 0; here we use index 2 to stay consistent
    # with the coupling variable. The 2D slice is now in Z-Z space.
    x0[:, p.slice_node_x, 2] = torch.tensor(Xg.ravel(), device=device)
    x0[:, p.slice_node_y, 2] = torch.tensor(Yg.ravel(), device=device)

    print(f"[burn-in]  {p.steps_transient} steps ...")
    for _ in tqdm(range(p.steps_transient), desc="burn-in"):
        x0 = rk4_step_batched(x0, L_gpu, p)

    print(f"[stream]   {p.steps_record} steps ...")
    vectors_gpu = run_sweep_streaming(x0, L_gpu, p, device)
    vectors     = vectors_gpu.cpu().numpy()
    print(f"[vps]      shape = {vectors.shape}")

    print(f"[kmeans]   k={args.k_clusters} ...")
    labels_gpu, cluster_inertia = kmeans_gpu(vectors_gpu, k=args.k_clusters)
    labels = labels_gpu.cpu().numpy().reshape(m, m)
    print(f"[kmeans]   inertia = {cluster_inertia:.4f}")

    # ── Elbow — FIX: use returned inertia directly ───────────────────────
    print("[elbow]    k=2..15 ...")
    inertias = []
    for k in tqdm(range(2, 16), desc="elbow"):
        _, _inertia = kmeans_gpu(vectors_gpu, k=k, n_iter=100)
        inertias.append(_inertia)

    fig_el, ax_el = plt.subplots(figsize=(6, 4))
    ax_el.plot(range(2, 16), inertias, "o-", color="steelblue")
    ax_el.set_xlabel("k")
    ax_el.set_ylabel("Inertia")
    ax_el.set_title("Elbow Method — Z-coupled")
    ax_el.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_plot_path("lorenz_vps_clustering_z", f"elbow_curve_{timestamp}.png", args.outdir), dpi=150)
    plt.close(fig_el)
    print(f"[saved]    elbow_curve_{timestamp}.png")

    fig_bm, ax_bm = plt.subplots(figsize=(7, 6))
    im = ax_bm.imshow(
        labels, origin="lower", cmap="tab20",
        extent=[-9.0, 9.0, -9.0, 9.0], interpolation="nearest",
    )
    ax_bm.set_xlabel(f"Node {p.slice_node_x}  Z perturbation")
    ax_bm.set_ylabel(f"Node {p.slice_node_y}  Z perturbation")
    ax_bm.set_title(
        f"K-Means Basin Map — Z-coupled (k={args.k_clusters})\n"
        f"N={p.n_osc}  coupling={p.coupling}  grid={m}²"
    )
    plt.colorbar(im, ax=ax_bm, label="Basin label")
    plt.tight_layout()
    plt.savefig(get_plot_path("lorenz_vps_clustering_z", f"basin_map_kmeans_{timestamp}.png", args.outdir), dpi=150)
    plt.close(fig_bm)
    print(f"[saved]    basin_map_kmeans_{timestamp}.png")

    boundary  = extract_boundary(labels)
    r, n      = boxcount_2d_gpu(boundary, device)
    D_f, r_sq = fractal_dimension(r, n)
    print(f"[fractal]  D_f = {D_f:.4f}  (R² = {r_sq:.4f})")
    print(
        "           D_f ≈ 1.0 → smooth  |  "
        "D_f ≈ 2.0 → space-filling  |  "
        "chimera range: ~1.2–1.8"
    )

    fig_bd, ax_bd = plt.subplots(figsize=(7, 6))
    ax_bd.imshow(
        boundary, origin="lower", cmap="binary",
        extent=[-9.0, 9.0, -9.0, 9.0], interpolation="nearest",
    )
    ax_bd.set_xlabel(f"Node {p.slice_node_x}  Z perturbation")
    ax_bd.set_ylabel(f"Node {p.slice_node_y}  Z perturbation")
    ax_bd.set_title(
        f"Basin Boundary — Z-coupled\n"
        f"N={p.n_osc}  coupling={p.coupling}  grid={m}²"
    )
    plt.tight_layout()
    plt.savefig(get_plot_path("lorenz_vps_clustering_z", f"basin_boundary_{timestamp}.png", args.outdir), dpi=150)
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
    ax_bc.set_title("Box-Counting — Z-coupled")
    ax_bc.legend()
    ax_bc.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_plot_path("lorenz_vps_clustering_z", f"boxcount_loglog_{timestamp}.png", args.outdir), dpi=150)
    plt.close(fig_bc)
    print(f"[saved]    boxcount_loglog_{timestamp}.png")

    print(f"[boxdiv2]  p={args.boxdiv_p} ...")
    seed = np.ones((256, 256), dtype=bool)
    frac = boxdiv2(seed, p=args.boxdiv_p)
    fig_bx, ax_bx = plt.subplots(figsize=(6, 6))
    ax_bx.imshow(frac, origin="lower", cmap="gray", interpolation="nearest")
    ax_bx.set_title(f"Synthetic Fractal (boxdiv2, p={args.boxdiv_p})")
    ax_bx.axis("off")
    plt.tight_layout()
    plt.savefig(get_plot_path("lorenz_vps_clustering_z", "boxdiv2_synthetic.png", args.outdir), dpi=150)
    plt.close(fig_bx)
    print("[saved]    boxdiv2_synthetic.png")

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
    plt.savefig(get_plot_path("lorenz_vps_clustering_z", "rewired_adjacency.png", args.outdir), dpi=150)
    plt.close(fig_rw)
    print("[saved]    rewired_adjacency.png")

    # ── Save — FIX: coupling-tagged filename, coupling saved in array ────
    out_name = f"basin_data_c{args.coupling:.2f}_{timestamp}.npz"
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