#!/usr/bin/env python3
"""
Fine coupling sweep of the DTI-coupled Lorenz basin geometry.

Sweeps the Laplacian coupling strength K over a fine ladder in the
near-critical window (default K in [0.45, 0.65]) while perturbing a specific
pair of connectome nodes along the two axes of a 2-D initial-condition slice
(default node 73 vs node 81 of ``DTI_A.mat``). For every K it

    1. integrates the 65,536-IC slice on the GPU (RK4, transient burn-in),
    2. accumulates the VPS coherence features via Welford streaming,
    3. partitions the slice into basins with the dynamic Elbow+BIC+Silhouette
       basin-count selection (or a user-forced K), and
    4. reports the box-counting fractal dimension D_f and the grid-independent
       uncertainty exponent gamma (D_f = d - gamma).

Each K produces its own ``.npz`` record and one animation frame; the frames are
compiled into an animated ``.gif``. The swept-node identities AND the live
coupling value are encoded directly into every output filename, so archives are
self-describing:

    lorenz_basins_n73_n81_K0.4500.npz
    lorenz_basins_n73_n81_K0.4750.npz
    ...
    lorenz_basins_n73_n81_K0.45-0.65.gif

Usage
-----
    python3 -m pythongpu.pipeline.lorenz_fine_coupling_sweep
    python3 -m pythongpu.pipeline.lorenz_fine_coupling_sweep \
        --node-x 73 --node-y 81 --k-start 0.45 --k-stop 0.65 --k-step 0.025 \
        --grid-n 128 --outdir data/derivatives
    python3 -m pythongpu.pipeline.lorenz_fine_coupling_sweep --smoke
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pythongpu.networks.static_adjacency import load_dti_laplacian
from pythongpu.networks.desikan_killiany import labels_for
from pythongpu.pipeline.lorenz_sweep import (
    LorenzParams,
    rk4_step_batched,
    run_sweep_streaming,
)
from pythongpu.processing.basin_clustering import select_optimal_clusters
from pythongpu.processing.box_counting import (
    extract_boundary,
    boxcount_2d_gpu,
    fractal_dimension,
    uncertainty_exponent,
)


# ── slice construction ───────────────────────────────────────
def build_slice(p: LorenzParams, grid_n: int, grid_lo: float, grid_hi: float,
                device: torch.device):
    """
    Build the (B, N, 3) IC batch for a 2-D affine slice: the X component of
    ``p.slice_node_x`` sweeps the grid x-axis and ``p.slice_node_y`` the y-axis;
    every other state entry sits near the (1,1,1) base point with a small jitter.
    """
    ax = np.linspace(grid_lo, grid_hi, grid_n, dtype=np.float32)
    Xg, Yg = np.meshgrid(ax, ax)
    B, N = grid_n * grid_n, p.n_osc
    x0 = torch.ones((B, N, 3), dtype=torch.float32, device=device)
    x0 += 0.05 * torch.randn_like(x0)
    x0[:, p.slice_node_x, 0] = torch.tensor(Xg.ravel(), device=device)
    x0[:, p.slice_node_y, 0] = torch.tensor(Yg.ravel(), device=device)
    return x0, Xg, Yg


# ── one coupling value ───────────────────────────────────────
def observe_coupling(L_gpu, coupling: float, p_base: LorenzParams, grid_n: int,
                     grid_lo: float, grid_hi: float, k_clusters,
                     cluster_criterion: str, device: torch.device) -> dict:
    """Integrate, cluster, and quantify the basin geometry for one K."""
    p = LorenzParams(
        sigma=p_base.sigma, rho=p_base.rho, beta=p_base.beta,
        coupling=coupling, dt=p_base.dt,
        t_transient=p_base.t_transient, tmax=p_base.tmax,
        slice_node_x=p_base.slice_node_x, slice_node_y=p_base.slice_node_y,
        n_osc=p_base.n_osc,
    )
    x0, Xg, Yg = build_slice(p, grid_n, grid_lo, grid_hi, device)

    for _ in range(p.steps_transient):
        x0 = rk4_step_batched(x0, L_gpu, p)
    vectors_gpu = run_sweep_streaming(x0, L_gpu, p, device)
    vectors = vectors_gpu.cpu().numpy()

    if str(k_clusters).lower() == "auto":
        sel = select_optimal_clusters(vectors, k_min=2, k_max=12,
                                      criterion=cluster_criterion)
        k_used = sel.best_k
        labels = sel.labels.reshape(grid_n, grid_n).astype(np.int32)
    else:
        from pythongpu.pipeline.lorenz_sweep import kmeans_gpu
        k_used = int(k_clusters)
        labels = kmeans_gpu(vectors_gpu, k=k_used).cpu().numpy().reshape(grid_n, grid_n)

    boundary = extract_boundary(labels)
    r, n = boxcount_2d_gpu(boundary, device)
    D_f, r_sq = fractal_dimension(r, n)
    ue = uncertainty_exponent(labels, d=2)

    if device.type == "cuda":
        torch.cuda.synchronize()
    return dict(
        coupling=float(coupling), Xg=Xg, Yg=Yg, labels=labels, boundary=boundary,
        boxcount_r=r, boxcount_n=n, fractal_dim=float(D_f), r_squared=float(r_sq),
        gamma=ue.gamma, gamma_r2=ue.r_squared, D_f_gamma=ue.D_f,
        k_used=int(k_used), grid_lo=grid_lo, grid_hi=grid_hi,
    )


# ── filename encoding ────────────────────────────────────────
def npz_name(node_x: int, node_y: int, coupling: float) -> str:
    """Encode swept-node ids and the live coupling into the .npz filename."""
    return f"lorenz_basins_n{node_x}_n{node_y}_K{coupling:.4f}.npz"


def gif_name(node_x: int, node_y: int, k_start: float, k_stop: float) -> str:
    """Encode swept-node ids and the coupling range into the .gif filename."""
    return f"lorenz_basins_n{node_x}_n{node_y}_K{k_start:g}-{k_stop:g}.gif"


# ── frame rendering ──────────────────────────────────────────
def render_frame(rec: dict, node_x: int, node_y: int, lbl_x: str, lbl_y: str,
                 fig, canvas) -> Image.Image:
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    ext = [rec["grid_lo"], rec["grid_hi"], rec["grid_lo"], rec["grid_hi"]]
    im = ax.imshow(rec["labels"], origin="lower", cmap="tab20",
                   extent=ext, interpolation="nearest")
    bd = rec["boundary"]
    if bd.any():
        ov = np.zeros((*bd.shape, 4), dtype=np.float32)
        ov[bd, 3] = 0.85
        ax.imshow(ov, origin="lower", extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Basin label")
    g = f"{rec['gamma']:.3f}" if rec["gamma"] is not None else "n/a"
    ax.set_xlabel(f"Node {node_x} ({lbl_x})  X perturbation")
    ax.set_ylabel(f"Node {node_y} ({lbl_y})  X perturbation")
    ax.set_title(
        f"DTI_A Lorenz basins   coupling K = {rec['coupling']:.4f}\n"
        f"K_basins={rec['k_used']}   $D_f$={rec['fractal_dim']:.3f}   "
        f"$\\gamma$={g}  ($D_f$=2−γ={rec['D_f_gamma'] if rec['D_f_gamma'] is None else round(rec['D_f_gamma'],3)})")
    fig.tight_layout()
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3].copy())


# ── CLI ──────────────────────────────────────────────────────
def _fine_ladder(k_start: float, k_stop: float, k_step: float) -> np.ndarray:
    n = int(round((k_stop - k_start) / k_step)) + 1
    return np.round(k_start + k_step * np.arange(n), 6)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dti-path", default="data/DTI_A.mat")
    ap.add_argument("--node-x", type=int, default=73, help="grid x-axis node")
    ap.add_argument("--node-y", type=int, default=81, help="grid y-axis node")
    ap.add_argument("--k-start", type=float, default=0.45)
    ap.add_argument("--k-stop", type=float, default=0.65)
    ap.add_argument("--k-step", type=float, default=0.025)
    ap.add_argument("--grid-n", type=int, default=128)
    ap.add_argument("--grid-lo", type=float, default=-9.0)
    ap.add_argument("--grid-hi", type=float, default=9.0)
    ap.add_argument("--k-clusters", default="auto",
                    help="'auto' (dynamic Elbow+BIC+Silhouette) or an integer.")
    ap.add_argument("--cluster-criterion", default="consensus",
                    choices=["consensus", "elbow", "bic", "silhouette"])
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--outdir", default="data/derivatives")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end pass (small grid, short integration).")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L_gpu, n_dti = load_dti_laplacian(args.dti_path, device)
    for node in (args.node_x, args.node_y):
        if not 0 <= node < n_dti:
            raise ValueError(f"node {node} out of range for N={n_dti}")

    p_base = LorenzParams(slice_node_x=args.node_x, slice_node_y=args.node_y,
                          n_osc=n_dti)
    grid_n = args.grid_n
    if args.smoke:
        grid_n = 24
        p_base = LorenzParams(coupling=0.5, dt=0.05, t_transient=5.0, tmax=10.0,
                              slice_node_x=args.node_x, slice_node_y=args.node_y,
                              n_osc=n_dti)

    ladder = _fine_ladder(args.k_start, args.k_stop, args.k_step)
    dk = labels_for(n_dti)
    lbl_x = dk[args.node_x] if args.node_x < len(dk) else f"node{args.node_x}"
    lbl_y = dk[args.node_y] if args.node_y < len(dk) else f"node{args.node_y}"
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device]   {device}")
    print(f"[fine]     nodes ({args.node_x}:{lbl_x}) vs ({args.node_y}:{lbl_y})  "
          f"K={ladder.tolist()}  grid={grid_n}²  N={n_dti}")

    fig = plt.figure(figsize=(7.5, 6.5), dpi=120)
    canvas = FigureCanvasAgg(fig)
    frames: list[Image.Image] = []
    for K in ladder:
        rec = observe_coupling(
            L_gpu, float(K), p_base, grid_n, args.grid_lo, args.grid_hi,
            args.k_clusters, args.cluster_criterion, device)
        cfg = dict(coupling=rec["coupling"], slice_node_x=args.node_x,
                   slice_node_y=args.node_y, n_osc=n_dti, grid_n=grid_n,
                   k_clusters=rec["k_used"], grid_lo=args.grid_lo,
                   grid_hi=args.grid_hi, sigma=p_base.sigma, rho=p_base.rho,
                   beta=p_base.beta, dt=p_base.dt, tmax=p_base.tmax)
        npz_path = out_dir / npz_name(args.node_x, args.node_y, float(K))
        np.savez_compressed(
            npz_path, Xg=rec["Xg"], Yg=rec["Yg"], labels=rec["labels"],
            boundary=rec["boundary"], boxcount_r=rec["boxcount_r"],
            boxcount_n=rec["boxcount_n"], fractal_dim=np.array([rec["fractal_dim"]]),
            r_squared=np.array([rec["r_squared"]]),
            gamma=np.array([np.nan if rec["gamma"] is None else rec["gamma"]]),
            config=np.array(cfg, dtype=object))
        g = f"{rec['gamma']:.3f}" if rec["gamma"] is not None else "  -  "
        print(f"  [K={K:.4f}]  K_basins={rec['k_used']}  D_f={rec['fractal_dim']:.3f}  "
              f"gamma={g}  ->  {npz_path.name}")
        frames.append(render_frame(rec, args.node_x, args.node_y, lbl_x, lbl_y,
                                   fig, canvas))
    plt.close(fig)

    gif_path = out_dir / gif_name(args.node_x, args.node_y, args.k_start, args.k_stop)
    duration_ms = int(1000.0 / max(args.fps, 1e-6))
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    print(f"[gif]      {len(frames)} frames -> {gif_path}  "
          f"({gif_path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
