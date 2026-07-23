#!/usr/bin/env python3
"""
Fine coupling sweep of the DTI-coupled Hindmarsh-Rose basin geometry.

Strict parity with lorenz_fine_coupling_sweep.py -- same pipeline, same CLI
shape, same output-file convention -- with the Lorenz-63 vector field
swapped for the canonical chaotic-bursting Hindmarsh-Rose neuron (see
hr_sweep.py's module docstring for the physics). Built to bring HR up to
the same true-VPS/lobe-locking capability Lorenz got tonight, since HR is
the system this project can most directly compare against the literature.

Differences from the Lorenz CLI, both inherited from hr_sweep.py's own
design rather than introduced here:
  - grid_lo/grid_hi default to [-2, 2] (HR's fast membrane potential X
    lives roughly in [-1.6, 2.0], vs Lorenz's [-9, 9]).
  - t_transient/tmax default to HindmarshRoseParams' own 200s/1000s
    (HR's slow adaptation variable, time-constant r=0.006, needs a longer
    window to average over several bursts than Lorenz's 100s/500s).
  - No --vps-norm generalization: that was a Lorenz-only exploration of
    the paper's "L2 chosen by fiat" gap, not yet ported here.

Usage
-----
    python3 -m pythongpu.pipeline.hr_fine_coupling_sweep
    python3 -m pythongpu.pipeline.hr_fine_coupling_sweep \
        --node-x 73 --node-y 81 --k-start 0.45 --k-stop 0.65 --k-step 0.025 \
        --grid-n 128 --outdir data/derivatives
    python3 -m pythongpu.pipeline.hr_fine_coupling_sweep --smoke
    python3 -m pythongpu.pipeline.hr_fine_coupling_sweep --vps-method true
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
from pythongpu.pipeline.hr_sweep import (
    HindmarshRoseParams,
    rk4_step_batched,
    run_sweep_streaming,
    run_sweep_true_vps,
)
from pythongpu.processing.basin_clustering import select_optimal_clusters
from pythongpu.processing.box_counting import (
    extract_boundary,
    boxcount_2d_gpu,
    fractal_dimension,
    uncertainty_exponent,
)


# ── slice construction ───────────────────────────────────────
def build_slice(p: HindmarshRoseParams, grid_n: int, grid_lo: float, grid_hi: float,
                device: torch.device):
    """
    Build the (B, N, 3) IC batch for a 2-D affine slice: the X component of
    ``p.slice_node_x`` sweeps the grid x-axis and ``p.slice_node_y`` the y-axis;
    every other state entry sits at the (1,1,1) base point so each batch element
    maps to exactly one grid cell.
    """
    ax = np.linspace(grid_lo, grid_hi, grid_n, dtype=np.float32)
    Xg, Yg = np.meshgrid(ax, ax)
    B, N = grid_n * grid_n, p.n_osc
    x0 = torch.full((B, N, 3), 1.0, dtype=torch.float32, device=device)

    Xg_r, Yg_r = Xg.ravel(), Yg.ravel()
    x0[:, p.slice_node_x, 0] = torch.as_tensor(Xg_r, device=device)
    x0[:, p.slice_node_y, 0] = torch.as_tensor(Yg_r, device=device)
    return x0, Xg, Yg


# ── one coupling value ───────────────────────────────────────
def observe_coupling(L_gpu, coupling: float, p_base: HindmarshRoseParams, grid_n: int,
                     grid_lo: float, grid_hi: float, k_clusters,
                     cluster_criterion: str, device: torch.device,
                     kmeans_seed: int = 42,
                     vps_method: str = "streaming", true_vps_alignment: str = "corrected",
                     true_vps_chunk_size: int | None = None,
                     true_vps_pair_chunk_size: int | None = None) -> dict:
    """Integrate, cluster, and quantify the basin geometry for one K.

    vps_method: "streaming" (default, O(1)-in-T Welford surrogate, no lag
    search) or "true" (the paper's own lag-based cross-correlation
    statistic, run_sweep_true_vps, chunked over both the IC batch and node
    pairs to stay within memory).
    """
    p = HindmarshRoseParams(
        a=p_base.a, b=p_base.b, c=p_base.c, d=p_base.d, s=p_base.s,
        r=p_base.r, x_rest=p_base.x_rest, I=p_base.I,
        coupling=coupling, dt=p_base.dt,
        t_transient=p_base.t_transient, tmax=p_base.tmax,
        slice_node_x=p_base.slice_node_x, slice_node_y=p_base.slice_node_y,
        n_osc=p_base.n_osc,
    )
    x0, Xg, Yg = build_slice(p, grid_n, grid_lo, grid_hi, device)

    for _ in range(p.steps_transient):
        x0 = rk4_step_batched(x0, L_gpu, p)

    if vps_method == "streaming":
        vectors_gpu, mean_x_gpu = run_sweep_streaming(
            x0, L_gpu, p, device, return_mean_x=True)
    elif vps_method == "true":
        vectors_gpu, mean_x_gpu = run_sweep_true_vps(
            x0, L_gpu, p, device, alignment=true_vps_alignment,
            chunk_size=true_vps_chunk_size, pair_chunk_size=true_vps_pair_chunk_size,
            return_mean_x=True)
    else:
        raise ValueError(f"vps_method must be 'streaming' or 'true', got {vps_method!r}")
    vectors = vectors_gpu.cpu().numpy()

    # ── clustering-free lobe-locking label field ──────────────────────
    # Same convention as Lorenz: each node locks to the attractor lobe whose
    # X-sign matches its time-mean X (coupling acts through X only here too).
    mean_x = mean_x_gpu.cpu().numpy()                       # (B, N)
    signs = (mean_x > 0.0)                                  # (B, N) bool
    nx, ny = p.slice_node_x, p.slice_node_y
    sign_ids = signs[:, nx].astype(np.int32) * 2 + signs[:, ny].astype(np.int32)
    sign_labels = sign_ids.reshape(grid_n, grid_n).astype(np.int32)
    ue_sign = uncertainty_exponent(sign_labels, d=2)
    n_pair_basins = int(np.unique(sign_ids).size)
    n_full_cfg = int(np.unique(signs, axis=0).shape[0])

    if str(k_clusters).lower() == "auto":
        sel = select_optimal_clusters(vectors, k_min=2, k_max=12,
                                      criterion=cluster_criterion)
        k_used = sel.best_k
        labels = sel.labels.reshape(grid_n, grid_n).astype(np.int32)
    else:
        from pythongpu.pipeline.hr_sweep import kmeans_gpu
        k_used = int(k_clusters)
        labels = kmeans_gpu(vectors_gpu, k=k_used, seed=kmeans_seed).cpu().numpy().reshape(grid_n, grid_n)

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
        sign_labels=sign_labels, n_lobe_configs=n_pair_basins,
        n_full_cfg=n_full_cfg, mean_x=mean_x,
        gamma_sign=ue_sign.gamma, gamma_sign_r2=ue_sign.r_squared,
        D_f_gamma_sign=ue_sign.D_f,
        k_used=int(k_used), grid_lo=grid_lo, grid_hi=grid_hi,
    )


# ── filename encoding ────────────────────────────────────────
def npz_name(node_x: int, node_y: int, coupling: float) -> str:
    return f"hr_basins_n{node_x}_n{node_y}_K{coupling:.4f}.npz"


def gif_name(node_x: int, node_y: int, k_start: float, k_stop: float) -> str:
    return f"hr_basins_n{node_x}_n{node_y}_K{k_start:g}-{k_stop:g}.gif"


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
        f"DTI_A Hindmarsh-Rose basins   coupling K = {rec['coupling']:.4f}\n"
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
    ap.add_argument("--dti-path", default="data/DTI-og.mat")
    ap.add_argument("--node-x", type=int, default=73, help="grid x-axis node")
    ap.add_argument("--node-y", type=int, default=81, help="grid y-axis node")
    ap.add_argument("--k-start", type=float, default=0.45)
    ap.add_argument("--k-stop", type=float, default=0.65)
    ap.add_argument("--k-step", type=float, default=0.025)
    ap.add_argument("--grid-n", type=int, default=128)
    ap.add_argument("--grid-lo", type=float, default=-2.0,
                    help="HR's fast membrane potential X lives roughly in [-1.6, 2.0] "
                         "(vs Lorenz's [-9, 9]).")
    ap.add_argument("--grid-hi", type=float, default=2.0)
    ap.add_argument("--kmeans-seed", type=int, default=42)
    ap.add_argument("--vps-method", default="streaming", choices=["streaming", "true"],
                    help="'streaming' (default): O(1)-in-T Welford surrogate, no lag search. "
                         "'true': the paper's own lag-based cross-correlation VPS.")
    ap.add_argument("--true-vps-alignment", default="corrected", choices=["corrected", "matlab"])
    ap.add_argument("--true-vps-chunk-size", type=int, default=None)
    ap.add_argument("--true-vps-pair-chunk-size", type=int, default=None)
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

    p_base = HindmarshRoseParams(slice_node_x=args.node_x, slice_node_y=args.node_y,
                                 n_osc=n_dti)
    grid_n = args.grid_n
    if args.smoke:
        grid_n = 24
        p_base = HindmarshRoseParams(coupling=0.5, dt=0.05, t_transient=5.0, tmax=10.0,
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
            args.k_clusters, args.cluster_criterion, device,
            kmeans_seed=args.kmeans_seed,
            vps_method=args.vps_method, true_vps_alignment=args.true_vps_alignment,
            true_vps_chunk_size=args.true_vps_chunk_size,
            true_vps_pair_chunk_size=args.true_vps_pair_chunk_size)
        cfg = dict(coupling=rec["coupling"], slice_node_x=args.node_x,
                   slice_node_y=args.node_y, n_osc=n_dti, grid_n=grid_n,
                   k_clusters=rec["k_used"], grid_lo=args.grid_lo,
                   grid_hi=args.grid_hi, a=p_base.a, b=p_base.b, c=p_base.c,
                   d=p_base.d, s=p_base.s, r=p_base.r, x_rest=p_base.x_rest,
                   I=p_base.I, dt=p_base.dt, tmax=p_base.tmax,
                   kmeans_seed=args.kmeans_seed,
                   vps_method=args.vps_method, true_vps_alignment=args.true_vps_alignment)
        npz_path = out_dir / npz_name(args.node_x, args.node_y, float(K))
        np.savez_compressed(
            npz_path, Xg=rec["Xg"], Yg=rec["Yg"], labels=rec["labels"],
            boundary=rec["boundary"], boxcount_r=rec["boxcount_r"],
            boxcount_n=rec["boxcount_n"], fractal_dim=np.array([rec["fractal_dim"]]),
            r_squared=np.array([rec["r_squared"]]),
            gamma=np.array([np.nan if rec["gamma"] is None else rec["gamma"]]),
            sign_labels=rec["sign_labels"],
            n_lobe_configs=np.array([rec["n_lobe_configs"]]),
            n_full_cfg=np.array([rec["n_full_cfg"]]),
            mean_x=rec["mean_x"],
            gamma_sign=np.array(
                [np.nan if rec["gamma_sign"] is None else rec["gamma_sign"]]),
            D_f_gamma_sign=np.array(
                [np.nan if rec["D_f_gamma_sign"] is None else rec["D_f_gamma_sign"]]),
            config=np.array(cfg, dtype=object))
        g = f"{rec['gamma']:.3f}" if rec["gamma"] is not None else "  -  "
        gs = f"{rec['gamma_sign']:.3f}" if rec["gamma_sign"] is not None else "  -  "
        print(f"  [K={K:.4f}]  K_basins={rec['k_used']}  D_f={rec['fractal_dim']:.3f}  "
              f"gamma={g}  |  lobe_configs={rec['n_lobe_configs']}  "
              f"gamma_sign={gs}  ->  {npz_path.name}")
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
