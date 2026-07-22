#!/usr/bin/env python3
"""
perturbation_sensitivity.py
----------------------------
Tests the paper's unverified control-theoretic claim: that riddled basins
admit vanishingly small perturbations able to switch attractor.

Hypothesis
----------
P_flip(delta) = P[perturbing an IC by magnitude delta changes its lobe-locking
label] should, in the small-delta limit:
  - decay to 0 for an ordinary (non-riddled) basin interior — a point
    strictly interior to a basin needs a finite kick to reach the boundary.
  - stay bounded away from 0 near a genuinely riddled boundary — every
    neighborhood of every point already touches another basin.

REVISION (2026-07-22): the original design compared this signature across
couplings (K=0.0 uncoupled control vs K=0.1 vs K=0.5), reusing the same base
ICs -- located from the K=0.5 boundary field -- at every K. That control was
invalid: at K=0.0 there is no locking at all (0.0% locked, Lambda~0.026 per
the onset-curve data), so the lobe-sign label is already a coin flip on
mean(X)~=0 before any perturbation is applied, and flips trivially at any
delta. The K=0.0 curve was indistinguishable from K=0.5 for that reason, not
because riddling was confirmed.

The fixed design compares boundary vs interior points *within the same
coupling* (--compare-boundary-interior), so both point sets have a valid
label to begin with -- this is the actual riddling test: do boundary points
flip more easily, and at smaller delta, than interior points, at a K where
locking is already established.

Labeling reuses the exact, clustering-free lobe-locking label from
attractor_id.py / lorenz_fine_coupling_sweep.py: sign(time-mean X) per node.
No VPS/k-means — cheaper, and immune to k-cluster artifacts.

Integration reuses rk4_step_batched from lorenz_sweep.py unchanged. The whole
(base points x directions x delta) set is one batch of independent ICs, so it
drops into the same batched-IC pattern as the basin sweeps.

Run:
    python3 -m pythongpu.pipeline.perturbation_sensitivity --smoke
    python3 -m pythongpu.pipeline.perturbation_sensitivity \
        --compare-boundary-interior --boundary-coupling 0.5 --couplings 0.5 \
        --slice-grid-n 96 --dti-path data/DTI-og.mat
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from pythongpu.networks.static_adjacency import load_dti_laplacian
from pythongpu.pipeline.lorenz_sweep import LorenzParams, rk4_step_batched
from pythongpu.processing.box_counting import extract_boundary

DEFAULT_COUPLINGS = (0.0, 0.1, 0.5)


# ═══════════════════════════════════════════════════════════════════════════
# labeling — sign(time-mean X) per node, exact and clustering-free
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def integrate_and_label(x0: torch.Tensor, L_gpu: torch.Tensor, p: LorenzParams,
                        ) -> torch.Tensor:
    """
    Run the transient + recording window, return the (B, N) bool lobe-sign
    label (mean_x > 0) for every IC in the batch. Mirrors the mean_x logic in
    run_sweep_streaming(..., return_mean_x=True), stripped of the VPS
    accumulation this experiment doesn't need.
    """
    x = x0
    for _ in range(p.steps_transient):
        x = rk4_step_batched(x, L_gpu, p)

    B, N, _ = x.shape
    mean_x = torch.zeros(B, N, device=x.device)
    count = 0
    for _ in range(p.steps_record):
        x = rk4_step_batched(x, L_gpu, p)
        count += 1
        mean_x += (x[:, :, 0] - mean_x) / count

    return mean_x > 0.0  # (B, N) bool


# ═══════════════════════════════════════════════════════════════════════════
# base ICs — where perturbations start from
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def build_sign_slice(coupling: float, node_x: int, node_y: int, grid_n: int,
                     grid_lo: float, grid_hi: float, p_base: LorenzParams,
                     L_gpu: torch.Tensor, device: torch.device):
    """
    Integrate a 2D IC slice at `coupling` and return the clustering-free
    lobe-sign label field, exactly as in lorenz_fine_coupling_sweep.py's
    observe_coupling: sign_ids = signs[:, node_x]*2 + signs[:, node_y].
    No VPS/k-means involved, so this is immune to the k-cluster D_f
    saturation seen in the existing basin_data*.npz files.

    Returns (Xg, Yg, sign_labels, boundary) — sign_labels/boundary are
    (grid_n, grid_n) arrays over the (node_x, node_y) IC plane.
    """
    N = p_base.n_osc
    p = LorenzParams(
        sigma=p_base.sigma, rho=p_base.rho, beta=p_base.beta,
        coupling=coupling, dt=p_base.dt, t_transient=p_base.t_transient,
        tmax=p_base.tmax, slice_node_x=node_x, slice_node_y=node_y, n_osc=N,
    )
    ax = np.linspace(grid_lo, grid_hi, grid_n, dtype=np.float32)
    Xg, Yg = np.meshgrid(ax, ax)
    B = grid_n * grid_n
    x0 = torch.ones((B, N, 3), device=device) + 0.05 * torch.randn((B, N, 3), device=device)
    x0[:, node_x, 0] = torch.tensor(Xg.ravel(), device=device)
    x0[:, node_y, 0] = torch.tensor(Yg.ravel(), device=device)

    signs = integrate_and_label(x0, L_gpu, p).cpu().numpy()  # (B, N) bool
    sign_ids = signs[:, node_x].astype(np.int32) * 2 + signs[:, node_y].astype(np.int32)
    sign_labels = sign_ids.reshape(grid_n, grid_n).astype(np.int32)
    boundary = extract_boundary(sign_labels)
    return Xg, Yg, sign_labels, boundary


def sample_base_ics_from_slice(mode: str, n_points: int, N: int,
                               rng: np.random.Generator, node_x: int, node_y: int,
                               Xg: np.ndarray, Yg: np.ndarray, boundary: np.ndarray,
                               ) -> np.ndarray:
    """
    Sample n_points ICs whose (node_x, node_y) coordinates sit either on the
    lobe-sign boundary ('boundary') or strictly away from it ('interior'),
    using the field returned by build_sign_slice. Falls back to sampling from
    whatever pool is non-empty, and raises if the slice is degenerate
    (e.g. K=0 with no boundary at all — that IS the expected control case,
    so callers should not use 'boundary' mode with a control coupling).
    """
    mask = boundary if mode == "boundary" else ~boundary
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(
            f"no '{mode}' pixels found in this slice (grid may be too coarse, "
            f"or the coupling has no boundary at all — use --base-ic-mode random "
            f"or a different --boundary-coupling)")
    chosen = rng.choice(idx, size=min(n_points, idx.size), replace=idx.size < n_points)

    base = np.ones((len(chosen), N, 3), dtype=np.float64)
    base += 0.05 * rng.standard_normal(base.shape)
    flat_x, flat_y = Xg.ravel(), Yg.ravel()
    base[:, node_x, 0] = flat_x[chosen]
    base[:, node_y, 0] = flat_y[chosen]
    return base


def sample_base_ics(mode: str, n_points: int, N: int, rng: np.random.Generator,
                    grid_lo: float, grid_hi: float) -> np.ndarray:
    """Return (n_points, N, 3) base ICs, jittered near the ones-vector baseline
    used throughout the sweep scripts. mode='random' draws uniformly on the
    grid range used elsewhere (matches basin-sweep slice ICs)."""
    base = np.ones((n_points, N, 3), dtype=np.float64)
    base += 0.05 * rng.standard_normal(base.shape)
    if mode == "random":
        x_ics = rng.uniform(grid_lo, grid_hi, size=n_points)
        y_ics = rng.uniform(grid_lo, grid_hi, size=n_points)
        base[:, 0, 0] = x_ics
        base[:, 1, 0] = y_ics
    return base


# ═══════════════════════════════════════════════════════════════════════════
# core experiment — one coupling value
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SweepResult:
    coupling: float
    deltas: np.ndarray
    p_flip: np.ndarray        # (n_delta,) fraction of (point, direction) pairs that flipped
    flip_frac: np.ndarray     # (n_delta, n_points, n_directions) per-node flip fraction
    base_labels: np.ndarray   # (n_points, N) baseline labels


@torch.no_grad()
def run_coupling(coupling: float, base_ics: np.ndarray, deltas: np.ndarray,
                 n_directions: int, p_base: LorenzParams, L_gpu: torch.Tensor,
                 device: torch.device, rng: np.random.Generator) -> SweepResult:
    n_points, N, _ = base_ics.shape
    n_delta = len(deltas)

    p = LorenzParams(
        sigma=p_base.sigma, rho=p_base.rho, beta=p_base.beta,
        coupling=coupling, dt=p_base.dt, t_transient=p_base.t_transient,
        tmax=p_base.tmax, slice_node_x=p_base.slice_node_x,
        slice_node_y=p_base.slice_node_y, n_osc=N,
    )

    # baseline label per base point (no perturbation)
    x0_base = torch.tensor(base_ics, device=device, dtype=torch.float32)
    base_labels = integrate_and_label(x0_base, L_gpu, p).cpu().numpy()  # (n_points, N)

    # random unit perturbation directions, shared across delta so the only
    # thing that varies per delta is the step size along a fixed direction
    directions = rng.standard_normal((n_points, n_directions, N, 3))
    directions /= np.linalg.norm(directions.reshape(n_points, n_directions, -1),
                                 axis=-1, keepdims=True).reshape(n_points, n_directions, 1, 1)

    flip_frac = np.zeros((n_delta, n_points, n_directions), dtype=np.float64)

    for di, delta in enumerate(deltas):
        perturbed = base_ics[:, None, :, :] + delta * directions   # (n_points, n_directions, N, 3)
        batch = perturbed.reshape(n_points * n_directions, N, 3)
        x0 = torch.tensor(batch, device=device, dtype=torch.float32)
        labels = integrate_and_label(x0, L_gpu, p).cpu().numpy()   # (B, N)
        labels = labels.reshape(n_points, n_directions, N)

        diff = labels != base_labels[:, None, :]                  # (n_points, n_directions, N)
        flip_frac[di] = diff.mean(axis=-1)

    p_flip = (flip_frac > 0.0).mean(axis=(1, 2))                  # (n_delta,)

    return SweepResult(coupling=coupling, deltas=deltas, p_flip=p_flip,
                       flip_frac=flip_frac, base_labels=base_labels)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dti-path", default="data/DTI-og.mat")
    ap.add_argument("--couplings", type=float, nargs="+", default=list(DEFAULT_COUPLINGS))
    ap.add_argument("--delta-min", type=float, default=1e-8)
    ap.add_argument("--delta-max", type=float, default=1e-1)
    ap.add_argument("--n-delta", type=int, default=15)
    ap.add_argument("--n-points", type=int, default=5)
    ap.add_argument("--n-directions", type=int, default=8)
    ap.add_argument("--base-ic-mode", default="random",
                    choices=["random", "boundary", "interior"],
                    help="'boundary'/'interior' locate points via the clustering-free "
                         "lobe-sign field at --boundary-coupling (see build_sign_slice). "
                         "Ignored if --compare-boundary-interior is set.")
    ap.add_argument("--compare-boundary-interior", action="store_true",
                    help="the fixed experiment: sample BOTH boundary and interior ICs "
                         "from the same reference slice and test both at the same "
                         "--couplings, producing p_flip_boundary/p_flip_interior side "
                         "by side. Avoids the invalid-control problem of comparing "
                         "across couplings (see module docstring).")
    ap.add_argument("--boundary-coupling", type=float, default=0.5,
                    help="coupling used to locate boundary/interior ICs (default: the "
                         "measured riddled regime, K=0.5).")
    ap.add_argument("--slice-grid-n", type=int, default=64,
                    help="grid resolution for locating boundary/interior pixels.")
    ap.add_argument("--node-x", type=int, default=28)
    ap.add_argument("--node-y", type=int, default=79)
    ap.add_argument("--grid-lo", type=float, default=-9.0)
    ap.add_argument("--grid-hi", type=float, default=9.0)
    ap.add_argument("--t-transient", type=float, default=100.0)
    ap.add_argument("--tmax", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="data/derivatives")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end pass (few points/directions/deltas, short integration).")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L_gpu, N = load_dti_laplacian(args.dti_path, device)
    rng = np.random.default_rng(args.seed)

    t_transient, tmax = args.t_transient, args.tmax
    n_points, n_directions, n_delta = args.n_points, args.n_directions, args.n_delta
    slice_grid_n = args.slice_grid_n
    base_ic_mode = args.base_ic_mode
    compare = args.compare_boundary_interior
    if args.smoke:
        t_transient, tmax = 5.0, 10.0
        n_points, n_directions, n_delta = 2, 2, 4
        slice_grid_n = 16
        if compare or base_ic_mode in ("boundary", "interior"):
            print("[perturbation_sensitivity] --smoke: integration is too short to "
                 "form a real lobe-sign boundary. Proceeding anyway so the sampling/"
                 "masking/serialization LOGIC gets exercised end-to-end; the boundary "
                 "vs interior split at this scale is not a scientific result.")

    p_base = LorenzParams(t_transient=t_transient, tmax=tmax, n_osc=N,
                          slice_node_x=args.node_x, slice_node_y=args.node_y)
    deltas = np.geomspace(args.delta_min, args.delta_max, n_delta)

    from pathlib import Path
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Native-dtype config fields (no dict, no object dtype, no pickle) so the
    # npz loads on any numpy version -- a dict value forces np.savez through
    # allow_pickle, and a pickle written by one numpy major version is not
    # guaranteed loadable by another (hit exactly this: numpy>=2 on ACRES vs
    # 1.24 here raised "No module named 'numpy._core'" on the config field
    # alone; the plain-dtype arrays loaded fine even then).
    cfg_kwargs = dict(
        cfg_dti_path=np.array(args.dti_path),
        cfg_t_transient=np.float64(t_transient),
        cfg_tmax=np.float64(tmax),
        cfg_grid_lo=np.float64(args.grid_lo),
        cfg_grid_hi=np.float64(args.grid_hi),
        cfg_seed=np.int64(args.seed),
        cfg_boundary_coupling=np.float64(args.boundary_coupling),
        cfg_node_x=np.int64(args.node_x),
        cfg_node_y=np.int64(args.node_y),
        cfg_slice_grid_n=np.int64(slice_grid_n),
    )

    if compare:
        print(f"[perturbation_sensitivity] --compare-boundary-interior: locating both "
             f"pools from the lobe-sign field at K={args.boundary_coupling:.3f} "
             f"(nodes {args.node_x},{args.node_y}, grid={slice_grid_n}) ...")
        Xg, Yg, sign_labels, boundary = build_sign_slice(
            args.boundary_coupling, args.node_x, args.node_y, slice_grid_n,
            args.grid_lo, args.grid_hi, p_base, L_gpu, device)
        n_boundary_px = int(boundary.sum())
        print(f"    {n_boundary_px}/{boundary.size} boundary pixels "
             f"({100*n_boundary_px/boundary.size:.1f}%), "
             f"{boundary.size - n_boundary_px} interior pixels")

        # Verify the mask actually partitions the grid the way sample_base_ics_from_slice
        # will read it: boundary + interior must cover every pixel exactly once.
        interior_mask = ~boundary
        assert (boundary | interior_mask).all() and not (boundary & interior_mask).any(), \
            "boundary/interior masks must partition the slice with no overlap or gap"

        boundary_ics = sample_base_ics_from_slice(
            "boundary", n_points, N, rng, args.node_x, args.node_y, Xg, Yg, boundary)
        interior_ics = sample_base_ics_from_slice(
            "interior", n_points, N, rng, args.node_x, args.node_y, Xg, Yg, boundary)
        print(f"    sampled {boundary_ics.shape[0]} boundary ICs, "
             f"{interior_ics.shape[0]} interior ICs")

        results_boundary: list[SweepResult] = []
        results_interior: list[SweepResult] = []
        for coupling in args.couplings:
            print(f"[perturbation_sensitivity] K={coupling:.3f}  boundary ...")
            rb = run_coupling(coupling, boundary_ics, deltas, n_directions, p_base,
                              L_gpu, device, rng)
            results_boundary.append(rb)
            for d, pf in zip(rb.deltas, rb.p_flip):
                print(f"    delta={d:.3e}  P_flip(boundary)={pf:.3f}")

            print(f"[perturbation_sensitivity] K={coupling:.3f}  interior ...")
            ri = run_coupling(coupling, interior_ics, deltas, n_directions, p_base,
                              L_gpu, device, rng)
            results_interior.append(ri)
            for d, pf in zip(ri.deltas, ri.p_flip):
                print(f"    delta={d:.3e}  P_flip(interior)={pf:.3f}")

        p_flip_boundary = np.stack([r.p_flip for r in results_boundary])   # (n_K, n_delta)
        p_flip_interior = np.stack([r.p_flip for r in results_interior])   # (n_K, n_delta)
        print(f"[perturbation_sensitivity] p_flip_boundary.shape={p_flip_boundary.shape}  "
             f"p_flip_interior.shape={p_flip_interior.shape}")

        out_path = outdir / "perturbation_sensitivity_boundary_interior.npz"
        np.savez(
            out_path,
            couplings=np.array([r.coupling for r in results_boundary]),
            deltas=deltas,
            p_flip_boundary=p_flip_boundary,
            p_flip_interior=p_flip_interior,
            flip_frac_boundary=np.stack([r.flip_frac for r in results_boundary]),
            flip_frac_interior=np.stack([r.flip_frac for r in results_interior]),
            base_labels_boundary=np.stack([r.base_labels for r in results_boundary]),
            base_labels_interior=np.stack([r.base_labels for r in results_interior]),
            base_ics_boundary=boundary_ics,
            base_ics_interior=interior_ics,
            sign_labels=sign_labels,
            boundary_mask=boundary,
            n_points=n_points, n_directions=n_directions,
            **cfg_kwargs,
        )
        print(f"[perturbation_sensitivity] wrote {out_path}")
        return 0

    # ── original single-pool path (unchanged behaviour, fixed serialization) ──
    if base_ic_mode in ("boundary", "interior"):
        print(f"[perturbation_sensitivity] locating '{base_ic_mode}' ICs from the "
             f"lobe-sign field at K={args.boundary_coupling:.3f} "
             f"(nodes {args.node_x},{args.node_y}, grid={slice_grid_n}) ...")
        Xg, Yg, sign_labels, boundary = build_sign_slice(
            args.boundary_coupling, args.node_x, args.node_y, slice_grid_n,
            args.grid_lo, args.grid_hi, p_base, L_gpu, device)
        n_boundary_px = int(boundary.sum())
        print(f"    {n_boundary_px}/{boundary.size} boundary pixels "
             f"({100*n_boundary_px/boundary.size:.1f}%)")
        base_ics = sample_base_ics_from_slice(
            base_ic_mode, n_points, N, rng, args.node_x, args.node_y,
            Xg, Yg, boundary)
    else:
        base_ics = sample_base_ics(base_ic_mode, n_points, N, rng,
                                   args.grid_lo, args.grid_hi)

    results: list[SweepResult] = []
    for coupling in args.couplings:
        print(f"[perturbation_sensitivity] K={coupling:.3f} ...")
        res = run_coupling(coupling, base_ics, deltas, n_directions, p_base,
                           L_gpu, device, rng)
        results.append(res)
        for d, pf in zip(res.deltas, res.p_flip):
            print(f"    delta={d:.3e}  P_flip={pf:.3f}")

    out_path = outdir / "perturbation_sensitivity.npz"
    np.savez(
        out_path,
        couplings=np.array([r.coupling for r in results]),
        deltas=deltas,
        p_flip=np.stack([r.p_flip for r in results]),          # (n_K, n_delta)
        flip_frac=np.stack([r.flip_frac for r in results]),    # (n_K, n_delta, n_points, n_directions)
        base_labels=np.stack([r.base_labels for r in results]),# (n_K, n_points, N)
        base_ics=base_ics,                                     # (n_points, N, 3)
        n_points=n_points, n_directions=n_directions,
        cfg_base_ic_mode=np.array(base_ic_mode),
        **cfg_kwargs,
    )
    print(f"[perturbation_sensitivity] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
