#!/usr/bin/env python3
"""
attractor_id.py — count attractors by where trajectories actually settle, with
the count *emerging* from a distance threshold rather than imposed as k.

Motivation
----------
k-means always returns exactly k groups, so it can never tell you "there is only
one attractor here." The trustworthy way to count basins is to (1) describe each
initial condition by an *attractor-invariant* summary of its long-time state and
(2) group those summaries with a method whose cluster count emerges from the
data and can legitimately be 1. This module uses DBSCAN with an auto-selected
neighborhood radius (k-distance knee): points denser than the radius merge into
one attractor, and the number of attractors is an output, not an input.

Two entry points, same grouping core:

  * ``--npz`` : group the VPS `vectors` already saved in basin_data.npz. These
    are time-averaged coherence features — attractor-invariant by construction —
    so this re-counts a completed sweep with no re-integration.

  * ``--integrate`` : integrate the real DTI-coupled Lorenz network for one K,
    build a compact final-state descriptor per IC (per-node mean X, std X and
    mean |X| over the recording window — all long-time averages, hence constant
    on a single attractor), and group that. This is the "watch where the marble
    settles" path from scratch.

The emergent count is only trustworthy once it is stable under the radius, the
integration length and the grid — the tool reports the radius sensitivity so
that check is visible.

Usage
-----
    python3 -m pythongpu.pipeline.attractor_id --root /mnt/data/tmp/$USER/coupling_sweep
    python3 -m pythongpu.pipeline.attractor_id --integrate --coupling 0.5 --grid-n 64
    python3 -m pythongpu.pipeline.attractor_id --integrate --coupling 0.6 \
        --grid-n 48 --tmax 60 --t-transient 20
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_K_FROM_DIR = re.compile(r"_K(-?\d+\.?\d*)")


# ── grouping core (shared by both entry points) ──────────────────────────────
def _kneedle(y_sorted: np.ndarray) -> int:
    """Index of the knee of a sorted-ascending curve (max distance to chord)."""
    n = len(y_sorted)
    x = np.linspace(0.0, 1.0, n)
    y = (y_sorted - y_sorted.min()) / (np.ptp(y_sorted) + 1e-12)
    num = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return int(np.argmax(num / (np.hypot(y[-1] - y[0], x[-1] - x[0]) + 1e-12)))


def _auto_eps(X: np.ndarray, min_samples: int) -> tuple[float, np.ndarray]:
    """DBSCAN radius from the knee of the sorted k-distance graph."""
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    kdist = np.sort(nn.kneighbors(X)[0][:, -1])
    return float(kdist[_kneedle(kdist)]), kdist


def group_emergent(desc: np.ndarray, pca_dim: int = 10, min_samples: int = 10,
                   eps_scales=(0.7, 1.0, 1.4)) -> dict:
    """
    Group descriptors so the attractor count emerges. Returns the count at the
    knee radius plus counts at scaled radii, so radius-sensitivity is visible.
    """
    Xs = StandardScaler().fit_transform(desc)
    if Xs.shape[1] > pca_dim:
        Xs = PCA(n_components=pca_dim, random_state=0).fit_transform(Xs)

    eps0, kdist = _auto_eps(Xs, min_samples)
    counts = {}
    for s in eps_scales:
        lab = DBSCAN(eps=eps0 * s, min_samples=min_samples).fit_predict(Xs)
        n_clust = int(len(set(lab)) - (1 if -1 in lab else 0))
        noise = float(np.mean(lab == -1))
        counts[round(s, 2)] = dict(n_attractors=n_clust, noise_frac=noise, labels=lab)

    at_knee = counts[1.0]
    scan = {s: counts[s]["n_attractors"] for s in counts}
    stable = len(set(scan.values())) == 1
    return dict(eps_knee=eps0, counts=counts, n_attractors=at_knee["n_attractors"],
                noise_frac=at_knee["noise_frac"], radius_scan=scan,
                radius_stable=stable, labels=at_knee["labels"])


# ── entry point 1: saved VPS features ────────────────────────────────────────
def _coupling_of(npz_path: Path, cfg):
    if cfg is not None and "coupling" in cfg:
        return float(cfg["coupling"])
    m = _K_FROM_DIR.search(npz_path.parent.name) or _K_FROM_DIR.search(npz_path.name)
    return float(m.group(1)) if m else None


def count_from_npz(npz_path: Path, n_sub: int, min_samples: int, rng) -> dict:
    with np.load(npz_path, allow_pickle=True) as data:
        if "vectors" not in data.files:
            raise KeyError("no 'vectors' array — predates feature persistence")
        vectors = np.asarray(data["vectors"], dtype=np.float64)
        cfg = dict(data["config"].item()) if "config" in data.files else None
    n = vectors.shape[0]
    sub = rng.choice(n, size=min(n_sub, n), replace=False)
    res = group_emergent(vectors[sub], min_samples=min_samples)
    res.update(coupling=_coupling_of(npz_path, cfg), n_ics=n, source=str(npz_path))
    return res


# ── entry point 2: integrate the real system ─────────────────────────────────
def count_from_integration(coupling: float, grid_n: int, grid_lo: float,
                           grid_hi: float, tmax: float, t_transient: float,
                           record_stride: int, node_x: int, node_y: int,
                           min_samples: int, dti_path: str) -> dict:
    import torch
    from pythongpu.networks.static_adjacency import load_dti_laplacian
    from pythongpu.pipeline.lorenz_sweep import LorenzParams, rk4_step_batched

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L_gpu, N = load_dti_laplacian(dti_path, device)
    p = LorenzParams(coupling=coupling, dt=0.05, t_transient=t_transient,
                     tmax=tmax, slice_node_x=node_x, slice_node_y=node_y, n_osc=N)

    ax = np.linspace(grid_lo, grid_hi, grid_n, dtype=np.float32)
    Xg, Yg = np.meshgrid(ax, ax)
    B = grid_n * grid_n
    x = torch.ones((B, N, 3), device=device) + 0.05 * torch.randn((B, N, 3), device=device)
    x[:, node_x, 0] = torch.tensor(Xg.ravel(), device=device)
    x[:, node_y, 0] = torch.tensor(Yg.ravel(), device=device)

    for _ in range(p.steps_transient):
        x = rk4_step_batched(x, L_gpu, p)

    # Accumulate long-time (attractor-invariant) per-node statistics of X.
    s1 = torch.zeros((B, N), device=device)   # sum X
    s2 = torch.zeros((B, N), device=device)   # sum X^2
    sa = torch.zeros((B, N), device=device)   # sum |X|
    cnt = 0
    for step in range(p.steps_record):
        x = rk4_step_batched(x, L_gpu, p)
        if step % record_stride == 0:
            Xc = x[..., 0]
            s1 += Xc
            s2 += Xc * Xc
            sa += Xc.abs()
            cnt += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    meanX = (s1 / cnt)
    stdX = torch.sqrt(torch.clamp(s2 / cnt - meanX * meanX, min=0.0))
    meanAbs = (sa / cnt)
    desc = torch.cat([meanX, stdX, meanAbs], dim=1).cpu().numpy()  # (B, 3N)

    res = group_emergent(desc, min_samples=min_samples)
    res.update(coupling=coupling, n_ics=B, source=f"integrate(K={coupling},grid={grid_n})")
    res["_desc"] = desc
    return res


# ── radius plateau: the test that decides "real count" vs "continuum" ────────
def scan_radius(desc: np.ndarray, min_samples: int = 10, pca_dim: int = 10,
                lo: float = 0.25, hi: float = 4.0, n_steps: int = 30) -> dict:
    """
    Sweep the DBSCAN radius over a wide range and record the count at each.

    This is the decisive test. N genuinely well-separated attractors produce a
    PLATEAU: the same count over a broad radius range, because real density gaps
    do not care where you put the threshold. A continuum — one connected cloud
    of varying density — instead fragments monotonically as the radius shrinks,
    with no flat region, and whatever count you report is just an artifact of
    where the k-distance knee happened to land.
    """
    Xs = StandardScaler().fit_transform(desc)
    if Xs.shape[1] > pca_dim:
        Xs = PCA(n_components=pca_dim, random_state=0).fit_transform(Xs)
    eps0, _ = _auto_eps(Xs, min_samples)

    scales = np.geomspace(lo, hi, n_steps)
    counts, noises = [], []
    for s in scales:
        lab = DBSCAN(eps=eps0 * s, min_samples=min_samples).fit_predict(Xs)
        counts.append(int(len(set(lab)) - (1 if -1 in lab else 0)))
        noises.append(float(np.mean(lab == -1)))
    counts = np.asarray(counts)

    # Longest run of an identical count >= 1, measured in radius decades so the
    # verdict does not depend on how finely we sampled.
    best_len, best_val, best_span = 0, None, (None, None)
    i = 0
    while i < len(counts):
        j = i
        while j + 1 < len(counts) and counts[j + 1] == counts[i]:
            j += 1
        if counts[i] >= 1 and (j - i + 1) > best_len:
            best_len, best_val = j - i + 1, int(counts[i])
            best_span = (float(scales[i]), float(scales[j]))
        i = j + 1
    decades = (np.log10(best_span[1] / best_span[0])
               if best_span[0] and best_span[1] and best_span[1] > best_span[0] else 0.0)
    return dict(eps_knee=eps0, scales=scales, counts=counts, noise=np.asarray(noises),
                plateau_count=best_val, plateau_steps=best_len,
                plateau_span=best_span, plateau_decades=float(decades))


# ── reporting ────────────────────────────────────────────────────────────────
def _report(res: dict) -> None:
    K = f"{res['coupling']:.4f}" if res.get("coupling") is not None else "  ?  "
    flag = "stable" if res["radius_stable"] else "RADIUS-SENSITIVE → not yet confident"
    print(f"[K={K}]  n_attractors={res['n_attractors']}  "
          f"noise_frac={res['noise_frac']:.2f}  eps_knee={res['eps_knee']:.3f}  "
          f"radius_scan={res['radius_scan']}  [{flag}]")


def _write_rows(results: list[dict], csv_path: Path, extra: dict | None = None) -> None:
    """
    Flatten result dicts to a CSV, dropping the bulky per-point arrays
    (`labels`, `counts`) that belong in memory, not a table.
    """
    import csv as _csv
    rows = []
    for res in results:
        row = {k: v for k, v in res.items() if k not in ("labels", "counts")}
        row["radius_scan"] = ";".join(f"{s}:{n}" for s, n in
                                      sorted(res["radius_scan"].items()))
        if extra:
            row.update(extra)
        rows.append(row)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv]      {len(rows)} row(s) -> {csv_path}")


def _discover(root, npz_globs):
    paths = []
    if root:
        paths += glob.glob(os.path.join(root, "**", "basin_data.npz"), recursive=True)
    for g in (npz_globs or []):
        paths += glob.glob(g)
    seen, uniq = set(), []
    for p in sorted(map(Path, paths)):
        rp = p.resolve()
        if rp in seen or re.search(r"basin_data_\w+\.npz$", p.name):
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="scan */basin_data.npz recursively")
    ap.add_argument("--npz", nargs="+", default=None, help="explicit npz path(s)/glob(s)")
    ap.add_argument("--integrate", action="store_true",
                    help="integrate the real DTI-Lorenz system for --coupling")
    ap.add_argument("--coupling", type=float, default=0.5)
    ap.add_argument("--grid-n", type=int, default=64)
    ap.add_argument("--grid-lo", type=float, default=-9.0)
    ap.add_argument("--grid-hi", type=float, default=9.0)
    ap.add_argument("--tmax", type=float, default=500.0)
    ap.add_argument("--t-transient", type=float, default=100.0)
    ap.add_argument("--record-stride", type=int, default=5)
    ap.add_argument("--node-x", type=int, default=28)
    ap.add_argument("--node-y", type=int, default=79)
    ap.add_argument("--dti-path", default="data/DTI_A.mat")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--n-sub", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-desc", default=None,
                    help="integrate mode: save the final-state descriptor (.npy, "
                         "~130 MB at 361^2) so radius analysis is free later "
                         "instead of costing another GPU-hour per K.")
    ap.add_argument("--scan-desc", default=None,
                    help="load a saved descriptor and sweep the DBSCAN radius to "
                         "test for a PLATEAU (a real count) vs a continuum (no "
                         "flat region -> the count is an artifact). No GPU.")
    ap.add_argument("--scan-csv", default=None,
                    help="where --scan-desc writes its count-vs-radius table")
    ap.add_argument("--csv", default=None,
                    help="write the result row(s) here. Without this an array "
                         "job's results live only in its stdout log.")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    if args.scan_desc:
        desc = np.load(args.scan_desc)
        r = scan_radius(desc, min_samples=args.min_samples)
        print(f"[scan]     descriptor {desc.shape} from {args.scan_desc}")
        print(f"[scan]     eps_knee={r['eps_knee']:.3f}")
        for s_, c_, nz in zip(r["scales"], r["counts"], r["noise"]):
            print(f"    scale={s_:6.3f}  eps={r['eps_knee']*s_:7.3f}  "
                  f"n_attractors={c_:4d}  noise={nz:5.1%}")
        dec = r["plateau_decades"]
        print(f"\n[verdict]  longest plateau: count={r['plateau_count']} over "
              f"scales {r['plateau_span'][0]:.2f}-{r['plateau_span'][1]:.2f} "
              f"({dec:.2f} radius decades, {r['plateau_steps']} steps)")
        if dec >= 0.3:
            print(f"[verdict]  PLATEAU -> the count {r['plateau_count']} is robust "
                  f"to the radius and can be reported.")
        else:
            print("[verdict]  NO PLATEAU -> continuum. The descriptor shows no "
                  "well-separated attractors; any single count is an artifact of "
                  "where the knee landed. Do NOT report a number.")
        if args.scan_csv:
            import csv as _csv
            with open(args.scan_csv, "w", newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(["scale", "eps", "n_attractors", "noise_frac"])
                for s_, c_, nz in zip(r["scales"], r["counts"], r["noise"]):
                    w.writerow([f"{s_:.6f}", f"{r['eps_knee']*s_:.6f}", c_, f"{nz:.6f}"])
            print(f"[csv]      -> {args.scan_csv}")
        return 0

    if args.integrate:
        res = count_from_integration(
            args.coupling, args.grid_n, args.grid_lo, args.grid_hi, args.tmax,
            args.t_transient, args.record_stride, args.node_x, args.node_y,
            args.min_samples, args.dti_path)
        _report(res)
        if args.save_desc:
            Path(args.save_desc).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_desc, res["_desc"].astype(np.float32))
            print(f"[desc]     saved {res['_desc'].shape} -> {args.save_desc}")
        res.pop("_desc", None)
        if args.csv:
            _write_rows([res], Path(args.csv),
                        extra=dict(grid_n=args.grid_n, tmax=args.tmax,
                                   t_transient=args.t_transient))
        return 0

    if not args.root and not args.npz:
        ap.error("provide --root/--npz, or --integrate")
    sources = _discover(args.root, args.npz)
    if not sources:
        ap.error("no basin_data.npz found")
    ok = False
    rows: list[dict] = []
    for npz_path in sources:
        try:
            res = count_from_npz(npz_path, args.n_sub, args.min_samples, rng)
        except KeyError as exc:
            print(f"  [skip]  {npz_path}: {exc}")
            continue
        _report(res)
        rows.append(res)
        ok = True
    if ok and args.csv:
        _write_rows(rows, Path(args.csv))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
