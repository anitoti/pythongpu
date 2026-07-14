#!/usr/bin/env python3
"""
recluster_basins.py — re-partition an already-integrated coupling sweep at a
new basin count *without* re-integrating a single trajectory.

Motivation
----------
``lorenz_sweep.py`` persists the standardised VPS feature matrix ``vectors``
(shape ``(B, 2*C(N,2))``) inside every ``basin_data.npz``. Clustering,
boundary extraction and the box-counting fractal dimension are all pure
functions of that matrix — the GPU RK4 integration is not. So changing the
basin count k (or the auto-selection criterion) only requires re-running
k-means + box-counting on the saved features, which is seconds per K rather
than the tens of minutes an integration costs.

This is the fix for the flat ``D_f ≈ 1.89`` artefact: ``--k-clusters auto``
over-fragmented the 0.45–0.65 window into 6–8 basins, pinning the
box-counting dimension. Re-clustering the *same* saved features at a fixed
k=2 (or 3, 4) recovers a genuine D_f(K) curve — with no new SLURM jobs.

Usage
-----
    # Re-cluster every task in a completed sweep at fixed k=2, emit a CSV +
    # D_f-vs-K plot, and (default) drop a basin_data_k2.npz beside each source:
    python3 -m pythongpu.pipeline.recluster_basins \
        --root /mnt/data/tmp/$USER/coupling_sweep --k 2

    # Compare several k on one figure without writing per-task npz:
    python3 -m pythongpu.pipeline.recluster_basins \
        --root /mnt/data/tmp/$USER/coupling_sweep --k 2 3 4 --no-write-npz

    # Point at explicit files and re-run the auto selector under a different
    # criterion (writes basin_data_auto.npz):
    python3 -m pythongpu.pipeline.recluster_basins \
        --npz data/task_*/basin_data.npz --k auto --criterion silhouette
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
import torch

from pythongpu.pipeline.lorenz_sweep import kmeans_gpu
from pythongpu.processing.basin_clustering import select_optimal_clusters
from pythongpu.processing.box_counting import (
    extract_boundary,
    boxcount_2d_gpu,
    fractal_dimension,
    uncertainty_exponent,
)

# task_0007_lorenz_K0.5237  ->  0.5237  (K parsed from the directory name as a
# fallback when the embedded config is missing or unreadable).
_K_FROM_DIR = re.compile(r"_K(-?\d+\.?\d*)")


def _coupling_of(npz_path: Path, config: dict | None) -> float | None:
    if config is not None and "coupling" in config:
        return float(config["coupling"])
    m = _K_FROM_DIR.search(npz_path.parent.name) or _K_FROM_DIR.search(npz_path.name)
    return float(m.group(1)) if m else None


def _read_config(data) -> dict | None:
    """Unpickle the object-array ``config`` if present; tolerate its absence."""
    if "config" not in data.files:
        return None
    try:
        return dict(data["config"].item())
    except Exception:
        return None


def _grid_n_of(vectors: np.ndarray, config: dict | None) -> int:
    if config is not None and "grid_n" in config:
        return int(config["grid_n"])
    n = int(round(np.sqrt(vectors.shape[0])))
    if n * n != vectors.shape[0]:
        raise ValueError(
            f"cannot infer square grid: B={vectors.shape[0]} is not a perfect square "
            f"and config carried no grid_n")
    return n


def recluster_one(npz_path: Path, k_spec: str, criterion: str,
                  device: torch.device) -> dict:
    """Re-cluster a single basin_data.npz from its saved VPS features."""
    with np.load(npz_path, allow_pickle=True) as data:
        if "vectors" not in data.files:
            raise KeyError(
                f"{npz_path} has no 'vectors' array — it predates feature "
                f"persistence and cannot be re-clustered without re-integrating.")
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        config = _read_config(data)
        Xg = data["Xg"] if "Xg" in data.files else None
        Yg = data["Yg"] if "Yg" in data.files else None

    grid_n = _grid_n_of(vectors, config)
    coupling = _coupling_of(npz_path, config)

    if str(k_spec).lower() == "auto":
        sel = select_optimal_clusters(vectors, k_min=2, k_max=12, criterion=criterion)
        k_used = int(sel.best_k)
        labels = sel.labels.reshape(grid_n, grid_n).astype(np.int32)
    else:
        k_used = int(k_spec)
        Xg_gpu = torch.from_numpy(vectors).to(device)
        labels = kmeans_gpu(Xg_gpu, k=k_used).cpu().numpy().reshape(grid_n, grid_n).astype(np.int32)
        if device.type == "cuda":
            torch.cuda.synchronize()

    boundary = extract_boundary(labels)
    r, n = boxcount_2d_gpu(boundary, device)
    D_f, r_sq = fractal_dimension(r, n)
    ue = uncertainty_exponent(labels, d=2)

    return dict(
        npz_path=npz_path, coupling=coupling, grid_n=grid_n,
        k_used=k_used, labels=labels, boundary=boundary,
        boxcount_r=r, boxcount_n=n,
        fractal_dim=float(D_f), r_squared=float(r_sq),
        gamma=ue.gamma, gamma_r2=ue.r_squared, D_f_gamma=ue.D_f,
        Xg=Xg, Yg=Yg, config=config,
    )


def _write_recluster_npz(rec: dict, k_spec: str) -> Path:
    """Emit a schema-compatible sibling npz the aggregator can consume."""
    tag = "auto" if str(k_spec).lower() == "auto" else f"k{rec['k_used']}"
    out = rec["npz_path"].with_name(f"basin_data_{tag}.npz")
    cfg = dict(rec["config"] or {})
    cfg.update(k_clusters=int(rec["k_used"]),
               coupling=rec["coupling"] if rec["coupling"] is not None else cfg.get("coupling"))
    payload = dict(
        labels=rec["labels"], boundary=rec["boundary"],
        boxcount_r=rec["boxcount_r"], boxcount_n=rec["boxcount_n"],
        fractal_dim=np.array([rec["fractal_dim"]]),
        r_squared=np.array([rec["r_squared"]]),
        gamma=np.array([np.nan if rec["gamma"] is None else rec["gamma"]]),
        config=np.array(cfg, dtype=object),
    )
    if rec["Xg"] is not None:
        payload["Xg"] = rec["Xg"]
    if rec["Yg"] is not None:
        payload["Yg"] = rec["Yg"]
    np.savez_compressed(out, **payload)
    return out


def _discover(root: str | None, npz_globs: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    if root:
        paths += [Path(p) for p in glob.glob(
            os.path.join(root, "**", "basin_data.npz"), recursive=True)]
    for g in (npz_globs or []):
        paths += [Path(p) for p in glob.glob(g)]
    # De-dup, drop any basin_data_<tag>.npz we may have written earlier.
    seen, uniq = set(), []
    for p in sorted(paths):
        rp = p.resolve()
        if rp in seen or re.search(r"basin_data_\w+\.npz$", p.name):
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def _plot_curves(rows: list[dict], k_specs: list[str], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=140)
    for k_spec in k_specs:
        pts = sorted((r["coupling"], r["fractal_dim"]) for r in rows
                     if r["k_spec"] == k_spec and r["coupling"] is not None)
        if not pts:
            continue
        ks, dfs = zip(*pts)
        ax.plot(ks, dfs, "-o", ms=4,
                label=f"k={k_spec}" if k_spec != "auto" else "auto")
    ax.set_xlabel("coupling strength  K")
    ax.set_ylabel(r"box-counting fractal dimension  $D_f$")
    ax.set_title("Re-clustered basin fractal dimension vs. coupling")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="sweep root scanned recursively for */basin_data.npz")
    ap.add_argument("--npz", nargs="+", default=None,
                    help="explicit basin_data.npz path(s) or glob(s)")
    ap.add_argument("--k", nargs="+", default=["2"],
                    help="basin count(s): integers and/or 'auto' (default: 2)")
    ap.add_argument("--criterion", default="consensus",
                    choices=["consensus", "elbow", "bic", "silhouette"],
                    help="selector used only when a --k value is 'auto'")
    ap.add_argument("--no-write-npz", action="store_true",
                    help="do not emit per-task basin_data_<tag>.npz siblings")
    ap.add_argument("--csv", default=None,
                    help="summary CSV path (default: <root>/recluster_summary.csv)")
    ap.add_argument("--plot", default=None,
                    help="D_f-vs-K PNG path (default: <root>/recluster_Df_vs_K.png)")
    args = ap.parse_args(argv)

    if not args.root and not args.npz:
        ap.error("provide --root and/or --npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = _discover(args.root, args.npz)
    if not sources:
        ap.error("no basin_data.npz found")

    out_base = Path(args.root) if args.root else sources[0].parent
    csv_path = Path(args.csv) if args.csv else out_base / "recluster_summary.csv"
    plot_path = Path(args.plot) if args.plot else out_base / "recluster_Df_vs_K.png"

    print(f"[device]   {device}")
    print(f"[found]    {len(sources)} basin_data.npz   k={args.k}   "
          f"write_npz={not args.no_write_npz}")

    rows: list[dict] = []
    for npz_path in sources:
        for k_spec in args.k:
            try:
                rec = recluster_one(npz_path, k_spec, args.criterion, device)
            except (KeyError, ValueError) as exc:
                print(f"  [skip]  {npz_path}: {exc}")
                continue
            written = None
            if not args.no_write_npz:
                written = _write_recluster_npz(rec, k_spec)
            g = f"{rec['gamma']:.3f}" if rec["gamma"] is not None else "  -  "
            Kstr = f"{rec['coupling']:.4f}" if rec["coupling"] is not None else "  ?   "
            print(f"  [K={Kstr}]  k={rec['k_used']:<2}  D_f={rec['fractal_dim']:.3f}  "
                  f"gamma={g}  R2={rec['r_squared']:.3f}"
                  + (f"  -> {written.name}" if written else ""))
            rows.append(dict(
                npz=str(npz_path), k_spec=str(k_spec), coupling=rec["coupling"],
                k_used=rec["k_used"], fractal_dim=rec["fractal_dim"],
                r_squared=rec["r_squared"],
                gamma="" if rec["gamma"] is None else rec["gamma"],
                gamma_r2="" if rec["gamma_r2"] is None else rec["gamma_r2"],
                D_f_gamma="" if rec["D_f_gamma"] is None else rec["D_f_gamma"]))

    if not rows:
        print("[done]     nothing re-clustered (no usable 'vectors' found).")
        return 1

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv]      {len(rows)} rows -> {csv_path}")

    try:
        _plot_curves(rows, [str(k) for k in args.k], plot_path)
        print(f"[plot]     {plot_path}")
    except Exception as exc:
        print(f"[plot]     skipped ({exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
