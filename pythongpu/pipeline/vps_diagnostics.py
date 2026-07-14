#!/usr/bin/env python3
"""
vps_diagnostics.py — is there any basin structure in the VPS features, and how
many attractors do they actually support?

The number of coexisting attractors is a property of the flow, not a knob. This
tool refuses to *impose* a k and instead asks whether the saved VPS feature
matrix `vectors` (inside basin_data.npz) carries separable cluster structure at
all, and if so how many groups the evidence supports. It answers three
questions that the elbow/consensus selector silently papers over:

  1. STRUCTURE vs NOISE.  Compare the best achievable silhouette on the real
     features against a null built by independently shuffling each feature
     column (destroys joint cluster structure, preserves every marginal). If
     the real silhouette is not clearly above the null band, the features are
     consistent with a single blob — e.g. a near-synchronized regime — and the
     attractor count is simply *not resolved* by VPS. No clusterer can fix that.

  2. HOW MANY, assumption-light.  The spectral eigengap of an RBF affinity
     graph (median-bandwidth) estimates the group count from the Laplacian
     spectrum without assuming convex, equal-size blobs the way k-means /
     elbow do.

  3. IS THERE A KNEE.  Dump the inertia / silhouette / BIC curves so you can
     *see* whether the elbow is fitting a real corner or the curvature of a
     knee-less curve.

It writes a verdict, a CSV, and a diagnostics figure per input. Pure function
of `vectors`, so it runs in seconds on everything already on disk — no
re-integration.

Usage
-----
    python3 -m pythongpu.pipeline.vps_diagnostics \
        --root /mnt/data/tmp/$USER/coupling_sweep
    python3 -m pythongpu.pipeline.vps_diagnostics \
        --npz data/task_0020_lorenz_K0.6500/basin_data.npz --k-max 12
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

_K_FROM_DIR = re.compile(r"_K(-?\d+\.?\d*)")


def _coupling_of(npz_path: Path, cfg: dict | None) -> float | None:
    if cfg is not None and "coupling" in cfg:
        return float(cfg["coupling"])
    m = _K_FROM_DIR.search(npz_path.parent.name) or _K_FROM_DIR.search(npz_path.name)
    return float(m.group(1)) if m else None


def _load(npz_path: Path):
    with np.load(npz_path, allow_pickle=True) as data:
        if "vectors" not in data.files:
            raise KeyError("no 'vectors' array — predates feature persistence")
        vectors = np.asarray(data["vectors"], dtype=np.float64)
        cfg = None
        if "config" in data.files:
            try:
                cfg = dict(data["config"].item())
            except Exception:
                cfg = None
    return vectors, cfg


def _preprocess(vectors: np.ndarray, pca_dim: int, rng) -> np.ndarray:
    """z-score then PCA-reduce, matching the selector's front end."""
    Xs = StandardScaler().fit_transform(vectors)
    if Xs.shape[1] > pca_dim:
        Xs = PCA(n_components=pca_dim, random_state=0).fit_transform(Xs)
    return Xs


def _sweep_k(Xs: np.ndarray, k_max: int, rng):
    """Inertia, silhouette value, and BIC across k (k=1 inertia = total SS)."""
    k1 = KMeans(n_clusters=1, n_init=1, random_state=0).fit(Xs)
    ks = list(range(2, k_max + 1))
    inertia, silh, bic = [float(k1.inertia_)], [np.nan], [np.nan]
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
        inertia.append(float(km.inertia_))
        lab = km.labels_
        silh.append(float(silhouette_score(Xs, lab)) if np.unique(lab).size > 1 else -1.0)
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              reg_covar=1e-4, random_state=0).fit(Xs)
        bic.append(float(gmm.bic(Xs)))
    return np.array([1] + ks), np.array(inertia), np.array(silh), np.array(bic)


def _best_silhouette(Xs: np.ndarray, k_max: int) -> float:
    best = -1.0
    for k in range(2, k_max + 1):
        lab = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(Xs)
        if np.unique(lab).size > 1:
            best = max(best, float(silhouette_score(Xs, lab)))
    return best


def _null_band(Xs: np.ndarray, k_max: int, reps: int, rng):
    """Best silhouette over k on column-shuffled copies (no joint structure)."""
    out = np.empty(reps)
    for r in range(reps):
        Z = np.empty_like(Xs)
        for j in range(Xs.shape[1]):
            Z[:, j] = Xs[rng.permutation(Xs.shape[0]), j]
        out[r] = _best_silhouette(Z, k_max)
    return out


def _eigengap(Xs: np.ndarray, k_max: int, rng, n_nbr: int = 7) -> tuple[int, np.ndarray]:
    """Spectral eigengap group-count estimate from an RBF-affinity Laplacian.

    Uses Zelnik-Manor & Perona local ("self-tuning") scaling — the bandwidth at
    point i is its distance to the n_nbr-th nearest neighbour, so W_ij =
    exp(-d_ij^2 / (sigma_i sigma_j)). A single global (e.g. median-of-all-pairs)
    bandwidth over-widens when clusters are far apart and washes the block
    structure out to a spurious k=1; local scaling resolves well-separated and
    multi-scale groups alike.
    """
    D = pairwise_distances(Xs)
    kk = min(n_nbr, D.shape[0] - 1)
    sigma = np.sort(D, axis=1)[:, kk] + 1e-12          # dist to kk-th neighbour
    W = np.exp(-(D ** 2) / (sigma[:, None] * sigma[None, :]))
    np.fill_diagonal(W, 0.0)
    d = W.sum(1)
    d_inv_sqrt = 1.0 / np.sqrt(d + 1e-12)
    L = np.eye(W.shape[0]) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
    evals = np.sort(np.linalg.eigvalsh(L))
    m = min(k_max + 1, len(evals) - 1)
    gaps = np.diff(evals[:m + 1])
    k_gap = int(np.argmax(gaps) + 1)
    return k_gap, evals[:m + 1]


def diagnose(npz_path: Path, k_max: int, n_sub: int, n_eig: int,
             null_reps: int, rng) -> dict:
    vectors, cfg = _load(npz_path)
    coupling = _coupling_of(npz_path, cfg)
    Xs_full = _preprocess(vectors, pca_dim=10, rng=rng)

    n = Xs_full.shape[0]
    sub = rng.choice(n, size=min(n_sub, n), replace=False)
    Xs = Xs_full[sub]
    eig_idx = rng.choice(Xs.shape[0], size=min(n_eig, Xs.shape[0]), replace=False)

    k_values, inertia, silh, bic = _sweep_k(Xs, k_max, rng)
    real_best = float(np.nanmax(silh))
    k_silhouette = int(k_values[1:][int(np.nanargmax(silh[1:]))])
    null = _null_band(Xs, k_max, null_reps, rng)
    null_mean, null_p95 = float(null.mean()), float(np.percentile(null, 95))
    k_gap, evals = _eigengap(Xs[eig_idx], k_max, rng)

    # Verdict: structure only if the real optimum clears the null band AND the
    # silhouette is not itself vanishingly weak.
    structured = (real_best > null_p95) and (real_best >= 0.10)
    if not structured:
        verdict = ("NO SEPARABLE STRUCTURE — consistent with a single blob "
                   "(near-sync / degenerate). Attractor count NOT resolved by VPS.")
        k_supported = 1
    else:
        agree = (k_silhouette == k_gap)
        verdict = (f"STRUCTURED — silhouette k={k_silhouette}, eigengap k={k_gap} "
                   f"({'agree' if agree else 'DISAGREE → needs convergence check'}).")
        k_supported = k_silhouette

    return dict(
        npz=str(npz_path), coupling=coupling, n_ics=int(n),
        real_best_silhouette=real_best, null_mean_silhouette=null_mean,
        null_p95_silhouette=null_p95, k_silhouette=k_silhouette,
        k_eigengap=k_gap, k_supported=k_supported, structured=bool(structured),
        verdict=verdict,
        _curves=dict(k=k_values, inertia=inertia, silh=silh, bic=bic, evals=evals),
    )


def _plot(rec: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = rec["_curves"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), dpi=130)
    ax[0].plot(c["k"], c["inertia"], "-o", ms=4)
    ax[0].set(xlabel="k", ylabel="k-means inertia W(k)", title="Elbow curve\n(is there a knee?)")
    ax[1].plot(c["k"][1:], c["silh"][1:], "-o", ms=4, label="real")
    ax[1].axhline(rec["null_p95_silhouette"], ls="--", color="crimson", label="null 95%")
    ax[1].axhline(rec["null_mean_silhouette"], ls=":", color="gray", label="null mean")
    ax[1].set(xlabel="k", ylabel="silhouette", title="Silhouette vs noise floor")
    ax[1].legend(fontsize=8)
    ax[2].plot(range(1, len(c["evals"]) + 1), c["evals"], "-o", ms=4)
    ax[2].set(xlabel="eigenvalue index", ylabel=r"$\lambda$",
              title=f"Spectral eigengap\n(k≈{rec['k_eigengap']})")
    K = f"{rec['coupling']:.4f}" if rec["coupling"] is not None else "?"
    fig.suptitle(f"VPS structure diagnostics  K={K}   →   {rec['verdict']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="scan */basin_data.npz recursively")
    ap.add_argument("--npz", nargs="+", default=None, help="explicit npz path(s)/glob(s)")
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--n-sub", type=int, default=6000,
                    help="ICs subsampled for the k-sweep / silhouette / null")
    ap.add_argument("--n-eig", type=int, default=1500,
                    help="landmarks for the spectral eigengap (O(m^2) memory)")
    ap.add_argument("--null-reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args(argv)

    if not args.root and not args.npz:
        ap.error("provide --root and/or --npz")
    sources = _discover(args.root, args.npz)
    if not sources:
        ap.error("no basin_data.npz found")

    rng = np.random.default_rng(args.seed)
    out_base = Path(args.root) if args.root else sources[0].parent
    csv_path = Path(args.csv) if args.csv else out_base / "vps_diagnostics.csv"

    rows = []
    for npz_path in sources:
        try:
            rec = diagnose(npz_path, args.k_max, args.n_sub, args.n_eig,
                           args.null_reps, rng)
        except KeyError as exc:
            print(f"  [skip]  {npz_path}: {exc}")
            continue
        png = npz_path.with_name(npz_path.stem + "_vps_diag.png")
        _plot(rec, png)
        K = f"{rec['coupling']:.4f}" if rec["coupling"] is not None else "  ?   "
        print(f"[K={K}]  sil_real={rec['real_best_silhouette']:.3f}  "
              f"null_p95={rec['null_p95_silhouette']:.3f}  "
              f"k_sil={rec['k_silhouette']}  k_gap={rec['k_eigengap']}  "
              f"-> k_supported={rec['k_supported']}\n"
              f"          {rec['verdict']}  [{png.name}]")
        rec.pop("_curves")
        rows.append(rec)

    if rows:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[csv]  {len(rows)} rows -> {csv_path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
