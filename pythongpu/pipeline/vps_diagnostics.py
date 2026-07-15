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
     features against a null band measured on a structureless reference cloud
     (uniform over the PCA bounding box — the gap-statistic reference). The
     verdict is driven by EFFECT SIZE, not a bare inequality. Only BALANCED
     partitions count: a split whose smallest cluster is a handful of points
     isolates outliers and scores a meaningless ~0.95. If the real optimum does
     not clear the null by --margin, the features are consistent with a single
     blob — e.g. a near-synchronized regime — and the attractor count is simply
     *not resolved* by VPS. No clusterer can fix that.

     History: the original test used a column-shuffle null and no balance check.
     Shuffling preserves each marginal, so on outlier-heavy features k-means
     scored ~0.95 by splitting outliers off the REAL and the SHUFFLED data
     alike; the null inflated to 0.94 and STRUCTURED was declared on margins of
     0.018. Winsorising + robust scaling, the balance filter, and the uniform
     reference each remove one leg of that failure.

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
from sklearn.preprocessing import RobustScaler

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


def _preprocess(vectors: np.ndarray, pca_dim: int, winsor: float = 0.005) -> np.ndarray:
    """
    Winsorise, robust-scale, then PCA-reduce.

    The original z-score + PCA front end let a handful of extreme ICs dominate
    the components. k-means then maximises silhouette by splitting those
    outliers off — scoring ~0.95 on REAL and SHUFFLED data alike, which is what
    inflated the null band to 0.94 and made the STRUCTURED verdict meaningless.
    Clipping the tails and scaling by the median/IQR removes that leverage.
    """
    X = np.asarray(vectors, dtype=np.float64)
    if winsor > 0:
        lo = np.quantile(X, winsor, axis=0)
        hi = np.quantile(X, 1.0 - winsor, axis=0)
        X = np.clip(X, lo, hi)
    Xs = RobustScaler().fit_transform(X)
    # Constant (zero-IQR) columns become 0 and carry no information; drop them
    # so PCA is not fed dead dimensions.
    keep = Xs.std(axis=0) > 1e-12
    if keep.any():
        Xs = Xs[:, keep]
    if Xs.shape[1] > pca_dim:
        Xs = PCA(n_components=pca_dim, random_state=0).fit_transform(Xs)
    return Xs


def _fit_k(mat: np.ndarray, k: int):
    """Cluster at k; return (labels, silhouette, smallest-cluster fraction)."""
    lab = KMeans(n_clusters=int(k), n_init=5, random_state=0).fit_predict(mat)
    if np.unique(lab).size < 2:
        return None, -1.0, 0.0
    counts = np.bincount(lab, minlength=int(k))
    min_frac = float(counts[counts > 0].min()) / len(lab)
    return lab, float(silhouette_score(mat, lab)), min_frac


def _best_balanced_sil(mat: np.ndarray, k_values, min_cluster_frac: float):
    """
    Best silhouette over k, considering only BALANCED partitions.

    A partition whose smallest cluster is a handful of points is an outlier
    split, not a basin: it scores a near-perfect silhouette while telling us
    nothing. Requiring every cluster to hold at least `min_cluster_frac` of the
    points is what makes the silhouette mean "there are real groups here".
    """
    best_sil, best_k, best_frac = -1.0, None, 0.0
    for k in k_values:
        lab, sil, frac = _fit_k(mat, int(k))
        if lab is None or frac < min_cluster_frac:
            continue
        if sil > best_sil:
            best_sil, best_k, best_frac = sil, int(k), frac
    return best_sil, best_k, best_frac


def _sweep_k(Xs: np.ndarray, k_max: int, min_cluster_frac: float):
    """Inertia, silhouette, BIC and smallest-cluster fraction across k."""
    k1 = KMeans(n_clusters=1, n_init=1, random_state=0).fit(Xs)
    ks = list(range(2, k_max + 1))
    inertia, silh, bic, fracs = [float(k1.inertia_)], [np.nan], [np.nan], [1.0]
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
        inertia.append(float(km.inertia_))
        lab = km.labels_
        if np.unique(lab).size > 1:
            silh.append(float(silhouette_score(Xs, lab)))
            counts = np.bincount(lab, minlength=k)
            fracs.append(float(counts[counts > 0].min()) / len(lab))
        else:
            silh.append(-1.0)
            fracs.append(0.0)
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              reg_covar=1e-4, random_state=0).fit(Xs)
        bic.append(float(gmm.bic(Xs)))
    return (np.array([1] + ks), np.array(inertia), np.array(silh),
            np.array(bic), np.array(fracs))


def _reference(Xs: np.ndarray, rng, kind: str) -> np.ndarray:
    """
    A featureless reference cloud with no cluster structure.

    'gaussian' (default) draws from a single multivariate normal matched to the
    data's mean and covariance. This is the right null for the question actually
    being asked — "one blob, or several?" — because it is unimodal BY
    CONSTRUCTION yet reproduces the real elliptical spread. Splitting a unimodal
    elliptical cloud in half already earns a decent silhouette, so a genuine
    multi-basin structure must beat THAT, not merely beat a uniform box.

    'uniform' draws over the PCA bounding box (Tibshirani gap-statistic
    reference). Shape-mismatched: a box is easier to beat than an ellipsoid, so
    a heavy-tailed unimodal blob can clear it and register a false positive.

    'shuffle' (legacy) permutes each column. It inherits the real marginals, so
    on outlier-heavy features it inflates to ~0.94 and destroys the test.
    """
    if kind == "gaussian":
        mu = Xs.mean(axis=0)
        C = np.cov(Xs, rowvar=False)
        C = np.atleast_2d(C) + 1e-9 * np.eye(Xs.shape[1])   # keep it PSD
        return rng.multivariate_normal(mu, C, size=Xs.shape[0])
    if kind == "uniform":
        return rng.uniform(Xs.min(axis=0), Xs.max(axis=0), size=Xs.shape)
    Z = Xs.copy()                                   # 'shuffle' (legacy)
    for j in range(Z.shape[1]):
        Z[:, j] = Z[rng.permutation(Z.shape[0]), j]
    return Z


def _null_band(Xs: np.ndarray, k_values, reps: int, rng,
               min_cluster_frac: float, kind: str):
    """Best BALANCED silhouette achievable on a structureless reference."""
    out = np.empty(reps)
    for r in range(reps):
        s, _, _ = _best_balanced_sil(_reference(Xs, rng, kind), k_values,
                                     min_cluster_frac)
        out[r] = s
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
             null_reps: int, rng, min_cluster_frac: float = 0.05,
             margin: float = 0.15, null_kind: str = "gaussian") -> dict:
    vectors, cfg = _load(npz_path)
    coupling = _coupling_of(npz_path, cfg)

    # Subsample the RAW vectors before preprocessing: the full matrix is
    # 130321 x 6806, and winsorising/scaling it in float64 costs ~7 GB for no
    # gain — the subsample estimates the same geometry.
    n = vectors.shape[0]
    sub = rng.choice(n, size=min(n_sub, n), replace=False)
    Xs = _preprocess(vectors[sub], pca_dim=10)
    del vectors
    eig_idx = rng.choice(Xs.shape[0], size=min(n_eig, Xs.shape[0]), replace=False)

    k_values, inertia, silh, bic, fracs = _sweep_k(Xs, k_max, min_cluster_frac)
    ks = k_values[1:]
    real_best, k_silhouette, best_frac = _best_balanced_sil(Xs, ks, min_cluster_frac)
    null = _null_band(Xs, ks, null_reps, rng, min_cluster_frac, null_kind)
    null_mean, null_p95 = float(null.mean()), float(np.percentile(null, 95))
    k_gap, evals = _eigengap(Xs[eig_idx], k_max, rng)

    # Verdict is driven by EFFECT SIZE, not a bare inequality: the old test
    # declared STRUCTURED on margins as thin as 0.018 between two ~0.94 numbers.
    effect = real_best - null_p95
    if k_silhouette is None:
        structured, k_supported = False, 1
        verdict = ("NO SEPARABLE STRUCTURE — every partition is an outlier split "
                   f"(no cluster holds >= {min_cluster_frac:.0%} of points). "
                   "Attractor count NOT resolved by VPS.")
    elif effect < margin:
        structured, k_supported = False, 1
        verdict = (f"NO SEPARABLE STRUCTURE — effect size {effect:+.3f} < "
                   f"{margin:.3f} (real {real_best:.3f} vs null p95 {null_p95:.3f}); "
                   "consistent with a single blob. Attractor count NOT resolved.")
    else:
        structured, k_supported = True, int(k_silhouette)
        agree = (k_silhouette == k_gap)
        verdict = (f"STRUCTURED — effect {effect:+.3f}, silhouette k={k_silhouette} "
                   f"(smallest cluster {best_frac:.1%}), eigengap k={k_gap} "
                   f"({'agree' if agree else 'DISAGREE → needs convergence check'}).")

    return dict(
        npz=str(npz_path), coupling=coupling, n_ics=int(n),
        real_best_silhouette=float(real_best), null_mean_silhouette=null_mean,
        null_p95_silhouette=null_p95, effect_size=float(effect),
        min_cluster_fraction=float(best_frac),
        k_silhouette=(0 if k_silhouette is None else int(k_silhouette)),
        k_eigengap=k_gap, k_supported=k_supported, structured=bool(structured),
        null_kind=null_kind, verdict=verdict,
        _curves=dict(k=k_values, inertia=inertia, silh=silh, bic=bic,
                     evals=evals, fracs=fracs),
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


def _write_csv(rows: list[dict], csv_path) -> None:
    """Rewrite the summary CSV from scratch; called after every file so a late
    failure (or a Ctrl-C) never discards verdicts already computed."""
    if not rows:
        return
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


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
    ap.add_argument("--min-cluster-frac", type=float, default=0.05,
                    help="reject a partition whose smallest cluster holds less "
                         "than this fraction of points — such a split isolates "
                         "outliers and scores a meaningless ~0.95 silhouette")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="required effect size (real silhouette minus null p95) "
                         "to call a slice STRUCTURED. Default 0.15 because the "
                         "raw statistic's seed-to-seed scatter on heavy-tailed "
                         "features is ~0.1: on one fixed configuration the real "
                         "silhouette moved 0.172 -> 0.076 on the RNG draw alone, "
                         "so a 0.05 margin sat inside the noise and could be "
                         "crossed by reseeding. A calibrated null puts genuine "
                         "structure at effect ~ +0.4, far above this floor.")
    ap.add_argument("--null-kind", default="gaussian",
                    choices=["gaussian", "uniform", "shuffle"],
                    help="reference cloud for the null band. 'gaussian' "
                         "(default) = a single covariance-matched normal: "
                         "unimodal by construction, so real structure must beat "
                         "a one-blob ellipsoid. 'uniform' = PCA bounding box "
                         "(shape-mismatched; a heavy-tailed blob can clear it). "
                         "'shuffle' = legacy column shuffle, inflated by "
                         "outliers to ~0.94 — do not use.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the diagnostics PNGs entirely (verdicts and CSV "
                         "are still produced). Useful where matplotlib's imaging "
                         "stack is unavailable — e.g. a login node without "
                         "libjpeg-turbo loaded.")
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
                           args.null_reps, rng,
                           min_cluster_frac=args.min_cluster_frac,
                           margin=args.margin, null_kind=args.null_kind)
        except KeyError as exc:
            print(f"  [skip]  {npz_path}: {exc}")
            continue
        # Print the verdict BEFORE plotting: the science must never be lost to a
        # cosmetic failure. (A missing libjpeg made matplotlib's import explode
        # mid-loop and threw away every computed verdict.)
        K = f"{rec['coupling']:.4f}" if rec["coupling"] is not None else "  ?   "
        print(f"[K={K}]  sil_real={rec['real_best_silhouette']:.3f}  "
              f"null_p95={rec['null_p95_silhouette']:.3f}  "
              f"effect={rec['effect_size']:+.3f}  "
              f"min_clu={rec['min_cluster_fraction']:.1%}  "
              f"k_sil={rec['k_silhouette']}  k_gap={rec['k_eigengap']}  "
              f"-> k_supported={rec['k_supported']}\n"
              f"          {rec['verdict']}")
        if not args.no_plot:
            png = npz_path.with_name(npz_path.stem + "_vps_diag.png")
            try:
                _plot(rec, png)
                print(f"          [{png.name}]")
            except Exception as exc:  # plotting is optional, never fatal
                print(f"          [plot skipped: {type(exc).__name__}: {exc}]")
        rec.pop("_curves")
        rows.append(rec)
        # Flush the CSV as we go so a late failure cannot discard earlier work.
        _write_csv(rows, csv_path)

    if rows:
        print(f"[csv]  {len(rows)} rows -> {csv_path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
