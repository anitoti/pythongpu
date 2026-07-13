"""
Data-driven basin-count selection for the Lorenz / Kuramoto sweep pipelines.

Historically the number of basins K was hard-wired (``--k-clusters 5`` with a
comment that "k=8 was found to be optimal using the elbow method", i.e. a value
transcribed once by hand and then frozen). This module replaces that fixed
choice with a dynamic model-selection sweep that estimates the optimal cluster
count directly from the feature geometry using three complementary criteria:

    Elbow (inertia knee)   — the largest-curvature point of the within-cluster
                             sum-of-squares curve W(k), located by the Kneedle
                             chord-distance heuristic (Satopää et al., 2011).
    Bayesian Information    — argmin_k BIC of a Gaussian-mixture model, which
      Criterion (BIC)        penalises free parameters and so guards against
                             the elbow's tendency to over-segment.
    Silhouette coefficient  — argmax_k of the mean silhouette s(k) (Rousseeuw,
                             1987), a separation/compactness ratio in [-1, 1].

A consensus estimate (the rounded median of the three) is returned alongside
the individual optima and the full diagnostic curves so the selection is
auditable rather than a single opaque number.

High-dimensional inputs (e.g. the 2*C(N,2) VPS feature block) are z-scored and
projected to a modest PCA subspace before the sweep, which keeps the silhouette
and mixture fits tractable and grid-resolution independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterSelection:
    """Result of a dynamic basin-count model-selection sweep."""
    best_k: int                       # consensus optimal cluster count
    labels: np.ndarray                # (n_samples,) int labels at best_k
    k_values: np.ndarray              # candidate k grid that was swept
    inertia: np.ndarray               # within-cluster SSE  W(k)
    bic: np.ndarray                   # Gaussian-mixture BIC(k)
    silhouette: np.ndarray            # mean silhouette s(k)
    k_elbow: int                      # elbow-criterion optimum
    k_bic: int                        # BIC-criterion optimum
    k_silhouette: int                 # silhouette-criterion optimum
    criterion: str                    # criterion used to set best_k
    meta: dict = field(default_factory=dict)


def _kneedle_elbow(k_values: np.ndarray, w: np.ndarray) -> int:
    """
    Locate the knee of a monotone-decreasing inertia curve W(k) as the point of
    maximal distance to the chord joining its endpoints (the Kneedle heuristic).
    Both axes are min-max normalised so the distance is scale-free.
    """
    if len(k_values) < 3:
        return int(k_values[0])
    x = (k_values - k_values.min()) / (np.ptp(k_values) + 1e-12)
    y = (w - w.min()) / (np.ptp(w) + 1e-12)
    # Distance of each (x, y) from the straight chord (x0,y0)->(x1,y1).
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    num = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    den = np.hypot(y1 - y0, x1 - x0) + 1e-12
    return int(k_values[int(np.argmax(num / den))])


def select_optimal_clusters(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 15,
    criterion: str = "consensus",
    pca_dim: int = 10,
    silhouette_sample: int = 4096,
    random_state: int = 42,
) -> ClusterSelection:
    """
    Estimate the optimal number of basins for the feature matrix ``X``
    (n_samples, n_features) by sweeping k in [k_min, k_max] and scoring each
    candidate with the elbow, BIC and silhouette criteria.

    Parameters
    ----------
    criterion : which optimum sets ``best_k`` — one of
        {"elbow", "bic", "silhouette", "consensus"}. "consensus" (default) is
        the rounded median of the three individual optima.
    pca_dim : if ``X`` has more features than this, it is z-scored and
        PCA-reduced to ``pca_dim`` components before the sweep, making the
        selection independent of the raw feature dimensionality (and hence of
        grid resolution / node count).
    silhouette_sample : cap on the number of points fed to the O(n^2)
        silhouette computation; a fixed-seed subsample is used above this size.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n_samples = X.shape[0]
    k_max = min(k_max, n_samples - 1)
    if k_max < k_min:
        raise ValueError(f"need at least {k_min + 1} samples, got {n_samples}")

    # Standardise, then PCA-reduce wide feature blocks for tractability.
    Xs = StandardScaler().fit_transform(X)
    if Xs.shape[1] > pca_dim:
        Xs = PCA(n_components=pca_dim, random_state=random_state).fit_transform(Xs)

    rng = np.random.default_rng(random_state)
    if n_samples > silhouette_sample:
        sil_idx = rng.choice(n_samples, size=silhouette_sample, replace=False)
    else:
        sil_idx = np.arange(n_samples)

    k_values = np.arange(k_min, k_max + 1)
    inertia, bic, silhouette = [], [], []
    labels_by_k: dict[int, np.ndarray] = {}

    for k in k_values:
        km = KMeans(n_clusters=int(k), n_init=10, random_state=random_state)
        lab = km.fit_predict(Xs)
        labels_by_k[int(k)] = lab
        inertia.append(float(km.inertia_))

        gmm = GaussianMixture(n_components=int(k), covariance_type="full",
                              reg_covar=1e-4, random_state=random_state)
        gmm.fit(Xs)
        bic.append(float(gmm.bic(Xs)))

        sub_lab = lab[sil_idx]
        if np.unique(sub_lab).size > 1:
            silhouette.append(float(silhouette_score(Xs[sil_idx], sub_lab)))
        else:
            silhouette.append(-1.0)

    inertia = np.asarray(inertia)
    bic = np.asarray(bic)
    silhouette = np.asarray(silhouette)

    k_elbow = _kneedle_elbow(k_values, inertia)
    k_bic = int(k_values[int(np.argmin(bic))])
    k_silhouette = int(k_values[int(np.argmax(silhouette))])

    if criterion == "elbow":
        best_k = k_elbow
    elif criterion == "bic":
        best_k = k_bic
    elif criterion == "silhouette":
        best_k = k_silhouette
    elif criterion == "consensus":
        best_k = int(round(float(np.median([k_elbow, k_bic, k_silhouette]))))
    else:
        raise ValueError(f"unknown criterion {criterion!r}")
    best_k = int(np.clip(best_k, k_min, k_max))

    return ClusterSelection(
        best_k=best_k, labels=labels_by_k[best_k], k_values=k_values,
        inertia=inertia, bic=bic, silhouette=silhouette,
        k_elbow=k_elbow, k_bic=k_bic, k_silhouette=k_silhouette,
        criterion=criterion,
        meta={"n_samples": n_samples, "pca_dim": int(min(pca_dim, X.shape[1]))},
    )


def plot_selection(sel: ClusterSelection, path, title: str = "") -> None:
    """Render the three model-selection curves with their optima marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(sel.k_values, sel.inertia, "o-", color="steelblue")
    axes[0].axvline(sel.k_elbow, ls="--", color="crimson",
                    label=f"elbow k={sel.k_elbow}")
    axes[0].set_ylabel("Inertia  W(k)")
    axes[0].set_title("Elbow (Kneedle)")

    axes[1].plot(sel.k_values, sel.bic, "o-", color="darkorange")
    axes[1].axvline(sel.k_bic, ls="--", color="crimson", label=f"BIC k={sel.k_bic}")
    axes[1].set_ylabel("Gaussian-mixture BIC(k)")
    axes[1].set_title("Bayesian Information Criterion")

    axes[2].plot(sel.k_values, sel.silhouette, "o-", color="seagreen")
    axes[2].axvline(sel.k_silhouette, ls="--", color="crimson",
                    label=f"silhouette k={sel.k_silhouette}")
    axes[2].set_ylabel("Mean silhouette  s(k)")
    axes[2].set_title("Silhouette coefficient")

    for ax in axes:
        ax.set_xlabel("k  (number of basins)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    suptitle = title or "Dynamic basin-count selection"
    fig.suptitle(f"{suptitle}   →   consensus k={sel.best_k} "
                 f"(elbow={sel.k_elbow}, BIC={sel.k_bic}, "
                 f"silhouette={sel.k_silhouette})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
