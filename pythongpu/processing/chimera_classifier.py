"""
VPS clustering + chimera classification (methods plan item 4).

Coherence measure: neighbor-relative variance (not phase-based). For each
node i, over a time window:

    local_field_i(t) = sum_j W[i,j] * x_j(t) / sum_j W[i,j]      (strength-
                                                                    weighted
                                                                    neighbor
                                                                    mean)
    coherence_i       = exp( -Var(x_i(t) - local_field_i(t)) / Var(x_i(t)) )

Bounded in (0, 1]: -> 1 when node i moves with its neighbors (synchronized),
-> 0 when its fluctuations relative to neighbors are large relative to its
own dynamic range. Scale-invariant by construction, so it applies unchanged
across Lorenz, Rossler, Kuramoto, and Van der Pol regardless of each
system's natural amplitude. Isolated nodes (no neighbors) return 1.0 — a
node cannot be "out of step" with an empty neighbor set.

A VPS (vector pattern state) is the N-vector of per-node coherence values
for one initial condition / trajectory. classify_chimera operates on a
single VPS; cluster_vps_population operates on a matrix of many VPS vectors
(one per initial condition across a basin sweep) for attractor-type/basin
labeling.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def local_coherence(trajectory, W, window: tuple | None = None) -> np.ndarray:
    """
    Per-node neighbor-relative-variance coherence.

    Parameters
    ----------
    trajectory : torch.Tensor or np.ndarray, shape (T, D, N)
        D is the state dimension (e.g. 3 for Lorenz/Rossler, 2 for Van der
        Pol, 1 for Kuramoto). Index 0 along axis 1 is treated as x, matching
        the coupling-through-x convention used by every oscillator in this
        repo (oscillators/lorenz.py, rossler.py, vanderpol.py).
    W : (N, N) array
        Weighted or binary adjacency/coupling matrix. Row i's nonzero
        entries define node i's neighbors.
    window : (start, end) tuple of timestep indices, optional
        Restricts the coherence computation to trajectory[start:end].
        Default: the full trajectory. Sweeps should pass the post-transient
        tail.

    Returns
    -------
    coherence : np.ndarray, shape (N,)
    """
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach().cpu().numpy()
    trajectory = np.asarray(trajectory)
    W = np.asarray(W, dtype=np.float64)

    if window is not None:
        start, end = window
        trajectory = trajectory[start:end]

    x = trajectory[:, 0, :]  # (T, N) — x-component of every node
    n = x.shape[1]

    row_sum = W.sum(axis=1)  # (N,)
    has_neighbors = row_sum > 0

    coherence = np.ones(n)  # default: isolated / degenerate nodes -> 1.0

    if not np.any(has_neighbors):
        return coherence

    with np.errstate(invalid="ignore", divide="ignore"):
        local_field = x @ W.T / row_sum  # (T, N), broadcasts (N,) over columns

    deviation = x - local_field
    var_dev = deviation.var(axis=0)  # (N,)
    var_own = x.var(axis=0)  # (N,)

    valid = has_neighbors & (var_own > 1e-12)
    coherence[valid] = np.exp(-var_dev[valid] / var_own[valid])

    return coherence


def bimodality_coefficient(values) -> float:
    """
    Sample-corrected bimodality coefficient (Pfister et al. 2013 / SAS
    formula):

        BC = (skew^2 + 1) / (kurtosis + 3*(n-1)^2 / ((n-2)*(n-3)))

    kurtosis here is excess kurtosis (normal distribution = 0). BC > 5/9
    (~0.555) is the standard "substantially bimodal" cutoff.

    Requires n > 3. Returns np.nan for n <= 3 (correction term undefined).
    """
    values = np.asarray(values).flatten()
    n = values.shape[0]
    if n <= 3:
        return float("nan")

    # Near-constant input (e.g. a fully-synchronized coherence vector, all
    # ~1.0) triggers a benign scipy precision-loss warning on the moment
    # calculation — expected here, not a bug, so it's suppressed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        g = skew(values)
        k = kurtosis(values, fisher=True)  # excess kurtosis
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g ** 2 + 1) / (k + correction))


def classify_chimera(
    coherence_vector,
    min_group_frac: float = 0.1,
    gap_threshold: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Classify a single VPS (per-node coherence vector) as a chimera state.

    Splits the coherence vector into 2 clusters via k-means. is_chimera is
    True only if BOTH clusters hold >= min_group_frac of nodes AND the
    cluster-mean gap >= gap_threshold — this rejects fully-synchronized
    (all-high) and fully-incoherent (all-low) states, which a 2-means split
    would otherwise partition arbitrarily despite there being no real
    structure to split.

    Returns
    -------
    dict with keys: is_chimera, labels, cluster_means, cluster_fractions,
    gap, bimodality_coefficient.
    """
    coherence_vector = np.asarray(coherence_vector).reshape(-1, 1)
    n = coherence_vector.shape[0]

    km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    labels = km.fit_predict(coherence_vector)

    means = np.array([coherence_vector[labels == k].mean() for k in range(2)])
    fractions = np.array([(labels == k).sum() / n for k in range(2)])
    gap = float(abs(means[0] - means[1]))

    is_chimera = bool(fractions.min() >= min_group_frac and gap >= gap_threshold)

    return {
        "is_chimera": is_chimera,
        "labels": labels,
        "cluster_means": means,
        "cluster_fractions": fractions,
        "gap": gap,
        "bimodality_coefficient": bimodality_coefficient(coherence_vector.flatten()),
    }


def cluster_vps_population(
    vps_matrix,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
) -> dict:
    """
    Cluster a population of VPS vectors (one basin sweep's worth of
    initial conditions) into attractor-type labels, for basin-map coloring.

    Parameters
    ----------
    vps_matrix : (M, N) array — M initial conditions x N-node coherence
        vector each.
    k_min, k_max : int
        Range of cluster counts to try; k is selected by silhouette score.

    Returns
    -------
    dict with keys: labels (M,), k (selected cluster count),
    silhouette_scores (dict k -> score).
    """
    vps_matrix = np.asarray(vps_matrix)
    m = vps_matrix.shape[0]
    k_max = min(k_max, m - 1)

    if k_max < k_min:
        km = KMeans(n_clusters=k_min, random_state=random_state, n_init=10)
        labels = km.fit_predict(vps_matrix)
        return {"labels": labels, "k": k_min, "silhouette_scores": {}}

    silhouette_scores = {}
    best_k, best_score, best_labels = k_min, -1.0, None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(vps_matrix)
        score = silhouette_score(vps_matrix, labels)
        silhouette_scores[k] = score
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    return {"labels": best_labels, "k": best_k, "silhouette_scores": silhouette_scores}
