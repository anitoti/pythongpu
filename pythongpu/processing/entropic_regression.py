#!/usr/bin/env python3
"""
Entropic regression for directed brain-network reconstruction.

Entropic regression (ER) is the forward-then-backward variant of optimal
causation entropy (oCSE). The forward pass greedily grows each target node's
causal parent set by adding the lagged source that maximally reduces the
target's conditional entropy, provided the gain is significant. The backward
pass then removes redundant sources by re-testing each retained source while
conditioning on the others. A source survives only if its conditional mutual
information with the target remains significant after the rest of the selected
set has been accounted for.

This implementation follows Fish, DeWitt, AlMomani, Laurienti & Bollt (2021),
"Entropic regression with neurologically motivated applications," Chaos 31,
113105.

Relation to the existing oCSE backend (``processing/causation_entropy.py``):
    - oCSE implements only the forward greedy aggregation step.
    - ER adds the backward elimination step that prunes indirect or spurious
      edges left behind by the greedy sweep.

Edge semantics match ``causation_entropy.py`` so the two outputs are
interchangeable downstream:
    ``adj[source, sink] = CMI weight of the retained lagged edge source -> sink``
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from pythongpu.processing.causation_entropy import (
    conditional_mutual_information,
    _cmi_significant,
    plot_causal_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _forward_pass(
    target: int,
    X_current: np.ndarray,
    X_lagged: np.ndarray,
    n_regions: int,
    n_samples: int,
    alpha: float,
) -> tuple[list[int], dict[int, float]]:
    """Run the greedy forward oCSE pass for one target node.

    Args:
        target: Index of the sink region being reconstructed.
        X_current: Target-aligned samples at times ``t``.
        X_lagged: Candidate sources at times ``t-1``.
        n_regions: Number of ROIs in the parcellation.
        n_samples: Number of time samples after lag alignment.
        alpha: Significance level for the conditional-mutual-information test.

    Returns:
        A pair ``(causal_set, weights)`` where ``causal_set`` contains the
        accepted source indices and ``weights[source]`` stores the retained CMI
        value for that source.
    """
    x_target = X_current[:, [target]]
    causal_set: list[int] = []
    weights: dict[int, float] = {}

    while True:
        best_cmi, best_source = -1.0, -1
        for source in range(n_regions):
            if source == target or source in causal_set:
                continue
            Z = X_lagged[:, causal_set] if causal_set else np.empty((n_samples, 0))
            cmi = conditional_mutual_information(x_target, X_lagged[:, [source]], Z)
            if cmi > best_cmi:
                best_cmi, best_source = cmi, source
        if best_source == -1 or best_cmi <= 0:
            break
        if _cmi_significant(best_cmi, n_samples, alpha):
            causal_set.append(best_source)
            weights[best_source] = best_cmi
        else:
            break
    return causal_set, weights


def _backward_pass(
    target: int,
    causal_set: list[int],
    X_current: np.ndarray,
    X_lagged: np.ndarray,
    n_samples: int,
    alpha: float,
) -> tuple[list[int], dict[int, float]]:
    """Run the ER backward elimination pass.

    The backward pass tests each retained source ``s`` against the conditioning
    set formed by the other retained sources. If ``I(X_t; X_{s,t-1} | Z)`` is no
    longer significant, then ``s`` is treated as redundant and removed.

    This is the mechanism that turns the greedy forward sweep into a true
    forward-then-backward selector: indirect edges that only appeared useful
    because a missing mediator was absent from the conditioning set are pruned
    away once the mediator has been admitted.

    Args:
        target: Index of the sink region being reconstructed.
        causal_set: Sources selected by the forward pass.
        X_current: Target-aligned samples at times ``t``.
        X_lagged: Candidate sources at times ``t-1``.
        n_samples: Number of time samples after lag alignment.
        alpha: Significance level for the conditional-mutual-information test.

    Returns:
        A pair ``(kept, weights)`` where ``kept`` is the pruned source list and
        ``weights[source]`` stores the backward-pass CMI conditioned on the
        remaining retained sources.
    """
    x_target = X_current[:, [target]]
    kept = list(causal_set)
    new_weights: dict[int, float] = {}
    changed = True
    # iterate to a fixed point: removing one edge can make another redundant
    while changed and kept:
        changed = False
        for source in list(kept):
            others = [s for s in kept if s != source]
            Z = X_lagged[:, others] if others else np.empty((n_samples, 0))
            cmi = conditional_mutual_information(x_target, X_lagged[:, [source]], Z)
            if not _cmi_significant(cmi, n_samples, alpha):
                kept.remove(source)
                changed = True
                break
            new_weights[source] = cmi
    return kept, new_weights


def _process_single_target(
    args: tuple[int, np.ndarray, np.ndarray, int, int, float]
) -> tuple[int, list[tuple[int, float]]]:
    """Reconstruct the incoming edges for a single sink node."""
    target, X_current, X_lagged, n_regions, n_samples, alpha = args
    causal_set, _ = _forward_pass(target, X_current, X_lagged, n_regions, n_samples, alpha)
    kept, weights = _backward_pass(target, causal_set, X_current, X_lagged, n_samples, alpha)
    return target, [(s, weights[s]) for s in kept]


def entropic_regression(
    timeseries: np.ndarray,
    max_lag: int = 1,
    alpha: float = 0.05,
    n_jobs: int = 4,
) -> np.ndarray:
    """Reconstruct a directed weighted adjacency matrix with ER.

    The input is a ``(T, N)`` time series array. The series is shifted by
    ``max_lag`` steps so each sink sample ``x_t`` is paired with its candidate
    lagged sources ``x_{t-1}``. Each target ROI is reconstructed independently,
    then the retained source weights are assembled into a directed adjacency
    matrix with ``adj[source, sink]`` semantics.

    Args:
        timeseries: Parcellated ROI signals with shape ``(T, N)``.
        max_lag: Temporal lag enforcing cause-precedes-effect.
        alpha: Significance level for the chi-squared causation-entropy test.
        n_jobs: Worker processes used for the per-target reconstruction.

    Returns:
        A ``(N, N)`` NumPy array of directed edge weights.
    """
    timeseries = np.asarray(timeseries, dtype=np.float64)
    n_timepoints, n_regions = timeseries.shape
    X_current = timeseries[max_lag:]
    X_lagged = timeseries[:-max_lag]
    n_samples = X_current.shape[0]
    logging.info(f"ER on {n_timepoints} TRs x {n_regions} ROIs (lag={max_lag}, alpha={alpha})")

    adj = np.zeros((n_regions, n_regions), dtype=np.float64)
    worker_args = [
        (t, X_current, X_lagged, n_regions, n_samples, alpha) for t in range(n_regions)
    ]
    if n_jobs == 1:
        results = [_process_single_target(a) for a in worker_args]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_process_single_target, a) for a in worker_args]
            for f in as_completed(futures):
                results.append(f.result())
    for target, edges in results:
        for source, weight in edges:
            adj[source, target] = weight

    n_edges = int(np.count_nonzero(adj))
    density = n_edges / (n_regions * (n_regions - 1)) if n_regions > 1 else 0.0
    logging.info(f"ER network: {n_edges} directed edges (density {density:.4f})")
    return adj


def fuse_structural_functional(
    functional_adj: np.ndarray,
    structural_adj: np.ndarray,
    mode: str = "gate",
) -> np.ndarray:
    """Fuse a functional ER graph with a structural connectome.

    Args:
        functional_adj: Directed functional adjacency from entropic regression.
        structural_adj: Structural adjacency or tractography matrix.
        mode: Fusion rule. ``"gate"`` keeps functional edges only where a
            structural edge exists. ``"weight"`` scales by row-normalized
            structural weights.

    Returns:
        A fused adjacency matrix with the same shape as the inputs.
    """
    F = np.asarray(functional_adj, dtype=np.float64)
    S = np.asarray(structural_adj, dtype=np.float64)
    if F.shape != S.shape:
        raise ValueError(f"shape mismatch: functional {F.shape} vs structural {S.shape}")
    if mode == "gate":
        return np.where(S > 0, F, 0.0)
    if mode == "weight":
        row = S.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        return F * (S / row)
    raise ValueError("mode must be 'gate' or 'weight'")


def main() -> None:
    """Parse CLI arguments, run ER, and write the outputs to disk."""
    ap = argparse.ArgumentParser(description="Entropic regression directed-network reconstruction.")
    ap.add_argument("--input", required=True, help="Parcellated timeseries CSV (T x N).")
    ap.add_argument("--out_csv", required=True, help="Output adjacency matrix CSV.")
    ap.add_argument("--out_img", default=None, help="Optional heatmap PNG.")
    ap.add_argument("--structural", default=None, help="Optional DTI structural adjacency CSV to fuse.")
    ap.add_argument("--fuse_mode", default="gate", choices=["gate", "weight"])
    ap.add_argument("--max_lag", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_jobs", type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        logging.error(f"Input not found: {args.input}. Run parcellation.py first.")
        return

    t0 = time.time()
    ts = pd.read_csv(args.input).values.astype(np.float64)
    adj = entropic_regression(ts, args.max_lag, args.alpha, args.n_jobs)

    if args.structural:
        S = pd.read_csv(args.structural, header=None).values.astype(np.float64)
        adj = fuse_structural_functional(adj, S, mode=args.fuse_mode)
        logging.info(f"Fused with structural connectome ({args.fuse_mode}); "
                     f"{int(np.count_nonzero(adj))} edges survive.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    pd.DataFrame(adj).to_csv(args.out_csv, index=False)
    logging.info(f"Adjacency saved -> {args.out_csv}")

    if args.out_img:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_img)), exist_ok=True)
        plot_causal_heatmap(adj, args.out_img)
        logging.info(f"Heatmap saved -> {args.out_img}")

    logging.info(f"Done in {(time.time() - t0) / 60:.2f} min.")


if __name__ == "__main__":
    main()
