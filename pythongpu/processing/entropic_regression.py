#!/usr/bin/env python3
"""
Entropic Regression (ER) for directed brain-network reconstruction.

Method: Fish, DeWitt, AlMomani, Laurienti & Bollt (2021), "Entropic regression
with neurologically motivated applications," Chaos 31, 113105. ER recovers the
DIRECTED information-flow graph among parcellated regions and was shown there to
out-recover correlation and LASSO on DTI-coupled Kuramoto brain models.

Relation to the existing oCSE module (processing/causation_entropy.py):
  - causation_entropy.py implements only the FORWARD greedy pass of optimal
    causation entropy: keep adding the source that maximally reduces the
    target's conditional entropy while the added causation entropy is
    significant.
  - Entropic regression adds a BACKWARD elimination pass: after the forward
    aggregation over-selects, re-test every kept source conditioned on all the
    *others* and drop any whose causation entropy is no longer significant.
    Forward-then-backward is what removes indirect/spurious edges that a purely
    forward sweep leaves behind — this is the ER contribution.

Edge semantics (matches causation_entropy.py so the two are interchangeable
downstream):
    adj[source, sink] = CMI weight of the retained lagged edge source -> sink.

The Gaussian conditional-mutual-information / significance estimators are reused
from causation_entropy.py rather than re-derived, so ER and oCSE share one
entropy backend.
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


def _forward_pass(target, X_current, X_lagged, n_regions, n_samples, alpha):
    """Standard oCSE forward aggregation: greedily grow the causal set."""
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


def _backward_pass(target, causal_set, X_current, X_lagged, n_samples, alpha):
    """ER backward elimination: drop any source that is redundant given the rest.

    For each retained source s, recompute CMI(s -> target | all other retained
    sources). If it is no longer significant, s was an indirect/spurious pickup
    from the forward greedy order and is removed. Re-weight the survivors with
    their conditional-on-the-rest CMI (the honest direct-influence weight).
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


def _process_single_target(args):
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
    """Reconstruct the directed edge-weighted adjacency via entropic regression.

    Parameters
    ----------
    timeseries : (T, N) array of parcellated ROI signals (z-scored upstream).
    max_lag    : temporal lag enforcing cause-precedes-effect (Granger-style).
    alpha      : significance level for the chi-squared causation-entropy test.
    n_jobs     : worker processes (keep modest on shared HPC nodes).

    Returns
    -------
    adj : (N, N) array, adj[source, sink] = retained ER causation-entropy weight.
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
    """Combine the ER functional graph with the DTI structural connectome.

    Matches the Fish/Bollt premise that information flow is *carried by* the
    physical white-matter wiring: functional edges with no structural substrate
    are down-weighted / removed.

    mode:
      "gate"     -> keep functional weight only where a structural edge exists
                    (structural_adj > 0); a hard anatomical mask.
      "weight"   -> multiply functional weight by the (row-normalised)
                    structural weight; a soft anatomical prior.
    Structural adjacency is assumed symmetric (undirected tractography); the
    directedness comes entirely from the functional ER side.
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


def main():
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
