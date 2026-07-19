import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from tqdm import tqdm

# =============================================================================
# THEORETICAL BASIS — Optimal Causation Entropy (oCSE)
# =============================================================================
# Sun, J., Taylor, D., & Bollt, E. M. (2015). Causal network inference by
# optimal causation entropy. SIAM Journal on Applied Dynamical Systems, 14(1),
# 73–106.
#
#   C(X_j -> X_i | X_S) = H(X_i | X_S) - H(X_i | X_S ∪ {X_j})
#
# where H is Shannon differential entropy and X_S is the current "causal set"
# of source nodes already selected for target X_i.
#
# oCSE distinguishes DIRECT from INDIRECT influence:
#   - If X_j -> X_i is mediated by X_k, then conditioning on X_k eliminates
#     the CMI: C(X_j -> X_i | X_k) ≈ 0.
#   - Greedy forward selection: at each step, add the source that maximally
#     reduces conditional entropy of the target.
#   - Significance test via chi-squared approximation of the null distribution
#     (n_samples * CMI ~ χ²(1) under H0: no causal link).
#
# In the fMRI context (parcellated brain regions):
#   - Nodes = parcellated ROIs (reduced from 200k+ voxels)
#   - Edge X_j -> X_i exists iff X_j is in the oCSE-discovered causal set of X_i
#   - max_lag=1 enforces Granger-style temporal precedence (cause precedes effect)
#   - Entropy estimated under Gaussian assumption (closed-form from covariance)
# =============================================================================

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ---------------------------------------------------------------------------
# Entropy and CMI estimators (Gaussian assumption)
# ---------------------------------------------------------------------------

def gaussian_entropy(cov: np.ndarray) -> float:
    """Differential entropy of a multivariate Gaussian.
    H(X) = 0.5 * log((2πe)^k * |Σ|)
    """
    cov = np.atleast_2d(cov)
    k = cov.shape[0]
    # Add small ridge for numerical stability if singular
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        # Fallback: regularize
        cov_reg = cov + 1e-10 * np.eye(k)
        _, logdet = np.linalg.slogdet(cov_reg)
    return 0.5 * (k * np.log(2 * np.pi * np.e) + logdet)

def conditional_entropy(X: np.ndarray, Y: np.ndarray) -> float:
    """Conditional entropy H(X|Y) under Gaussian assumption.
    H(X|Y) = H(X, Y) - H(Y)
    """
    Z = np.hstack([X, Y])
    return gaussian_entropy(np.cov(Z.T)) - gaussian_entropy(np.cov(Y.T))

def conditional_mutual_information(X: np.ndarray, Y: np.ndarray, Z: np.ndarray = None) -> float:
    """Conditional mutual information I(X; Y | Z) under Gaussian assumption.
    I(X; Y | Z) = H(X | Z) - H(X | Y, Z)
    Returns CMI in nats (≥ 0, 0 iff conditional independence)
    """
    if Z is None or Z.shape[1] == 0:
        # Unconditional: I(X; Y) = H(X) - H(X | Y)
        H_X = gaussian_entropy(np.cov(X.T))
        H_X_given_Y = conditional_entropy(X, Y)
        return max(0.0, H_X - H_X_given_Y)
    else:
        H_X_given_Z = conditional_entropy(X, Z)
        H_X_given_YZ = conditional_entropy(X, np.hstack([Y, Z]))
        return max(0.0, H_X_given_Z - H_X_given_YZ)

def _cmi_significant(cmi_value: float, n_samples: int, alpha: float = 0.05) -> bool:
    """Significance test for CMI via chi-squared approximation.
    Under H0 (conditional independence): n_samples * CMI ~ χ²(1).
    """
    stat = n_samples * cmi_value
    p_value = 1.0 - chi2.cdf(stat, 1)
    return p_value < alpha

# ---------------------------------------------------------------------------
# Worker Function for Parallelization
# ---------------------------------------------------------------------------
def _process_single_target(args):
    """
    Isolated function to find causes for a single target node.
    This enables embarrassingly parallel execution across HPC CPU cores.
    """
    target, X_current, X_lagged, n_regions, n_samples, alpha = args
    # Current (unlagged) value of the target node
    x_target = X_current[:, [target]]
    
    # Indices of source nodes already selected for this target
    causal_set = []
    edges = [] # list of tuples: (source, cmi_weight)

    while True:
        best_cmi = -1.0
        best_source = -1

        for source in range(n_regions):
            if source == target or source in causal_set:
                continue

            # Lagged candidate driver
            x_source_lagged = X_lagged[:, [source]]
            
            # Conditioning set: lagged values of already-selected sources
            Z = X_lagged[:, causal_set] if causal_set else np.empty((n_samples, 0))
            
            cmi = conditional_mutual_information(x_target, x_source_lagged, Z)

            if cmi > best_cmi:
                best_cmi = cmi
                best_source = source

        # If no candidate found or CMI not significant, stop
        if best_source == -1 or best_cmi <= 0:
            break

        if _cmi_significant(best_cmi, n_samples, alpha):
            causal_set.append(best_source)
            edges.append((best_source, best_cmi))
        else:
            break
            
    return target, edges

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_causal_heatmap(adj_matrix: np.ndarray, out_path: str) -> None:
    """Plot directed weighted adjacency matrix as a heatmap."""
    import seaborn as sns  # plotting-only dep; kept out of the core-estimator import path
    n_regions = adj_matrix.shape[0]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        adj_matrix, cmap="RdPu",
        cbar_kws={'label': 'CMI (nats) — Conditional Mutual Information'}, 
        xticklabels=False, yticklabels=False
    )
    plt.title(f"Directed Functional Connectivity (Optimal Causation Entropy)\nN={n_regions} ROIs | max_lag=1 TR")
    plt.xlabel("Sink (target) node")
    plt.ylabel("Source (causal) node")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Discover directed causal network using oCSE.")
    parser.add_argument("--input", required=True, help="Path to parcellated timeseries CSV")
    parser.add_argument("--out_img", required=True, help="Path to save heatmap PNG")
    parser.add_argument("--out_csv", required=True, help="Path to save adjacency matrix CSV")
    parser.add_argument("--max_lag", type=int, default=1, help="Maximum TR lag (default: 1)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of cores to use. Avoid -1 on shared nodes to prevent OOM errors.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}\nRun parcellation script first.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out_img)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    start_time = time.time()
    
    # Load data
    df_ts = pd.read_csv(args.input)
    timeseries = df_ts.values.astype(np.float64)
    
    # Columns = ROI nodes (output of parcellation step)
    n_timepoints, n_regions = timeseries.shape
    logging.info(f"Loaded timeseries: {n_timepoints} TRs × {n_regions} ROIs")

    # Build lagged dataset
    X_current = timeseries[args.max_lag:]
    X_lagged = timeseries[:-args.max_lag]
    n_samples = X_current.shape[0]

    # adj_matrix[source, sink] = CMI(source -> sink | S_sink)
    adj_matrix = np.zeros((n_regions, n_regions), dtype=np.float64)

    # Prepare arguments for parallel workers
    worker_args = [
        (target, X_current, X_lagged, n_regions, n_samples, args.alpha) 
        for target in range(n_regions)
    ]

    logging.info(f"Starting oCSE discovery using {args.n_jobs} cores...")
    
    # Process nodes in parallel
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(_process_single_target, arg): arg[0] for arg in worker_args}
        
        # tqdm progress bar wrapped around as_completed yields as they finish
        for future in tqdm(as_completed(futures), total=n_regions, desc="Processing ROIs"):
            target, edges = future.result()
            for source, weight in edges:
                adj_matrix[source, target] = weight

    n_edges = np.count_nonzero(adj_matrix)
    density = n_edges / (n_regions * (n_regions - 1)) if n_regions > 1 else 0.0
    
    logging.info(f"Discovered network: {n_edges} edges (Density: {density:.4f})")

    # Save matrix data for dynamical systems use later
    pd.DataFrame(adj_matrix).to_csv(args.out_csv, index=False)
    logging.info(f"Adjacency matrix saved → {args.out_csv}")

    # Plot and save image
    plot_causal_heatmap(adj_matrix, args.out_img)
    logging.info(f"Heatmap saved → {args.out_img}")

    elapsed = (time.time() - start_time) / 60
    logging.info(f"Total time elapsed: {elapsed:.2f} minutes.")

if __name__ == "__main__":
    main()