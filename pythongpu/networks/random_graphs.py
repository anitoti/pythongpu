import networkx as nx
from pythongpu.utils import get_laplacian, get_plot_path
import matplotlib # this and Agg dont open a new window in the server, so just a file saves
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_gnm_laplacian(n, m, norm=None, device="cpu", plot=True, seed=None):
    g = nx.gnm_random_graph(n, m, seed=seed)
    adj = nx.to_numpy_array(g)

    # uses the pytorch utils.py we made earlier
    L = get_laplacian(adj, norm=norm, device=device)

    if plot:
        # generate image of nodes and edges per run
        nx.draw(g, node_size=20)
        plt.savefig(get_plot_path("random_graphs", "network.png"))
        plt.clf()
        plt.close()

    # calculate probability
    p = m / (n * (n - 1) / 2)
    print("probability = ",p)

    return L

def generate_gnp_laplacian(n, p, norm=None, device="cpu", plot=True, seed=None):
    g = nx.gnp_random_graph(n, p, seed=seed)
    adj = nx.to_numpy_array(g)
    L = get_laplacian(adj, norm=norm, device=device)

    if plot:
        nx.draw(g, node_size=20)
        plt.savefig(get_plot_path("random_graphs", "gnp_network.png"))
        plt.clf()
        plt.close()

    return L

def generate_ba_graph(n, m, norm=None, device="cpu", plot=True, seed=None):
    """
    Barabási–Albert preferential-attachment graph: a topology reference
    (not a strength/density-exact match) — used to compare the empirical
    connectome's basin structure against a scale-free null. `m` is BA's
    native parameter (edges attached per new node), so total edge count
    is only approximately m*n for large n, not exact like gnm's `m`.
    """
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    adj = nx.to_numpy_array(g)
    L = get_laplacian(adj, norm=norm, device=device)

    if plot:
        nx.draw(g, node_size=20)
        plt.savefig(get_plot_path("random_graphs", "ba_network.png"))
        plt.clf()
        plt.close()

    return L


def generate_ws_graph(n, k, p, norm=None, device="cpu", plot=True, seed=None):
    """
    Watts–Strogatz small-world graph: a topology reference preserving
    exact edge count n*k/2 (only rewired, not added/removed) while
    interpolating between a ring lattice (p=0) and random rewiring (p=1).
    `k` must be even (each node starts connected to k nearest neighbors).
    """
    g = nx.watts_strogatz_graph(n, k, p, seed=seed)
    adj = nx.to_numpy_array(g)
    L = get_laplacian(adj, norm=norm, device=device)

    if plot:
        nx.draw(g, node_size=20)
        plt.savefig(get_plot_path("random_graphs", "ws_network.png"))
        plt.clf()
        plt.close()

    return L


def _binarize_symmetrize(A):
    """Collapse a (possibly directed, weighted) connectome to an undirected
    simple graph for topology-baseline matching: edge iff A[i,j] or A[j,i] > 0."""
    import numpy as np
    A = np.asarray(A, dtype=float)
    B = ((A > 0) | (A.T > 0)).astype(int)
    np.fill_diagonal(B, 0)
    return B


def _graph_stats(g):
    """Compact topology fingerprint used to compare empirical vs. null graphs."""
    import numpy as np
    degs = np.array([d for _, d in g.degree()])
    n = g.number_of_nodes()
    return {
        "n_nodes": n,
        "n_edges": g.number_of_edges(),
        "density": nx.density(g),
        "mean_degree": float(degs.mean()) if n else 0.0,
        # degree heterogeneity <k^2>/<k>^2: ~1 for regular, >>1 for scale-free hubs
        "degree_heterogeneity": float((degs**2).mean() / (degs.mean() ** 2))
        if degs.mean() > 0 else 0.0,
        "mean_clustering": nx.average_clustering(g),
    }


def match_baselines_from_adjacency(A, norm=None, device="cpu", seed=None, plot=False):
    """Build Erdős–Rényi and Barabási–Albert nulls matched to an EMPIRICAL
    connectome, for baseline comparison against the reconstructed brain network.

    The empirical adjacency (from entropic_regression / oCSE, or DTI tractography)
    sets the targets:
      - N   = number of nodes
      - E   = number of undirected edges (after binarise+symmetrise)
      - ER  (gnm): same N and exactly E random edges  -> density-matched null,
                   answers "is the brain's structure more than its edge count?"
      - BA  (m = round(E / N)): same N, preferential attachment -> scale-free
                   null with hubs, answers "are the brain's hubs Barabási-like?"

    Returns
    -------
    dict with keys 'empirical', 'er', 'ba', each mapping to
        {'adj': ndarray, 'L': Tensor, 'stats': dict}
    so callers can feed any of the three Laplacians into the same LorenzNetwork
    / CLV pipeline and compare basin structure on identical footing.
    """
    import numpy as np
    B = _binarize_symmetrize(A)
    n = B.shape[0]
    g_emp = nx.from_numpy_array(B)
    E = g_emp.number_of_edges()

    # ER: exact edge-count match
    g_er = nx.gnm_random_graph(n, E, seed=seed)
    # BA: match mean degree via m ~= E/N (clamped to [1, n-1])
    m_ba = int(np.clip(round(E / n), 1, n - 1)) if n > 1 else 1
    g_ba = nx.barabasi_albert_graph(n, m_ba, seed=seed)

    out = {}
    for name, g in (("empirical", g_emp), ("er", g_er), ("ba", g_ba)):
        adj = nx.to_numpy_array(g)
        out[name] = {
            "adj": adj,
            "L": get_laplacian(adj, norm=norm, device=device),
            "stats": _graph_stats(g),
        }
    out["ba"]["stats"]["ba_m"] = m_ba

    if plot:
        import numpy as np
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, name in zip(axes, ("empirical", "er", "ba")):
            g = nx.from_numpy_array(out[name]["adj"])
            nx.draw(g, ax=ax, node_size=15)
            s = out[name]["stats"]
            ax.set_title(f"{name}\nE={s['n_edges']} het={s['degree_heterogeneity']:.2f}")
        plt.tight_layout()
        plt.savefig(get_plot_path("random_graphs", "baseline_comparison.png"), dpi=150)
        plt.clf(); plt.close()

    return out


if __name__ == "__main__":
    generate_gnm_laplacian(100, 500)

# to run,
# python3 -m pythongpu.networks.random_graphs