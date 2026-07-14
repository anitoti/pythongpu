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


if __name__ == "__main__":
    generate_gnm_laplacian(100, 500)

# to run,
# python3 -m pythongpu.networks.random_graphs