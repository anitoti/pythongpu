from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import seeded_random_graph_sweep as sweep


def test_same_seed_gives_identical_graph():
    """The whole point of this script: a seed must reproduce the exact
    same graph on a fresh call, not just a graph with similar stats."""
    for kind in ("ba", "gnm", "ws"):
        g1 = sweep.build_graph(kind, seed=7)
        g2 = sweep.build_graph(kind, seed=7)
        import networkx as nx
        assert nx.to_numpy_array(g1).sum() == nx.to_numpy_array(g2).sum()
        assert set(g1.edges()) == set(g2.edges())


def test_different_seeds_give_different_graphs():
    for kind in ("ba", "gnm", "ws"):
        g1 = sweep.build_graph(kind, seed=0)
        g2 = sweep.build_graph(kind, seed=1)
        assert set(g1.edges()) != set(g2.edges())


def test_node_count_matches_dti_target():
    for kind in ("ba", "gnm", "ws"):
        g = sweep.build_graph(kind, seed=0)
        assert g.number_of_nodes() == sweep.N_NODES == 83


def test_gnm_hits_exact_edge_target():
    g = sweep.build_graph("gnm", seed=0)
    assert g.number_of_edges() == sweep.N_EDGES_TARGET == 850
