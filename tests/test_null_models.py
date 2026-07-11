"""
Tests for topological null-model generation (methods plan item 3 —
ER/BA/WS references matched to the empirical structural connectome) and
for dynamics-agnostic validation of the neighbor-relative-variance
coherence metric (processing/chimera_classifier.py, item 4) across three
independently-implemented oscillators.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pytest
import torch
from scipy.io import loadmat

from pythongpu.networks.random_graphs import (
    generate_ba_graph,
    generate_gnm_laplacian,
    generate_ws_graph,
)
from pythongpu.oscillators.lorenz import LorenzNetwork
from pythongpu.oscillators.rossler import RosslerNetwork
from pythongpu.oscillators.vanderpol import VanDerPolNetwork
from pythongpu.processing.chimera_classifier import local_coherence

DTI_PATH = "data/DTI_A.mat"
TOPOLOGY_SEED = 42


def _reference_connectome_stats() -> tuple[int, int]:
    """N and undirected edge count of the empirical structural connectome
    that the ER/BA/WS null models below are matched against."""
    A = loadmat(DTI_PATH)["A"].astype(np.float64)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    edges = int((A > 0).sum() // 2)
    return n, edges


N_REF, EDGES_REF = _reference_connectome_stats()
DENSITY_REF = EDGES_REF / (N_REF * (N_REF - 1) / 2)

# BA generates exactly m*(n-m) edges; WS generates exactly n*k/2 edges
# (k forced even). Both solved here so the null models' native integer
# parameters land as close to EDGES_REF as an integer choice allows.
_BA_M = round((N_REF - (N_REF**2 - 4 * EDGES_REF) ** 0.5) / 2)
_WS_K = round(2 * EDGES_REF / N_REF)
if _WS_K % 2 == 1:
    _WS_K += 1


def _build_null_model(model: str, seed: int = TOPOLOGY_SEED) -> torch.Tensor:
    if model == "er":
        return generate_gnm_laplacian(N_REF, EDGES_REF, plot=False, seed=seed)
    if model == "ba":
        return generate_ba_graph(N_REF, _BA_M, plot=False, seed=seed)
    if model == "ws":
        return generate_ws_graph(N_REF, _WS_K, 0.1, plot=False, seed=seed)
    raise ValueError(model)


def _adjacency_from_laplacian(L: torch.Tensor) -> np.ndarray:
    A = -L.cpu().numpy().copy()
    np.fill_diagonal(A, 0.0)
    return A


def _edge_count(A: np.ndarray) -> int:
    return int((A > 0).sum() // 2)


# --------------------------------------------------------------------
# Part 1: null-model topology validated against the structural connectome
# --------------------------------------------------------------------


@pytest.mark.parametrize("model", ["er", "ba", "ws"])
def test_null_model_matches_reference_node_count(model):
    L = _build_null_model(model)
    assert L.shape == (N_REF, N_REF)


@pytest.mark.parametrize("model", ["er", "ba", "ws"])
def test_null_model_matches_reference_density(model):
    """Each null model's edge density should approximate the empirical
    DTI connectome's density (~25%): exact for ER (m is its native
    parameter), approximate for BA/WS since their native parameters
    (m, k) are integers that can't hit EDGES_REF exactly."""
    A = _adjacency_from_laplacian(_build_null_model(model))
    density = _edge_count(A) / (N_REF * (N_REF - 1) / 2)
    assert abs(density - DENSITY_REF) / DENSITY_REF < 0.1


def test_er_null_model_edge_count_is_exact():
    A = _adjacency_from_laplacian(_build_null_model("er"))
    assert _edge_count(A) == EDGES_REF


@pytest.mark.parametrize("model", ["er", "ba", "ws"])
def test_null_model_has_no_self_loops(model):
    A = _adjacency_from_laplacian(_build_null_model(model))
    assert np.all(np.diag(A) == 0)


@pytest.mark.parametrize("model", ["er", "ba", "ws"])
def test_null_model_is_undirected(model):
    A = _adjacency_from_laplacian(_build_null_model(model))
    np.testing.assert_array_equal(A, A.T)


def test_ba_null_model_has_heavier_degree_tail_than_er():
    """Scale-free hallmark: at matched N/density, BA's preferential
    attachment produces hub nodes and a much wider degree distribution
    than ER's near-binomial spread -- this is what makes BA a distinct
    topology reference rather than just another random-graph label."""
    A_er = _adjacency_from_laplacian(_build_null_model("er"))
    A_ba = _adjacency_from_laplacian(_build_null_model("ba"))
    std_er = np.asarray([d for _, d in nx.from_numpy_array(A_er).degree()]).std()
    std_ba = np.asarray([d for _, d in nx.from_numpy_array(A_ba).degree()]).std()
    assert std_ba > 2 * std_er


def test_ws_null_model_has_higher_clustering_than_er():
    """Small-world hallmark: WS's ring-lattice-plus-light-rewiring (p=0.1)
    retains much higher clustering than ER at matched N/density."""
    A_er = _adjacency_from_laplacian(_build_null_model("er"))
    A_ws = _adjacency_from_laplacian(_build_null_model("ws"))
    clustering_er = nx.average_clustering(nx.from_numpy_array(A_er))
    clustering_ws = nx.average_clustering(nx.from_numpy_array(A_ws))
    assert clustering_ws > 2 * clustering_er


# --------------------------------------------------------------------
# Part 2: dynamics-agnostic synchronization phase transition
# --------------------------------------------------------------------
# Each oscillator has its own natural coupling scale -- Lorenz needs far
# stronger x-only coupling to fully synchronize than Rossler or Van der
# Pol, an established property of these systems' master stability
# functions, not a metric quirk -- so low/high coupling values are
# calibrated per oscillator (see scratch calibration; consistent across
# 3 IC seeds x 3 topologies before being fixed here). What's under test
# is dynamics-agnostic: the *same* local_coherence measure, with the
# *same* per-oscillator thresholds, correctly separates "desynchronized"
# from "synchronized" regardless of which null-model topology carries
# the coupling.


def _run_rossler(L, coupling, steps, dt, seed):
    torch.manual_seed(seed)
    n = L.shape[0]
    net = RosslerNetwork(L=L, coupling=coupling, device="cpu")
    init = torch.randn(3, n) * 0.5 + torch.tensor([[1.0], [1.0], [1.0]])
    return net.integrate(init, dt=dt, steps=steps)


def _run_vanderpol(L, coupling, steps, dt, seed):
    torch.manual_seed(seed)
    n = L.shape[0]
    net = VanDerPolNetwork(L=L, mu=1.5, coupling=coupling, device="cpu")
    init = torch.randn(2, n) * 0.5 + torch.tensor([[1.0], [0.0]])
    return net.integrate(init, dt=dt, steps=steps)


def _run_lorenz(L, coupling, steps, dt, seed):
    """LorenzNetwork uses a batched (B, 3, N) state convention (see
    pipeline/lorenz_basins_sweep.py), unlike Rossler/VanDerPol's (D, N).
    Run a single-trajectory batch (B=1) and squeeze it back out so the
    returned trajectory matches local_coherence's (T, D, N) shape."""
    torch.manual_seed(seed)
    n = L.shape[0]
    net = LorenzNetwork(L=L, coupling=coupling, dt=dt, device="cpu")
    state = torch.randn(1, 3, n) * 0.5 + torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1)
    history = torch.zeros((steps, 1, 3, n))
    for t in range(steps):
        state = net.rk4_step(state, dt)
        history[t] = state
    return history.squeeze(1)


# name -> (runner, low_coupling, high_coupling, low_ceiling, high_floor, min_gap)
OSCILLATOR_REGIMES = {
    "rossler": (_run_rossler, 0.0, 0.15, 0.85, 0.95, 0.15),
    "lorenz": (_run_lorenz, 0.0, 3.0, 0.5, 0.95, 0.4),
    "vanderpol": (_run_vanderpol, 0.0, 0.5, 0.85, 0.95, 0.15),
}


@pytest.mark.parametrize("null_model", ["er", "ba", "ws"])
@pytest.mark.parametrize("oscillator", ["rossler", "lorenz", "vanderpol"])
def test_coherence_metric_detects_sync_transition(oscillator, null_model):
    runner, low_c, high_c, low_ceiling, high_floor, min_gap = OSCILLATOR_REGIMES[oscillator]

    L = _build_null_model(null_model)
    A = _adjacency_from_laplacian(L)

    tail_low = runner(L, low_c, steps=4000, dt=0.01, seed=0)[2000:]
    tail_high = runner(L, high_c, steps=4000, dt=0.01, seed=0)[2000:]

    coherence_low = local_coherence(tail_low, A)
    coherence_high = local_coherence(tail_high, A)

    assert coherence_low.mean() < low_ceiling, (
        f"{oscillator}/{null_model}: expected a desynchronized regime at "
        f"coupling={low_c}, got mean coherence {coherence_low.mean():.3f}"
    )
    assert coherence_high.mean() > high_floor, (
        f"{oscillator}/{null_model}: expected a synchronized regime at "
        f"coupling={high_c}, got mean coherence {coherence_high.mean():.3f}"
    )
    assert coherence_high.mean() - coherence_low.mean() > min_gap, (
        f"{oscillator}/{null_model}: coherence did not rise enough across "
        f"the coupling sweep to count as a detected phase transition"
    )
