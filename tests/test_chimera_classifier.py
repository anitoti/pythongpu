"""
Tests for src/pythongpu/processing/chimera_classifier.py (methods plan item 4).
"""

import numpy as np
import pytest
import torch

from pythongpu.oscillators.vanderpol import VanDerPolNetwork
from pythongpu.processing.chimera_classifier import (
    bimodality_coefficient,
    classify_chimera,
    cluster_vps_population,
    local_coherence,
)
from pythongpu.utils import get_laplacian


def _ring_adjacency(n: int) -> np.ndarray:
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[i, (i - 1) % n] = 1.0
    return A


def test_local_coherence_fully_synchronized():
    """All nodes share an identical trajectory -> coherence == 1 exactly for all."""
    n, T = 6, 200
    t = np.linspace(0, 10, T)
    shared_x = np.sin(t)
    # (T, D, N) with D=3; only index 0 (x) should matter — Y/Z are garbage per node
    trajectory = np.zeros((T, 3, n))
    for i in range(n):
        trajectory[:, 0, i] = shared_x
        trajectory[:, 1, i] = np.random.randn(T) * 100  # should be ignored
        trajectory[:, 2, i] = i * 999  # should be ignored

    W = _ring_adjacency(n)
    coherence = local_coherence(trajectory, W)

    assert coherence.shape == (n,)
    np.testing.assert_allclose(coherence, 1.0, atol=1e-8)


def test_local_coherence_independent_nodes():
    """Independent uncorrelated node trajectories -> coherence well below 1."""
    rng = np.random.default_rng(0)
    n, T = 20, 2000
    trajectory = np.zeros((T, 3, n))
    trajectory[:, 0, :] = rng.standard_normal((T, n))

    W = np.ones((n, n)) - np.eye(n)  # fully connected
    coherence = local_coherence(trajectory, W)

    assert coherence.shape == (n,)
    # With many independent neighbors, the neighbor-mean field has much
    # smaller variance than each node's own signal, so coherence should
    # land well below 1 fairly uniformly (theoretical value ~= exp(-1) for
    # a large fully-connected independent-noise network).
    assert np.all(coherence < 0.7)
    assert coherence.std() < 0.15  # roughly uniform across nodes


def test_local_coherence_isolated_node_defaults_to_one():
    """A node with zero-weight row (no neighbors) returns coherence 1.0."""
    n, T = 4, 100
    trajectory = np.zeros((T, 3, n))
    trajectory[:, 0, :] = np.random.randn(T, n)

    W = np.zeros((n, n))
    W[0, 1] = W[1, 0] = 1.0  # nodes 0,1 connected; nodes 2,3 isolated

    coherence = local_coherence(trajectory, W)
    assert coherence[2] == 1.0
    assert coherence[3] == 1.0


def test_classify_chimera_detects_bimodal_split():
    """A clearly bimodal coherence vector should be flagged as a chimera."""
    rng = np.random.default_rng(1)
    high = 0.95 + rng.normal(0, 0.01, size=10)
    low = 0.05 + rng.normal(0, 0.01, size=10)
    coherence_vector = np.concatenate([high, low])

    result = classify_chimera(coherence_vector)

    assert result["is_chimera"] is True
    assert result["gap"] > 0.2
    assert result["bimodality_coefficient"] > 5 / 9

    labels = result["labels"]
    # the first 10 entries (high group) must all share one label, the last
    # 10 (low group) the other — label values themselves are arbitrary
    assert len(set(labels[:10])) == 1
    assert len(set(labels[10:])) == 1
    assert labels[0] != labels[10]


@pytest.mark.parametrize("center", [0.9, 0.1])
def test_classify_chimera_rejects_unimodal_states(center):
    """Fully-synchronized (all-high) and fully-incoherent (all-low) states
    are not chimeras, even though a 2-means split will still partition
    them into two arbitrary groups."""
    rng = np.random.default_rng(2)
    coherence_vector = center + rng.normal(0, 0.02, size=20)

    result = classify_chimera(coherence_vector)

    assert result["is_chimera"] is False


def test_bimodality_coefficient_requires_more_than_three_samples():
    assert np.isnan(bimodality_coefficient([0.1, 0.2, 0.3]))


def test_cluster_vps_population_recovers_three_clusters():
    """Three well-separated synthetic 'attractor types', each repeated
    with noise, should be recovered as k=3 via silhouette selection."""
    rng = np.random.default_rng(3)
    n_nodes = 10
    templates = [
        np.full(n_nodes, 0.9),
        np.full(n_nodes, 0.1),
        np.array([0.9, 0.1] * (n_nodes // 2)),
    ]
    per_type = 15
    rows = []
    true_labels = []
    for label, template in enumerate(templates):
        for _ in range(per_type):
            rows.append(template + rng.normal(0, 0.02, size=n_nodes))
            true_labels.append(label)
    vps_matrix = np.array(rows)
    true_labels = np.array(true_labels)

    result = cluster_vps_population(vps_matrix, k_min=2, k_max=6)

    assert result["k"] == 3
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(true_labels, result["labels"]) > 0.95


def test_chimera_pipeline_on_real_vanderpol_synchronized():
    """Integration test: a densely-coupled Van der Pol network with similar
    initial conditions should synchronize -> high coherence, not a chimera."""
    torch.manual_seed(1)
    np.random.seed(1)

    n = 8
    A = np.ones((n, n)) - np.eye(n)
    L = get_laplacian(A, norm=None, device="cpu")
    net = VanDerPolNetwork(L=L, mu=1.5, coupling=2.0, device="cpu")

    init_state = torch.randn(2, n) * 0.05 + torch.tensor([[1.0], [0.0]])
    traj = net.integrate(init_state, dt=0.01, steps=3000)
    tail = traj[1500:]  # post-transient tail

    coherence = local_coherence(tail, A)
    result = classify_chimera(coherence)

    assert np.all(coherence > 0.9)
    assert result["is_chimera"] is False


def test_chimera_pipeline_on_real_vanderpol_split_network():
    """Integration test: one strongly-coupled group with similar ICs
    (synchronizes) plus one weakly-coupled group with divergent ICs (stays
    incoherent relative to its own neighbors) on a single real network ->
    classify_chimera should detect the split.

    This is a controlled, reliable split (real dynamics, real topology, real
    classifier), not a claim that Van der Pol networks spontaneously
    produce chimeras under arbitrary parameters — that's a much harder,
    system-specific research question outside the scope of a unit test.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    n_a, n_b = 5, 5
    n = n_a + n_b
    A = np.zeros((n, n))
    for i in range(n_a):
        for j in range(n_a):
            if i != j:
                A[i, j] = 1.0  # strong intra-group-A coupling
    for i in range(n_a, n):
        for j in range(n_a, n):
            if i != j:
                A[i, j] = 0.02  # weak intra-group-B coupling (not isolated, not enough to sync)

    L = get_laplacian(A, norm=None, device="cpu")
    net = VanDerPolNetwork(L=L, mu=1.5, coupling=2.0, device="cpu")

    init_state = torch.zeros(2, n)
    init_state[:, :n_a] = torch.randn(2, n_a) * 0.05 + torch.tensor([[1.0], [0.0]])
    init_state[:, n_a:] = torch.randn(2, n_b) * 2.0  # divergent ICs

    traj = net.integrate(init_state, dt=0.01, steps=3000)
    tail = traj[1500:]

    coherence = local_coherence(tail, A)
    result = classify_chimera(coherence)

    assert np.all(coherence[:n_a] > 0.9)  # group A: synchronized
    assert np.all(coherence[n_a:] < 0.7)  # group B: meaningfully desynchronized
    assert result["is_chimera"] is True
