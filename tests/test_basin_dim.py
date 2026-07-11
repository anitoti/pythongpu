"""
Tests for src/pythongpu/processing/basin_dim.py (methods plan items 5/6 --
basin mapping over an IC grid + basin-boundary box-counting), covering
arbitrary BaseOscillator subclasses and topological null-model topologies.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pythongpu.networks.random_graphs import (
    generate_ba_graph,
    generate_gnm_laplacian,
    generate_ws_graph,
)
from pythongpu.oscillators.lorenz import LorenzNetwork
from pythongpu.oscillators.rossler import RosslerNetwork
from pythongpu.oscillators.vanderpol import VanDerPolNetwork
from pythongpu.processing.basin_dim import (
    build_ic_grid,
    coherence_vps_matrix,
    integrate_grid,
    map_basins,
    sweep_coupling,
)
from pythongpu.processing.box_counting import extract_boundary
from pythongpu.processing.chimera_classifier import local_coherence
from pythongpu.utils import get_laplacian


def _adjacency_from_laplacian(L: torch.Tensor) -> np.ndarray:
    A = -L.cpu().numpy().copy()
    np.fill_diagonal(A, 0.0)
    return A


def _small_network(n: int = 10) -> tuple[torch.Tensor, np.ndarray]:
    A = np.ones((n, n)) - np.eye(n)
    return get_laplacian(A, norm=None, device="cpu"), A


# --------------------------------------------------------------------
# build_ic_grid
# --------------------------------------------------------------------


def test_build_ic_grid_shape():
    n_nodes, resolution = 12, 5
    state0, Xg, Yg = build_ic_grid(
        base_state=(1.0, 1.0, 1.0), slice_node_x=0, slice_node_y=1,
        n_nodes=n_nodes, bounds=(-5.0, 5.0), resolution=resolution,
    )
    assert state0.shape == (resolution * resolution, 3, n_nodes)
    assert Xg.shape == (resolution, resolution)
    assert Yg.shape == (resolution, resolution)


def test_build_ic_grid_grid_coordinates_match_bounds():
    _, Xg, Yg = build_ic_grid(
        base_state=(0.0, 0.0), slice_node_x=0, slice_node_y=1,
        n_nodes=4, bounds=(-3.0, 7.0), resolution=6,
    )
    assert Xg.min() == pytest.approx(-3.0)
    assert Xg.max() == pytest.approx(7.0)
    assert Yg.min() == pytest.approx(-3.0)
    assert Yg.max() == pytest.approx(7.0)


def test_build_ic_grid_base_state_broadcast_to_non_slice_nodes():
    """Every node except the two slice nodes should sit exactly at
    base_state for every grid point, on every state component."""
    n_nodes, resolution = 8, 4
    base_state = (2.0, -1.0, 0.5)
    state0, _, _ = build_ic_grid(
        base_state=base_state, slice_node_x=0, slice_node_y=1,
        n_nodes=n_nodes, resolution=resolution,
    )
    other_nodes = [i for i in range(n_nodes) if i not in (0, 1)]
    for node in other_nodes:
        for d, value in enumerate(base_state):
            assert torch.all(state0[:, d, node] == value)
    # slice nodes: non-x components (d >= 1) still sit at base_state
    for node in (0, 1):
        for d in range(1, len(base_state)):
            assert torch.all(state0[:, d, node] == base_state[d])


def test_build_ic_grid_slice_nodes_carry_grid_values():
    n_nodes, resolution = 6, 5
    state0, Xg, Yg = build_ic_grid(
        base_state=(0.0, 0.0), slice_node_x=2, slice_node_y=4,
        n_nodes=n_nodes, bounds=(-1.0, 1.0), resolution=resolution,
    )
    np.testing.assert_allclose(state0[:, 0, 2].numpy(), Xg.ravel())
    np.testing.assert_allclose(state0[:, 0, 4].numpy(), Yg.ravel())


@pytest.mark.parametrize("base_state", [(1.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)])
def test_build_ic_grid_supports_arbitrary_state_dimension(base_state):
    """D = len(base_state); must work for Van der Pol's D=2, Lorenz/
    Rossler's D=3, and an arbitrary future D=4 oscillator alike."""
    resolution = 4
    state0, _, _ = build_ic_grid(
        base_state=base_state, slice_node_x=0, slice_node_y=1,
        n_nodes=5, resolution=resolution,
    )
    assert state0.shape == (resolution * resolution, len(base_state), 5)


# --------------------------------------------------------------------
# integrate_grid
# --------------------------------------------------------------------


def test_integrate_grid_tail_shape():
    L, _ = _small_network(n=6)
    net = VanDerPolNetwork(L=L, mu=1.5, coupling=0.5, device="cpu")
    state0, _, _ = build_ic_grid(
        base_state=(1.0, 0.0), slice_node_x=0, slice_node_y=1, n_nodes=6, resolution=4,
    )
    steps, tail_frac = 200, 0.25
    tail = integrate_grid(net, state0, dt=0.02, steps=steps, tail_frac=tail_frac)
    assert tail.shape == (int(steps * tail_frac), 4 * 4, 2, 6)


# --------------------------------------------------------------------
# coherence_vps_matrix
# --------------------------------------------------------------------


def test_coherence_vps_matrix_matches_per_point_local_coherence():
    """coherence_vps_matrix should reproduce exactly what calling
    local_coherence once per grid point would give -- it's a batched
    convenience wrapper, not a different computation."""
    T, B, D, N = 50, 6, 3, 5
    rng = np.random.default_rng(0)
    trajectory_tail = torch.tensor(rng.standard_normal((T, B, D, N)), dtype=torch.float32)
    A = np.ones((N, N)) - np.eye(N)

    vps = coherence_vps_matrix(trajectory_tail, A)
    assert vps.shape == (B, N)

    for b in range(B):
        expected = local_coherence(trajectory_tail[:, b, :, :], A)
        np.testing.assert_allclose(vps[b], expected)


# --------------------------------------------------------------------
# map_basins -- across arbitrary BaseOscillator subclasses
# --------------------------------------------------------------------

# (oscillator_cls, base_state, coupling, dt, bounds, osc_kwargs)
OSCILLATOR_CASES = [
    (VanDerPolNetwork, (1.0, 0.0), 0.5, 0.02, (-2.0, 2.0), {"mu": 1.5}),
    (RosslerNetwork, (1.0, 1.0, 1.0), 0.15, 0.01, (-5.0, 5.0), {}),
    (LorenzNetwork, (1.0, 1.0, 1.0), 3.0, 0.01, (-5.0, 5.0), {}),
]


@pytest.mark.parametrize(
    "oscillator_cls,base_state,coupling,dt,bounds,osc_kwargs",
    OSCILLATOR_CASES,
    ids=["vanderpol", "rossler", "lorenz"],
)
def test_map_basins_valid_output(oscillator_cls, base_state, coupling, dt, bounds, osc_kwargs):
    L, A = _small_network(n=10)
    resolution = 8
    bm = map_basins(
        oscillator_cls, L, A, base_state, slice_node_x=0, slice_node_y=1,
        coupling=coupling, dt=dt, steps=400, resolution=resolution, bounds=bounds,
        device="cpu", osc_kwargs=osc_kwargs, seed=0,
    )
    assert bm.label_grid.shape == (resolution, resolution)
    assert bm.vps_matrix.shape == (resolution * resolution, 10)
    assert bm.k >= 2
    # box-counting on a bounded 2D binary image is mathematically confined
    # to [0, 2] regardless of the oscillator's native state dimension --
    # this is exactly item 6's cross-system comparability guarantee.
    assert 0.0 <= bm.fractal_dim <= 2.0
    assert np.isfinite(bm.r_squared)


def test_map_basins_boundary_matches_extract_boundary_orchestration():
    """The boundary attached to a BasinMap should be exactly what calling
    extract_boundary on its own label_grid produces -- regression-checks
    the box-counting orchestration wiring, not just that it runs."""
    L, A = _small_network(n=8)
    bm = map_basins(
        VanDerPolNetwork, L, A, base_state=(1.0, 0.0), slice_node_x=0, slice_node_y=1,
        coupling=0.5, dt=0.02, steps=300, resolution=6, bounds=(-2.0, 2.0),
        device="cpu", osc_kwargs={"mu": 1.5}, seed=0,
    )
    np.testing.assert_array_equal(bm.boundary, extract_boundary(bm.label_grid))


# --------------------------------------------------------------------
# map_basins -- topological null models (item 3 generators feeding item 5)
# --------------------------------------------------------------------


def _null_model_laplacian(model: str, n: int = 12, seed: int = 1) -> torch.Tensor:
    if model == "er":
        return generate_gnm_laplacian(n, 25, plot=False, seed=seed)
    if model == "ba":
        return generate_ba_graph(n, 3, plot=False, seed=seed)
    if model == "ws":
        return generate_ws_graph(n, 4, 0.1, plot=False, seed=seed)
    raise ValueError(model)


@pytest.mark.parametrize("null_model", ["er", "ba", "ws"])
def test_map_basins_supports_topological_null_models(null_model):
    """map_basins takes L generically -- it must run identically whether L
    comes from an empirical connectome or an ER/BA/WS topological null
    model, since that substitutability is the whole point of a null-model
    comparison."""
    L = _null_model_laplacian(null_model)
    A = _adjacency_from_laplacian(L)
    resolution = 6

    bm = map_basins(
        VanDerPolNetwork, L, A, base_state=(1.0, 0.0), slice_node_x=0, slice_node_y=1,
        coupling=0.5, dt=0.02, steps=300, resolution=resolution, bounds=(-2.0, 2.0),
        device="cpu", osc_kwargs={"mu": 1.5}, seed=0,
    )
    assert bm.label_grid.shape == (resolution, resolution)
    assert 0.0 <= bm.fractal_dim <= 2.0


# --------------------------------------------------------------------
# sweep_coupling
# --------------------------------------------------------------------


def test_sweep_coupling_returns_one_result_per_coupling_value_in_order():
    L, A = _small_network(n=8)
    coupling_values = [0.0, 0.2, 0.5]
    results = sweep_coupling(
        VanDerPolNetwork, L, A, base_state=(1.0, 0.0), slice_node_x=0, slice_node_y=1,
        coupling_values=coupling_values, dt=0.02, steps=300, resolution=5,
        bounds=(-2.0, 2.0), device="cpu", osc_kwargs={"mu": 1.5}, seed=0,
    )
    assert len(results) == len(coupling_values)
    assert [r.coupling for r in results] == coupling_values
    assert all(0.0 <= r.fractal_dim <= 2.0 for r in results)
