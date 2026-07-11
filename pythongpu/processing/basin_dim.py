"""
Basin mapping + box-counting API (methods plan items 5 and 6).

Refactors the grid-slice generation from pipeline/lorenz_basins_sweep.py
and the box-counting orchestration duplicated across pipeline/lorenz_sweep.py
and pipeline/rossler_sweep.py into oscillator-agnostic, reusable functions.
Those scripts predate the BaseOscillator hierarchy and each hand-roll their
own physics/state-shape convention; this module instead drives any
BaseOscillator subclass directly (LorenzNetwork, RosslerNetwork,
VanDerPolNetwork, ...) through its own rhs/integrate.

Pipeline for one coupling value:
    build_ic_grid      -> (B, D, N) initial-condition batch over a 2D slice
    integrate_grid      -> post-transient trajectory tail, (T, B, D, N)
    coherence_vps_matrix -> per-grid-point coherence vectors, (B, N)
    cluster_vps_population (chimera_classifier, item 4) -> basin labels
    extract_boundary + boxcount_2d_gpu + fractal_dimension (box_counting,
        item 6) -> basin-boundary fractal dimension D_f

sweep_coupling repeats this across a range of coupling values sigma so
D_f(sigma) can be plotted. Because the grid always varies exactly 2 scalar
initial conditions regardless of the oscillator's native state dimension D,
the boundary is always a 2D image and D_f always lands in [1, 2] -- this is
what keeps cross-system (e.g. Lorenz vs. Rossler) comparison meaningful
(item 6) without any D-dependent special-casing.

L (and its matching adjacency A) can come from anywhere with the right node
count -- an empirical connectome or a topological null model built via
networks/random_graphs.py (ER/BA/WS) -- this module has no opinion on the
source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pythongpu.oscillators.base_oscillator import BaseOscillator
from pythongpu.processing.box_counting import (
    boxcount_2d_gpu,
    extract_boundary,
    fractal_dimension,
)
from pythongpu.processing.chimera_classifier import (
    cluster_vps_population,
    local_coherence,
)


def build_ic_grid(
    base_state: tuple[float, ...],
    slice_node_x: int,
    slice_node_y: int,
    n_nodes: int,
    bounds: tuple[float, float] = (-10.0, 10.0),
    resolution: int = 32,
    device: str = "cpu",
    jitter: float = 0.0,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Build a batched (B, D, N) initial-condition grid: the x (index-0)
    component of two representative nodes is swept over a 2D affine slice,
    every other state entry held at base_state. D = len(base_state), so
    this works unchanged for a 2-component state (Van der Pol) or a
    3-component state (Lorenz, Rossler). Generalizes
    pipeline/lorenz_basins_sweep.py::build_ic_grid beyond Lorenz.

    jitter adds small Gaussian noise to every entry (matches the +-0.05
    perturbation used in pipeline/lorenz_sweep.py to avoid landing exactly
    on an unstable fixed point); 0.0 (default) reproduces the deterministic
    grid lorenz_basins_sweep.py builds.

    Returns
    -------
    state0 : (B, D, N) tensor, B = resolution**2
    Xg, Yg : (resolution, resolution) grid coordinate arrays
    """
    D = len(base_state)
    lo, hi = bounds
    ax = np.linspace(lo, hi, resolution, dtype=np.float32)
    Xg, Yg = np.meshgrid(ax, ax)
    B = resolution * resolution

    state0 = torch.zeros((B, D, n_nodes), dtype=torch.float32, device=device)
    for d, value in enumerate(base_state):
        state0[:, d, :] = value
    if jitter > 0:
        state0 += jitter * torch.randn_like(state0)

    state0[:, 0, slice_node_x] = torch.tensor(Xg.ravel(), device=device)
    state0[:, 0, slice_node_y] = torch.tensor(Yg.ravel(), device=device)

    return state0, Xg, Yg


def integrate_grid(
    oscillator: BaseOscillator,
    state0: torch.Tensor,
    dt: float,
    steps: int,
    tail_frac: float = 0.25,
) -> torch.Tensor:
    """
    Integrate a batched (B, D, N) IC grid through `oscillator` and return
    the post-transient tail, (T_tail, B, D, N). Every BaseOscillator
    subclass in this repo exposes integrate(state0, dt, steps) with the
    same signature regardless of its native single-trajectory state
    convention, so this is the one call site that works for all of them.
    """
    traj = oscillator.integrate(state0, dt, steps)  # (steps, B, D, N)
    tail_steps = max(1, int(steps * tail_frac))
    return traj[-tail_steps:]


def coherence_vps_matrix(trajectory_tail: torch.Tensor, A: np.ndarray) -> np.ndarray:
    """
    trajectory_tail : (T, B, D, N)
    A               : (N, N) adjacency defining node neighborhoods

    Returns the (B, N) matrix of per-grid-point local_coherence vectors --
    one VPS per initial condition -- that cluster_vps_population classifies
    into basins.
    """
    if isinstance(trajectory_tail, torch.Tensor):
        trajectory_tail = trajectory_tail.detach().cpu().numpy()
    T, B, D, N = trajectory_tail.shape

    vps = np.zeros((B, N))
    for b in range(B):
        vps[b] = local_coherence(trajectory_tail[:, b, :, :], A)
    return vps


@dataclass
class BasinMap:
    coupling: float
    Xg: np.ndarray
    Yg: np.ndarray
    label_grid: np.ndarray
    vps_matrix: np.ndarray
    k: int
    boundary: np.ndarray
    fractal_dim: float
    r_squared: float


def map_basins(
    oscillator_cls: type[BaseOscillator],
    L: torch.Tensor,
    A: np.ndarray,
    base_state: tuple[float, ...],
    slice_node_x: int,
    slice_node_y: int,
    coupling: float,
    dt: float,
    steps: int,
    resolution: int = 32,
    bounds: tuple[float, float] = (-10.0, 10.0),
    tail_frac: float = 0.25,
    device: str = "cpu",
    osc_kwargs: dict | None = None,
    k_min: int = 2,
    k_max: int = 8,
    r_min: int | None = None,
    r_max: int | None = None,
    jitter: float = 0.0,
    seed: int | None = None,
) -> BasinMap:
    """
    Full basin map for one coupling value: build the IC grid, integrate
    oscillator_cls over it, classify each grid point's post-transient
    coherence signature into basins (chimera_classifier, item 4), then
    measure the basin-boundary fractal dimension via box-counting (item 6).
    """
    if seed is not None:
        torch.manual_seed(seed)

    n_nodes = L.shape[0]
    osc_kwargs = osc_kwargs or {}
    net = oscillator_cls(L=L, coupling=coupling, device=device, **osc_kwargs)

    state0, Xg, Yg = build_ic_grid(
        base_state, slice_node_x, slice_node_y, n_nodes, bounds, resolution, device, jitter
    )
    tail = integrate_grid(net, state0, dt, steps, tail_frac)
    vps = coherence_vps_matrix(tail, A)

    cluster_result = cluster_vps_population(vps, k_min=k_min, k_max=k_max)
    label_grid = cluster_result["labels"].reshape(resolution, resolution)

    boundary = extract_boundary(label_grid)
    r, n = boxcount_2d_gpu(boundary, torch.device(device))
    D_f, r_sq = fractal_dimension(r, n, r_min=r_min, r_max=r_max)

    return BasinMap(
        coupling=coupling,
        Xg=Xg,
        Yg=Yg,
        label_grid=label_grid,
        vps_matrix=vps,
        k=cluster_result["k"],
        boundary=boundary,
        fractal_dim=D_f,
        r_squared=r_sq,
    )


def sweep_coupling(
    oscillator_cls: type[BaseOscillator],
    L: torch.Tensor,
    A: np.ndarray,
    base_state: tuple[float, ...],
    slice_node_x: int,
    slice_node_y: int,
    coupling_values,
    dt: float,
    steps: int,
    **kwargs,
) -> list[BasinMap]:
    """
    Item 5/6 orchestration: repeat map_basins across a coupling sweep so
    D_f can be plotted vs. coupling. Cross-system comparison (item 6) is
    just calling this with the same slice/grid spec for different
    oscillator_cls values -- the 2D IC slice keeps every boundary's
    dimension in [1, 2] regardless of each system's native state dimension.
    """
    return [
        map_basins(
            oscillator_cls, L, A, base_state, slice_node_x, slice_node_y,
            coupling, dt, steps, **kwargs,
        )
        for coupling in coupling_values
    ]
