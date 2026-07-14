"""
Author: anitoti
Date: summer 2026

this code evaluates initial condition samples
on the lorenz system in pytorch
the outcome is is a basin of attraction map
using fractal dimension to
quantify the nimble brain theory

it tracks how trajectories evolve from 4096
initial conditions into global attractors
via laplacian flow

python3 pipeline/lorenz_basins_sweep.py
"""

import os
import numpy as np
import torch
from dataclasses import dataclass, asdict

from pythongpu.oscillators.lorenz import LorenzNetwork
from pythongpu.networks.random_graphs import generate_gnm_laplacian


# -- 1. CONFIG ------------------------------------------------
@dataclass
class Config:
    """
    Central parameter store.
    """
    n_nodes        : int   = 100
    n_edges        : int   = 50 # to control connectedness
    graph_seed     : int   = 7
    sweep_points   : int   = 1024 # increase for better resolution
    tmax           : float = 200.0 # increase if basins need more time to separate into distinct attractors
    dt             : float = 0.02
    coupling       : float = 0.0001 # originally at 0.1
    sigma          : float = 10.0
    rho            : float = 28.0
    beta           : float = 8.0/3.0
    out_path       : str   = "/home/atotilca/pythongpu/data/lorenz_basins.npz"
    device         : str   = "cuda"
    # -- Affine slice parameters ---------------------------------
    slice_node_x   : int   = 73    # node index varied on the x-axis of the grid
    slice_node_y   : int   = 81    # node index varied on the y-axis of the grid
    base_state     : tuple = (1.0, 1.0, 1.0)  # (X, Y, Z) for all unvaried state entries

    # run this to get top 5 connected hub nodes for
    # testing where one node doesn't dominate the other:
    # python3 -c "import networkx as nx; G = nx.gnm_random_graph(100, 50, seed=7); print(sorted(G.degree, key=lambda x: x[1], reverse=True)[:5])"

    # run this to find neighbors of a node:
    # python3 -c "import networkx as nx; G = nx.gnm_random_graph(100, 50, seed=7); print(list(G.neighbors([NODE_NUMBER]])))"
    # 81 is a leaf of 73


# -- 2. AFFINE SLICE GRID SETUP --------------------------------
def build_ic_grid(cfg: Config):
    """
    Construct a 2-D affine slice through the (3, N)-dim state space.
    Two nodes' X components are varied over [-10, 10]; all other
    state entries are held at cfg.base_state.

    Parameters
    ----------
    cfg : Config
        Must provide sweep_points, n_nodes, slice_node_x, slice_node_y,
        base_state, and device.

    Returns
    -------
    state0 : Tensor (B, 3, N)   -- GPU batch of initial conditions
    Xg     : ndarray (m, m)      -- grid x-coordinates (for plotting)
    Yg     : ndarray (m, m)      -- grid y-coordinates (for plotting)
    """
    m   = cfg.sweep_points
    ax  = np.linspace(-10.0, 10.0, m, dtype=np.float32)
    Xg, Yg = np.meshgrid(ax, ax)                     # (m, m)

    B = m * m
    N = cfg.n_nodes
    dev = torch.device(cfg.device)

    # Tile base_state across all (B, 3, N) positions
    bx, by, bz = cfg.base_state
    state0 = torch.full((B, 3, N), bx, dtype=torch.float32, device=dev)
    state0[:, 1, :] = by
    state0[:, 2, :] = bz

    # Imprint the 2-D slice onto the chosen nodes' X components
    state0[:, 0, cfg.slice_node_x] = torch.tensor(Xg.ravel(), device=dev)
    state0[:, 0, cfg.slice_node_y] = torch.tensor(Yg.ravel(), device=dev)

    return state0, Xg, Yg


# -- 3. MAIN INTEGRATION LOOP ---------------------------------
def run_sweep(cfg: Config):
    A = generate_gnm_laplacian(cfg.n_nodes, cfg.n_edges,
                               device=cfg.device, seed=cfg.graph_seed,
                               norm=None, plot=False)

    # Build Laplacian (unweighted)
    D = torch.diag(A.sum(dim=1))
    L = D - A

    net = LorenzNetwork(L,
                        sigma=cfg.sigma, rho=cfg.rho, beta=cfg.beta,
                        coupling=cfg.coupling, dt=cfg.dt, device=cfg.device)

    state, Xg, Yg = build_ic_grid(cfg)   # (B, 3, N)

    steps = int(cfg.tmax / cfg.dt)
    print(f"Integrating {steps} RK4 steps over {state.shape[0]} ICs ...")

    for step in range(steps):
        state = net.rk4_step(state)
        if (step + 1) % 250 == 0:
            print(f"  step {step+1}/{steps}")

    # -- 4. SAVE ----------------------------------------------
    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)

    # Final state per IC: (B, 3, N) -> flatten to (B, 3*N) for clustering
    state_np = state.cpu().numpy()
    state_flat = state_np.reshape(state_np.shape[0], -1)  # (B, 3*N)

    np.savez_compressed(cfg.out_path,
        X           = Xg,                       # (m, m) grid coords
        Y           = Yg,                       # (m, m) grid coords
        state_final = state_np,                 # (B, 3, N) final state
        state_flat  = state_flat,               # (B, 3*N) flattened for k-means
        config      = asdict(cfg),
    )
    print(f"Saved -> {cfg.out_path}")


# -- 5. ENTRY POINT -------------------------------------------
if __name__ == "__main__":
    cfg = Config()
    run_sweep(cfg)
