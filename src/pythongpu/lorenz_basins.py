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

python3 src/pythongpu/lorenz_basins.py
"""

import os
import math
import numpy as np
import torch
import networkx as nx
from dataclasses import dataclass, asdict


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

# -- 2. NETWORK -----------------------------------------------
class LorenzNetwork:
    def __init__(self, L, sigma=10.0, rho=28.0, beta=8.0/3.0, coupling=0.1, dt=0.02, device='cpu'):
        self.L = torch.as_tensor(L, dtype=torch.float32, device=device)
        self.N = self.L.shape[0]
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.coupling = coupling
        self.dt = dt
        self.device = device

    def forward(self, state):
        # slice x, y, z as state[0], state[1], state[2]
        X = state[:, 0, :]   # all X values (one per node)
        Y = state[:, 1, :]   # all Y values (one per node)
        Z = state[:, 2, :]   # all Z values (one per node)

        # Lorenz internal dynamics
        dX = self.sigma * (Y - X)
        dY = X * (self.rho - Z) - Y
        dZ = X * Y - self.beta * Z

        # Network coupling (diffusive coupling applied to the X component)
        network_influence = torch.matmul(X, self.L.T)
        dX = dX - self.coupling * network_influence

        return torch.stack([dX, dY, dZ], dim=1)

    def rk4_step(self, state):
        dt = self.dt
        k1 = self.forward(state)
        k2 = self.forward(state + 0.5 * dt * k1)
        k3 = self.forward(state + 0.5 * dt * k2)
        k4 = self.forward(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# -- 3. GRAPH LOADER ------------------------------------------
def load_dti_laplacian(mat_path: str, device: torch.device) -> tuple[torch.Tensor, int]:
    """
    Load DTI_A.mat, build graph Laplacian, return L and node count n.

    Replicates MATLAB exactly:
        load('DTI_A.mat')
        A = double(A)
        L = diag(sum(A,2)) - A
        gel = 0.5
        H = [0 0 0; 0 1 0; 0 0 0]
        gelLH = gel * kron(L, H)
    [Page 28, full_.m_script.pdf]

    We do NOT form the full kron(L,H) — instead we apply the H selection
    (coupling only through X component) directly in lorenz_rhs_batched
    by only subtracting the Laplacian term from dX.

    Args
    ----
    mat_path : path to DTI_A.mat
    device   : torch.device

    Returns
    -------
    L_gpu : (n, n) float32 Laplacian tensor on device
    n     : number of nodes
    """
    mat = loadmat(mat_path)

    # DTI_A.mat stores the adjacency matrix as variable 'A'
    # [Page 28, full_.m_script.pdf: "load('DTI_A.mat') A = double(A)"]
    A = mat["A"].astype(np.float64)
    n = A.shape[1]   # "n = size(A,2)" [Page 28, full_.m_script.pdf]

    # Graph Laplacian
    # [Page 28, full_.m_script.pdf: "L = diag(sum(A,2)) - A"]
    L = np.diag(A.sum(axis=1)) - A

    print(f"[dti]      loaded DTI_A.mat  n={n}  edges={int(A.sum()//2)}")
    return torch.tensor(L, dtype=torch.float32, device=device), n


# -- 4. AFFINE SLICE GRID SETUP --------------------------------
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


# -- 5. MAIN INTEGRATION LOOP ---------------------------------
def run_sweep(cfg: Config):
    dev = torch.device(cfg.device)
    A   = load_graph(cfg)

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

    # -- 6. SAVE ----------------------------------------------
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


# -- 7. ENTRY POINT -------------------------------------------
if __name__ == "__main__":
    cfg = Config()
    run_sweep(cfg)