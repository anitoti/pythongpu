# ============================================================
#  Kuramoto Basin Mapper — RK4 GPU Sweep
#  Project : Nimble Brain (REU @ Clarkson)
#  Author  : anitoti
#  Purpose : 2-D slice of phase-space → basin stability map
#            Node 0 & Node 1 phases swept over [-π, π]²
#            All other nodes initialized to θ=0
# ============================================================

# CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 /home/atotilca/pythongpu/kuramoto_basins.py

import os
import math
import numpy as np
import torch
import networkx as nx
from dataclasses import dataclass, asdict


# ── 1. CONFIG ────────────────────────────────────────────────
@dataclass
class Config:
    """
    Central parameter store.  Pass one Config instance through
    the whole pipeline so nothing is hardcoded downstream.

    Fields
    ------
    n_nodes        : number of Kuramoto oscillators
    n_edges        : target edge count for the random graph
    graph_seed     : RNG seed → reproducible topology
    sweep_points   : grid resolution per axis  (m × m total ICs)
    tmax           : total integration time  [arb. units]
    dt             : RK4 step size
    coupling       : global coupling strength K
    omega_scale    : std-dev of natural frequencies ~ N(0, ω_scale)
    sync_threshold : order-parameter cutoff to call a state 'synced'
    out_path       : .npy save path on the HPC node
    device         : 'cuda' or 'cpu'
    """
    n_nodes        : int   = 100
    n_edges        : int   = 500
    graph_seed     : int   = 7
    sweep_points   : int   = 64
    tmax           : float = 20.0
    dt             : float = 0.02
    coupling       : float = 17.6438 # transition point is between 17.6438 and 17.6439
    omega_scale    : float = 0.5
    sync_threshold : float = 0.9
    out_path       : str   = "/home/atotilca/pythongpu/data/kuramoto_basins.npy"
    device         : str   = "cuda"


# ── 2. GRAPH LOADER ──────────────────────────────────────────
def load_graph(cfg: Config) -> torch.Tensor:
    """
    Build a G(n,M) random graph with EXACTLY cfg.n_edges edges.
    
    G(n,p) — gnp_random_graph : edges added independently with prob p
                                 actual edge count varies run-to-run
    G(n,M) — gnm_random_graph : exactly M edges sampled uniformly
                                 edge count is guaranteed = cfg.n_edges ✓

    Returns
    -------
    A : (n_nodes × n_nodes) float32 tensor on cfg.device
    """
    G = nx.gnm_random_graph(cfg.n_nodes, cfg.n_edges, seed=cfg.graph_seed)
    A = nx.to_numpy_array(G).astype(np.float32)
    return torch.tensor(A, device=cfg.device)


# ── 3. DYNAMICS ──────────────────────────────────────────────
def kuramoto_rhs(theta: torch.Tensor,
                 omega: torch.Tensor,
                 A    : torch.Tensor,
                 K    : float) -> torch.Tensor:
    """
    Vectorized Kuramoto right-hand side (all ICs simultaneously):

        dθᵢ/dt = ωᵢ + (K/N) Σⱼ Aᵢⱼ sin(θⱼ − θᵢ)

    Parameters
    ----------
    theta : (B × N)  — phases for B initial conditions, N oscillators
    omega : (N,)     — natural frequencies (broadcast over batch)
    A     : (N × N)  — adjacency matrix
    K     : float    — coupling strength

    Returns
    -------
    dtheta : (B × N) — phase velocities

    Shape bookkeeping
    -----------------
    theta.unsqueeze(-1)            →  (B, N, 1)
    theta.unsqueeze(-2)            →  (B, 1, N)
    diff = θᵢ − θⱼ  (broadcast)  →  (B, N, N)   diff[b,i,j] = θᵢ - θⱼ
    sin(-diff) = sin(θⱼ − θᵢ)    →  (B, N, N)   ✓ correct Kuramoto sign
    A * sin(-diff)                 →  (B, N, N)   mask non-edges
    sum(dim=-1)                    →  (B, N)      row-sum over j
    """
    diff    = theta.unsqueeze(-1) - theta.unsqueeze(-2)   # (B, N, N)
    dtheta  = omega + (K / theta.shape[-1]) * torch.sum(A * torch.sin(-diff), dim=-1)
    return dtheta                                          # (B, N)


def rk4_step(theta: torch.Tensor,
             dt   : float,
             omega: torch.Tensor,
             A    : torch.Tensor,
             K    : float) -> torch.Tensor:
    """
    Classic 4th-order Runge-Kutta step:

        k1 = f(θ)
        k2 = f(θ + dt/2 · k1)
        k3 = f(θ + dt/2 · k2)
        k4 = f(θ + dt   · k3)
        θ_new = θ + (dt/6)(k1 + 2k2 + 2k3 + k4)

    Local truncation error ~ O(dt⁵),  global error ~ O(dt⁴).
    All intermediates stay on GPU; no Python loops over B or N.

    Parameters / Returns: same shapes as kuramoto_rhs.
    """
    k1 = kuramoto_rhs(theta,                   omega, A, K)
    k2 = kuramoto_rhs(theta + 0.5 * dt * k1,  omega, A, K)
    k3 = kuramoto_rhs(theta + 0.5 * dt * k2,  omega, A, K)
    k4 = kuramoto_rhs(theta +        dt * k3,  omega, A, K)
    return theta + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# ── 4. GRID SETUP ────────────────────────────────────────────
def build_ic_grid(cfg: Config) -> torch.Tensor:
    """
    Construct the 2-D slice of initial conditions.

    The full phase space is R^N (N = n_nodes).
    We fix a random plane through it by varying only two
    coordinates (nodes 0 and 1) over [-π, π]² while all
    other nodes start at θ = 0.

    This mirrors the MATLAB pattern [Page 67]:
        S1(NodexVals(Randnodei)) = Xg(j);
        S1(NodexVals(Randnodej)) = Yg(j);

    Returns
    -------
    theta0 : (m² × N) float32 tensor on cfg.device
             Row j = one initial condition from the grid
    X, Y   : (m × m) numpy arrays  — grid coordinates for plotting
    """
    m   = cfg.sweep_points
    ax  = np.linspace(-np.pi, np.pi, m, dtype=np.float32)
    X, Y = np.meshgrid(ax, ax)                             # (m, m)

    # (m², N) — all zeros; then overwrite the two swept nodes
    theta0 = torch.zeros((m * m, cfg.n_nodes),
                         device=cfg.device, dtype=torch.float32)
    theta0[:, 0] = torch.tensor(X.ravel(), device=cfg.device)
    theta0[:, 1] = torch.tensor(Y.ravel(), device=cfg.device)
    return theta0, X, Y


# ── 5. SYNCHRONY CLASSIFIER ──────────────────────────────────
def order_parameter(theta: torch.Tensor) -> torch.Tensor:
    """
    Kuramoto order parameter R ∈ [0, 1]:

        R = |⟨e^{iθ}⟩|  =  (1/N) |Σⱼ e^{iθⱼ}|

        R ≈ 1  →  global synchrony
        R ≈ 0  →  incoherence
        0 < R < 1  →  chimera / partial sync  ← region of interest

    Parameters
    ----------
    theta : (B × N)  complex exponentials computed inline

    Returns
    -------
    R : (B,) real float32  — one value per initial condition
    """
    # torch.exp works on complex tensors created via 1j trick
    R = torch.abs(torch.exp(1j * theta).mean(dim=-1))
    return R                                               # (B,)


# ── 6. MAIN INTEGRATION LOOP ─────────────────────────────────
def run_sweep(cfg: Config):
    """
    Full pipeline:
      build graph → sample ω → build IC grid → integrate → classify → save

    Memory note:
        B = m² = 4096,  N = 100  → theta is (4096 × 100) float32 = ~1.6 MB
        Intermediate (B, N, N)   =  (4096 × 100 × 100) float32 = ~164 MB
        Both fit comfortably in TITAN Xp's 12 GB VRAM.
        [Page 67]: 'a pytorch tensor can be instantly beamed directly
        into the titan xp gpu memory ... the gpu run[s] math on the
        entire grid at the exact same time instead of going row by row!'
    """
    dev  = torch.device(cfg.device)
    A    = load_graph(cfg)                                  # (N, N)
    omega = torch.randn(cfg.n_nodes, device=dev) * cfg.omega_scale  # (N,)

    theta, X, Y = build_ic_grid(cfg)                       # (B, N)

    steps = int(cfg.tmax / cfg.dt)
    print(f"Integrating {steps} RK4 steps over {theta.shape[0]} ICs ...")

    for step in range(steps):
        theta = rk4_step(theta, cfg.dt, omega, A, cfg.coupling)
        # Wrap phases back to [-π, π] to prevent float overflow
        # over long integrations
        theta = torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi

    # ── 7. CLASSIFY & SAVE ──────────────────────────────────
    R = order_parameter(theta).cpu().numpy()               # (B,)

    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)

    # Save dict-style .npy (use allow_pickle=True to reload)
    np.save(cfg.out_path, {
        "X"        : X,                                    # (m, m) grid coords
        "Y"        : Y,                                    # (m, m) grid coords
        "theta"    : theta.cpu().numpy(),
        "R"        : R.reshape(cfg.sweep_points,
                               cfg.sweep_points),          # (m, m) order param
        "config"   : asdict(cfg),                          # full param record
    })
    print(f"Saved → {cfg.out_path}")

    synced_frac = (R >= cfg.sync_threshold).mean()
    print(f"Basin stability estimate (R ≥ {cfg.sync_threshold}): "
          f"{synced_frac:.3f}")


# ── 8. ENTRY POINT ───────────────────────────────────────────
if __name__ == "__main__":
    cfg = Config()
    run_sweep(cfg)