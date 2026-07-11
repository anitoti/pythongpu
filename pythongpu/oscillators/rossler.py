import torch

from pythongpu.oscillators.base_oscillator import BaseOscillator


class RosslerNetwork(BaseOscillator):
    def __init__(self, L, a=0.2, b=0.2, c=5.7, coupling=0.1, device='cpu'):
        """
        Initializes the Rössler Network.

        Parameters:
        - L (Tensor): Laplacian or adjacency matrix representing network connectivity (N x N).
        - a, b, c (float): Internal parameters of the chaotic Rössler oscillator.
        - coupling (float): Global coupling strength between nodes.
        - device (str): 'cpu' or 'cuda' for GPU acceleration.
        """
        super().__init__(L, device=device, a=a, b=b, c=c, coupling=coupling)

    def rhs(self, state):
        """
        Calculates the derivatives for the network.

        Parameters:
        - state (Tensor): shape (3, N) for a single trajectory (row 0=X,
                          row 1=Y, row 2=Z), or (B, 3, N) for a batch of
                          B trajectories (e.g. an IC grid) -- auto-detected
                          via state.dim() so basin-mapping sweeps can
                          integrate an entire grid in one GPU call.

        Returns:
        - dstate (Tensor): derivatives (dX, dY, dZ), same shape as state.
        """
        comp_dim = 1 if state.dim() == 3 else 0
        X = state.select(comp_dim, 0)
        Y = state.select(comp_dim, 1)
        Z = state.select(comp_dim, 2)

        # Rössler internal dynamics
        dX = -Y - Z
        dY = X + self.a * Y
        dZ = self.b + Z * (X - self.c)

        # Network coupling (diffusive coupling applied to the X component).
        # X @ L.T == L @ X for a single (N,) vector, and is the batched
        # (B, N) @ (N, N) form for a grid -- one expression covers both.
        network_influence = torch.matmul(X, self.L.T)
        dX = dX - self.coupling * network_influence

        return torch.stack([dX, dY, dZ], dim=comp_dim)

    def integrate(self, initial_state, dt, steps):
        """
        Integrates the system over time using the 4th-order Runge-Kutta (RK4) method.
        """
        state0 = torch.as_tensor(initial_state, dtype=torch.float32, device=self.device)
        return super().integrate(state0, dt, steps, method="rk4")


def rossler_single_step(state, L, a=0.2, b=0.2, c=5.7, coupling=0.1):
    """
    Functional single-step RHS, kept for call sites that don't need a
    RosslerNetwork instance (e.g. simple ER-graph sweeps in pipeline/run_sim.py).
    """
    x, y, z = state[0], state[1], state[2]
    dx = -y - z - coupling * (L @ x)
    dy = x + a * y
    dz = b + z * (x - c)
    return torch.stack([dx, dy, dz])


def simulate(L, steps=2000, dt=0.01):
    n = L.shape[0]

    # Initialize random state on the same device as the Laplacian (L)
    state = torch.randn((3, n), device=L.device)

    for _ in range(steps):
        # Euler integration step
        state = state + rossler_single_step(state, L) * dt

    return state


if __name__ == "__main__":
    # Create a simple 3-node ring network Laplacian matrix
    # L = Degree Matrix - Adjacency Matrix
    L_matrix = torch.tensor([
        [ 2., -1., -1.],
        [-1.,  2., -1.],
        [-1., -1.,  2.]
    ])

    # 3 nodes, each needs an initial (x, y, z) coordinate close to the attractor
    init_state = torch.tensor([
        [0.1, 0.2, 0.3],  # X coordinates for nodes 1, 2, 3
        [0.1, 0.2, 0.3],  # Y coordinates for nodes 1, 2, 3
        [0.1, 0.2, 0.3]   # Z coordinates for nodes 1, 2, 3
    ])

    net = RosslerNetwork(L=L_matrix, coupling=0.05)
    traj = net.integrate(init_state, dt=0.01, steps=1000)
    print("Simulation finished. Output shape (steps, variables, nodes):", traj.shape)
