import torch

from pythongpu.oscillators.base_oscillator import BaseOscillator


class VanDerPolNetwork(BaseOscillator):
    """
    Network of diffusively-coupled Van der Pol relaxation oscillators.

    Ported from jefish003/GenerateDynamics (GenerateDynamics.py,
    laplacian_dynamics.vanderpol, dispatched via
    continuous_time_nonlinear_dynamics(dynamics_type='VanDerPol')):

        dx/dt = y
        dy/dt = mu*(1 - x**2)*y - x

    Coupling: LH = coupling_strength * kron(L, H) with H = diag([1, 0])
    — the reference couples only through the x (position) component
    ("assume coupling only through the x component" in the source),
    not through y. L is a graph Laplacian built from a (possibly
    directed) adjacency matrix, same convention as get_laplacian().

    mu = 1.5 is the reference's own documented default
    ("Suggested example value is [1.5]" in its README).
    """

    def __init__(self, L, mu=1.5, coupling=1.0, device='cpu'):
        super().__init__(L, device=device, mu=mu, coupling=coupling)

    def rhs(self, state):
        """
        Parameters:
        - state (Tensor): shape (2, N) for a single trajectory (row 0=x
                          position, row 1=y velocity), or (B, 2, N) for a
                          batch of B trajectories (e.g. an IC grid) --
                          auto-detected via state.dim() so basin-mapping
                          sweeps can integrate an entire grid in one call.

        Returns:
        - dstate (Tensor): derivatives (dx, dy), same shape as state.
        """
        comp_dim = 1 if state.dim() == 3 else 0
        X = state.select(comp_dim, 0)
        Y = state.select(comp_dim, 1)

        # Intrinsic Van der Pol dynamics
        dX = Y
        dY = self.mu * (1 - X ** 2) * Y - X

        # Network coupling through the x component only (H = diag([1, 0])).
        # X @ L.T == L @ X for a single (N,) vector, and is the batched
        # (B, N) @ (N, N) form for a grid -- one expression covers both.
        network_influence = torch.matmul(X, self.L.T)
        dX = dX - self.coupling * network_influence

        return torch.stack([dX, dY], dim=comp_dim)

    def integrate(self, initial_state, dt, steps):
        """
        Integrates the system over time using the 4th-order Runge-Kutta (RK4) method.
        """
        state0 = torch.as_tensor(initial_state, dtype=torch.float32, device=self.device)
        return super().integrate(state0, dt, steps, method="rk4")


if __name__ == "__main__":
    # Simple 3-node ring network Laplacian matrix
    L_matrix = torch.tensor([
        [ 2., -1., -1.],
        [-1.,  2., -1.],
        [-1., -1.,  2.]
    ])

    # 3 nodes, each needs an initial (x, y) coordinate
    init_state = torch.tensor([
        [0.1, 0.2, 0.3],  # x for nodes 1, 2, 3
        [0.1, 0.2, 0.3],  # y for nodes 1, 2, 3
    ])

    net = VanDerPolNetwork(L=L_matrix, mu=1.5, coupling=0.05)
    traj = net.integrate(init_state, dt=0.01, steps=1000)
    print("Simulation finished. Output shape (steps, variables, nodes):", traj.shape)
