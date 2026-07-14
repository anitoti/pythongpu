import torch

from pythongpu.oscillators.base_oscillator import BaseOscillator


class ChuaNetwork(BaseOscillator):
    def __init__(self, L, alpha=15.6, beta=28.0, m0=-8.0 / 7.0, m1=-5.0 / 7.0,
                 coupling=0.1, device='cpu'):
        """
        Initializes the Chua's-circuit network.

        Dimensionless Chua equations (double-scroll form):
            dx/dt = alpha * (y - x - f(x))
            dy/dt = x - y + z
            dz/dt = -beta * y
        with the piecewise-linear Chua-diode characteristic
            f(x) = m1*x + 0.5*(m0 - m1)*(|x + 1| - |x - 1|)

        Parameters:
        - L (Tensor): Laplacian/adjacency matrix of network connectivity (N x N).
        - alpha, beta (float): circuit parameters; (15.6, 28) gives the
          canonical double-scroll chaotic attractor.
        - m0, m1 (float): inner/outer slopes of the Chua diode; the canonical
          values are m0 = -8/7 ≈ -1.143 and m1 = -5/7 ≈ -0.714.
        - coupling (float): global coupling strength between nodes.
        - device (str): 'cpu' or 'cuda'.

        Ref: L. O. Chua, "Chua circuit", Scholarpedia; canonical double-scroll
        parameters alpha=15.6, beta=28, m0=-8/7, m1=-5/7.
        """
        super().__init__(L, device=device, alpha=alpha, beta=beta,
                         m0=m0, m1=m1, coupling=coupling)

    def _f(self, X):
        """Piecewise-linear Chua-diode characteristic f(x)."""
        return self.m1 * X + 0.5 * (self.m0 - self.m1) * (
            torch.abs(X + 1.0) - torch.abs(X - 1.0))

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

        # Chua internal dynamics
        dX = self.alpha * (Y - X - self._f(X))
        dY = X - Y + Z
        dZ = -self.beta * Y

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


def chua_single_step(state, L, alpha=15.6, beta=28.0, m0=-8.0 / 7.0,
                     m1=-5.0 / 7.0, coupling=0.1):
    """
    Functional single-step RHS, kept for call sites that don't need a
    ChuaNetwork instance (mirrors rossler_single_step).
    """
    x, y, z = state[0], state[1], state[2]
    fx = m1 * x + 0.5 * (m0 - m1) * (torch.abs(x + 1.0) - torch.abs(x - 1.0))
    dx = alpha * (y - x - fx) - coupling * (L @ x)
    dy = x - y + z
    dz = -beta * y
    return torch.stack([dx, dy, dz])


def simulate(L, steps=2000, dt=0.02):
    n = L.shape[0]

    # Initialize small random state on the same device as the Laplacian (L)
    state = 0.1 * torch.randn((3, n), device=L.device)

    for _ in range(steps):
        # Euler integration step
        state = state + chua_single_step(state, L) * dt

    return state


if __name__ == "__main__":
    # Single uncoupled node (L = [[0]]) should reproduce the classic
    # double-scroll: a bounded chaotic trajectory whose x-coordinate visits
    # BOTH scrolls (x takes both signs with |x| well above 1).
    L_matrix = torch.tensor([[0.0]])
    init_state = torch.tensor([[0.7], [0.0], [0.0]])  # (x, y, z) for 1 node

    net = ChuaNetwork(L=L_matrix, coupling=0.0)
    traj = net.integrate(init_state, dt=0.02, steps=20000)
    x = traj[:, 0, 0]

    # Discard transient, then summarize the attractor.
    xs = x[5000:]
    print("Simulation finished. Output shape (steps, variables, nodes):", traj.shape)
    print(f"x range: [{xs.min():.3f}, {xs.max():.3f}]")
    print(f"visits negative scroll: {(xs < -1).any().item()}  "
          f"visits positive scroll: {(xs > 1).any().item()}")
    print(f"bounded: {torch.isfinite(xs).all().item() and xs.abs().max() < 10}")
