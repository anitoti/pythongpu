import torch

from pythongpu.oscillators.base_oscillator import BaseOscillator


class HindmarshRoseNetwork(BaseOscillator):
    """
    Network of diffusively-coupled Hindmarsh–Rose neurons.

    A three-dimensional slow–fast vector field whose isolated dynamics
    (I in the bursting window) admit a chaotic invariant set — the fourth
    exemplar in this suite's cross-system universality study, alongside the
    Lorenz, Rössler, and Van der Pol flows. The state manifold factorises as
    (x, y, z): x the fast membrane potential, y the fast recovery current,
    z the slow adaptation variable whose small time-constant r separates the
    time scales and organises the spiking/bursting foliation.

        dx/dt = y - a*x**3 + b*x**2 - z + I
        dy/dt = c - d*x**2 - y
        dz/dt = r*( s*(x - x_rest) - z )

    Diffusive coupling acts through the fast component only (H = diag[1,0,0]),
    the same projection convention every oscillator in this package uses:
    the network term subtracts coupling * (L @ x) from dx/dt, with L the
    graph Laplacian supplied by get_laplacian(). The default parameter set
    (a=1, b=3, c=1, d=5, s=4, x_rest=-1.6, r=0.006, I=3.2) places the
    uncoupled unit in its canonical chaotic-bursting regime.
    """

    def __init__(
        self,
        L,
        a=1.0,
        b=3.0,
        c=1.0,
        d=5.0,
        r=0.006,
        s=4.0,
        x_rest=-1.6,
        I=3.2,  # noqa: E741
        coupling=0.1,
        device="cpu",
    ):
        super().__init__(
            L, device=device, a=a, b=b, c=c, d=d, r=r, s=s,
            x_rest=x_rest, I=I, coupling=coupling,
        )

    def rhs(self, state):
        """
        Parameters:
        - state (Tensor): shape (3, N) for a single trajectory (row 0=x
                          membrane potential, row 1=y recovery, row 2=z
                          adaptation), or (B, 3, N) for a batch of B
                          trajectories (an initial-condition ensemble) --
                          auto-detected via state.dim() so a whole basin
                          grid integrates in one GPU call.

        Returns:
        - dstate (Tensor): derivatives (dx, dy, dz), same shape as state.
        """
        comp_dim = 1 if state.dim() == 3 else 0
        X = state.select(comp_dim, 0)
        Y = state.select(comp_dim, 1)
        Z = state.select(comp_dim, 2)

        # Intrinsic Hindmarsh–Rose slow–fast dynamics
        dX = Y - self.a * X ** 3 + self.b * X ** 2 - Z + self.I
        dY = self.c - self.d * X ** 2 - Y
        dZ = self.r * (self.s * (X - self.x_rest) - Z)

        # Network coupling through the fast (x) component only.
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


if __name__ == "__main__":
    # Simple 3-node ring network Laplacian matrix
    L_matrix = torch.tensor([
        [ 2., -1., -1.],
        [-1.,  2., -1.],
        [-1., -1.,  2.],
    ])

    # 3 nodes, each needs an initial (x, y, z) coordinate
    init_state = torch.tensor([
        [-1.0, -1.1, -0.9],  # x (membrane potential) for nodes 1, 2, 3
        [ 0.0,  0.1,  0.0],  # y (recovery) for nodes 1, 2, 3
        [ 3.0,  3.1,  2.9],  # z (adaptation) for nodes 1, 2, 3
    ])

    net = HindmarshRoseNetwork(L=L_matrix, coupling=0.05)
    traj = net.integrate(init_state, dt=0.01, steps=1000)
    print("Simulation finished. Output shape (steps, variables, nodes):", traj.shape)
