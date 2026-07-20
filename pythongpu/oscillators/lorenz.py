import torch

from pythongpu.oscillators.base_oscillator import BaseOscillator


class LorenzNetwork(BaseOscillator):
    def __init__(
        self,
        L,
        sigma=10.0,
        rho=28.0,
        beta=8.0 / 3.0,
        coupling=0.1,
        coupling_mode: str = 'x',
        dt=0.02,
        device='cpu',
    ):
        # coupling_mode: 'x', 'xy', 'z', or 'sigmoidal'
        super().__init__(L, device=device, sigma=sigma, rho=rho, beta=beta, coupling=coupling, dt=dt)
        self.coupling_mode = coupling_mode

    def rhs(self, state):
        # slice x, y, z as state[:, 0, :], state[:, 1, :], state[:, 2, :]
        X = state[:, 0, :]   # all X values (one per node)
        Y = state[:, 1, :]   # all Y values (one per node)
        Z = state[:, 2, :]   # all Z values (one per node)

        # Lorenz internal dynamics
        dX = self.sigma * (Y - X)
        dY = X * (self.rho - Z) - Y
        dZ = X * Y - self.beta * Z

        # Network coupling: modular wrapper supporting different modes
        mode = getattr(self, 'coupling_mode', 'x')
        L = self.L
        c = self.coupling

        if mode == 'x':
            # diffusive coupling on X only
            network_influence = torch.matmul(X, L.T)
            dX = dX - c * network_influence
        elif mode == 'xy':
            # diffusive coupling on X and Y components
            network_influence_X = torch.matmul(X, L.T)
            network_influence_Y = torch.matmul(Y, L.T)
            dX = dX - c * network_influence_X
            dY = dY - c * network_influence_Y
        elif mode == 'z':
            # diffusive coupling on Z only
            network_influence = torch.matmul(Z, L.T)
            dZ = dZ - c * network_influence
        elif mode == 'sigmoidal':
            # Nonlinear sigmoidal coupling on X using tanh(x_j - x_i)
            # Build matrix of x_j - x_i: shape (N, N) where row i col j = x_j - x_i
            # X is shape (N,) when state has batch dim 1
            if X.dim() == 2:  # (batch, N)
                x_vec = X.squeeze(0)
            else:
                x_vec = X
            diff = x_vec[None, :] - x_vec[:, None]
            tanh_mat = torch.tanh(diff)
            # network influence for each node i is sum_j L_ij * tanh(x_j - x_i)
            network_influence = (L * tanh_mat).sum(dim=1)
            dX = dX - c * network_influence
        else:
            raise ValueError(f"Unknown coupling_mode: {mode}")

        return torch.stack([dX, dY, dZ], dim=1)

    def rk4_step(self, state, dt=None):
        return super().rk4_step(state, dt if dt is not None else self.dt)
