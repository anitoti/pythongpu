import torch

from pythongpu.oscillators.base_oscillator import BaseOscillator


class LorenzNetwork(BaseOscillator):
    def __init__(self, L, sigma=10.0, rho=28.0, beta=8.0 / 3.0, coupling=0.1, dt=0.02, device='cpu'):
        super().__init__(L, device=device, sigma=sigma, rho=rho, beta=beta, coupling=coupling, dt=dt)

    def rhs(self, state):
        # slice x, y, z as state[:, 0, :], state[:, 1, :], state[:, 2, :]
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

    def rk4_step(self, state, dt=None):
        return super().rk4_step(state, dt if dt is not None else self.dt)
