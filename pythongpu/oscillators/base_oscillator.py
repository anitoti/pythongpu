from abc import ABC, abstractmethod

import torch


class BaseOscillator(ABC):
    """
    Common interface for network-coupled dynamical systems.

    Generalizes the forward()/rk4_step()/integrate() pattern independently
    implemented by LorenzNetwork, RosslerNetwork, and KuramotoSimulator.
    """

    def __init__(self, L, device="cpu", **params):
        self.L = torch.as_tensor(L, dtype=torch.float32, device=device)
        self.N = self.L.shape[0]
        self.device = device
        for k, v in params.items():
            setattr(self, k, v)

    @abstractmethod
    def rhs(self, state):
        """Return d(state)/dt given the current state. Shape is subclass-defined."""
        raise NotImplementedError

    def rk4_step(self, state, dt):
        k1 = self.rhs(state)
        k2 = self.rhs(state + 0.5 * dt * k1)
        k3 = self.rhs(state + 0.5 * dt * k2)
        k4 = self.rhs(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def euler_step(self, state, dt):
        return state + self.rhs(state) * dt

    def integrate(self, state0, dt, steps, method="rk4"):
        step_fn = self.rk4_step if method == "rk4" else self.euler_step
        state = state0
        history = torch.zeros((steps, *state0.shape), device=self.device)
        for t in range(steps):
            state = step_fn(state, dt)
            history[t] = state
        return history
