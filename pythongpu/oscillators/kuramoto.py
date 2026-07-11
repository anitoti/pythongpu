import numpy as np
import torch

from pythongpu.oscillators.base_oscillator import BaseOscillator


class KuramotoSimulator(BaseOscillator):
    def __init__(self, adj, K=1.0, device="cpu"):
        super().__init__(adj, device=device, K=K)
        self.adj = self.L  # Kuramoto couples via the adjacency matrix, not a Laplacian

        # Placeholders for simulation states
        self.phases = None
        self.frequencies = None

    def initialize_states(self, initial_phases=None, frequencies=None):
        """Initializes natural frequencies and starting phases."""
        if initial_phases is None:
            # Random phases between 0 and 2*pi
            self.phases = torch.rand(self.N, device=self.device) * 2 * np.pi
        else:
            self.phases = torch.as_tensor(initial_phases, dtype=torch.float32, device=self.device)

        if frequencies is None:
            # Standard normal distribution for natural frequencies
            self.frequencies = torch.randn(self.N, device=self.device)
        else:
            self.frequencies = torch.as_tensor(frequencies, dtype=torch.float32, device=self.device)

    def rhs(self, state):
        """Kuramoto coupling equation: d(theta_i)/dt = omega_i + (K/N) * sum_j(A_ij * sin(theta_j - theta_i))."""
        phase_diff = state.unsqueeze(0) - state.unsqueeze(1)
        interaction = torch.sin(phase_diff) * self.adj
        coupling_term = (self.K / self.N) * torch.sum(interaction, dim=1)
        return self.frequencies + coupling_term

    def compute_derivatives(self):
        """Calculates the phase derivatives d(theta)/dt for the current internal state."""
        return self.rhs(self.phases)

    def step(self, dt=0.01):
        """Advances the simulation by a single time step using Euler integration."""
        self.phases = self.euler_step(self.phases, dt) % (2 * np.pi)
        return self.phases

    def simulate(self, steps=1000, dt=0.01):
        """Runs the simulation for a set number of steps and tracks phase history."""
        if self.phases is None or self.frequencies is None:
            self.initialize_states()

        history = torch.zeros((steps, self.N), device=self.device)
        for t in range(steps):
            history[t] = self.step(dt)
        return history
