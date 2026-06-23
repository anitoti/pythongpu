import torch
import numpy as np

class KuramotoSimulator:
    def __init__(self, adj, K=1.0, device="cpu"):
        # Ensure adjacency matrix is a PyTorch tensor
        self.adj = torch.as_tensor(adj, dtype=torch.float32, device=device)
        self.K = K
        self.N = self.adj.shape[0]
        self.device = device
        
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

    def compute_derivatives(self):
        """Calculates the phase derivatives d(theta)/dt using matrix operations."""
        # Pairwise phase differences: phase_diff[i, j] = phase[j] - phase[i]
        phase_diff = self.phases.unsqueeze(0) - self.phases.unsqueeze(1)
        
        # Kuramoto coupling equation: d(theta_i)/dt = omega_i + (K/N) * sum_j(A_ij * sin(theta_j - theta_i))
        interaction = torch.sin(phase_diff) * self.adj
        coupling_term = (self.K / self.N) * torch.sum(interaction, dim=1)
        
        return self.frequencies + coupling_term

    def step(self, dt=0.01):
        """Advances the simulation by a single time step using Euler integration."""
        dtheta_dt = self.compute_derivatives()
        self.phases = (self.phases + dtheta_dt * dt) % (2 * np.pi)
        return self.phases

    def simulate(self, steps=1000, dt=0.01):
        """Runs the simulation for a set number of steps and tracks phase history."""
        if self.phases is None or self.frequencies is None:
            self.initialize_states()
            
        history = torch.zeros((steps, self.N), device=self.device)
        for t in range(steps):
            history[t] = self.step(dt)
        return history
