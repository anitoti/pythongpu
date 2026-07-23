"""
Exact reproduction of ~/matlab_fractalbasin/HR_ElCh_network.m and
HR_ElCh_distribution_network.m -- the actual model behind the paper's
headline DTI-connectome result (Fig. 2), per talk/notes/matlab_source_audit.md.

Fixes two structural differences found between the original and this
project's existing pythongpu.oscillators.hindmarsh_rose.HindmarshRoseNetwork
(that class is intentionally left alone -- it's what tonight's validated
streaming/true-VPS/CLV results for HR are built on):

  1. Diffusive/"electrical" coupling acts on Y (the recovery variable),
     not X (membrane potential). The original's H = [0,0,0; 0,1,0; 0,0,0]
     has its one nonzero entry at the Y slot within each node's (x,y,z)
     block -- confirmed directly from HR_ElCh_network.m's state unpacking
     (x = w(1:3:end-2), y = w(2:3:end-1), z = w(3:3:end)).
  2. A chemical/sigmoidal synaptic coupling term exists, acting on X,
     using the RAW adjacency matrix A (not the graph Laplacian):
         gch * (x_i - Vsyn) * sum_j A_ij * sigmoid(lambda*(x_j - thetasyn))
     This is a real physical term in the original, entirely absent from
     the simplified pythongpu HR model.

Equations (matches HR_ElCh_network.m line for line):
    dx_i = y_i - a*x_i^3 + b*x_i^2 - z_i + Iext
           - gch*(x_i - Vsyn) * sum_j A_ij * sigmoid(lambda*(x_j - thetasyn))
    dy_i = c - d*x_i^2 - y_i  - gel * (L @ y)_i
    dz_i = r*(s*(x_i - p0) - z_i)

Default parameters are the original's own values (a=1, b=3, c=1, d=5, s=4,
p0=-1.6, Iext=3.25, r=0.005, Vsyn=2, thetasyn=-0.25, lambda=10), citing
Hizanidis et al., "Chimera-like States in Modular Neural Networks" -- NOT
this project's earlier HR defaults (r=0.006, I=3.2), which were a close
but not exact approximation.
"""
from __future__ import annotations

import torch

from pythongpu.oscillators.base_oscillator import BaseOscillator


class HindmarshRoseNetworkExact(BaseOscillator):
    def __init__(
        self,
        A,
        gel: float = 0.5,
        gch: float = 0.0,
        a: float = 1.0,
        b: float = 3.0,
        c: float = 1.0,
        d: float = 5.0,
        r: float = 0.005,
        s: float = 4.0,
        p0: float = -1.6,
        Iext: float = 3.25,
        Vsyn: float = 2.0,
        thetasyn: float = -0.25,
        lam: float = 10.0,
        device="cpu",
    ):
        A_t = torch.as_tensor(A, dtype=torch.float32, device=device)
        L = torch.diag(A_t.sum(dim=1)) - A_t
        super().__init__(
            L, device=device, a=a, b=b, c=c, d=d, r=r, s=s, p0=p0,
            Iext=Iext, gel=gel, gch=gch, Vsyn=Vsyn, thetasyn=thetasyn, lam=lam,
        )
        # BaseOscillator only keeps the Laplacian; the chemical coupling
        # term needs the raw adjacency matrix directly (A*sigmoid(...), not
        # L*sigmoid(...)) -- confirmed from the original's own line
        # `A*(1./(1+exp(-lambda*(x-thetasyn))))`.
        self.A = A_t

    def rhs(self, state: torch.Tensor) -> torch.Tensor:
        """
        state : (3, N) for a single trajectory (row 0=x, 1=y, 2=z), or
                (B, 3, N) for a batch -- same convention as
                pythongpu.oscillators.hindmarsh_rose.HindmarshRoseNetwork.
        """
        comp_dim = 1 if state.dim() == 3 else 0
        X = state.select(comp_dim, 0)
        Y = state.select(comp_dim, 1)
        Z = state.select(comp_dim, 2)

        dX = Y - self.a * X**3 + self.b * X**2 - Z + self.Iext
        dY = self.c - self.d * X**2 - Y
        dZ = self.r * (self.s * (X - self.p0) - Z)

        # Chemical/sigmoidal coupling -- acts on X, uses the RAW adjacency.
        # Skippable at gch=0 (matches ForPlottingPNAS_PeaksFigure.m panel a).
        if self.gch != 0.0:
            sig = torch.sigmoid(self.lam * (X - self.thetasyn))
            chem_influence = torch.matmul(sig, self.A.T)
            dX = dX - self.gch * (X - self.Vsyn) * chem_influence

        # Diffusive/electrical coupling -- acts on Y, NOT X (see docstring).
        dY = dY - self.gel * torch.matmul(Y, self.L.T)

        return torch.stack([dX, dY, dZ], dim=comp_dim)

    def integrate(self, initial_state, dt, steps):
        state0 = torch.as_tensor(initial_state, dtype=torch.float32, device=self.device)
        return super().integrate(state0, dt, steps, method="rk4")
