from __future__ import annotations

import numpy as np
import torch
from pythongpu.pipeline.clv_diagnostics import CLVCalculator


def test_clv_linear_smoke(tmp_path):
    # simple linear system x' = A x where A is constant; CLVs should align with eigenvectors
    n = 4
    rng = np.random.RandomState(0)
    A = rng.randn(n, n).astype(np.float32)
    # make A upper-triangular to ensure distinct Lyapunov spectrum
    A = np.triu(A)
    A_torch = torch.from_numpy(A)

    def rhs(x: torch.Tensor) -> torch.Tensor:
        return A_torch.matmul(x)

    def jac(x: torch.Tensor) -> torch.Tensor:
        return A_torch

    # initial state
    init = torch.randn(n, dtype=torch.float32)

    clv = CLVCalculator(rhs_fn=rhs, jac_fn=jac, n=n, dt=0.01, device=torch.device('cpu'), qr_interval=5)
    Q_list, R_half_list, qr_steps = clv.run_forward(initial_state=init, total_steps=30, m=2)
    assert len(Q_list) == len(R_half_list) == len(qr_steps) and len(Q_list) > 0

    clv_list = clv.run_backward_reconstruct(Q_list, R_half_list, qr_steps, leading_m=2)
    assert isinstance(clv_list, list) and len(clv_list) == len(Q_list)

    min_angles = clv.compute_transversality_angles(clv_list, K=2, out_prefix=str(tmp_path / 'clv_angles_'))
    assert min_angles.shape[0] == len(clv_list)
