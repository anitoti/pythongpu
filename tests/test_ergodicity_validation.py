from __future__ import annotations

import torch

from pythongpu.processing.ergodicity_validation import (
    validate_ergodicity,
    compare_scaling_to_control,
    checkpoint_batch_split,
)


def _tiny_lorenz_rhs_step(x, L_gpu, p):
    """Minimal uncoupled/coupled Lorenz-63 step, no DTI file needed -- keeps
    this test fast and independent of data/DTI-og.mat."""
    sigma, rho, beta, dt, K = p["sigma"], p["rho"], p["beta"], p["dt"], p["coupling"]

    def rhs(x):
        X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
        dX = sigma * (Y - X) - K * torch.matmul(X, L_gpu.T)
        dY = X * (rho - Z) - Y
        dZ = X * Y - beta * Z
        return torch.stack([dX, dY, dZ], dim=-1)

    k1 = rhs(x)
    k2 = rhs(x + 0.5 * dt * k1)
    k3 = rhs(x + 0.5 * dt * k2)
    k4 = rhs(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class _P(dict):
    """Attribute-style dict so validate_ergodicity's `p.steps_record` works
    without needing the real LorenzParams dataclass."""
    def __getattr__(self, k):
        return self[k]


def test_checkpoint_batch_split_covers_batch_disjointly():
    groups = checkpoint_batch_split(100, 5)
    assert len(groups) == 5
    covered = set()
    for g in groups:
        idx = set(range(g.start, g.stop))
        assert not (idx & covered), "checkpoint groups must be disjoint"
        covered |= idx
    assert covered == set(range(100))


def test_ergodicity_distinguishes_uncoupled_from_locked():
    """Uncoupled (K=0, N independent ergodic oscillators, no real
    multistability) must show real 1/sqrt(T) scaling; coupled (K=0.5, real
    lobe-locking on a small ring network) must show scaling that's
    statistically distinguishable from the K=0 control -- this is the
    formal, non-visual version of the K=0-vs-K=0.5 check this project's
    own methodology depends on (see math_textbook.md's ergodicity section)."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    N = 8
    L_gpu = (torch.eye(N) * 2 - torch.roll(torch.eye(N), 1, 0) - torch.roll(torch.eye(N), -1, 0))

    results = {}
    for K in (0.0, 0.5):
        p = _P(sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.05, coupling=K, steps_record=640)
        B = 400
        x0 = (torch.rand(B, N, 3) - 0.5) * 4
        for _ in range(200):
            x0 = _tiny_lorenz_rhs_step(x0, L_gpu, p)
        results[K] = validate_ergodicity(x0, L_gpu, p, _tiny_lorenz_rhs_step, device,
                                         n_scaling_checkpoints=4)

    # K=0 must look genuinely ergodic: slope close to -0.5, clean fit.
    assert results[0.0]["scaling_exponent"] < -0.35
    assert results[0.0]["scaling_fit_r_squared"] > 0.9

    # K=0.5's scaling must be statistically distinguishable from the K=0
    # control -- the correct test (see module docstring), not an absolute
    # test against the idealized -0.5 asymptote.
    cmp = compare_scaling_to_control(results[0.5], results[0.0])
    assert not cmp.matches_control
    assert abs(cmp.z_vs_control) > 4
