from __future__ import annotations
"""Shared CLV-on-a-Laplacian driver.

Factors the Lorenz-network rhs/Jacobian closures and the full CLV topological
pipeline (forward Ginelli pass -> Lyapunov spectrum -> Kaplan-Yorke dimension ->
transversality angles -> k-means riddling verdict) out of clv_cli.py so that both
the single-network CLI (clv_cli.py) and the baseline comparison CLI
(baseline_models.py) call one implementation. The analytic Jacobian below is the
vectorised block form verified against finite differences.
"""

from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch

from pythongpu.oscillators.lorenz import LorenzNetwork
from pythongpu.pipeline.clv_diagnostics import (
    CLVCalculator,
    lyapunov_spectrum,
    kaplan_yorke_dimension,
    detect_riddling_kmeans,
)


def lorenz_clv_closures(
    lor: LorenzNetwork, N: int, device: torch.device
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Build (rhs_flat, jac_flat) for a Lorenz network over a flat 3N state.

    State layout is [X(1..N), Y(1..N), Z(1..N)] to match clv_cli. Coupling is
    diffusive on the X component only (H = e_1 e_1^T), so the coupling term
    enters exactly one Jacobian block.
    """
    L = lor.L.to(device=device, dtype=torch.float32)

    def rhs_flat(state_flat: torch.Tensor) -> torch.Tensor:
        x = state_flat.view(1, 3, N).to(device=device, dtype=torch.float32)
        return lor.rhs(x).view(3 * N)

    def jac_flat(state_flat: torch.Tensor) -> torch.Tensor:
        s = state_flat.view(3, N)
        X, Y, Z = s[0], s[1], s[2]
        sigma, rho, beta, coupling = lor.sigma, lor.rho, lor.beta, lor.coupling
        n3 = 3 * N
        J = torch.zeros((n3, n3), dtype=torch.float32, device=device)
        eye = torch.eye(N, dtype=torch.float32, device=device)
        xs, ys, zs = slice(0, N), slice(N, 2 * N), slice(2 * N, 3 * N)
        # dX rows
        J[xs, xs] = -sigma * eye - coupling * L
        J[xs, ys] = sigma * eye
        # dY rows
        J[ys, xs] = torch.diag(rho - Z)
        J[ys, ys] = -eye
        J[ys, zs] = torch.diag(-X)
        # dZ rows
        J[zs, xs] = torch.diag(Y)
        J[zs, ys] = torch.diag(X)
        J[zs, zs] = -beta * eye
        return J

    return rhs_flat, jac_flat


def run_clv_topology(
    L: torch.Tensor,
    N: int,
    *,
    coupling: float,
    steps: int,
    m: int,
    K: int,
    qr_interval: int,
    device: torch.device,
    out_prefix: str,
    label: str,
    seed: int = 0,
) -> dict:
    """Run the full CLV topological diagnostic on a network Laplacian L.

    Returns a JSON-serialisable summary dict: sorted Lyapunov spectrum, number
    of positive exponents, Kaplan-Yorke dimension (+ ceiling flag), and the
    k-means riddling verdict. Saves the minimum-angle series to
    ``{out_prefix}{K}.npy`` via the CLVCalculator.
    """
    lor = LorenzNetwork(L, device=device, coupling=coupling)
    rhs_flat, jac_flat = lorenz_clv_closures(lor, N, device)

    clv = CLVCalculator(
        rhs_fn=rhs_flat, jac_fn=jac_flat, n=3 * N, dt=lor.dt,
        device=device, qr_interval=qr_interval,
    )

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    init_state = 0.1 * torch.randn(3 * N, dtype=torch.float32, generator=rng)

    Q_list, R_half_list, qr_steps = clv.run_forward(
        initial_state=init_state, total_steps=steps, m=m
    )
    clv_list = clv.run_backward_reconstruct(Q_list, R_half_list, qr_steps, leading_m=m)
    min_angles = clv.compute_transversality_angles(clv_list, K=K, out_prefix=out_prefix)

    exponents = lyapunov_spectrum(R_half_list, qr_interval=qr_interval, dt=lor.dt)
    exponents_sorted = np.sort(exponents)[::-1]
    d_ky = kaplan_yorke_dimension(exponents)
    n_positive = int((exponents > 0).sum())
    ky_ceiling = bool(d_ky >= m)
    riddle = detect_riddling_kmeans(min_angles)

    return {
        "label": label,
        "coupling": coupling,
        "n_nodes": int(N),
        "n_clvs_computed": int(m),
        "lyapunov_exponents": exponents_sorted.tolist(),
        "lambda_max": float(exponents_sorted[0]),
        "n_positive_exponents": n_positive,
        "kaplan_yorke_dimension": d_ky,
        "kaplan_yorke_is_ceiling": ky_ceiling,
        "riddling": riddle,
    }


def format_topology_line(summary: dict, m: int) -> str:
    """One-line human-readable rendering of a run_clv_topology summary."""
    ky = (f">= {m} (ceiling)" if summary["kaplan_yorke_is_ceiling"]
          else f"{summary['kaplan_yorke_dimension']:.3f}")
    r = summary["riddling"]
    return (
        f"{summary['label']:<16s} "
        f"lam_max={summary['lambda_max']:+.4f}  "
        f"n_pos={summary['n_positive_exponents']:>3d}/{m:<3d}  "
        f"D_KY={ky:<14s}  "
        f"{r['verdict']:<13s} burst={r['burst_fraction']:.3f}"
    )
