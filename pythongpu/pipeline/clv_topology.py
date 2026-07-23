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
from pythongpu.oscillators.hindmarsh_rose import HindmarshRoseNetwork
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
    # Preserve the Laplacian matrix and move to the target device without changing values
    L = lor.L.to(device=device, dtype=torch.float32)

    def rhs_flat(state_flat: torch.Tensor) -> torch.Tensor:
        x = state_flat.view(1, 3, N).to(device=device, dtype=torch.float32)
        return lor.rhs(x).view(3 * N)

    def jac_flat(state_flat: torch.Tensor) -> torch.Tensor:
        s = state_flat.view(3, N)
        X, Y, Z = s[0], s[1], s[2]
        sigma, rho, beta, coupling = lor.sigma, lor.rho, lor.beta, lor.coupling
        mode = getattr(lor, 'coupling_mode', 'x')
        n3 = 3 * N
        J = torch.zeros((n3, n3), dtype=torch.float32, device=device)
        eye = torch.eye(N, dtype=torch.float32, device=device)
        xs, ys, zs = slice(0, N), slice(N, 2 * N), slice(2 * N, 3 * N)

        # dX rows -- depends on coupling mode
        if mode == 'x':
            J[xs, xs] = -sigma * eye - coupling * L
        elif mode == 'xy':
            J[xs, xs] = -sigma * eye - coupling * L
        elif mode == 'z':
            J[xs, xs] = -sigma * eye
        elif mode == 'sigmoidal':
            # Nonlinear sigmoidal coupling contributes a state-dependent Jacobian block
            # Build sech^2 matrix: S[i,j] = sech^2(x_j - x_i) = 1 - tanh^2(x_j - x_i)
            diff = X[None, :] - X[:, None]  # shape (N, N): row i col j = x_j - x_i
            tanh_mat = torch.tanh(diff)
            sech2 = 1.0 - tanh_mat * tanh_mat
            A = L * sech2  # elementwise
            df_matrix = A - torch.diag(A.sum(dim=1))
            J[xs, xs] = -sigma * eye - coupling * df_matrix
        else:
            raise ValueError(f"Unknown coupling_mode in jacobian: {mode}")

        # dX-dY block unchanged
        J[xs, ys] = sigma * eye

        # dY rows
        J[ys, xs] = torch.diag(rho - Z)
        if mode == 'xy':
            J[ys, ys] = -eye - coupling * L
        else:
            J[ys, ys] = -eye
        J[ys, zs] = torch.diag(-X)

        # dZ rows
        J[zs, xs] = torch.diag(Y)
        J[zs, ys] = torch.diag(X)
        if mode == 'z':
            J[zs, zs] = -beta * eye - coupling * L
        else:
            J[zs, zs] = -beta * eye

        return J

    return rhs_flat, jac_flat


def hr_clv_closures(
    hr: HindmarshRoseNetwork, N: int, device: torch.device
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Build (rhs_flat, jac_flat) for a Hindmarsh-Rose network over a flat 3N state.

    State layout is [X(1..N), Y(1..N), Z(1..N)], same convention as
    lorenz_clv_closures. Coupling is diffusive on X only (this package's one
    coupling mode for HR, unlike Lorenz's four), so it enters exactly one
    Jacobian block, same shape as Lorenz's mode='x' case.

    Analytic Jacobian, from HindmarshRoseNetwork.rhs's
        dX_i = Y_i - a X_i^3 + b X_i^2 - Z_i + I - coupling*(L@X)_i
        dY_i = c - d X_i^2 - Y_i
        dZ_i = r*(s*(X_i - x_rest) - Z_i)
    Per-node partials (i=j diagonal terms; coupling only touches dX/dX):
        d(dX)/dX = diag(-3a X^2 + 2b X) - coupling*L
        d(dX)/dY = I          d(dX)/dZ = -I
        d(dY)/dX = diag(-2d X)   d(dY)/dY = -I        d(dY)/dZ = 0
        d(dZ)/dX = r*s*I         d(dZ)/dY = 0          d(dZ)/dZ = -r*I
    """
    L = hr.L.to(device=device, dtype=torch.float32)

    def rhs_flat(state_flat: torch.Tensor) -> torch.Tensor:
        x = state_flat.view(1, 3, N).to(device=device, dtype=torch.float32)
        return hr.rhs(x).view(3 * N)

    def jac_flat(state_flat: torch.Tensor) -> torch.Tensor:
        s = state_flat.view(3, N)
        X = s[0]
        a, b, c, d, r, coupling = hr.a, hr.b, hr.c, hr.d, hr.r, hr.coupling
        n3 = 3 * N
        J = torch.zeros((n3, n3), dtype=torch.float32, device=device)
        eye = torch.eye(N, dtype=torch.float32, device=device)
        xs, ys, zs = slice(0, N), slice(N, 2 * N), slice(2 * N, 3 * N)

        J[xs, xs] = torch.diag(-3.0 * a * X * X + 2.0 * b * X) - coupling * L
        J[xs, ys] = eye
        J[xs, zs] = -eye

        J[ys, xs] = torch.diag(-2.0 * d * X)
        J[ys, ys] = -eye

        J[zs, xs] = hr.r * hr.s * eye
        J[zs, zs] = -r * eye

        return J

    return rhs_flat, jac_flat


def run_hr_clv_topology(
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
    dt: float = 0.05,
    seed: int = 0,
) -> dict:
    """Same as run_clv_topology, for the Hindmarsh-Rose network."""
    hr = HindmarshRoseNetwork(L, device=device, coupling=coupling)
    hr.dt = dt
    rhs_flat, jac_flat = hr_clv_closures(hr, N, device)

    clv = CLVCalculator(
        rhs_fn=rhs_flat, jac_fn=jac_flat, n=3 * N, dt=dt,
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

    exponents = lyapunov_spectrum(R_half_list, qr_interval=qr_interval, dt=dt)
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
    coupling_mode: str = 'x',
    seed: int = 0,
) -> dict:
    """Run the full CLV topological diagnostic on a network Laplacian L.

    Returns a JSON-serialisable summary dict: sorted Lyapunov spectrum, number
    of positive exponents, Kaplan-Yorke dimension (+ ceiling flag), and the
    k-means riddling verdict. Saves the minimum-angle series to
    ``{out_prefix}{K}.npy`` via the CLVCalculator.
    """
    # Construct Lorenz network preserving the supplied Laplacian exactly
    lor = LorenzNetwork(L, device=device, coupling=coupling, coupling_mode=coupling_mode)
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
