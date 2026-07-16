from __future__ import annotations

import argparse
from pathlib import Path
import torch
import numpy as np

from pythongpu.pipeline.clv_diagnostics import CLVCalculator
from pythongpu.networks.static_adjacency import load_dti_laplacian
from pythongpu.oscillators.lorenz import LorenzNetwork


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Run CLV diagnostics on the Lorenz DTI network')
    parser.add_argument('--mat', type=str, default='data/DTI-og.mat', help='path to DTI .mat file')
    parser.add_argument('--outdir', type=str, default='output', help='output directory')
    parser.add_argument('--steps', type=int, default=500, help='number of integration steps')
    parser.add_argument('--m', type=int, default=10, help='number of tangent vectors / CLVs to compute')
    parser.add_argument('--K', type=int, default=10, help='leading K CLVs for transversality')
    parser.add_argument('--qr-interval', type=int, default=10, help='QR orthonormalization interval')
    parser.add_argument('--device', type=str, default=None, help='torch device (cpu or cuda)')
    parser.add_argument('--coupling', type=float, default=0.1, help='network coupling strength (passed to LorenzNetwork)')
    args = parser.parse_args(argv)

    device = torch.device(args.device) if args.device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load Laplacian (L) and node count
    L, N = load_dti_laplacian(args.mat, device=device)

    # build Lorenz network with user-specified coupling
    lor = LorenzNetwork(L, device=device, coupling=args.coupling)

    # wrap rhs and jac to operate on flat state vector of size 3*N
    def rhs_flat(state_flat: torch.Tensor) -> torch.Tensor:
        # state_flat shape (3*N,)
        x = state_flat.view(1, 3, N).to(device=device, dtype=torch.float32)
        dx = lor.rhs(x)  # shape (1,3,N)
        return dx.view(3 * N)

    def jac_flat(state_flat: torch.Tensor) -> torch.Tensor:
        # analytic Jacobian for Lorenz network with diffusive coupling applied to X component
        s = state_flat.view(3, N)
        X = s[0, :]
        Y = s[1, :]
        Z = s[2, :]
        sigma = lor.sigma
        rho = lor.rho
        beta = lor.beta
        coupling = lor.coupling

        # build block Jacobian of shape (3N,3N)
        n3 = 3 * N
        J = torch.zeros((n3, n3), dtype=torch.float32, device=device)

        # indices helpers
        def idx(i, comp):
            # comp: 0->x,1->y,2->z
            return comp * N + i

        # Fill blocks
        # For i,j nodes:
        # d(dX_i)/dX_j = -sigma*delta_ij - coupling * L[i,j]
        # d(dX_i)/dY_j = sigma*delta_ij
        # d(dX_i)/dZ_j = 0
        # d(dY_i)/dX_j = (rho - Z_i) * delta_ij
        # d(dY_i)/dY_j = -delta_ij
        # d(dY_i)/dZ_j = -X_i * delta_ij
        # d(dZ_i)/dX_j = Y_i * delta_ij
        # d(dZ_i)/dY_j = X_i * delta_ij
        # d(dZ_i)/dZ_j = -beta * delta_ij

        # Precompute diagonal contributions
        for i in range(N):
            for j in range(N):
                delta = 1.0 if i == j else 0.0
                Lij = float(L[i, j].item())
                J[idx(i, 0), idx(j, 0)] = (-sigma * delta) - coupling * Lij
                J[idx(i, 0), idx(j, 1)] = sigma * delta
                J[idx(i, 0), idx(j, 2)] = 0.0

                J[idx(i, 1), idx(j, 0)] = (rho - float(Z[i].item())) * delta
                J[idx(i, 1), idx(j, 1)] = -1.0 * delta
                J[idx(i, 1), idx(j, 2)] = -float(X[i].item()) * delta

                J[idx(i, 2), idx(j, 0)] = float(Y[i].item()) * delta
                J[idx(i, 2), idx(j, 1)] = float(X[i].item()) * delta
                J[idx(i, 2), idx(j, 2)] = -beta * delta

        return J

    n_total = 3 * N
    # initial state: small random seed for reproducibility
    rng = torch.Generator(device='cpu')
    rng.manual_seed(0)
    init_state = 0.1 * torch.randn(n_total, dtype=torch.float32, generator=rng)

    # Instantiate CLVCalculator
    clv_calc = CLVCalculator(rhs_fn=rhs_flat, jac_fn=jac_flat, n=n_total, dt=lor.dt, device=device, qr_interval=args.qr_interval)

    print('Running forward integration...')
    Q_list, R_half_list, qr_steps = clv_calc.run_forward(initial_state=init_state, total_steps=args.steps, m=args.m)

    print('Reconstructing CLVs...')
    clv_list = clv_calc.run_backward_reconstruct(Q_list=Q_list, R_half_list=R_half_list, qr_steps=qr_steps, leading_m=args.m)

    print(f'Computing transversality angles (K={args.K}) and saving to output...')
    out_prefix = str(Path(outdir) / 'clv_angles_')
    min_angles = clv_calc.compute_transversality_angles(clv_list, K=args.K, out_prefix=out_prefix)

    out_file = Path(f"{out_prefix}{min(args.K, args.m)}.npy")
    print(f'Done. Saved min-angle time series to {out_file}')


if __name__ == '__main__':
    main()
