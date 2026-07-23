from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
import numpy as np

from pythongpu.pipeline.clv_topology import run_hr_clv_topology
from pythongpu.networks.static_adjacency import load_dti_laplacian
from pythongpu.networks.random_graphs import generate_ba_graph


def load_seeded_graph_laplacian(npz_path: str, device: torch.device):
    """Load one graph produced by scripts/seeded_random_graph_sweep.py and
    build its Laplacian the same way load_dti_laplacian does (L = diag(sum(A,2)) - A),
    so the seeded BA/GNM/WS nulls drop into the exact same CLV pipeline as
    the DTI connectome and the ad hoc --null-model BA graph."""
    d = np.load(npz_path)
    A = torch.as_tensor(d["adjacency"], dtype=torch.float32, device=device)
    L = torch.diag(A.sum(dim=1)) - A
    return L, A.shape[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Run CLV diagnostics on the Hindmarsh-Rose DTI network '
                                                  '(mirrors clv_cli.py, HR vector field instead of Lorenz)')
    parser.add_argument('--mat', type=str, default='data/DTI-og.mat', help='path to DTI .mat file')
    parser.add_argument('--outdir', type=str, default='output', help='output directory')
    parser.add_argument('--steps', type=int, default=500, help='number of integration steps')
    parser.add_argument('--m', type=int, default=10, help='number of tangent vectors / CLVs to compute')
    parser.add_argument('--K', type=int, default=10, help='leading K CLVs for transversality')
    parser.add_argument('--qr-interval', type=int, default=10, help='QR orthonormalization interval')
    parser.add_argument('--device', type=str, default=None, help='torch device (cpu or cuda)')
    parser.add_argument('--coupling', type=float, default=0.1, help='network coupling strength (passed to HindmarshRoseNetwork)')
    parser.add_argument('--dt', type=float, default=0.05, help='integration step -- HindmarshRoseParams default (see hr_sweep.py)')
    parser.add_argument('--null-model', action='store_true', help='use a BA scale-free null model instead of the DTI connectome')
    parser.add_argument('--random-graph-file', type=str, default=None,
                        help='path to a seeded null-model graph produced by '
                             'scripts/seeded_random_graph_sweep.py (e.g. '
                             'data/derivatives/random_graph_seeds/ba/seed003.npz) -- '
                             'takes priority over --null-model/--mat if given.')
    args = parser.parse_args(argv)

    device = torch.device(args.device) if args.device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.random_graph_file:
        L, N = load_seeded_graph_laplacian(args.random_graph_file, device=device)
        stem = Path(args.random_graph_file).parent.name + "_" + Path(args.random_graph_file).stem
        label = f'seeded null ({stem})'
    elif args.null_model:
        print('Using BA scale-free null model (n=83, m=6)')
        L = generate_ba_graph(n=83, m=6, device=device, plot=False)
        N = L.shape[0]
        label = 'BA null model'
    else:
        L, N = load_dti_laplacian(args.mat, device=device)
        label = 'DTI connectome'

    print('Running forward integration + CLV reconstruction + topology (Hindmarsh-Rose)...')

    coupling_str = f"c{args.coupling:.2f}".replace('.', '_')
    mode_dir = outdir / coupling_str
    mode_dir.mkdir(parents=True, exist_ok=True)

    summary = run_hr_clv_topology(
        L, N, coupling=args.coupling, steps=args.steps, m=args.m, K=args.K,
        qr_interval=args.qr_interval, device=device,
        out_prefix=str(mode_dir / 'clv_angles_'), label=label,
        dt=args.dt,
    )

    exps = np.asarray(summary['lyapunov_exponents'])
    riddle = summary['riddling']
    print(f'\n=== Topological summary [{label}, coupling={args.coupling}, system=hindmarsh-rose] ===')
    print(f'  leading exponents : {np.round(exps[:min(5, len(exps))], 4)}')
    print(f'  positive exponents: {summary["n_positive_exponents"]} / {args.m} computed  ->  '
          f'{"chaotic" if summary["n_positive_exponents"] >= 1 else "non-chaotic"}')
    if summary['kaplan_yorke_is_ceiling']:
        print(f'  Kaplan–Yorke dim  : >= {args.m} (CEILING — running exponent sum '
              f'still positive at all {args.m} computed CLVs; rerun with larger '
              f'--m to resolve D_KY)')
    else:
        print(f'  Kaplan–Yorke dim  : {summary["kaplan_yorke_dimension"]:.4f}')
    print(f'  riddling verdict  : {riddle["verdict"]}  '
          f'(burst_fraction={riddle["burst_fraction"]:.3f}, '
          f'centroid_sep={riddle["centroid_separation_rad"]:.3f} rad)')

    summary_path = mode_dir / 'clv_topology_summary.json'
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'Saved topological summary to {summary_path}')


if __name__ == '__main__':
    main()
