#!/usr/bin/env python3
"""
Van der Pol coupling sweep driver.

Loads a structural connectivity matrix ("subject 01" analog — see
pipeline/run_vdp_sweep.sh; no subject-tagged fMRI timeseries exists in
this repo yet, so DTI_A.mat stands in, same as the Lorenz/Rossler
sweeps), builds its graph Laplacian, and integrates a VanDerPolNetwork
across a range of coupling strengths, logging progress and saving
per-coupling final-state summaries.

Run:
    python3 pipeline/vdp_sweep.py
    python3 pipeline/vdp_sweep.py --coupling-min 0 --coupling-max 2 --coupling-steps 41
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from pythongpu.oscillators.vanderpol import VanDerPolNetwork
from pythongpu.networks.static_adjacency import load_dti_laplacian


def synchronization_index(state: torch.Tensor) -> float:
    """
    Cheap order-parameter-like synchronization measure for a Van der Pol
    network: 1 - (std of final x across nodes) / (mean |x| across nodes),
    clipped to [0, 1]. 1 = fully synchronized x, 0 = fully desynchronized.
    """
    x = state[0, :]
    spread = x.std().item()
    scale = x.abs().mean().item() + 1e-8
    return float(max(0.0, min(1.0, 1.0 - spread / scale)))


def main():
    ap = argparse.ArgumentParser(description="Van der Pol network coupling sweep.")
    ap.add_argument("--dti-path", type=str, default="data/DTI_A.mat", help="Path to structural connectivity .mat file.")
    ap.add_argument("--mu", type=float, default=1.5, help="Van der Pol nonlinearity/damping parameter (reference default: 1.5).")
    ap.add_argument("--coupling-min", type=float, default=0.0, help="Minimum coupling strength.")
    ap.add_argument("--coupling-max", type=float, default=1.0, help="Maximum coupling strength.")
    ap.add_argument("--coupling-steps", type=int, default=21, help="Number of coupling values to sweep.")
    ap.add_argument("--dt", type=float, default=0.01, help="Integration time step.")
    ap.add_argument("--steps", type=int, default=20000, help="Number of RK4 integration steps per coupling value.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for initial conditions.")
    ap.add_argument("--outdir", type=str, default="data/", help="Output directory for results and logs.")
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device to run on.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.outdir)
    deriv_dir = out_dir / "derivatives"
    deriv_dir.mkdir(parents=True, exist_ok=True)
    log_path = deriv_dir / "vdp_sweep_log.txt"

    def log(msg: str) -> None:
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"Loading structural connectivity from {args.dti_path} ...")
    L, n = load_dti_laplacian(args.dti_path, device=device)
    log(f"Loaded Laplacian: {n} nodes, device={device}")

    couplings = np.linspace(args.coupling_min, args.coupling_max, args.coupling_steps)
    torch.manual_seed(args.seed)
    init_state = (torch.rand(2, n, device=device) - 0.5) * 2.0  # x, y in [-1, 1]

    all_coupling = []
    all_sync = []
    all_final_state = []
    start_time = time.time()

    for i, coupling in enumerate(couplings):
        coupling = float(coupling)
        net = VanDerPolNetwork(L=L, mu=args.mu, coupling=coupling, device=device)
        traj = net.integrate(init_state.clone(), dt=args.dt, steps=args.steps)
        final_state = traj[-1]
        sync = synchronization_index(final_state)

        all_coupling.append(coupling)
        all_sync.append(sync)
        all_final_state.append(final_state.cpu().numpy())

        elapsed = time.time() - start_time
        log(f"[{i + 1}/{len(couplings)}] coupling={coupling:.4f}  "
            f"sync_index={sync:.4f}  elapsed={elapsed:.1f}s")

    out_path = deriv_dir / "vdp_sweep_results.npz"
    np.savez_compressed(
        out_path,
        coupling=np.array(all_coupling),
        sync_index=np.array(all_sync),
        final_state=np.array(all_final_state),
        mu=args.mu,
        dt=args.dt,
        steps=args.steps,
    )
    log(f"Saved sweep results -> {out_path}")
    log("Van der Pol coupling sweep complete!")


if __name__ == "__main__":
    main()
