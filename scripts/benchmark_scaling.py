#!/usr/bin/env python3
"""
benchmark_scaling.py
---------------------
Turns the "distributed computing solves the paper's basin-mapping bottleneck"
pitch into a number instead of an assertion.

Times the dominant cost of basin mapping -- the transient + recording
integration of a batch of B = grid_n^2 ICs via rk4_step_batched -- under two
regimes:

  serial      : one process integrates the FULL grid_n^2 batch alone.
  chunk       : one process integrates only ITS 1/n_chunks slice of the same
                grid_n^2 batch. Run as an ACRES array job with n_chunks tasks,
                each lands on a different node; wall-clock for the distributed
                run is the max over the array (they run concurrently), not
                the sum -- that max is what actually gates when the full
                basin map is ready.

Each invocation does exactly one timing and writes one small record, so this
composes directly with a SLURM array job (see submit_benchmark_scaling.sh):
one task per (grid_n, chunk_index) pair. No clustering/box-counting here --
only the integration cost, which dominates at any interesting resolution.

Run:
    python3 scripts/benchmark_scaling.py --grid-n 64 --mode serial
    python3 scripts/benchmark_scaling.py --grid-n 64 --mode chunk \
        --n-chunks 8 --chunk-index 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from pythongpu.networks.static_adjacency import load_dti_laplacian
from pythongpu.pipeline.lorenz_sweep import LorenzParams, rk4_step_batched
from pythongpu.pipeline.lorenz_fine_coupling_sweep import build_slice


@torch.no_grad()
def time_integration(x0: torch.Tensor, L_gpu: torch.Tensor, p: LorenzParams,
                     device: torch.device) -> float:
    """Wall-clock seconds for the full transient + recording integration."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    x = x0
    for _ in range(p.steps_transient):
        x = rk4_step_batched(x, L_gpu, p)
    for _ in range(p.steps_record):
        x = rk4_step_batched(x, L_gpu, p)

    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dti-path", default="data/DTI-og.mat")
    ap.add_argument("--coupling", type=float, default=0.5)
    ap.add_argument("--node-x", type=int, default=28)
    ap.add_argument("--node-y", type=int, default=79)
    ap.add_argument("--grid-n", type=int, required=True,
                    help="full grid resolution; the batch this measures is grid_n^2 "
                         "ICs (or 1/n_chunks of that, in chunk mode).")
    ap.add_argument("--grid-lo", type=float, default=-9.0)
    ap.add_argument("--grid-hi", type=float, default=9.0)
    ap.add_argument("--t-transient", type=float, default=100.0)
    ap.add_argument("--tmax", type=float, default=500.0)
    ap.add_argument("--mode", choices=["serial", "chunk"], default="serial")
    ap.add_argument("--n-chunks", type=int, default=1,
                    help="chunk mode only: how many equal pieces the grid_n^2 batch "
                         "is split into (= array-job task count).")
    ap.add_argument("--chunk-index", type=int, default=0,
                    help="chunk mode only: which 1/n_chunks slice this task integrates.")
    ap.add_argument("--outdir", default="data/derivatives/scaling_benchmark")
    args = ap.parse_args(argv)

    if args.mode == "serial" and args.n_chunks != 1:
        raise ValueError("--mode serial requires --n-chunks 1 (it always does the full batch)")
    if not (0 <= args.chunk_index < args.n_chunks):
        raise ValueError(f"--chunk-index must be in [0, {args.n_chunks})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L_gpu, N = load_dti_laplacian(args.dti_path, device)
    p = LorenzParams(coupling=args.coupling, t_transient=args.t_transient,
                     tmax=args.tmax, slice_node_x=args.node_x,
                     slice_node_y=args.node_y, n_osc=N)

    x0_full, _, _ = build_slice(p, args.grid_n, args.grid_lo, args.grid_hi, device)
    B_full = x0_full.shape[0]

    # Split into n_chunks contiguous pieces (last chunk absorbs any remainder).
    bounds = np.linspace(0, B_full, args.n_chunks + 1, dtype=int)
    lo, hi = bounds[args.chunk_index], bounds[args.chunk_index + 1]
    x0 = x0_full[lo:hi]
    n_ics_this_task = x0.shape[0]

    elapsed = time_integration(x0, L_gpu, p, device)

    record = dict(
        mode=args.mode, grid_n=args.grid_n, n_chunks=args.n_chunks,
        chunk_index=args.chunk_index, n_ics_full=int(B_full),
        n_ics_this_task=int(n_ics_this_task), coupling=args.coupling,
        t_transient=args.t_transient, tmax=args.tmax,
        device=str(device), wall_clock_seconds=elapsed,
    )
    print(f"[benchmark_scaling] mode={args.mode} grid_n={args.grid_n} "
         f"n_chunks={args.n_chunks} chunk_index={args.chunk_index} "
         f"n_ics={n_ics_this_task}/{B_full} device={device} "
         f"wall_clock={elapsed:.3f}s")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"scaling_{args.mode}_grid{args.grid_n}_n{args.n_chunks}_c{args.chunk_index}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"[benchmark_scaling] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
