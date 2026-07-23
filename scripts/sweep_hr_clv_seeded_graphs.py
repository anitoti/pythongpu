#!/usr/bin/env python3
"""
Wire the seeded null-model graphs (scripts/seeded_random_graph_sweep.py:
BA, GNM, WS, matched to the real DTI connectome's size) into the actual
HR CLV/Kaplan-Yorke pipeline (pythongpu.pipeline.clv_topology), instead of
leaving the graph generation sitting unused.

Runs the full literature-window coupling ladder (K=0.45..0.65, matching
this project's own HR streaming/CLV production convention) for 3 seeds of
each of the 3 graph types = 9 (graph_type, seed) combinations x 9
couplings = 81 CLV runs, at this project's production CLV parameters
(--steps 1000 --m 83 --K 83, matching scripts/run_clv_sweep.sh). Answers:
does riddling (and how riddled) depend on which null-model FAMILY a graph
comes from, or on which specific random realization (seed) of that family
-- i.e. how much does "how the graph forms" actually matter, not just
"what type of graph is it."

Run:
    python3 scripts/sweep_hr_clv_seeded_graphs.py
    python3 scripts/sweep_hr_clv_seeded_graphs.py --n-seeds 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from pythongpu.pipeline.clv_topology import run_hr_clv_topology
from pythongpu.pipeline.hr_clv_cli import load_seeded_graph_laplacian

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = REPO_ROOT / "data" / "derivatives" / "random_graph_seeds"
OUTDIR = REPO_ROOT / "output" / "hr_clv_seeded_graphs"

COUPLINGS = [0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65]
GRAPH_TYPES = ("ba", "gnm", "ws")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=3, help="seeds per graph type (0..n-1)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--m", type=int, default=83)
    ap.add_argument("--K", type=int, default=83)
    ap.add_argument("--qr-interval", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    OUTDIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    n_total = len(GRAPH_TYPES) * args.n_seeds * len(COUPLINGS)
    n_done = 0
    for kind in GRAPH_TYPES:
        for seed in range(args.n_seeds):
            npz_path = GRAPH_DIR / kind / f"seed{seed:03d}.npz"
            if not npz_path.exists():
                print(f"[skip] {npz_path} not found -- run scripts/seeded_random_graph_sweep.py first")
                continue
            L, N = load_seeded_graph_laplacian(str(npz_path), device=device)

            for K in COUPLINGS:
                tag = f"{kind}_seed{seed:03d}_c{K:.3f}".replace(".", "_")
                run_dir = OUTDIR / kind / f"seed{seed:03d}" / f"c{K:.3f}".replace(".", "_")
                run_dir.mkdir(parents=True, exist_ok=True)

                summary = run_hr_clv_topology(
                    L, N, coupling=K, steps=args.steps, m=args.m, K=args.K,
                    qr_interval=args.qr_interval, device=device,
                    out_prefix=str(run_dir / "clv_angles_"),
                    label=f"{kind} seed{seed:03d}", dt=args.dt,
                )
                (run_dir / "clv_topology_summary.json").write_text(json.dumps(summary, indent=2))

                riddle = summary["riddling"]
                dky = "ceil" if summary["kaplan_yorke_is_ceiling"] else f"{summary['kaplan_yorke_dimension']:.2f}"
                n_done += 1
                print(f"[{n_done:3d}/{n_total}] {kind} seed={seed} K={K:.3f}  "
                     f"D_KY={dky:>6}  burst={riddle['burst_fraction']:.3f}  "
                     f"verdict={riddle['verdict']}")

                manifest.append(dict(
                    graph_type=kind, seed=seed, coupling=K,
                    n_positive_exponents=summary["n_positive_exponents"],
                    kaplan_yorke_is_ceiling=summary["kaplan_yorke_is_ceiling"],
                    kaplan_yorke_dimension=summary["kaplan_yorke_dimension"],
                    burst_fraction=riddle["burst_fraction"],
                    verdict=riddle["verdict"],
                ))

    manifest_path = OUTDIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {manifest_path}  ({len(manifest)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
