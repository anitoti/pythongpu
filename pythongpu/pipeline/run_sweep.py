#!/usr/bin/env python3
"""
Thin master CLI. Dispatches to the individual sweep scripts in this
directory by --system, forwarding all remaining args unchanged to that
script's own argparse. Additive convenience layer — each sweep script
below remains a fully independent, directly runnable entry point.

Usage:
    python3 pipeline/run_sweep.py --system lorenz -- --grid-n 64 --coupling 0.5
    python3 pipeline/run_sweep.py --system rossler -- --outdir data/rossler/
"""

import argparse
import runpy
import sys
from pathlib import Path

SYSTEMS = {
    "lorenz": "lorenz_sweep.py",
    "lorenz_z": "lorenz_sweep_z.py",
    "lorenz_xcoupled": "lorenz_sweep_xcoupled.py",
    "rossler": "rossler_sweep.py",
    "kuramoto": "kuramoto_sweep.py",
}


def main():
    ap = argparse.ArgumentParser(
        description="Master dispatcher for the sweep pipeline scripts.",
        usage="run_sweep.py --system {%s} [-- SCRIPT_ARGS...]" % ",".join(SYSTEMS),
    )
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS), help="Which sweep to run.")
    args, remaining = ap.parse_known_args()

    if remaining and remaining[0] == "--":
        remaining = remaining[1:]

    target = Path(__file__).resolve().parent / SYSTEMS[args.system]
    sys.argv = [str(target)] + remaining
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
