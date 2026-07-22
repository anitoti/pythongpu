#!/usr/bin/env python3
"""
Plot wall-clock vs grid resolution: serial (one node does the whole grid_n^2
batch) vs distributed (max over the N_CHUNKS array tasks that split the same
batch, since they run concurrently -- that max is what actually gates when
the result is ready). This is the "distributed computing" pitch made
concrete: reads whatever data/derivatives/scaling_benchmark/*.json records
exist so far, safe to run while the sweep is still filling in.

Usage:  python3 scripts/plot_scaling_benchmark.py [--glob PATTERN] [--out PNG]
"""
from __future__ import annotations
import argparse, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(pattern):
    serial, chunk = {}, {}   # grid_n -> wall_clock (serial) / list of wall_clock (chunk)
    n_ics = {}
    for path in glob.glob(pattern):
        rec = json.loads(open(path).read())
        g = rec["grid_n"]
        n_ics[g] = rec["n_ics_full"]
        if rec["mode"] == "serial":
            serial[g] = rec["wall_clock_seconds"]
        else:
            chunk.setdefault(g, []).append(rec["wall_clock_seconds"])
    return serial, chunk, n_ics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="data/derivatives/scaling_benchmark/scaling_*.json")
    ap.add_argument("--out", default="data/derivatives/scaling_benchmark.png")
    args = ap.parse_args(argv)

    serial, chunk, n_ics = load(args.glob)
    grid_sizes = sorted(set(serial) | set(chunk))
    if not grid_sizes:
        print(f"no records matched {args.glob!r} yet")
        return 0

    rows = []
    for g in grid_sizes:
        s = serial.get(g)
        c = chunk.get(g)
        d = max(c) if c else None          # distributed wall-clock = slowest task
        n_chunks = len(c) if c else None
        speedup = (s / d) if (s is not None and d is not None and d > 0) else None
        rows.append((g, n_ics.get(g), s, d, n_chunks, speedup))
        tag = f"n_ics={n_ics.get(g)}"
        print(f"grid_n={g:>4}  {tag:>14}  serial={s}  distributed(max of {n_chunks})={d}  "
             f"speedup={speedup}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    xs = [r[0] for r in rows]
    s_ys = [r[2] for r in rows]
    d_ys = [r[3] for r in rows]
    if any(v is not None for v in s_ys):
        ax1.plot(xs, s_ys, "o-", label="serial (1 node, full batch)", color="#d62728")
    if any(v is not None for v in d_ys):
        ax1.plot(xs, d_ys, "s-", label="distributed (max over array)", color="#1f77b4")
    ax1.set_xlabel("grid resolution (grid_n, so grid_n^2 ICs)")
    ax1.set_ylabel("wall-clock (s)")
    ax1.set_title("Basin-mapping integration cost: serial vs distributed")
    ax1.legend()
    ax1.grid(alpha=0.3)

    sp_ys = [r[5] for r in rows]
    if any(v is not None for v in sp_ys):
        ax2.plot(xs, sp_ys, "^-", color="#2ca02c")
        ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("grid resolution (grid_n)")
    ax2.set_ylabel("speedup (serial / distributed)")
    ax2.set_title("Distributed speedup vs resolution")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
