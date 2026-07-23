#!/usr/bin/env python3
"""
Generate and track the three random-graph null models this project's
pipeline actually uses (imported by lorenz_sweep.py: BA, GNM, WS -- a
fourth, GNP, exists in random_graphs.py but isn't wired into the main
pipeline), across an explicit set of seeds, so every graph that gets fed
into a dynamics/VPS/CLV run later can be exactly reproduced and its
formation tracked -- not just "a random graph," but "seed 3's BA graph,"
reproducible on demand.

Matches each null's size to the real DTI connectome by default (83 nodes,
850 edges after binarize+symmetrize -- see
pythongpu.networks.random_graphs.match_baselines_from_adjacency for the
same convention used elsewhere in this project):
  - BA:  m = round(850 / 83) ~= 10 edges attached per new node
  - GNM: exactly 850 edges, n=83
  - WS:  k = round(2*850/83) ~= 20 (nearest-neighbor ring degree,
         matched to DTI's mean degree), p=0.1 (standard small-world regime)

For each (graph_type, seed) pair, saves:
  - the realized adjacency matrix (.npz)
  - topology stats (n_edges, density, mean_degree, degree_heterogeneity,
    mean_clustering) via random_graphs._graph_stats
  - a manifest (data/derivatives/random_graph_seeds/manifest.json) listing
    every (type, seed) -> stats, so "how did this graph form" is answered
    by looking the seed up, not re-deriving it.

Run:
    python3 scripts/seeded_random_graph_sweep.py
    python3 scripts/seeded_random_graph_sweep.py --n-seeds 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import numpy as np

from pythongpu.networks.random_graphs import _graph_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "data" / "derivatives" / "random_graph_seeds"

# Matched to the real DTI connectome (83 nodes, 850 edges after
# binarize+symmetrize -- confirmed via [dti] loader logs elsewhere in
# this project) so null-model comparisons are on equal footing.
N_NODES = 83
N_EDGES_TARGET = 850
BA_M = max(1, round(N_EDGES_TARGET / N_NODES))          # ~10
WS_K = max(2, round(2 * N_EDGES_TARGET / N_NODES))       # ~20, must be even
if WS_K % 2 == 1:
    WS_K += 1
WS_P = 0.1


def build_graph(kind: str, seed: int) -> nx.Graph:
    if kind == "ba":
        return nx.barabasi_albert_graph(N_NODES, BA_M, seed=seed)
    if kind == "gnm":
        return nx.gnm_random_graph(N_NODES, N_EDGES_TARGET, seed=seed)
    if kind == "ws":
        return nx.watts_strogatz_graph(N_NODES, WS_K, WS_P, seed=seed)
    raise ValueError(f"unknown graph kind: {kind!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="Number of seeds per graph type (seeds 0..n-1).")
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[params] BA: n={N_NODES} m={BA_M}  |  GNM: n={N_NODES} m={N_EDGES_TARGET}  |  "
         f"WS: n={N_NODES} k={WS_K} p={WS_P}")

    manifest = {}
    for kind in ("ba", "gnm", "ws"):
        manifest[kind] = []
        kind_dir = outdir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(args.n_seeds):
            g = build_graph(kind, seed)
            adj = nx.to_numpy_array(g)
            stats = _graph_stats(g)

            npz_path = kind_dir / f"seed{seed:03d}.npz"
            np.savez_compressed(npz_path, adjacency=adj, seed=np.array([seed]))

            record = dict(seed=seed, path=str(npz_path.relative_to(REPO_ROOT)), **stats)
            manifest[kind].append(record)
            print(f"[{kind}] seed={seed:3d}  edges={stats['n_edges']:4d}  "
                 f"density={stats['density']:.4f}  mean_deg={stats['mean_degree']:.2f}  "
                 f"het={stats['degree_heterogeneity']:.3f}  clust={stats['mean_clustering']:.3f}")

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
