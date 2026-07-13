#!/usr/bin/env python3
"""
Explicit adjacency-list report for the DTI_A structural connectome.

Loads ``data/DTI_A.mat`` (an 83 x 83 symmetric binary Desikan-Killiany
connectome), enumerates every undirected edge (i, j), i < j, where A_ij > 0,
and maps each endpoint to its Desikan-Killiany region label under the
convention documented in :mod:`pythongpu.networks.desikan_killiany`.

Usage
-----
    python3 scripts/run_dti_adjacency.py
    python3 scripts/run_dti_adjacency.py --mat data/DTI_A.mat --var A
    python3 scripts/run_dti_adjacency.py --labels my_lut.txt --out edges.tsv

The numeric node indices are authoritative; the region labels reflect an
assumed FreeSurfer aparc+aseg ordering (see the module header) and can be
overridden with ``--labels``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.io import loadmat

from pythongpu.networks.desikan_killiany import labels_for


def load_adjacency(mat_path: str, var: str) -> np.ndarray:
    """Load the raw binary adjacency (symmetrised, diagonal zeroed)."""
    A = loadmat(mat_path)[var]
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square adjacency, got shape {A.shape}")
    A = (0.5 * (A.astype(np.float64) + A.astype(np.float64).T) > 0).astype(np.uint8)
    np.fill_diagonal(A, 0)
    return A


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mat", default="data/DTI_A.mat", help="path to DTI_A.mat")
    ap.add_argument("--var", default="A", help="matrix variable name inside the .mat")
    ap.add_argument("--labels", default=None,
                    help="optional index->label lookup (one label per line)")
    ap.add_argument("--out", default=None,
                    help="optional TSV path to also write the full edge list")
    args = ap.parse_args(argv)

    A = load_adjacency(args.mat, args.var)
    n = A.shape[0]
    labels = labels_for(n, args.labels)

    iu, ju = np.triu_indices(n, k=1)
    edge_mask = A[iu, ju] > 0
    ei, ej = iu[edge_mask], ju[edge_mask]
    degree = A.sum(axis=1)

    assumed = args.labels is None and n == 83
    print(f"[dti]      {args.mat}  N={n}  undirected edges={ei.size}  "
          f"deg min/med/max = {degree.min()}/{int(np.median(degree))}/{degree.max()}")
    if assumed:
        print("[labels]   ASSUMED FreeSurfer aparc+aseg ordering "
              "(Lausanne-2008 scale-33); numeric indices are authoritative. "
              "Override with --labels if the true LUT is known.")
    print(f"\n{'edge':>6}  {'i':>3} {'j':>3}   {'label(i)':<28} {'label(j)':<28}")
    print("─" * 74)
    lines = []
    for k, (i, j) in enumerate(zip(ei.tolist(), ej.tolist())):
        li, lj = labels[i], labels[j]
        print(f"{k+1:>6}  {i:>3} {j:>3}   {li:<28} {lj:<28}")
        lines.append(f"{i}\t{j}\t{li}\t{lj}")

    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        header = "i\tj\tlabel_i\tlabel_j\n"
        out.write_text(header + "\n".join(lines) + "\n")
        print(f"\n[saved]    {ei.size} edges -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
