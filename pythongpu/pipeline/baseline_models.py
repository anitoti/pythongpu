#!/usr/bin/env python3
"""
Baseline comparison CLI: empirical connectome vs. Erdős–Rényi vs. Barabási–Albert.

Runs the full CLV topological diagnostic (Lyapunov spectrum, Kaplan–Yorke
dimension, k-means riddling verdict) on THREE networks on identical footing:

  1. the empirical connectome (DTI .mat, or a reconstructed adjacency CSV from
     entropic_regression.py / causation_entropy.py),
  2. an Erdős–Rényi G(n, m) null with the SAME node and edge count,
  3. a Barabási–Albert scale-free null with matched mean degree.

All three are binarised and symmetrised first (via
random_graphs.match_baselines_from_adjacency), so the ONLY thing that differs
between them is graph *topology* — this isolates "does the brain's specific
wiring shape the dynamics?" from edge-weight effects. (For the weighted-DTI
dynamics, use clv_cli.py --mat directly.)

For each network it also reports the Laplacian spectral summary that Part IV.7 of
math_textbook.md derives: algebraic connectivity λ₂ (Fiedler value), spectral
radius λ_N, and the synchronizability ratio λ_N/λ₂.

Run:
    python3 -m pythongpu.pipeline.baseline_models \\
        --mat data/DTI-og.mat --steps 700 --m 40 --K 12 \\
        --qr-interval 5 --coupling 0.1 --outdir output/baselines --plot
    # or from a reconstructed functional connectome:
    python3 -m pythongpu.pipeline.baseline_models \\
        --adjacency data/processed/100307/er_adjacency.csv --steps 700 --m 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pythongpu.networks.random_graphs import match_baselines_from_adjacency
from pythongpu.pipeline.clv_topology import run_clv_topology, format_topology_line

_ORDER = ("empirical", "er", "ba")
_PRETTY = {"empirical": "Empirical", "er": "Erdős–Rényi", "ba": "Barabási–Albert"}


def _load_empirical_adjacency(mat: str | None, adjacency: str | None) -> np.ndarray:
    """Load the empirical adjacency matrix from a DTI .mat (variable 'A') or a CSV.

    Handles both our own writers (entropic_regression / causation_entropy use
    ``to_csv(index=False)``, which prepends a header row of integer column names)
    and bare headerless matrices, by keeping whichever parse yields a square array.
    Integer column headers are themselves numeric, so a plain ``astype(float)``
    cannot distinguish the two — the squareness test is what disambiguates.
    """
    if adjacency is not None:
        import pandas as pd
        candidates = []
        for header in (0, None):  # 0 = our writers' format; None = bare matrix
            try:
                vals = pd.read_csv(adjacency, header=header).values.astype(float)
                candidates.append(vals)
            except (ValueError, TypeError):
                continue
        for vals in candidates:
            if vals.ndim == 2 and vals.shape[0] == vals.shape[1]:
                return vals
        if candidates:  # no square parse — surface the shape rather than guess
            raise ValueError(
                f"adjacency CSV {adjacency} is not square (got {candidates[0].shape}); "
                f"expected an N×N matrix.")
        raise ValueError(f"could not parse adjacency CSV: {adjacency}")
    if mat is not None:
        from scipy.io import loadmat
        return loadmat(mat)["A"].astype(float)
    raise ValueError("provide either --mat or --adjacency")


def _laplacian_spectral_summary(L: torch.Tensor) -> dict:
    """Algebraic connectivity λ₂, spectral radius λ_N, and synchronizability λ_N/λ₂.

    L is a symmetric PSD graph Laplacian, so its eigenvalues are real and
    0 = λ₁ ≤ λ₂ ≤ … ≤ λ_N. λ₂ (Fiedler value) governs how well the network can
    synchronise; λ_N/λ₂ is the master-stability-function eigenratio (smaller ⇒
    a wider stable coupling window). See math_textbook.md §IV.7.
    """
    ev = torch.linalg.eigvalsh(L.to(torch.float64)).cpu().numpy()
    ev = np.sort(ev)
    lam2 = float(ev[1]) if len(ev) > 1 else 0.0
    lamN = float(ev[-1])
    ratio = float(lamN / lam2) if lam2 > 1e-12 else float("inf")
    return {"algebraic_connectivity": lam2, "spectral_radius": lamN,
            "synchronizability_ratio": ratio}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--mat", type=str, default="data/DTI-og.mat",
                     help="DTI .mat with variable 'A' (default).")
    src.add_argument("--adjacency", type=str, default=None,
                     help="Empirical adjacency CSV (e.g. an ER/oCSE connectome).")
    ap.add_argument("--outdir", type=str, default="output/baselines")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--m", type=int, default=40, help="CLVs to compute per network.")
    ap.add_argument("--K", type=int, default=12, help="leading CLVs for transversality.")
    ap.add_argument("--qr-interval", type=int, default=5)
    ap.add_argument("--coupling", type=float, default=0.1)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0, help="seed for the ER/BA nulls.")
    ap.add_argument("--plot", action="store_true", help="save a comparison bar chart.")
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    A = _load_empirical_adjacency(args.mat if args.adjacency is None else None, args.adjacency)
    src_desc = args.adjacency or args.mat
    print(f"Loaded empirical adjacency {A.shape} from {src_desc}")

    # Build matched ER + BA nulls (binarised/symmetrised empirical included).
    nets = match_baselines_from_adjacency(A, device=device, seed=args.seed, plot=args.plot)
    N = nets["empirical"]["stats"]["n_nodes"]
    print(f"n_nodes={N}  |  empirical edges={nets['empirical']['stats']['n_edges']}  "
          f"ER edges={nets['er']['stats']['n_edges']}  "
          f"BA edges={nets['ba']['stats']['n_edges']} (m={nets['ba']['stats']['ba_m']})")

    results = {}
    print("\nRunning CLV topology on each network "
          f"(coupling={args.coupling}, m={args.m}, steps={args.steps})...")
    for name in _ORDER:
        L = nets[name]["L"]
        topo = run_clv_topology(
            L, N, coupling=args.coupling, steps=args.steps, m=args.m, K=args.K,
            qr_interval=args.qr_interval, device=device,
            out_prefix=str(outdir / f"clv_angles_{name}_"), label=_PRETTY[name],
            seed=args.seed,
        )
        topo["graph_stats"] = nets[name]["stats"]
        topo["spectral"] = _laplacian_spectral_summary(L)
        results[name] = topo
        print("  " + format_topology_line(topo, args.m))

    # ── comparison table ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'network':<16s}{'λ₂':>8s}{'λ_N':>9s}{'λ_N/λ₂':>9s}{'het':>7s}"
          f"{'λ_max':>9s}{'D_KY':>10s}{'riddling':>13s}")
    print("-" * 78)
    for name in _ORDER:
        r = results[name]
        sp, gs = r["spectral"], r["graph_stats"]
        ky = f">={args.m}" if r["kaplan_yorke_is_ceiling"] else f"{r['kaplan_yorke_dimension']:.2f}"
        print(f"{_PRETTY[name]:<16s}{sp['algebraic_connectivity']:>8.3f}"
              f"{sp['spectral_radius']:>9.2f}{sp['synchronizability_ratio']:>9.1f}"
              f"{gs['degree_heterogeneity']:>7.2f}{r['lambda_max']:>9.4f}"
              f"{ky:>10s}{r['riddling']['verdict']:>13s}")
    print("=" * 78)

    out_json = outdir / "baseline_comparison.json"
    with open(out_json, "w") as fh:
        json.dump({"config": vars(args), "results": results}, fh, indent=2)
    print(f"\nSaved comparison → {out_json}")

    if args.plot:
        _plot_comparison(results, outdir)


def _plot_comparison(results: dict, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [_PRETTY[n] for n in _ORDER]
    lam_max = [results[n]["lambda_max"] for n in _ORDER]
    burst = [results[n]["riddling"]["burst_fraction"] for n in _ORDER]
    ratio = [results[n]["spectral"]["synchronizability_ratio"] for n in _ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, vals, title in zip(
        axes, (lam_max, burst, ratio),
        ("Largest Lyapunov exponent λ_max", "Riddling burst fraction",
         "Synchronizability λ_N/λ₂"),
    ):
        ax.bar(names, vals, color=["#c1272d", "#0000a7", "#008176"])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Empirical connectome vs. ER / BA nulls — CLV & spectral comparison")
    fig.tight_layout()
    path = outdir / "baseline_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved bar chart → {path}")


if __name__ == "__main__":
    main()
