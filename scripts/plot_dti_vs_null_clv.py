#!/usr/bin/env python3
"""
Compares the real DTI-coupled Lorenz network's CLV riddling signature against
a null-model network (BA scale-free, n=83, m=6 -- matched node count, not
matched connectome structure) at the same 4 couplings and same production CLV
parameters (--steps 1000 --m 83 --K 83). Both datasets pre-exist from earlier
runs (data/derivatives/clv_c{K}/ for DTI, output/clv_null_results/clv_c{K}/
for the null model) -- this script only visualizes them, no new computation.

Answers: is the riddling signature specific to the real human connectome, or
would any similarly-sized random graph show the same thing?

Run:
    python3 scripts/plot_dti_vs_null_clv.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DTI_DIR = REPO_ROOT / "data" / "derivatives"
NULL_DIR = REPO_ROOT / "output" / "clv_null_results"

COUPLINGS = ["0.05", "0.10", "0.15", "0.20"]
TAGS = ["0_05", "0_10", "0_15", "0_20"]

DTI_COLOR = "#b1126b"
NULL_COLOR = "#888888"


def load(base: Path, tag: str) -> dict:
    return json.loads((base / f"clv_c{tag}" / "clv_topology_summary.json").read_text())


def main() -> int:
    ks = np.array([float(k) for k in COUPLINGS])
    dti = [load(DTI_DIR, t) for t in TAGS]
    null = [load(NULL_DIR, t) for t in TAGS]

    dti_burst = np.array([d["riddling"]["burst_fraction"] for d in dti])
    null_burst = np.array([d["riddling"]["burst_fraction"] for d in null])
    dti_resolved = np.array([not d["kaplan_yorke_is_ceiling"] for d in dti])
    null_resolved = np.array([not d["kaplan_yorke_is_ceiling"] for d in null])

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(ks, dti_burst, "o-", color=DTI_COLOR, ms=10, lw=2, label="Real DTI connectome")
    ax.plot(ks, null_burst, "s--", color=NULL_COLOR, ms=9, lw=2, label="Null model (BA scale-free, n=83)")

    if dti_resolved.any():
        ax.plot(ks[dti_resolved], dti_burst[dti_resolved], "*", color="gold",
               ms=22, mec="black", mew=1, zorder=5, label="D_KY resolved (not ceiling)")

    ax.set_xlabel("coupling K")
    ax.set_ylabel("riddling burst fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Both networks say RIDDLED everywhere -- but the real connectome's\n"
                "burst fraction is far noisier, and only IT ever resolves D_KY")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = DTI_DIR / "dti_vs_null_clv_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    print()
    print(f"{'K':>6} {'DTI burst':>10} {'null burst':>11} {'DTI D_KY':>10} {'null D_KY':>10}")
    for k, d, n in zip(COUPLINGS, dti, null):
        dky = "ceil@83" if d["kaplan_yorke_is_ceiling"] else f"{d['kaplan_yorke_dimension']:.1f}"
        nky = "ceil@83" if n["kaplan_yorke_is_ceiling"] else f"{n['kaplan_yorke_dimension']:.1f}"
        print(f"{k:>6} {d['riddling']['burst_fraction']:>10.2f} "
             f"{n['riddling']['burst_fraction']:>11.2f} {dky:>10} {nky:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
