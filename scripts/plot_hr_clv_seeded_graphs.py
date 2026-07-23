#!/usr/bin/env python3
"""
Visualizes the seeded null-model CLV sweep (scripts/sweep_hr_clv_seeded_graphs.py):
does riddling depend on graph TYPE (BA/GNM/WS), on the specific SEED within
a type, or on coupling K -- three genuinely different questions the seed
tracking makes answerable.

Run:
    python3 scripts/plot_hr_clv_seeded_graphs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "output" / "hr_clv_seeded_graphs" / "manifest.json"
OUT = REPO_ROOT / "data" / "derivatives" / "hr_clv_seeded_graphs_comparison.png"

COLORS = {"ba": "#b1126b", "gnm": "#2A1C12", "ws": "#1f77b4"}
MARKERS = {"ba": "o", "gnm": "s", "ws": "^"}


def main() -> int:
    if not MANIFEST.exists():
        print(f"missing {MANIFEST} -- run scripts/sweep_hr_clv_seeded_graphs.py first")
        return 1
    records = json.loads(MANIFEST.read_text())

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))

    seed_spread = {}
    for kind in ("ba", "gnm", "ws"):
        seeds = sorted(set(r["seed"] for r in records if r["graph_type"] == kind))
        all_dky = []
        for seed in seeds:
            rows = sorted((r for r in records if r["graph_type"] == kind and r["seed"] == seed),
                         key=lambda r: r["coupling"])
            ks = [r["coupling"] for r in rows]
            dky = [r["kaplan_yorke_dimension"] for r in rows]
            all_dky.extend(dky)
            a.plot(ks, dky, "-", marker=MARKERS[kind], color=COLORS[kind], alpha=0.55,
                  ms=6, label=f"{kind.upper()} seed={seed}" if seed == seeds[0] else None)
        seed_spread[kind] = (min(all_dky), max(all_dky), max(all_dky) - min(all_dky))

    a.set_xlabel("coupling K")
    a.set_ylabel(r"Kaplan-Yorke dimension $D_{KY}$")
    a.set_title("D_KY vs. coupling, all seeds shown\n(each line = one seed)")
    a.legend(loc="upper right", fontsize=8, ncol=1)
    a.grid(alpha=0.3)

    kinds = ["ba", "gnm", "ws"]
    spreads = [seed_spread[k][2] for k in kinds]
    bars = b.bar([k.upper() for k in kinds], spreads, color=[COLORS[k] for k in kinds], alpha=0.85)
    for bar, k in zip(bars, kinds):
        lo, hi, _ = seed_spread[k]
        b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
              f"[{lo:.2f}, {hi:.2f}]", ha="center", fontsize=9)
    b.set_ylabel(r"$D_{KY}$ range across ALL seeds & couplings")
    b.set_title("How much does the specific random\nrealization (seed) matter, by graph type?")
    b.grid(alpha=0.3, axis="y")

    verdicts = set(r["verdict"] for r in records)
    headline = f"all {len(records)} runs say {verdicts.pop()}" if len(verdicts) == 1 else "verdicts differ"
    fig.suptitle(f"Seeded null-model graphs (BA/GNM/WS) x 3 seeds x 9 couplings, HR CLV -- {headline}",
                fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")

    print()
    for kind in kinds:
        lo, hi, spread = seed_spread[kind]
        print(f"{kind.upper():>4}: D_KY range [{lo:.2f}, {hi:.2f}], spread={spread:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
