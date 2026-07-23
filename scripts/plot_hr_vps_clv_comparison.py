#!/usr/bin/env python3
"""
Hindmarsh-Rose analogue of plot_vps_clv_comparison.py: compares the
streaming-surrogate box-counting D_f against the CLV/Kaplan-Yorke
transversality method, same DTI connectome, same 9-point coupling ladder
(K=0.45..0.65 step 0.025) -- full overlap between the two methods here,
unlike the Lorenz version's 4-point subset, since both were run across the
whole literature-window ladder tonight (see run_hr_followup.sh).

NAMING, same caveat as the Lorenz version: the left panel is the streaming
surrogate (Definition C), NOT the paper's true VPS (Definition A). Unlike
Lorenz, HR's true VPS HAS now been checked against the surrogate too --
see plot_hr_true_vps_vs_surrogate.py -- at a smaller resolution-matched
grid (48 vs this ladder's 96), so treat that as the separate validation and
this figure as the surrogate-vs-CLV cross-method comparison.

Left panel:  surrogate-VPS box-counting D_f, data/derivatives/hr_basins_n73_n81_K{K}.npz
Right panel: CLV Kaplan-Yorke dimension + riddling burst-fraction,
             output/hr_clv_results/clv_c{K}/clv_topology_summary.json

Run:
    python3 scripts/plot_hr_vps_clv_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DERIV = REPO_ROOT / "data" / "derivatives"
CLV_OUT = REPO_ROOT / "output" / "hr_clv_results"

COUPLINGS = ["0.4500", "0.4750", "0.5000", "0.5250", "0.5500", "0.5750", "0.6000", "0.6250", "0.6500"]
CLV_TAGS  = ["0.45", "0.475", "0.50", "0.525", "0.55", "0.575", "0.60", "0.625", "0.65"]

DKY_COLOR = "#b1126b"
BURST_COLOR = "#e91e63"
SURROGATE_COLOR = "#2A1C12"


def load_surrogate_df(k: str) -> float:
    d = np.load(DERIV / f"hr_basins_n73_n81_K{k}.npz", allow_pickle=True)
    return float(np.asarray(d["fractal_dim"]).ravel()[0])


def load_clv(tag: str) -> dict:
    return json.loads((CLV_OUT / f"clv_c{tag.replace('.', '_')}" / "clv_topology_summary.json").read_text())


def main() -> int:
    missing = ([DERIV / f"hr_basins_n73_n81_K{k}.npz" for k in COUPLINGS if not (DERIV / f"hr_basins_n73_n81_K{k}.npz").exists()]
              + [CLV_OUT / f"clv_c{t.replace('.', '_')}" / "clv_topology_summary.json" for t in CLV_TAGS
                 if not (CLV_OUT / f"clv_c{t.replace('.', '_')}" / "clv_topology_summary.json").exists()])
    if missing:
        print("HR VPS-vs-CLV comparison data not ready yet -- missing:")
        for m in missing:
            print(f"  {m}")
        return 1

    ks = np.array([float(k) for k in COUPLINGS])
    surrogate_df = np.array([load_surrogate_df(k) for k in COUPLINGS])
    clv = [load_clv(t) for t in CLV_TAGS]
    ceiling = np.array([c["kaplan_yorke_is_ceiling"] for c in clv])
    d_ky = np.array([c["kaplan_yorke_dimension"] for c in clv])
    burst = np.array([c["riddling"]["burst_fraction"] for c in clv])
    verdicts = [c["riddling"]["verdict"] for c in clv]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))

    a.plot(ks, surrogate_df, "o-", color=SURROGATE_COLOR, ms=8)
    a.axhline(surrogate_df.mean(), color=SURROGATE_COLOR, ls="--", lw=1, alpha=0.5)
    a.set_xlabel("coupling K")
    a.set_ylabel(r"surrogate box-counting $D_f$")
    a.set_title("HR VPS surrogate (streaming approx.)")
    a.grid(alpha=0.3)

    b2 = b.twinx()
    resolved = ~ceiling
    if ceiling.any():
        b.plot(ks[ceiling], d_ky[ceiling], "^", color=DKY_COLOR, ms=11,
              mfc="none", mew=2, label="ceiling (D_KY >= 83, unresolved)")
    if resolved.any():
        b.plot(ks[resolved], d_ky[resolved], "o", color=DKY_COLOR, ms=9,
              label="resolved D_KY")
    b.plot(ks, d_ky, "-", color=DKY_COLOR, lw=1.5, alpha=0.6)
    b2.plot(ks, burst, "s-", color=BURST_COLOR, ms=7, label="riddling burst fraction")
    b2.set_ylim(0, 1)
    b.set_xlabel("coupling K")
    b.set_ylabel("Kaplan-Yorke dimension $D_{KY}$", color=DKY_COLOR)
    b2.set_ylabel("burst fraction", color=BURST_COLOR)
    b.set_title("HR CLV method")
    b.legend(loc="upper left", fontsize=8)
    b.grid(alpha=0.3)

    uniform = len(set(verdicts)) == 1
    headline = f"both say {verdicts[0]}" if uniform else f"verdicts differ: {sorted(set(verdicts))}"
    fig.suptitle(f"Hindmarsh-Rose: VPS surrogate vs. CLV -- {headline}",
                fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = DERIV / "hr_vps_surrogate_vs_clv_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    print()
    print(f"{'K':>6} {'surrogate D_f':>13} {'CLV D_KY':>10} {'CLV burst':>10} {'CLV verdict':>12}")
    for k, fd, cl in zip(COUPLINGS, surrogate_df, clv):
        ky = "ceil@83" if cl["kaplan_yorke_is_ceiling"] else f"{cl['kaplan_yorke_dimension']:.1f}"
        print(f"{k:>6} {fd:>13.3f} {ky:>10} {cl['riddling']['burst_fraction']:>10.2f} "
             f"{cl['riddling']['verdict']:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
