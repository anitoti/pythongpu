#!/usr/bin/env python3
"""
Compare the k-means/box-counting basin-mapping method against the CLV/
Kaplan-Yorke transversality method, on the same DTI connectome, at the same
coupling values (K=0.05, 0.10, 0.15, 0.20) -- the only four where both
methods have real data.

NAMING, READ THIS FIRST: the left panel is NOT the paper's true VPS. It is
the streaming surrogate (math_textbook.md Sec I.6a, "Definition C") forced
by the memory wall -- tau = mean|dX|/std|dX| instead of a time lag, L =
instantaneous mean instead of a lag-aligned norm. The true lag-based VPS
(Definition A) was transcribed and tested against the paper's own test
matrices (talk slide 7): L disagreed on 36/36 pairs, mean ratio 1.96, traced
to a lag=1 off-by-one. The streaming surrogate used here has never been
checked against the true VPS at all, at any scale -- that comparison does
not exist yet. Do not call this panel "the VPS method" out loud; call it
"the box-counting/k-means surrogate" or similar.

Left panel:  surrogate-VPS box-counting fractal dimension D_f (from the fine
             onset sweep, data/derivatives/lorenz_basins_n73_n81_K{K}.npz).
Right panel: CLV Kaplan-Yorke dimension (ceiling cases marked distinctly,
             per plot_publication_figs.py's convention) and riddling
             burst-fraction, from data/derivatives/clv_c{K}/clv_topology_summary.json.

Both methods independently return a RIDDLED verdict at all four couplings
(the surrogate route via D_f staying flat near 2; the CLV route via
detect_riddling_kmeans on the transversality angles). What's genuinely
different between them is visible here and nowhere else: the CLV method's
OWN resolvability has a discontinuity at K=0.20 (Kaplan-Yorke goes from an
unresolved ceiling to a real number) that the surrogate's D_f does not show
at all -- D_f stays flat across the same four points. That's not a
disagreement between the methods; it's a piece of information only one of
them can see -- and neither of them is validated against the paper's actual
VPS definition.

Run:
    python3 scripts/plot_vps_clv_comparison.py
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

COUPLINGS = ["0.0500", "0.1000", "0.1500", "0.2000"]
CLV_TAGS = ["0_05", "0_10", "0_15", "0_20"]

DKY_COLOR = "#b1126b"
BURST_COLOR = "#e91e63"
SURROGATE_COLOR = "#2A1C12"


def load_surrogate_df(k: str) -> float:
    """Box-counting D_f from the streaming VPS surrogate (NOT the true VPS
    -- see module docstring)."""
    d = np.load(DERIV / f"lorenz_basins_n73_n81_K{k}.npz", allow_pickle=True)
    return float(np.asarray(d["fractal_dim"]).ravel()[0])


def load_clv(tag: str) -> dict:
    return json.loads((DERIV / f"clv_c{tag}" / "clv_topology_summary.json").read_text())


def main() -> int:
    ks = np.array([float(k) for k in COUPLINGS])
    surrogate_df = np.array([load_surrogate_df(k) for k in COUPLINGS])
    clv = [load_clv(t) for t in CLV_TAGS]
    ceiling = np.array([c["kaplan_yorke_is_ceiling"] for c in clv])
    d_ky = np.array([c["kaplan_yorke_dimension"] for c in clv])
    burst = np.array([c["riddling"]["burst_fraction"] for c in clv])
    verdicts = [c["riddling"]["verdict"] for c in clv]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── left: streaming-surrogate box-counting D_f (NOT the true VPS) ───
    a.plot(ks, surrogate_df, "o-", color=SURROGATE_COLOR, ms=8)
    a.axhline(surrogate_df.mean(), color=SURROGATE_COLOR, ls="--", lw=1, alpha=0.5)
    a.set_ylim(1.5, 2.05)
    a.set_xlabel("coupling K")
    a.set_ylabel(r"surrogate/$k$-means box-counting $D_f$")
    a.set_title("VPS surrogate (streaming approx.): flat,\nnear-space-filling across all four couplings")
    a.grid(alpha=0.3)

    # ── right: CLV Kaplan-Yorke + burst fraction ────────────────────────
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
    b.set_ylim(0, 90)
    b2.set_ylim(0, 1)
    b.set_xlabel("coupling K")
    b.set_ylabel("Kaplan-Yorke dimension $D_{KY}$", color=DKY_COLOR)
    b2.set_ylabel("burst fraction", color=BURST_COLOR)
    b.set_title("CLV method: same RIDDLED verdict\nthroughout, but D_KY switches at K=0.20")
    b.legend(loc="upper left", fontsize=8)
    b.grid(alpha=0.3)

    assert len(set(verdicts)) == 1, "verdicts differ across couplings -- update the suptitle below"
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.suptitle(f"VPS surrogate vs. CLV: two riddling detectors, same DTI network, "
                f"both say {verdicts[0]}",
                fontsize=12, y=0.99)
    fig.text(0.5, 0.925, "(neither is checked against the paper's true VPS -- see module docstring)",
            ha="center", fontsize=8.5, style="italic", color="#666666")
    out = DERIV / "vps_surrogate_vs_clv_comparison.png"
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
