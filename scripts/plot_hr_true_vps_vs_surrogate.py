#!/usr/bin/env python3
"""
Hindmarsh-Rose analogue of plot_true_vps_vs_surrogate.py: compares the true
(lag-based) VPS against the streaming surrogate on the DTI-coupled HR
network, resolution-matched at grid_n=48 (smaller than the main 96-grid
literature-window ladder, since HR's true VPS costs ~2x Lorenz's per chunk
-- steps_record is 2x at the same dt -- see run_hr_followup.sh).

Reads:
  data/derivatives/hr_matched48/hr_basins_n73_n81_K{K}.npz            (surrogate)
  data/derivatives/hr_true_vps_matched48_c{K}/hr_basins_n73_n81_K{K}.npz  (true)
for K in 0.45, 0.50, 0.55, 0.60 -- produced by scripts/run_hr_followup.sh.

Run:
    python3 scripts/plot_hr_true_vps_vs_surrogate.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DERIV = REPO_ROOT / "data" / "derivatives"

COUPLINGS = ["0.4500", "0.5000", "0.5500", "0.6000"]

SURROGATE_COLOR = "#2A1C12"
TRUE_COLOR = "#b1126b"


def surrogate_path(k: str) -> Path:
    return DERIV / "hr_matched48" / f"hr_basins_n73_n81_K{k}.npz"


def true_vps_path(k: str) -> Path:
    tag = {"0.4500": "0.45", "0.5000": "0.50", "0.5500": "0.55", "0.6000": "0.60"}[k]
    return DERIV / f"hr_true_vps_matched48_c{tag.replace('.', '_')}" / f"hr_basins_n73_n81_K{k}.npz"


def load_df(path: Path) -> float:
    d = np.load(path, allow_pickle=True)
    return float(np.asarray(d["fractal_dim"]).ravel()[0])


def main() -> int:
    missing = [p for k in COUPLINGS for p in (surrogate_path(k), true_vps_path(k)) if not p.exists()]
    if missing:
        print("HR matched comparison data not ready yet -- missing:")
        for m in missing:
            print(f"  {m}")
        print("\nStill running? Check: ps aux | grep hr_fine, or tail /tmp/hr_followup.log")
        return 1

    ks = np.array([float(k) for k in COUPLINGS])
    surrogate_df = np.array([load_df(surrogate_path(k)) for k in COUPLINGS])
    true_df = np.array([load_df(true_vps_path(k)) for k in COUPLINGS])

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))

    a.plot(ks, surrogate_df, "o-", color=SURROGATE_COLOR, ms=8)
    a.axhline(surrogate_df.mean(), color=SURROGATE_COLOR, ls="--", lw=1, alpha=0.5)
    a.set_xlabel("coupling K")
    a.set_ylabel(r"surrogate box-counting $D_f$")
    a.set_title("Streaming surrogate (Definition C)\nHindmarsh-Rose, grid_n=48")
    a.grid(alpha=0.3)

    b.plot(ks, true_df, "o-", color=TRUE_COLOR, ms=8)
    b.axhline(true_df.mean(), color=TRUE_COLOR, ls="--", lw=1, alpha=0.5)
    b.set_xlabel("coupling K")
    b.set_ylabel(r"true-VPS box-counting $D_f$")
    b.set_title("Paper's true VPS (Definition A)\nHindmarsh-Rose, grid_n=48")
    b.grid(alpha=0.3)

    max_abs_diff = float(np.max(np.abs(surrogate_df - true_df)))
    agree = max_abs_diff < 0.1
    verdict = "AGREE" if agree else "DISAGREE"
    fig.suptitle(f"Hindmarsh-Rose: streaming surrogate vs. true VPS -- {verdict} "
                f"(max |dD_f| = {max_abs_diff:.3f})",
                fontsize=12, y=0.99)
    fig.text(0.5, 0.925, "resolution-matched at grid_n=48 (smaller than the main literature-window "
             "ladder -- see run_hr_followup.sh)",
             ha="center", fontsize=8.5, style="italic", color="#666666")
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out = DERIV / "hr_true_vps_vs_surrogate_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    print()
    print(f"{'K':>6} {'surrogate D_f':>13} {'true D_f':>10} {'|diff|':>8}")
    for k, sd, td in zip(COUPLINGS, surrogate_df, true_df):
        print(f"{k:>6} {sd:>13.3f} {td:>10.3f} {abs(sd-td):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
