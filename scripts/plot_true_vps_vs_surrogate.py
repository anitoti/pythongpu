#!/usr/bin/env python3
"""
Compare the paper's ACTUAL VPS (Definition A: FFT cross-correlation + lag
alignment, run_sweep_true_vps) against the streaming surrogate (Definition C,
run_sweep_streaming) that every other D_f figure in this project has used --
on the same real, integrated DTI-Lorenz network, not just the static
Example_A_3.mat test matrix (talk/figs/alignment_bias.png).

This is the comparison that plot_vps_clv_comparison.py's docstring said
"doesn't exist yet." Same four couplings, same node pair, same grid
resolution, same k_clusters -- only --vps-method differs between the two
sets of files, so any difference in D_f is attributable to the VPS
definition, not a confound from resolution/coupling/clustering.

Left panel:  streaming-surrogate box-counting D_f (existing files,
             data/derivatives/lorenz_basins_n73_n81_K{K}.npz).
Right panel: true-VPS box-counting D_f (new files,
             data/derivatives/true_vps_c{K}/lorenz_basins_n73_n81_K{K}.npz),
             produced by submit_true_vps_production.sh / run_sweep_true_vps.

If the true-VPS files aren't there yet (job still running on ACRES), this
prints which ones are missing and exits without writing a partial/misleading
figure -- pull them with scripts/pull_acres_outputs.sh once the array job
finishes.

Run:
    python3 scripts/plot_true_vps_vs_surrogate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DERIV = REPO_ROOT / "data" / "derivatives"

COUPLINGS = ["0.0500", "0.1000", "0.1500", "0.2000"]

SURROGATE_COLOR = "#2A1C12"
TRUE_COLOR = "#b1126b"


def surrogate_path(k: str) -> Path:
    return DERIV / f"lorenz_basins_n73_n81_K{k}.npz"


def true_vps_path(k: str) -> Path:
    k_tag = k.rstrip("0").rstrip(".") if "." in k else k
    # matches submit_true_vps_production.sh's OUTDIR="true_vps_c${K//./_}"
    # where K is the same "0.05"/"0.10"/... string passed on the CLI
    tag = {"0.0500": "0.05", "0.1000": "0.10", "0.1500": "0.15", "0.2000": "0.20"}[k]
    return DERIV / f"true_vps_c{tag.replace('.', '_')}" / f"lorenz_basins_n73_n81_K{k}.npz"


def load_df(path: Path) -> tuple[float, dict | None]:
    d = np.load(path, allow_pickle=True)
    df = float(np.asarray(d["fractal_dim"]).ravel()[0])
    try:
        cfg = d["config"].item()
    except Exception as e:
        # config is a pickled object array; a numpy-version mismatch between
        # the environment that wrote it (ACRES's venv) and this one raises
        # ModuleNotFoundError('numpy._core') rather than failing to load.
        # D_f itself is a plain float array and loads fine regardless --
        # only the confound-check below loses its input.
        print(f"  (warning: couldn't unpickle config in {path.name}: {e!r} -- "
              f"skipping the config-match sanity check for this file)")
        cfg = None
    return df, cfg


def main() -> int:
    missing = [true_vps_path(k) for k in COUPLINGS if not true_vps_path(k).exists()]
    if missing:
        print("True-VPS output not ready yet -- missing:")
        for m in missing:
            print(f"  {m}")
        print("\nStill running on ACRES (job 4556141)? Pull with:")
        print("  scripts/pull_acres_outputs.sh derivatives")
        return 1

    ks = np.array([float(k) for k in COUPLINGS])
    surrogate_df, surrogate_cfg = zip(*(load_df(surrogate_path(k)) for k in COUPLINGS))
    true_df, true_cfg = zip(*(load_df(true_vps_path(k)) for k in COUPLINGS))
    surrogate_df, true_df = np.array(surrogate_df), np.array(true_df)

    # Sanity check: everything except vps_method should match between the
    # two sets of files, or this comparison is confounded and not worth
    # trusting -- fail loudly rather than plot a misleading figure. Only
    # runs when both configs actually unpickled (see load_df) -- if not,
    # this falls back to the submit-script-level guarantee (same node
    # pair/grid_n/tmax were hardcoded into both submission scripts) instead
    # of a runtime check, and says so.
    check_keys = ("slice_node_x", "slice_node_y", "n_osc", "grid_n", "grid_lo",
                 "grid_hi", "sigma", "rho", "beta", "dt", "tmax", "k_clusters")
    any_skipped = False
    for k, sc, tc in zip(COUPLINGS, surrogate_cfg, true_cfg):
        if sc is None or tc is None:
            any_skipped = True
            continue
        mismatches = {key: (sc.get(key), tc.get(key)) for key in check_keys
                     if sc.get(key) != tc.get(key)}
        if mismatches:
            print(f"REFUSING to plot: K={k} configs differ outside vps_method: {mismatches}")
            return 1
    if any_skipped:
        print("NOTE: config-match check skipped for one or both file sets (unpickle failure "
              "above) -- relying on submit_true_vps_production.sh / the surrogate sweep's own "
              "script both hardcoding node_x=73, node_y=81, grid_n=96, tmax=500 rather than a "
              "runtime-verified match.")

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))

    a.plot(ks, surrogate_df, "o-", color=SURROGATE_COLOR, ms=8)
    a.axhline(surrogate_df.mean(), color=SURROGATE_COLOR, ls="--", lw=1, alpha=0.5)
    a.set_ylim(1.0, 2.05)
    a.set_xlabel("coupling K")
    a.set_ylabel(r"surrogate box-counting $D_f$")
    a.set_title("Streaming surrogate (Definition C)\nno lag search")
    a.grid(alpha=0.3)

    b.plot(ks, true_df, "o-", color=TRUE_COLOR, ms=8)
    b.axhline(true_df.mean(), color=TRUE_COLOR, ls="--", lw=1, alpha=0.5)
    b.set_ylim(1.0, 2.05)
    b.set_xlabel("coupling K")
    b.set_ylabel(r"true-VPS box-counting $D_f$")
    b.set_title("Paper's true VPS (Definition A)\nFFT cross-correlation + lag alignment")
    b.grid(alpha=0.3)

    max_abs_diff = float(np.max(np.abs(surrogate_df - true_df)))
    agree = max_abs_diff < 0.1
    verdict = "AGREE" if agree else "DISAGREE"
    fig.suptitle(f"First production comparison: streaming surrogate vs. the paper's true VPS "
                f"-- {verdict} (max |dD_f| = {max_abs_diff:.3f})",
                fontsize=12, y=0.99)
    fig.text(0.5, 0.925,
             "same node pair, grid, coupling ladder, and k_clusters -- only the VPS "
             "definition differs",
             ha="center", fontsize=8.5, style="italic", color="#666666")
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out = DERIV / "true_vps_vs_surrogate_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    print()
    print(f"{'K':>6} {'surrogate D_f':>13} {'true D_f':>10} {'|diff|':>8}")
    for k, sd, td in zip(COUPLINGS, surrogate_df, true_df):
        print(f"{k:>6} {sd:>13.3f} {td:>10.3f} {abs(sd-td):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
