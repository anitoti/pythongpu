#!/usr/bin/env python3
"""
Plot the DTI-coupled Lorenz onset sweep from whatever per-K npz records exist
so far — safe to run repeatedly while the sweep is still filling in.

Reads data/derivatives/lorenz_basins_n{X}_n{Y}_K*.npz and plots, vs coupling K:
    * gamma       — uncertainty exponent on the k-means basin labels
    * gamma_sign  — same, on the clustering-free sign(mean X) lobe label
                    (only present in npz written by the patched pipeline)
    * D_f         — box-counting fractal dimension of the k-means boundary
    * n_lobe_configs — number of distinct 83-bit lobe patterns on the slice

Usage:  python3 scripts/plot_onset_sweep.py [--glob PATTERN] [--out PNG]
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _scalar(d, key):
    if key not in d.files:
        return None
    v = np.asarray(d[key]).ravel()
    return float(v[0]) if v.size else None


def load(pattern):
    rows = []
    for path in glob.glob(pattern):
        d = np.load(path, allow_pickle=True)
        rows.append(dict(
            K=_scalar(d, "coupling") if "coupling" in d.files
              else float(np.asarray(d["config"]).item()["coupling"]),
            gamma=_scalar(d, "gamma"),
            gamma_sign=_scalar(d, "gamma_sign"),
            D_f=_scalar(d, "fractal_dim"),
            n_cfg=_scalar(d, "n_lobe_configs"),
        ))
    rows = [r for r in rows if r["K"] is not None]
    rows.sort(key=lambda r: r["K"])
    return rows


def col(rows, key):
    return np.array([np.nan if r[key] is None else r[key] for r in rows], float)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob",
                    default="data/derivatives/lorenz_basins_n73_n81_K*.npz")
    ap.add_argument("--out", default="data/derivatives/onset_sweep_curves.png")
    a = ap.parse_args(argv)

    rows = load(a.glob)
    if not rows:
        print(f"no npz matched {a.glob}"); return 1
    K = col(rows, "K")
    has_sign = np.isfinite(col(rows, "gamma_sign")).any()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(K, col(rows, "gamma"), "o-", color="#888", label=r"$\gamma$ (k-means)")
    if has_sign:
        ax1.plot(K, col(rows, "gamma_sign"), "o-", color="#c0392b", lw=2,
                 label=r"$\gamma_{\rm sign}$ (lobe label)")
    ax1.plot(K, col(rows, "D_f") , "s--", color="#2980b9", alpha=.7,
             label=r"$D_f$ (box-count)")
    ax1.axhline(0.5, color="k", ls=":", alpha=.4)
    ax1.set_ylabel("exponent / dimension"); ax1.legend(); ax1.grid(alpha=.3)
    ax1.set_title(f"Lorenz onset sweep  ({len(rows)} K values"
                  f"{'' if has_sign else ', pre-patch — no gamma_sign yet'})")

    n = col(rows, "n_cfg")
    if np.isfinite(n).any():
        ax2.plot(K, n, "o-", color="#27ae60")
        ax2.set_ylabel("# lobe configs on slice")
    else:
        ax2.text(.5, .5, "n_lobe_configs appears once patched npz land",
                 ha="center", va="center", transform=ax2.transAxes, color="#999")
    ax2.set_xlabel("coupling K"); ax2.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"[plot] {len(rows)} K in [{K.min():.3f}, {K.max():.3f}] "
          f"gamma_sign={'yes' if has_sign else 'not yet'} -> {a.out}")
    # terminal peek
    print(f"{'K':>7} {'gamma':>7} {'g_sign':>7} {'D_f':>7} {'n_cfg':>6}")
    for r in rows:
        f = lambda v: "   -  " if v is None else f"{v:6.3f}"
        print(f"{r['K']:7.3f} {f(r['gamma'])} {f(r['gamma_sign'])} "
              f"{f(r['D_f'])} {'' if r['n_cfg'] is None else int(r['n_cfg']):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
