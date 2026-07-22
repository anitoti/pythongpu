#!/usr/bin/env python3
"""Generate three high-resolution PNGs from ACRES staging data (m=83).

Reads data from ~/pythongpu/data/acres_staging/ and writes PNGs to
~/pythongpu/data/derivatives/ with a warm, publication-ready palette.

Produces: onset_curve.png, lobe_histogram.png, convergence_control.png
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = Path.home()
DATA = HOME / "pythongpu" / "data"
STAGING = DATA / "acres_staging"
OUTDIR = DATA / "derivatives"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Palette
CREAM = "#f5eedc"
DARK = "#2a1c12"   # dark brown
RUST = "#b24727"
OCHRE = "#c49a3c"

plt.rcParams.update({
    "figure.facecolor": CREAM,
    "axes.facecolor": CREAM,
    "savefig.facecolor": CREAM,
    "text.color": DARK,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "axes.edgecolor": DARK,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.color": DARK,
    "legend.frameon": False,
})

DPI = 300


def find_clv_dirs(staging_dir: Path):
    """Yield (coupling, path) for directories like clv_c0_05 under staging."""
    out = []
    if not staging_dir.exists():
        return out
    for p in staging_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith("clv_c"):
            # parse coupling from name like clv_c0_05 or clv_c0_250
            try:
                val = name.split("clv_c", 1)[1].replace("_", ".")
                coupling = float(val)
            except Exception:
                # try reading summary json for coupling
                coupling = None
            out.append((coupling, p))
    # try to fill missing couplings from json, and sort by coupling
    filled = []
    for coupling, p in out:
        if coupling is None:
            sj = p / "clv_topology_summary.json"
            if sj.exists():
                try:
                    j = json.loads(sj.read_text())
                    coupling = float(j.get("coupling", 0.0))
                except Exception:
                    coupling = 0.0
        filled.append((coupling, p))
    filled.sort(key=lambda x: x[0] if x[0] is not None else 0.0)
    return filled


def load_summary(p: Path):
    sj = p / "clv_topology_summary.json"
    if not sj.exists():
        return None
    try:
        return json.loads(sj.read_text())
    except Exception:
        return None


def onset_curve():
    dirs = find_clv_dirs(STAGING)
    if not dirs:
        print("No clv directories found in staging; skipping onset_curve")
        return
    Ks = []
    lam = []
    lock = []
    big = []
    for coupling, p in dirs:
        s = load_summary(p)
        if s is None:
            continue
        if int(s.get("n_nodes", 0)) != 83:
            continue
        Ks.append(float(s.get("coupling", coupling)))
        lam.append(float(s.get("lambda_max", np.nan)))
        # treat "largest cluster pop frac" as the max cluster fraction if available
        riddling = s.get("riddling", {})
        cluster_pop = riddling.get("cluster_pop_frac") or []
        if cluster_pop:
            lock.append(100.0 * max(cluster_pop))
        else:
            # fallback: use 100*(1 - burst_fraction) if available
            bf = float(riddling.get("burst_fraction", 0.0))
            lock.append(100.0 * (1.0 - bf))
        # big: use burst_fraction*100 as a proxy for largest basin size
        big.append(100.0 * float(riddling.get("burst_fraction", 0.0)))

    if not Ks:
        print("No suitable summaries for n_nodes=83 found; skipping onset_curve")
        return

    Ks = np.array(Ks)
    order = np.argsort(Ks)
    Ks = Ks[order]
    lam = np.array(lam)[order]
    lock = np.array(lock)[order]
    big = np.array(big)[order]

    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
    a.plot(Ks, lock, "o-", color=RUST, lw=2.2, label="% nodes locked (largest cluster)")
    a.set_ylabel("% nodes locked", color=RUST)
    a.tick_params(axis="y", labelcolor=RUST)
    a2 = a.twinx()
    a2.plot(Ks, lam, "s--", color=OCHRE, label="lambda_max")
    a2.set_ylabel(r"$\Lambda$ (lambda_max)", color=OCHRE)
    a2.tick_params(axis="y", labelcolor=OCHRE)
    a2.grid(False)
    a.set_title("Lobe-locking onset (m=83)")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1+h2, l1+l2, loc="center right")

    b.plot(Ks, big, "o-", color=DARK, lw=2)
    b.set(xlabel="coupling K", ylabel="burst fraction (%)",
          title="Burst fraction (proxy for large basin occupation)")

    out = OUTDIR / "onset_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def lobe_histogram():
    # Prefer a coupling with strong locking (largest lock). Fall back to first.
    dirs = find_clv_dirs(STAGING)
    if not dirs:
        print("No clv directories found in staging; skipping lobe_histogram")
        return
    # pick first directory with clv_angles_83.npy
    chosen = None
    for coupling, p in dirs:
        arr = p / "clv_angles_83.npy"
        if arr.exists():
            chosen = (coupling, arr)
            break
    if chosen is None:
        print("No clv_angles_83.npy found; skipping lobe_histogram")
        return
    coupling, arrpath = chosen
    data = np.load(arrpath)
    # Plot histogram of the CLV min-angles as a proxy for bimodality in locking
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0.0, float(np.nanmax(data) * 1.05), 20)
    ax.hist(data, bins=bins, color=RUST, edgecolor=DARK, alpha=0.9)
    ax.set(xlabel="min transversality angle (rad)", ylabel="count",
           title=f"Histogram of min CLV angles (m=83), coupling={coupling}")
    out = OUTDIR / "lobe_histogram.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def convergence_control():
    # Use clv_angles_83.npy time series to estimate convergence vs averaging window
    dirs = find_clv_dirs(STAGING)
    if not dirs:
        print("No clv directories found in staging; skipping convergence_control")
        return
    # pick an example directory with clv_angles_83.npy
    chosen = None
    for coupling, p in dirs:
        arr = p / "clv_angles_83.npy"
        if arr.exists():
            chosen = (coupling, arr)
            break
    if chosen is None:
        print("No clv_angles_83.npy found; skipping convergence_control")
        return
    coupling, arrpath = chosen
    data = np.load(arrpath)
    # define averaging windows relative to series length
    L = len(data)
    Ts = np.unique(np.clip((np.array([5, 10, 20, 50, 100, 200, 400]) ), 1, L)).astype(int)
    desc = []
    for T in Ts:
        # compute running mean over windows of length T and take std of those means as descriptor
        if T >= 2:
            means = np.convolve(data, np.ones(T)/T, mode='valid')
            # descriptor: variability across windows (lower => more converged)
            desc.append(np.std(means))
        else:
            desc.append(np.std(data))
    Ts = Ts.astype(float)
    desc = np.array(desc)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(Ts, desc, "o-", color=RUST, lw=2.2, label=f"coupling={coupling}")
    ax.set(xlabel="averaging window T", ylabel="descriptor variability",
           title="Convergence proxy from CLV min-angles (m=83)")
    ax.legend()
    out = OUTDIR / "convergence_control.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == '__main__':
    onset_curve()
    lobe_histogram()
    convergence_control()
