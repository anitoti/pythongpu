#!/usr/bin/env python3
"""Basin-plane figure for the K=0.45/0.475/0.50 reconnaissance sweep.

Same node pair (73, 81) and plotting convention as talk/make_figs.py's
fig_basin_map() (which only covers K=0.65): colour each IC pixel by its
full 83-node lobe-locking sign pattern sign(mean X_i), shuffled to
well-separated colour indices so adjacent-but-distinct patterns don't
blur together, then render one panel per coupling plus a D_f-vs-K line
so the "still flat right around K=0.5" result from the meeting notes is
visible directly.

Run:
    python3 scripts/plot_basin_maps_k045_050.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DERIV = REPO_ROOT / "data" / "derivatives"
OUT = DERIV / "basin_maps_K045_050.png"

COUPLINGS = ["0.4500", "0.4750", "0.5000"]

# Deck palette (talk/summer_talk.tex / talk/make_figs.py)
OCHRE = "#C49A3C"
RUST = "#B24727"
CREAM = "#F5EEDC"
DARK = "#2A1C12"

plt.rcParams.update({
    "figure.facecolor": CREAM, "axes.facecolor": CREAM,
    "savefig.facecolor": CREAM, "text.color": DARK,
    "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "axes.edgecolor": DARK, "font.size": 11, "axes.titlesize": 12,
    "axes.grid": False, "legend.frameon": False, "figure.dpi": 160,
})


def load(k: str) -> dict:
    path = DERIV / f"lorenz_basins_n73_n81_K{k}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def lobe_pattern_image(d: dict) -> tuple[np.ndarray, int, tuple[float, float]]:
    """Shuffle-coloured 83-bit lobe-sign pattern, same recipe as fig_basin_map()."""
    g = int(d["config"].item()["grid_n"])
    mean_x = d["mean_x"].reshape(g, g, -1)  # (g, g, 83), C-order
    lo, hi = float(d["Xg"].min()), float(d["Xg"].max())

    bits = (mean_x > 0).reshape(g * g, -1)  # (g*g, 83) boolean
    uniq, inv = np.unique(bits, axis=0, return_inverse=True)
    rng = np.random.default_rng(0)
    shuffle = rng.permutation(len(uniq))
    img = shuffle[inv].reshape(g, g)
    return img, len(uniq), (lo, hi)


def main() -> None:
    data = {k: load(k) for k in COUPLINGS}

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2),
                              gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})

    d_f_vals = []
    for ax, k in zip(axes[:3], COUPLINGS):
        d = data[k]
        img, n_patterns, (lo, hi) = lobe_pattern_image(d)
        g2 = img.shape[0] * img.shape[1]
        d_f = float(d["fractal_dim"])
        r2 = float(d["r_squared"])
        d_f_vals.append(d_f)

        ax.imshow(img, cmap="twilight_shifted", origin="lower",
                   extent=[lo, hi, lo, hi], interpolation="nearest")
        ax.set(xlabel=r"IC of node 73  ($X$)", ylabel=r"IC of node 81  ($X$)",
               title=f"$K={float(k):.3f}$: {n_patterns} patterns / {g2} ICs\n"
                     f"$D_f={d_f:.4f}$  ($R^2={r2:.4f}$)")
        ax.grid(False)

    ks = [float(k) for k in COUPLINGS]
    ax_df = axes[3]
    ax_df.plot(ks, d_f_vals, "o-", color=RUST, linewidth=2, markersize=7)
    ax_df.set(xlabel="coupling $K$", ylabel=r"box-counting $D_f$",
              title="$D_f$ vs $K$\n(flat near $K=0.5$)")
    ax_df.set_ylim(1.80, 1.95)
    ax_df.grid(True, alpha=0.25, color=DARK)

    fig.suptitle(
        "Basin reconnaissance near the paper's coupling: "
        "node pair (73, 81), $K \\in \\{0.45, 0.475, 0.50\\}$",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
