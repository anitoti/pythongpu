#!/usr/bin/env python3
"""Generate the talk's figures from measured data.

Every number here came out of a run in this repository; nothing is illustrative
except the two explicitly-labelled cartoons (basin_cartoon, lorenz_attractor),
which depict textbook geometry rather than results.

    python3 talk/make_figs.py            # writes talk/figs/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# Deck palette (summer_talk.tex)
OCHRE = "#C49A3C"
RUST = "#B24727"
OLIVE = "#5B6737"
CREAM = "#F5EEDC"
DARK = "#2A1C12"
SAND = "#E6DABE"

plt.rcParams.update({
    "figure.facecolor": CREAM, "axes.facecolor": CREAM,
    "savefig.facecolor": CREAM, "text.color": DARK,
    "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "axes.edgecolor": DARK, "font.size": 11, "axes.titlesize": 12,
    "axes.grid": True, "grid.alpha": 0.25, "grid.color": DARK,
    "legend.frameon": False, "figure.dpi": 160,
})


def save(fig, name):
    p = FIGS / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}")


# ── 1. Benchmark: the profiling story ────────────────────────────────────────
def fig_benchmark():
    T = np.array([256, 512, 1024, 2048])
    serial = np.array([0.252, 0.417, 0.912, 2.614])
    looped = np.array([0.083, 0.141, 0.249, 0.400])
    vect = np.array([0.0011, 0.0019, 0.0036, 0.0070])

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.loglog(T, serial, "o-", color=DARK, label="serial CPU reference")
    a.loglog(T, looped, "s-", color=OCHRE, label="GPU port (lag loop)")
    a.loglog(T, vect, "^-", color=RUST, lw=2.4, label="GPU vectorised")
    a.set(xlabel="time samples $T$", ylabel="runtime (s)",
          title="VPS runtime, $N{=}83$ (3403 pairs)")
    a.legend()

    x = np.arange(len(T))
    w = 0.38
    b.bar(x - w/2, serial/looped, w, color=OCHRE, label="port (lag loop)")
    b.bar(x + w/2, serial/vect, w, color=RUST, label="vectorised")
    b.axhline(1.0, color=DARK, lw=1, ls="--")
    b.text(0.02, 1.35, "no faster than serial", fontsize=8, color=DARK)
    for i, v in enumerate(serial/vect):
        b.text(i + w/2, v*1.05, f"{v:.0f}×", ha="center", fontsize=9,
               color=RUST, fontweight="bold")
    for i, v in enumerate(serial/looped):
        b.text(i - w/2, v*1.15, f"{v:.1f}×", ha="center", fontsize=8, color=DARK)
    b.set_yscale("log")
    b.set_xticks(x, [str(t) for t in T])
    b.set(xlabel="time samples $T$", ylabel="speedup over serial",
          title="The FFT was never the bottleneck")
    b.legend(loc="upper left")
    save(fig, "benchmark.png")


# ── 2. The lag-1 under-shift inflates L ──────────────────────────────────────
def fig_alignment():
    import scipy.io
    from scipy.signal import correlate

    def vps(x, matlab):
        T, n = x.shape
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        L = np.zeros(len(pairs))
        for k, (i, j) in enumerate(pairs):
            cx = correlate(x[:, i], x[:, j], mode="full")
            lag = int(np.arange(-(T-1), T)[int(np.argmax(cx))])
            s = max(abs(lag) - 1, 0) if matlab else abs(lag)
            if lag > 0:
                a, b_ = x[0:T-s, j], x[s:, i]
            elif lag < 0:
                a, b_ = x[0:T-s, i], x[s:, j]
            else:
                a, b_ = x[:, j], x[:, i]
            m = min(len(a), len(b_))
            L[k] = np.linalg.norm(a[:m] - b_[:m])
        return L

    A = scipy.io.loadmat(
        "/home/atotilca/matlab_fractalbasin/Example_A_3.mat")["A"].astype(float)
    Lm, Lc = vps(A, True), vps(A, False)

    fig, ax = plt.subplots(figsize=(5.4, 5))
    hi = max(Lm.max(), Lc.max()) * 1.08
    ax.plot([0, hi], [0, hi], "--", color=DARK, lw=1, label="agreement")
    ax.scatter(Lc, Lm, s=42, color=RUST, alpha=0.85, edgecolor=DARK, lw=0.4)
    ax.set(xlim=(0, hi), ylim=(0, hi),
           xlabel=r"$\ell_{ij}$  aligned by $|\tau|$   (corrected)",
           ylabel=r"$\ell_{ij}$  aligned by $|\tau|-1$   (reference)",
           title="The under-shift inflates every residual")
    ax.text(0.05*hi, 0.90*hi,
            f"{(Lm >= Lc-1e-12).sum()}/{len(Lm)} pairs above the line\n"
            f"mean ratio  {Lm.mean()/Lc.mean():.2f}×",
            fontsize=10, color=RUST, fontweight="bold")
    ax.legend(loc="lower right")
    save(fig, "alignment_bias.png")


# ── 3. Lobe-locking: the bimodal histogram (the money figure) ────────────────
def fig_lobe_hist():
    # Measured distribution of per-node <X>, 24^2 grid, transient 100, tmax 600.
    # K=0: 100% of nodes in |<X>|<1, mean|<X>|=0.025 (ergodic, both wings).
    # K=0.5: 97.8% locked, sharply bimodal at +-7; only 0.3% within |<X>|<1.
    edges = np.array([-12, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 12])
    k00 = np.array([0, 0, 0, 0, 0, 100.0, 0, 0, 0, 0, 0])
    k05 = np.array([11.3, 28.1, 9.6, 0.7, 0.3, 0.3, 0.2, 0.5, 9.5, 29.0, 10.4])
    ctr = (edges[:-1] + edges[1:]) / 2
    wid = np.diff(edges) * 0.92

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    a.bar(ctr, k00, width=wid, color=OLIVE, edgecolor=DARK, lw=0.5)
    a.set(title="$K=0$  uncoupled: every node ergodic",
          xlabel=r"per-node  $\langle X_i\rangle$", ylabel="% of nodes")
    a.text(0, 60, "100% within\n$|\\langle X\\rangle|<1$", ha="center",
           fontsize=10, color=OLIVE, fontweight="bold")

    b.bar(ctr, k05, width=wid, color=RUST, edgecolor=DARK, lw=0.5)
    for s in (-7.8, 7.8):
        b.axvline(s, color=DARK, ls="--", lw=1.2)
    b.text(7.8, 26, r"  $C^{+}$", fontsize=11, color=DARK)
    b.text(-7.8, 26, r"$C^{-}$  ", fontsize=11, color=DARK, ha="right")
    b.set(title="$K=0.5$  coupled: 97.8% locked to one wing",
          xlabel=r"per-node  $\langle X_i\rangle$")
    b.text(0, 60, "only 0.3%\nnear zero", ha="center", fontsize=10,
           color=RUST, fontweight="bold")
    fig.suptitle(r"Coupling pins each node to one wing "
                 r"$\Rightarrow$ the basin label is $\mathrm{sign}\langle X_i\rangle$",
                 fontsize=12)
    save(fig, "lobe_histogram.png")


# ── 4. The onset curve ───────────────────────────────────────────────────────
def fig_onset():
    K = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20])
    lam = np.array([0.026, 0.252, 0.258, 0.269, 0.413, 1.202, 2.916, 5.318,
                    6.368, 6.847, 7.115, 7.291])
    lock = np.array([0.0, 0.0, 0.0, 0.0, 1.7, 11.0, 32.8, 66.0, 80.7, 87.8, 91.6, 95.1])
    big = np.array([25.0, 0.1, 0.1, 0.1, 0.1, 0.1, 2.9, 4.7, 8.7, 9.3, 4.0, 0.4])

    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
    a.plot(K, lock, "o-", color=RUST, lw=2.2, label=r"% nodes locked")
    a.axvline(0.07, color=DARK, ls="--", lw=1.2)
    a.text(0.073, 45, "onset\n$K\\approx0.07$", fontsize=10, color=DARK)
    a.set_ylabel("% of nodes locked", color=RUST)
    a.tick_params(axis="y", labelcolor=RUST)
    a2 = a.twinx()
    a2.plot(K, lam, "s--", color=OLIVE, label=r"$\Lambda=\langle|\langle X_i\rangle|\rangle$")
    a2.axhline(7.8, color=OLIVE, ls=":", lw=1)
    a2.text(0.005, 7.9, r"single-wing mean $\pm7.8$", fontsize=8, color=OLIVE)
    a2.set_ylabel(r"$\Lambda$", color=OLIVE)
    a2.tick_params(axis="y", labelcolor=OLIVE)
    a2.grid(False)
    a.set_title("Lobe-locking onset: a sigmoid bifurcation")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1+h2, l1+l2, loc="center right")

    b.plot(K, big, "o-", color=OCHRE, lw=2)
    b.axvspan(0.08, 0.12, color=OCHRE, alpha=0.18)
    b.text(0.10, 6.5, "basins have\nfinite size", ha="center", fontsize=9, color=DARK)
    b.annotate("sign noise\n(not attractors)", xy=(0.02, 0.1), xytext=(0.02, 4.5),
               fontsize=8, color=DARK, ha="center",
               arrowprops=dict(arrowstyle="->", color=DARK, lw=0.8))
    b.annotate("riddled:\nunmappable", xy=(0.20, 0.4), xytext=(0.175, 5.5),
               fontsize=8, color=DARK, ha="center",
               arrowprops=dict(arrowstyle="->", color=DARK, lw=0.8))
    b.set(xlabel="coupling $K$", ylabel="largest basin (% of ICs)",
          title="Only $K\\approx0.08$–$0.12$ has mappable basins")
    save(fig, "onset_curve.png")


# ── 5. The validated metric: K=0 converges, K=0.5 does not ───────────────────
def fig_convergence():
    T = np.array([50, 200, 800, 1600])
    k0 = np.array([1.332, 0.548, 0.308, 0.241])
    k5 = np.array([86.748, 86.987, 87.187, 87.276])
    pred = k0[0] * np.sqrt(T[0] / T)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.loglog(T, k5, "s-", color=RUST, lw=2.2,
              label=r"$K=0.5$: flat $\Rightarrow$ IC-dependent")
    ax.loglog(T, k0, "o-", color=OLIVE, lw=2.2,
              label=r"$K=0$ (uncoupled): converges")
    ax.loglog(T, pred, ":", color=DARK, lw=1.4, label=r"$1/\sqrt{T}$ (ergodic)")
    ax.annotate("", xy=(1600, 87.276), xytext=(1600, 0.241),
                arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.2))
    ax.text(1150, 5, "362×", fontsize=12, color=DARK, fontweight="bold", ha="right")
    ax.set(xlabel="averaging window $T$",
           ylabel="descriptor distance, grid neighbours",
           title="Validating the ruler on a known answer")
    ax.legend(loc="center left")
    save(fig, "convergence_control.png")


# ── 6. Persistence: the label is as good as the locking ──────────────────────
def fig_persistence():
    K = np.array([0.01, 0.06, 0.08, 0.10, 0.12, 0.20, 0.50])
    bits = np.array([50.2, 91.1, 96.7, 98.0, 98.2, 98.7, 99.9])
    exact = np.array([0.0, 3.6, 18.1, 30.1, 35.0, 63.5, 94.6])
    lock = np.array([0.0, 32.8, 66.0, 80.7, 87.8, 95.1, 97.8])

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.semilogx(K, bits, "o-", color=RUST, lw=2.2, label="bits agreeing")
    ax.semilogx(K, exact, "^-", color=OCHRE, lw=2, label=r"exact 83-bit pattern")
    ax.semilogx(K, lock, "s--", color=OLIVE, lw=1.6, label="% nodes locked")
    ax.axhline(50, color=DARK, ls=":", lw=1.4)
    ax.text(0.011, 53, "coin flip", fontsize=9, color=DARK)
    ax.annotate("negative control\npasses: 50.2%", xy=(0.01, 50.2), xytext=(0.016, 22),
                fontsize=9, color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=0.9))
    ax.set(xlabel="coupling $K$", ylabel="% agreement across disjoint windows",
           ylim=(-4, 108),
           title=r"$\mathrm{sign}\langle X\rangle$ is exactly as trustworthy as the locking")
    ax.legend(loc="center right")
    save(fig, "persistence.png")


# ── 7. The DTI network ───────────────────────────────────────────────────────
def fig_network():
    import scipy.io
    import networkx as nx
    A = scipy.io.loadmat("data/DTI-og.mat")["A"].astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A)
    deg = np.array([d for _, d in G.degree()])
    pos = nx.spring_layout(G, seed=3, k=0.32, iterations=120)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.10, edge_color=DARK, width=0.5)
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=14 + 3.2*deg, node_color=deg,
        cmap="YlOrBr", edgecolors=DARK, linewidths=0.4)
    for n, label in ((int(np.argmax(deg)), "hub, deg 44"),
                     (int(np.argmin(deg)), "leaf, deg 2")):
        ax.annotate(label, xy=pos[n], xytext=(pos[n][0]+0.22, pos[n][1]+0.20),
                    fontsize=9, color=RUST, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=RUST, lw=1.2))
    ax.set_title("The DTI connectome: 83 regions, 850 edges\n"
                 "density 0.25, diameter 4, degree 2–44", fontsize=11)
    ax.axis("off")
    fig.colorbar(nodes, ax=ax, fraction=0.04, pad=0.02, label="degree")
    save(fig, "dti_network.png")


# ── 8. Lorenz attractor with its two wings (textbook geometry) ───────────────
def fig_lorenz():
    sig, rho, beta, dt = 10.0, 28.0, 8.0/3.0, 0.004
    x = np.empty((60000, 3))
    x[0] = (1.0, 1.0, 1.0)
    for i in range(len(x)-1):
        X, Y, Z = x[i]
        x[i+1] = x[i] + dt*np.array([sig*(Y-X), X*(rho-Z)-Y, X*Y-beta*Z])
    x = x[2000:]
    c = np.sqrt(beta*(rho-1))

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    left = x[:, 0] < 0
    ax.plot(x[left, 0], x[left, 2], ",", color=OLIVE, alpha=0.5)
    ax.plot(x[~left, 0], x[~left, 2], ",", color=RUST, alpha=0.5)
    for s, lab, col in ((c, r"$C^{+}$", RUST), (-c, r"$C^{-}$", OLIVE)):
        ax.plot(s, rho-1, "o", ms=9, color=col, mec=DARK, mew=1.2)
        ax.annotate(lab + f"\n$X={s:+.1f}$", xy=(s, rho-1),
                    xytext=(s + (7 if s > 0 else -7), rho-13),
                    fontsize=11, color=DARK, ha="center",
                    arrowprops=dict(arrowstyle="->", color=DARK, lw=1))
    ax.set(xlabel="$X$", ylabel="$Z$",
           title="Two wings. An ergodic trajectory visits both\n"
                 r"($\langle X\rangle=0$); a locked one does not ($\langle X\rangle\approx\pm7.8$)")
    save(fig, "lorenz_attractor.png")


# ── 9. Basin cartoon: smooth vs fractal ridgeline (illustrative) ─────────────
def fig_cartoon():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    g = 400
    xs = np.linspace(0, 1, g)
    ys = np.linspace(0, 1, g)
    X, Y = np.meshgrid(xs, ys)

    a.imshow((X > 0.5).astype(float), cmap=matplotlib.colors.ListedColormap([OLIVE, OCHRE]),
             origin="lower", extent=[0, 1, 0, 1], interpolation="nearest")
    a.axvline(0.5, color=DARK, lw=2.5)
    a.plot(0.30, 0.5, "o", ms=13, color=CREAM, mec=DARK, mew=2)
    a.annotate("", xy=(0.47, 0.5), xytext=(0.32, 0.5),
               arrowprops=dict(arrowstyle="->", color=DARK, lw=2.4))
    a.set_title("Smooth boundary\na big push is needed to switch")

    rng = np.random.default_rng(1)
    lab = np.zeros((g, g))
    for k in range(1, 9):                       # nested interleaving -> fine scales
        lab += ((X * 2**k).astype(int) % 2) * (0.5 ** k)
    lab = (lab > lab.mean()).astype(float)
    flip = rng.random((g, g)) < 0.06        # one mask, evaluated once
    lab[flip] = 1 - lab[flip]
    b.imshow(lab, cmap=matplotlib.colors.ListedColormap([OLIVE, OCHRE]),
             origin="lower", extent=[0, 1, 0, 1], interpolation="nearest")
    b.plot(0.30, 0.5, "o", ms=13, color=CREAM, mec=DARK, mew=2)
    b.annotate("", xy=(0.335, 0.5), xytext=(0.31, 0.5),
               arrowprops=dict(arrowstyle="->", color=RUST, lw=2.4))
    b.set_title("Fractal boundary\nan edge is always a hair away")
    for ax in (a, b):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("Crinkliness is nimbleness   (illustrative)", fontsize=12)
    save(fig, "basin_cartoon.png")


if __name__ == "__main__":
    print("generating figures ->", FIGS)
    for f in (fig_benchmark, fig_alignment, fig_lobe_hist, fig_onset,
              fig_convergence, fig_persistence, fig_network, fig_lorenz,
              fig_cartoon):
        try:
            f()
        except Exception as exc:      # one bad figure must not kill the batch
            print(f"  [skip] {f.__name__}: {type(exc).__name__}: {exc}")
    print("done")
