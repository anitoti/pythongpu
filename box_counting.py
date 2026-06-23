# ============================================================
#  Kuramoto Basin Boundary — Box-Counting Fractal Dimension
#  Project : Nimble Brain (REU @ Clarkson)
#  Author  : anitoti
#
#  Pipeline position: RUNS AFTER kuramoto_basins.py
#  Input  : /data/kuramoto_basins.npy — dict with R, config
#  Output : /data/box_counting_fit.png
#
#  Theory : The basin boundary is extracted as the set of grid
#           cells where adjacent cells belong to different
#           k-means clusters. The fractal dimension D_f of this
#           boundary is estimated via box-counting:
#
#               N(s) ~ s^{-D_f}   →   log N(s) = -D_f log(s) + C
#
#           A slope of D_f ∈ (1, 2) on the log-log plot indicates
#           a fractal boundary — more complex than a curve (D=1)
#           but not filling the plane (D=2).
#
#  MATLAB ref : "forPlotting_FractalBasins.m" — uses gscatter
#               with K=8 kmeans clusters, marker 's', size 2.
#               Box-counting logic mirrors boxcount.m 2D case:
#               "n(g+1) = sum(sum(c(1:siz:(width-siz+1), ...)))"
#               [full_.m_script.pdf, Page 24-25]
# ============================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ── 0. LOAD ──────────────────────────────────────────────────
data   = np.load('/home/atotilca/pythongpu/data/kuramoto_basins.npy',
                 allow_pickle=True).item()
cfg    = data['config']

# R : (m, m) order parameter grid — one scalar per IC
R      = data['R']
m      = R.shape[0]          # grid resolution (64)
K_clus = 8                   # MATLAB ref: "K = 8" [Plotting_New_Zooms.m]


# ── 1. K-MEANS CLUSTERING ────────────────────────────────────
# "Colors = kmeans(KmeansMat, K)" [Plotting_New_Zooms.m]
# Each IC's final R value is its feature. k-means partitions
# the (m²,) R values into K basin regions — same logic as:
# "we apply the k-means method to the set of VPS {el}^M_{l=1}
#  to cluster the space into k-regions" [fractal_basins...pdf, Page 9]
#
# NOTE: if you save final theta (B×N) in the .npy, replace
#       R.ravel().reshape(-1,1) with theta for richer clustering.

R_flat  = R.ravel().reshape(-1, 1)           # (m², 1)
km      = KMeans(n_clusters=K_clus,
                 random_state=cfg['graph_seed'],
                 n_init=10)
labels  = km.fit_predict(R_flat)             # (m²,) int ∈ [0, K-1]
labels2d = labels.reshape(m, m)              # (m, m) cluster index grid


# ── 2. BOUNDARY EXTRACTION ───────────────────────────────────
# A pixel is ON the boundary if any of its 4 cardinal neighbors
# belongs to a different cluster. This is the discrete analog of
# the basin boundary set ∂B_k = closure(B_k) ∩ closure(B_j), j≠k
#
# Implementation: compare each cell to its shifted neighbors.
# b[i,j] = True  ↔  cell (i,j) sits on a basin boundary.
#
# Equivalent to MATLAB boxcount.m preprocessing step before the
# coarsening loop: "c(i,j) = c(i,j) || c(i+siz2,j) || ..."
# [full_.m_script.pdf, Page 24-25]

b = np.zeros((m, m), dtype=bool)
b[1:,  :]  |= labels2d[1:,  :] != labels2d[:-1, :]   # top    neighbor
b[:-1, :]  |= labels2d[:-1, :] != labels2d[1:,  :]   # bottom neighbor
b[:,  1:]  |= labels2d[:,  1:] != labels2d[:, :-1]   # right  neighbor
b[:, :-1]  |= labels2d[:, :-1] != labels2d[:, 1:]    # left   neighbor

boundary_density = b.sum() / (m * m)
print(f"Boundary pixels : {b.sum()} / {m*m}  ({boundary_density:.3%})")


# ── 3. BOX-COUNTING ──────────────────────────────────────────
# For each box side length s, count how many s×s boxes contain
# at least one boundary pixel. Mirrors the 2D boxcount.m loop:
#
#   "for g=(p-1):-1:0
#      siz = 2^(p-g);
#      n(g+1) = sum(sum(c(1:siz:(width-siz+1), ...)))"
#   [full_.m_script.pdf, Page 24-25]
#
# Here we use powers of 2 up to m/2 so every box size divides
# the grid evenly — avoids boundary artifacts from partial boxes.
#
#   N(s) = number of boxes of side s covering the boundary
#   s    = box side length in grid units
#
# Fractal dimension via log-log regression:
#   log N(s) = D_f · log(1/s) + C
#   slope of log N vs log(1/s) → D_f

sizes  = [1, 2, 4, 8, 16, 32]   # powers of 2; max = m/2 = 32
counts = []

for s in sizes:
    c = 0
    for i in range(0, m, s):
        for j in range(0, m, s):
            # Does this s×s box contain any boundary pixel?
            if b[i:i+s, j:j+s].any():
                c += 1
    counts.append(c)
    print(f"  s={s:2d}  →  N(s) = {c}")

# Log-log regression: log N(s) = D_f * log(1/s) + intercept
x = np.log(1.0 / np.array(sizes, dtype=float))   # log(1/s) — positive slope
y = np.log(np.array(counts,      dtype=float))   # log N(s)

coeffs          = np.polyfit(x, y, 1)
D_f, intercept  = coeffs
fit_y           = D_f * x + intercept

# R² — goodness of fit; D_f is only meaningful if R² ≈ 1
ss_res = np.sum((y - fit_y) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
R2     = 1.0 - ss_res / ss_tot

print(f"\nFractal dimension  D_f = {D_f:.4f}")
print(f"Fit R²              = {R2:.4f}")
print(f"Interpretation      : ", end="")
if   D_f < 1.05: print("smooth curve — likely under-resolved or K too small")
elif D_f > 1.95: print("space-filling — likely noise, not fractal structure")
else:            print(f"fractal boundary  ✓  ({1:.0f} < {D_f:.3f} < {2:.0f})")


# ── 4. PLOT ──────────────────────────────────────────────────
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# ── 4a. Basin boundary binary image ─────────────────────────
ax0 = axes[0]
ax0.imshow(b.astype(float),
           origin='lower',
           extent=[-np.pi, np.pi, -np.pi, np.pi],
           cmap='magma',
           interpolation='nearest')
ax0.set_xlabel(r'$x_0(0)$ — Node 0 phase', fontsize=11)
ax0.set_ylabel(r'$x_1(0)$ — Node 1 phase', fontsize=11)
ax0.set_title(f'Basin Boundary  (K={K_clus} clusters)\n'
              f'Boundary density = {boundary_density:.2%}', fontsize=10)

ticks      = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
ticklabels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
ax0.set_xticks(ticks); ax0.set_xticklabels(ticklabels)
ax0.set_yticks(ticks); ax0.set_yticklabels(ticklabels)

# ── 4b. Box-counting log-log plot ────────────────────────────
# Mirrors MATLAB: "loglog(r, n, 's-')" [full_.m_script.pdf, Page 24-25]

ax1 = axes[1]
ax1.scatter(x, y,
            color='#ff69b4',
            zorder=3,
            s=60,
            label='$N(s)$ data')
ax1.plot(x, fit_y,
         color='magenta',
         linestyle='--',
         linewidth=1.8,
         label=f'Linear fit\n$D_f = {D_f:.4f}$\n$R^2 = {R2:.4f}$')

ax1.set_xlabel(r'$\log(1/s)$  — log inverse box size', fontsize=11)
ax1.set_ylabel(r'$\log\, N(s)$  — log box count',      fontsize=11)
ax1.set_title('Box-Counting Fractal Dimension\n'
              r'$N(s) \sim s^{-D_f}$',                 fontsize=10)
ax1.legend(fontsize=9, framealpha=0.4)
ax1.grid(True, alpha=0.2)

# Annotate each data point with its (s, N(s)) value
for xi, yi, s, c in zip(x, y, sizes, counts):
    ax1.annotate(f's={s}\nN={c}',
                 xy=(xi, yi),
                 xytext=(6, -14),
                 textcoords='offset points',
                 fontsize=7,
                 color='#aaaaaa')

fig.suptitle(f'Nimble Brain — Fractal Basin Analysis\n'
             f'N={cfg["n_nodes"]} nodes, M={cfg["n_edges"]} edges, '
             f'K_coup={cfg["coupling"]}, T={cfg["tmax"]}',
             fontsize=11, y=1.02)

plt.tight_layout()
plt.savefig('/home/atotilca/pythongpu/data/box_counting_fit.png',
            bbox_inches='tight')
plt.close()
print("\nSaved → box_counting_fit.png")
