# ============================================================
#  Kuramoto Basin Plotter — 2-D Phase Slice Visualization
#  Project : Nimble Brain (REU @ Clarkson)
#  Author  : anitoti
#
#  Pipeline position: RUNS AFTER kuramoto_basins.py
#  Input  : /data/kuramoto_basins.npy  — dict with X, Y, R, config
#  Output : /data/basin_map_continuous.png   — continuous R heatmap
#           /data/basin_map_kmeans.png       — k-means clustered basin plot
#
#  Theory : Basin plots partition phase space so that "in any like
#           colored region, the orbits of the initial conditions map
#           asymptotically to similar patterns." [Page 9]
#           K-means is applied to cluster the VPS (vector of
#           phase states) into k regions, forming a partition
#           function P : Z → {1, 2, .., k} [Page 9]
# ============================================================

# run command:
# /usr/bin/python3 ~/pythongpu/plot_kuramoto_basins.py

import numpy as np
import matplotlib
matplotlib.use('Agg')                        # no display — HPC node has no GUI
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from pythongpu.utils import get_plot_path


# ── 0. LOAD ──────────────────────────────────────────────────
data   = np.load('/home/atotilca/pythongpu/data/kuramoto_basins.npy',
                 allow_pickle=True).item()

X      = data['X']           # (m, m) node-0 phase grid  ∈ [-π, π]
Y      = data['Y']           # (m, m) node-1 phase grid  ∈ [-π, π]
R      = data['R']           # (m, m) Kuramoto order parameter R ∈ [0,1]
cfg    = data['config']      # dict — full Config snapshot from integration run

m      = R.shape[0]          # grid resolution per axis
K_clus = 8                   # number of basin clusters  (mirrors MATLAB: K=8)
                             # [Page 37]: "K = 8; Colors = kmeans(KmeansMat,K)"

SAVE   = '/home/atotilca/pythongpu/data/'


# ── 1. CONTINUOUS R HEATMAP ──────────────────────────────────
# Shows raw order parameter — no clustering.
# Bright regions → global sync (R≈1), dark → incoherence (R≈0)
# Fractal boundary structure visible as the transition band.

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

mesh = ax.pcolormesh(X, Y, R, cmap='magma', shading='auto',
                     vmin=0.0, vmax=1.0)

cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Order Parameter  $R = |\\langle e^{i\\theta}\\rangle|$',
               fontsize=11)
cbar.ax.axhline(cfg['sync_threshold'], color='cyan',
                linewidth=1.5, linestyle='--')        # mark sync cutoff
cbar.ax.text(1.6, cfg['sync_threshold'],
             f"R={cfg['sync_threshold']}", va='center',
             fontsize=8, color='cyan')

ax.set_xlabel("$x_0(0)$ — Node 0 initial phase", fontsize=12)
ax.set_ylabel("$x_1(0)$ — Node 1 initial phase", fontsize=12)
ax.set_title('Kuramoto Basin Map — Continuous $R$\n'
             f'N={cfg["n_nodes"]}, M={cfg["n_edges"]}, '
             f'K={cfg["coupling"]}, T={cfg["tmax"]}',
             fontsize=11)

# Axis ticks in units of π — matches standard phase-space convention
ticks     = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
ticklabels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels)
ax.set_yticks(ticks)
ax.set_yticklabels(ticklabels)

plt.tight_layout()
plt.savefig(get_plot_path('plot_kuramoto_basins', 'basin_map_continuous.png'), bbox_inches='tight')
plt.close()
print("Saved → basin_map_continuous.png")


# ── 2. K-MEANS CLUSTERED BASIN PLOT ─────────────────────────
# Mirrors the MATLAB pipeline exactly:
#   Colors = kmeans(KmeansMat, K);
#   h = gscatter(Xg, Yg, Colors);   [Page 37]
#
# "We apply the k-means method to the set of VPS {el}^M_{l=1}
#  to cluster the space into k-regions (colors) and map the
#  phase space by associating these colors to each corresponding
#  initial condition z_l(0)." [Page 9]
#
# KmeansMat = the full final theta (B×N) — each IC's complete phase
# vector, not just its scalar R value. Richer than clustering on R alone:
# two ICs can share an R value while landing in very different phase
# configurations, which a single scalar feature can't distinguish.

R_flat  = data['theta']                      # (m², N) — full phase vector per IC
Xg      = X.ravel()                          # (m²,)   — x coords for scatter
Yg      = Y.ravel()                          # (m²,)   — y coords for scatter

km      = KMeans(n_clusters=K_clus, random_state=cfg['graph_seed'], n_init=10)
labels  = km.fit_predict(R_flat)             # (m²,) int ∈ [0, K-1]

# Color palette — one distinct color per basin cluster
cmap_k  = plt.get_cmap('tab10')
colors  = [cmap_k(i / K_clus) for i in range(K_clus)]

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

for k in range(K_clus):
    mask = labels == k
    ax.scatter(Xg[mask], Yg[mask],
               color=colors[k],
               s=1.5,                        # MarkerSize=1.75 equivalent
               marker='s',                   # [Page 37]: h(i).Marker = 's'
               label=f'Basin {k+1}')

ax.set_xlabel("$x_{0}(0)$ — Node 0 initial phase", fontsize=12)
ax.set_ylabel("$x_{1}(0)$ — Node 1 initial phase", fontsize=12)
ax.set_title(f'Kuramoto Basin Map — $K={K_clus}$ Clusters\n'
             f'N={cfg["n_nodes"]}, M={cfg["n_edges"]}, '
             f'K_coup={cfg["coupling"]}, T={cfg["tmax"]}',
             fontsize=11)

ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels)
ax.set_yticks(ticks)
ax.set_yticklabels(ticklabels)
ax.set_xlim(Xg.min(), Xg.max())
ax.set_ylim(Yg.min(), Yg.max())             # [Page 37]: xlim/ylim from data
ax.legend(markerscale=4, fontsize=8,
          loc='upper right', framealpha=0.6)

plt.tight_layout()
plt.savefig(get_plot_path('plot_kuramoto_basins', 'basin_map_kmeans.png'), bbox_inches='tight')
plt.close()
print(f"Saved → basin_map_kmeans.png  (K={K_clus} clusters)")


# ── 3. SUMMARY STATS ─────────────────────────────────────────
synced = (R >= cfg['sync_threshold']).mean()
print("\n── Basin Statistics ──────────────────────────")
print(f"Grid resolution : {m}×{m} = {m*m} initial conditions")
print(f"R  mean         : {R.mean():.4f}")
print(f"R  std          : {R.std():.4f}")
print(f"R  min / max    : {R.min():.4f} / {R.max():.4f}")
print(f"Synced fraction : {synced:.4f}  (R ≥ {cfg['sync_threshold']})")
print(f"K-means K       : {K_clus} basins")