# ============================================================
#  Lorenz Basin Plotter -- 2-D IC Slice Visualization
#  Project : Nimble Brain (REU @ Clarkson)
#  Author  : anitoti
#
#  Loads precomputed data from lorenz_basins.npz, extracts
#  the config dictionary, and replots the continuous norm
#  heatmap + k-means clustered basin map from saved state.
#
#  Output : /data/lorenz_basin_map_continuous.png
#           /data/lorenz_basin_map_kmeans.png
# ============================================================

"""
run via
python3 ~/pythongpu/src/pythongpu/plot_lorenz_basins.py
"""

import os, math
import numpy as np
import torch
import networkx as nx
from dataclasses import dataclass, asdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from lorenz_basins import LorenzNetwork, Config, load_graph


# -- 0. LOAD PRECOMPUTED DATA ----------------------------------
DATA_PATH = '/home/atotilca/pythongpu/data/lorenz_basins.npz'
SAVE_DIR  = '/home/atotilca/pythongpu/data/'

data        = np.load(DATA_PATH, allow_pickle=True)
cfg_dict    = data["config"].item()               # unpickle the saved config dict
cfg         = Config(**cfg_dict)                  # reconstruct Config object

Xg          = data["X"]                            # (m, m) grid coords
Yg          = data["Y"]
state_np    = data["state_final"]                  # (B, 3, N) final state
state_flat  = data["state_flat"]                   # (B, 3*N) flattened for k-means
B           = state_flat.shape[0]
m           = int(math.sqrt(B))
N           = cfg.n_nodes

os.makedirs(SAVE_DIR, exist_ok=True)


# -- 1. CONTINUOUS HEATMAP --------------------------------------
# Euclidean norm of each final state (analogous to Kuramoto order param R)
state_norm = np.linalg.norm(state_flat, axis=1).reshape(m, m)

print("\nPlotting continuous norm heatmap ...")

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

mesh = ax.pcolormesh(Xg, Yg, state_norm, cmap='magma', shading='auto')

cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'Final state norm  $\|\mathbf{s}(T)\|_2$', fontsize=11)

ax.set_xlabel(f"$x_{{{cfg.slice_node_x}}}(0)$ -- Node {cfg.slice_node_x} initial $X$", fontsize=12)
ax.set_ylabel(f"$x_{{{cfg.slice_node_y}}}(0)$ -- Node {cfg.slice_node_y} initial $X$", fontsize=12)
ax.set_title(f'Lorenz Basin Map -- Final State Norm\n'
             f'N={cfg.n_nodes}, M={cfg.n_edges}, '
             f'K_coup={cfg.coupling}, T={cfg.tmax}',
             fontsize=11)

ax.set_xlim(Xg.min(), Xg.max())
ax.set_ylim(Yg.min(), Yg.max())

plt.tight_layout()
plt.savefig(SAVE_DIR + 'lorenz_basin_map_continuous.png', bbox_inches='tight')
plt.close()
print(f"Saved -> {SAVE_DIR}lorenz_basin_map_continuous.png")


# -- 2. K-MEANS CLUSTERED BASIN PLOT ----------------------------
# Cluster the full final state vector (3*N dims) per IC.
# Same logic as plot_kuramoto_basins.py: K=8, marker='s', tab10 colors.
print("Clustering with K-means ...")

K_clus = 8
Xg_flat = Xg.ravel()
Yg_flat = Yg.ravel()

km      = KMeans(n_clusters=K_clus,
                 random_state=cfg.graph_seed,
                 n_init=10)
labels  = km.fit_predict(state_flat)              # (B,) int in [0, K-1]

cmap_k  = plt.get_cmap('tab10')
colors  = [cmap_k(i / K_clus) for i in range(K_clus)]

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

for k in range(K_clus):
    mask = labels == k
    ax.scatter(Xg_flat[mask], Yg_flat[mask],
               color=colors[k],
               s=1.5,
               marker='s',
               label=f'Basin {k+1}')

ax.set_xlabel(f"$x_{{{cfg.slice_node_x}}}(0)$ -- Node {cfg.slice_node_x} initial $X$", fontsize=12)
ax.set_ylabel(f"$x_{{{cfg.slice_node_y}}}(0)$ -- Node {cfg.slice_node_y} initial $X$", fontsize=12)
ax.set_title(f'Lorenz Basin Map -- $K={K_clus}$ Clusters\n'
             f'N={cfg.n_nodes}, M={cfg.n_edges}, '
             f'K_coup={cfg.coupling}, T={cfg.tmax}',
             fontsize=11)

ax.set_xlim(Xg.min(), Xg.max())
ax.set_ylim(Yg.min(), Yg.max())
ax.legend(markerscale=4, fontsize=8,
          loc='upper right', framealpha=0.6)

plt.tight_layout()
plt.savefig(SAVE_DIR + 'lorenz_basin_map_kmeans.png', bbox_inches='tight')
plt.close()
print(f"Saved -> {SAVE_DIR}lorenz_basin_map_kmeans.png  (K={K_clus} clusters)")


# -- 3. SUMMARY STATS -------------------------------------------
print(f"\n-- Basin Statistics ------------------------------")
print(f"Grid resolution  : {m}x{m} = {B} initial conditions")
print(f"State dim (3*N)  : {state_flat.shape[1]}")
print(f"Norm mean        : {state_norm.mean():.4f}")
print(f"Norm std         : {state_norm.std():.4f}")
print(f"Norm min / max   : {state_norm.min():.4f} / {state_norm.max():.4f}")
print(f"K-means K        : {K_clus} basins")