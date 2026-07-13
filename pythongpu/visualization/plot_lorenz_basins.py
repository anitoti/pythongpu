# ============================================================
#  Lorenz Basin Plotter -- 2-D IC Slice Visualization
#  Project : Nimble Brain (REU @ Clarkson)
#  Author  : anitoti
#  filename: plot_lorenz_basins.py
#
#  Loads precomputed data from basin_data_{timestamp}.npz,
#  and replots the k-means clustered basin map, boundary,
#  box-counting log-log, and elbow curve from saved state.
#
#  Usage:
#    python3 -m pythongpu.visualization.plot_lorenz_basins
#    python3 -m pythongpu.visualization.plot_lorenz_basins --timestamp 20260629_125200
#    python3 -m pythongpu.visualization.plot_lorenz_basins --outdir /path/to/output
# ============================================================

import os
import re
import glob
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pythongpu.utils import get_plot_path
from pythongpu.networks.desikan_killiany import labels_for


# -- HELPERS ----------------------------------------------------
def _find_latest_npz(data_dir: str) -> str:
    """
    Scan *data_dir* for files matching ``basin_data_*.npz`` and return
    the path with the most recent timestamp suffix.
    """
    pattern = os.path.join(data_dir, "basin_data_*.npz")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No basin_data_*.npz files found in {data_dir}"
        )
    # Extract timestamp from filename and sort descending
    def _ts_key(fpath: str) -> str:
        base = os.path.basename(fpath)
        m = re.match(r"basin_data_(\d{8}_\d{6})\.npz", base)
        return m.group(1) if m else ""
    files.sort(key=_ts_key, reverse=True)
    return files[0]


# -- 0. ARGUMENTS ----------------------------------------------
parser = argparse.ArgumentParser(
    description="Plot Lorenz basin maps from precomputed timestamped data."
)
parser.add_argument("--timestamp", type=str, default=None,
                    help="Timestamp suffix YYYYMMDD_HHMMSS. "
                         "If omitted, loads the latest basin_data_*.npz.")
parser.add_argument("--outdir", type=str,
                    default="/home/atotilca/pythongpu/data/",
                    help="Output directory for figures "
                         "(default: /home/atotilca/pythongpu/data/)")
args = parser.parse_args()

SAVE_DIR = args.outdir
os.makedirs(SAVE_DIR, exist_ok=True)

# -- DETERMINE TIMESTAMP & DATA PATH ----------------------------
if args.timestamp is not None:
    ts = args.timestamp
    data_path = os.path.join(SAVE_DIR, f"basin_data_{ts}.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Specified data file not found: {data_path}")
else:
    data_path = _find_latest_npz(SAVE_DIR)
    # Extract timestamp from the found file
    base = os.path.basename(data_path)
    m = re.match(r"basin_data_(\d{8}_\d{6})\.npz", base)
    ts = m.group(1) if m else datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[data]     Auto-detected latest: {data_path}")

print(f"[data]     Loading {data_path} ...")
data = np.load(data_path, allow_pickle=True)

# -- EXTRACT DATA -----------------------------------------------
Xg          = data["Xg"]                            # (m, m) grid coords
Yg          = data["Yg"]
labels      = data["labels"]                        # (m, m) basin labels
boundary    = data["boundary"]                      # (m, m) bool boundary
vectors     = data["vectors"]                       # (B, 2*C) VPS features
boxcount_r  = data["boxcount_r"]                    # (p,) box sizes
boxcount_n  = data["boxcount_n"]                    # (p,) box counts
fractal_dim = float(data["fractal_dim"])            # D_f
r_squared   = float(data["r_squared"])              # R²

m = labels.shape[0]
B = m * m
N_osc = None  # not stored directly; can be inferred from vectors dim
# vectors shape: (B, 2*C) where C = N*(N-1)/2
C_pairs = vectors.shape[1] // 2
# N*(N-1)/2 = C_pairs  =>  N² - N - 2*C_pairs = 0
# N = (1 + sqrt(1 + 8*C_pairs)) / 2
N_osc = int((1 + np.sqrt(1 + 8 * C_pairs)) / 2)

# -- LIVE SIMULATION CONFIG ------------------------------------
# Read the configuration the producing sweep embedded in the .npz so every
# title/axis reflects the ACTUAL coupling and swept-node identities. Older
# archives without a `config` record fall back to sentinels and the plot is
# annotated as such rather than silently mislabelling coupling as 0.
if "config" in data.files:
    cfg = data["config"].item()
    coupling  = float(cfg.get("coupling", float("nan")))
    node_x    = int(cfg.get("slice_node_x", -1))
    node_y    = int(cfg.get("slice_node_y", -1))
    grid_lo   = float(cfg.get("grid_lo", -9.0))
    grid_hi   = float(cfg.get("grid_hi",  9.0))
    N_osc     = int(cfg.get("n_osc", N_osc))
    coupling_str = f"coupling={coupling:g}"
    config_present = True
else:
    coupling, node_x, node_y = float("nan"), -1, -1
    grid_lo, grid_hi = -9.0, 9.0
    coupling_str = "coupling=?? (legacy npz — no config record)"
    config_present = False

# Desikan-Killiany region labels for the two swept nodes (indices authoritative).
dk = labels_for(N_osc)
lbl_x = dk[node_x] if 0 <= node_x < len(dk) else f"node {node_x}"
lbl_y = dk[node_y] if 0 <= node_y < len(dk) else f"node {node_y}"
axis_x = f"Node {node_x} ({lbl_x})  X perturbation"
axis_y = f"Node {node_y} ({lbl_y})  X perturbation"
extent = [grid_lo, grid_hi, grid_lo, grid_hi]

print(f"[data]     grid={m}²  N={N_osc}  {coupling_str}  "
      f"nodes=({node_x},{node_y})  D_f={fractal_dim:.4f}  R²={r_squared:.4f}")
if not config_present:
    print("[warn]     no live config in npz; regenerate with the updated "
          "lorenz_sweep.py to embed coupling + swept-node ids.")

# -- 1. BASIN MAP (K-MEANS) ------------------------------------
print("\nPlotting basin map ...")
fig_bm, ax_bm = plt.subplots(figsize=(7, 6))
im = ax_bm.imshow(
    labels, origin="lower", cmap="tab20",
    extent=extent, interpolation="nearest",
)
ax_bm.set_xlabel(axis_x)
ax_bm.set_ylabel(axis_y)
ax_bm.set_title(
    f"K-Means Basin Map (k={len(np.unique(labels))})\n"
    f"N={N_osc}  {coupling_str}  grid={m}²"
)
plt.colorbar(im, ax=ax_bm, label="Basin label")
plt.tight_layout()
bm_path = get_plot_path("plot_lorenz_basins", f"basin_map_kmeans_{ts}.png", SAVE_DIR)
plt.savefig(bm_path, dpi=150)
plt.close(fig_bm)
print(f"[saved]    {bm_path}")

# -- 2. BASIN BOUNDARY -----------------------------------------
print("Plotting basin boundary ...")
fig_bd, ax_bd = plt.subplots(figsize=(7, 6))
ax_bd.imshow(
    boundary, origin="lower", cmap="binary",
    extent=extent, interpolation="nearest",
)
ax_bd.set_xlabel(axis_x)
ax_bd.set_ylabel(axis_y)
ax_bd.set_title(
    f"Basin Boundary — DTI_A\n"
    f"N={N_osc}  {coupling_str}  grid={m}²"
)
plt.tight_layout()
bd_path = get_plot_path("plot_lorenz_basins", f"basin_boundary_{ts}.png", SAVE_DIR)
plt.savefig(bd_path, dpi=150)
plt.close(fig_bd)
print(f"[saved]    {bd_path}")

# -- 3. BOX-COUNTING LOG-LOG -----------------------------------
print("Plotting box-counting log-log ...")
mask = boxcount_n > 0
log_r_fit = np.log(boxcount_r[mask].astype(float))
log_n_fit = np.polyval(
    np.polyfit(log_r_fit, np.log(boxcount_n[mask].astype(float)), 1),
    log_r_fit,
)
fig_bc, ax_bc = plt.subplots(figsize=(6, 5))
ax_bc.loglog(boxcount_r[mask], boxcount_n[mask], "o-",
             color="crimson", label="box count")
ax_bc.loglog(boxcount_r[mask], np.exp(log_n_fit), "--", color="navy",
             label=f"fit  D_f={fractal_dim:.3f}  R²={r_squared:.3f}")
ax_bc.set_xlabel("Box size r")
ax_bc.set_ylabel("Box count N(r)")
ax_bc.set_title("Box-Counting — Fractal Dimension")
ax_bc.legend()
ax_bc.grid(True, which="both", alpha=0.3)
plt.tight_layout()
bc_path = get_plot_path("plot_lorenz_basins", f"boxcount_loglog_{ts}.png", SAVE_DIR)
plt.savefig(bc_path, dpi=150)
plt.close(fig_bc)
print(f"[saved]    {bc_path}")

# -- 4. ELBOW CURVE (if available in data) ----------------------
# The elbow curve is not saved in the npz; it's generated during
# the run. We skip it here since it's a diagnostic plot from the
# main pipeline, not a re-plot from saved data.

# -- 5. SUMMARY STATS -------------------------------------------
print(f"\n-- Basin Statistics ------------------------------")
print(f"Grid resolution  : {m}x{m} = {B} initial conditions")
print(f"Number of nodes  : {N_osc}")
print(f"Coupling         : {coupling_str}")
print(f"Swept nodes      : {node_x} ({lbl_x})  vs  {node_y} ({lbl_y})")
print(f"Fractal dim D_f  : {fractal_dim:.4f}  (R² = {r_squared:.4f})")
print(f"K-means K        : {len(np.unique(labels))} basins")
print(f"Timestamp        : {ts}")