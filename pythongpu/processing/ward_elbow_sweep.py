"""
Ward parcellation elbow sweep: fit Ward clustering at multiple scales
and plot reconstruction error (variance not explained) to identify
the optimal number of parcels.
"""

import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend
import matplotlib.pyplot as plt
from nilearn import image, masking
from nilearn.regions import Parcellations
from tqdm import tqdm

from pythongpu.utils import get_plot_path


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
NIFTI_PATH = os.path.join(PROJECT_ROOT, "data",
                          "rfMRI_REST1_LR_hp2000_clean.nii.gz")
OUT_PNG = get_plot_path("ward_elbow_sweep", "ward_elbow_curve.png")
LOG_FILE = os.path.join(PROJECT_ROOT, "data", "derivatives",
                        "sweep_log.txt")

# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
N_PARCELLATIONS = np.arange(50, 501, 10)   # 50 … 500, step 10
RANDOM_STATE = 42


def reconstruction_error(labels, data):
    """Total within-cluster sum of squared distances (inertia).

    Parameters
    ----------
    labels : ndarray, shape (n_voxels,)
        Cluster assignment for each voxel.
    data : ndarray, shape (n_voxels, n_timepoints)
        Voxel time series (already masked).

    Returns
    -------
    inertia : float
        Sum over all clusters of the sum of squared Euclidean distances
        from each voxel to its cluster centroid time series.
    """
    inertia = 0.0
    for k in np.unique(labels):
        cluster_mask = labels == k
        cluster_data = data[cluster_mask]            # (n_k, n_timepoints)
        centroid = cluster_data.mean(axis=0)         # (n_timepoints,)
        diff = cluster_data - centroid               # (n_k, n_timepoints)
        inertia += np.sum(diff ** 2)
    return inertia


# ---------------------------------------------------------------------------
# load & prepare data
# ---------------------------------------------------------------------------
print("Loading 4D NIfTI …")
func_img = image.load_img(NIFTI_PATH)

# compute a quick brain mask from the mean image
print("Computing EPI mask …")
mean_img = image.mean_img(func_img)
mask_img = masking.compute_epi_mask(mean_img)

# extract masked voxel time series   (n_voxels, n_timepoints)
print("Applying mask …")
data_masked = masking.apply_mask(func_img, mask_img)
n_timepoints, n_voxels = data_masked.shape
print(f"  {n_voxels} brain voxels × {n_timepoints} timepoints")

# also keep the label volume template for future label image saving
# (just for reference, not used further)
print()

# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
n_parcels_list = []
errors = []

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

for n_parcels in tqdm(N_PARCELLATIONS, desc="Ward sweep", unit="parcels"):
    ward = Parcellations(
        method="ward",
        n_parcels=n_parcels,
        mask=mask_img,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    ward.fit(func_img)

    # labels_img_  : Nifti1Image of integer labels in voxel space
    labels_img = ward.labels_img_

    # apply the same brain mask to labels so shape matches data_masked
    labels_flat = masking.apply_mask(labels_img, mask_img).astype(np.int64)

    err = reconstruction_error(labels_flat, data_masked.T)

    n_parcels_list.append(n_parcels)
    errors.append(err)

    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp}  {n_parcels:4d}  {err:.3e}\n")

# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_parcels_list, errors, "o-", color="steelblue", markersize=4)
ax.set_xlabel("Number of parcels (Ward)")
ax.set_ylabel("Reconstruction error (inertia)")
ax.set_title("Ward parcellation elbow curve")
ax.grid(True, linestyle=":", alpha=0.6)

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_PNG}")
plt.close(fig)