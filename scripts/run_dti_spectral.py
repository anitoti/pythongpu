#!/usr/bin/env python3
"""
Spectral clustering diagnostics for the professor-provided DTI (data/DTI-og.mat).

Produces:
 - adjacency and Laplacian eigenvalues/eigenvectors (raw and normalized Laplacian)
 - plots: adjacency heatmap, adjacency reordered by clusters, eigenvalue spectrum, eigengap, embedding scatter, Fiedler vector
 - spectral clustering labels (k selected by eigengap by default) and diagnostics (silhouette)

Usage:
    python3 scripts/run_dti_spectral.py
    python3 scripts/run_dti_spectral.py --mat data/DTI-og.mat --outdir output/dti_spectral --k 4
"""

from pathlib import Path
import argparse
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mat", default="data/DTI-og.mat", help="path to .mat file (variable 'A')")
parser.add_argument("--var", default="A", help="variable name inside the .mat file")
parser.add_argument("--outdir", default="output/dti_spectral", help="output directory")
parser.add_argument("--k", type=int, default=None, help="force number of spectral clusters (default: eigengap heuristic)")
parser.add_argument("--max-k", type=int, default=10, help="max k to consider for eigengap heuristic")
args = parser.parse_args()

out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)
plots = out / "plots"
plots.mkdir(exist_ok=True)

mat = loadmat(args.mat)
A = np.asarray(mat[args.var]).astype(np.float64)
A = 0.5 * (A + A.T)
np.fill_diagonal(A, 0.0)

# weighted (here binary) adjacency
A_w = A.copy()

# Laplacian
D = np.diag(A_w.sum(axis=1))
L = D - A_w

# normalized symmetric Laplacian
deg = A_w.sum(axis=1)
with np.errstate(divide='ignore'):
    inv_sqrt = np.where(deg>0, 1.0/np.sqrt(deg), 0.0)
D_inv_sqrt = np.diag(inv_sqrt)
L_norm = np.eye(A.shape[0]) - D_inv_sqrt @ A_w @ D_inv_sqrt

# eigen decomposition
eigvals_A, eigvecs_A = np.linalg.eigh(A_w)
eigvals_L, eigvecs_L = np.linalg.eigh(L)
eigvals_Ln, eigvecs_Ln = np.linalg.eigh(L_norm)

# save spectra
np.save(out / 'eigvals_A.npy', eigvals_A)
np.save(out / 'eigvals_L.npy', eigvals_L)
np.save(out / 'eigvals_Lnorm.npy', eigvals_Ln)

np.savetxt(out / 'eigvals_A.txt', eigvals_A)
np.savetxt(out / 'eigvals_L.txt', eigvals_L)
np.savetxt(out / 'eigvals_Lnorm.txt', eigvals_Ln)

n = A.shape[0]
edges = int((A_w>0).sum()//2)
info_text = (
    f'n={n}\nedges={edges}\n'
    f'deg min/med/max = {deg.min()}/{int(np.median(deg))}/{deg.max()}\n'
    f'algebraic connectivity (Fiedler) = {np.sort(eigvals_L)[1]:.6f}\n'
)
(out / 'summary.txt').write_text(info_text)
print(info_text)

# plots
plt.figure(figsize=(6,6))
plt.imshow(A_w, cmap='viridis', interpolation='nearest')
plt.title('Adjacency (weighted)')
plt.colorbar()
plt.tight_layout()
plt.savefig(plots / 'adjacency_heatmap.png', dpi=150)
plt.close()

# eigenvalue spectrum (Laplacian)
plt.figure(figsize=(6,3))
vals = np.sort(eigvals_L)
plt.plot(vals, marker='o')
plt.xlabel('index')
plt.ylabel('Laplacian eigenvalue')
plt.title('Laplacian spectrum')
plt.tight_layout()
plt.savefig(plots / 'laplacian_spectrum.png', dpi=150)
plt.close()

# eigengap heuristic
k_choice = args.k
if k_choice is None:
    lam = vals
    m = min(args.max_k+1, len(lam))
    diffs = np.diff(lam[:m])
    # skip the trivial first gap (between 0 and lambda1) when choosing k>1
    if len(diffs) >= 2:
        gap_idx = np.argmax(diffs[1:]) + 1
    else:
        gap_idx = 1
    k_choice = max(2, gap_idx+1)

plt.figure(figsize=(6,3))
plt.plot(np.arange(len(vals)), vals, marker='o')
plt.vlines(np.arange(len(vals))[1:args.max_k+1], ymin=vals.min(), ymax=vals.max(), colors='lightgray', linewidth=0.5)
plt.title(f'Laplacian spectrum (chosen k={k_choice})')
plt.tight_layout()
plt.savefig(plots / 'laplacian_spectrum_chosen_k.png', dpi=150)
plt.close()

# spectral clustering using first k eigenvectors (skip trivial first)
k = int(k_choice)
idxs = np.argsort(eigvals_L)[1:k]
if idxs.size == 0:
    idxs = np.argsort(eigvals_L)[1:k+1]
embedding = eigvecs_L[:, idxs]

# ensure sklearn available
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'scikit-learn'])
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(embedding)
labels = km.labels_

s_score = silhouette_score(embedding, labels) if k>1 and embedding.shape[0]>k else float('nan')

# save labels
np.savetxt(out / 'spectral_labels.txt', labels, fmt='%d')

# plot embedding scatter
plt.figure(figsize=(5,4))
if embedding.shape[1] >= 2:
    plt.scatter(embedding[:,0], embedding[:,1], c=labels, cmap='tab10')
    plt.xlabel('eigvec 1')
    plt.ylabel('eigvec 2')
else:
    plt.scatter(np.arange(len(embedding)), embedding[:,0], c=labels, cmap='tab10')
    plt.xlabel('node')
    plt.ylabel('eigvec 1')
plt.title(f'Spectral embedding (k={k})  silhouette={s_score:.3f}')
plt.tight_layout()
plt.savefig(plots / 'spectral_embedding.png', dpi=150)
plt.close()

# fiedler vector
fiedler = eigvecs_L[:, np.argsort(eigvals_L)[1]]
plt.figure(figsize=(8,3))
plt.bar(range(n), fiedler)
plt.xlabel('Node index')
plt.ylabel('Fiedler vector')
plt.tight_layout()
plt.savefig(plots / 'fiedler_vector.png', dpi=150)
plt.close()

# adjacency reordered by cluster
order = np.argsort(labels)
A_re = A_w[order][:, order]
plt.figure(figsize=(6,6))
plt.imshow(A_re, cmap='viridis', interpolation='nearest')
plt.title(f'Adjacency reordered by spectral clusters (k={k})')
plt.colorbar()
plt.tight_layout()
plt.savefig(plots / 'adjacency_reordered.png', dpi=150)
plt.close()

# write diagnostics
with open(out / 'diagnostics.txt', 'w') as f:
    f.write(f'k={k}\n')
    f.write(f'silhouette={s_score}\n')
    f.write(f'algebraic_connectivity={np.sort(eigvals_L)[1]:.6f}\n')

print(f'Done. Results in {out.resolve()}')
