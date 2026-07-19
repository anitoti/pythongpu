#!/usr/bin/env python3
"""
Spectral clustering diagnostics for a DTI matrix (.mat).

New features:
 - Compute SHA256 fingerprint of the adjacency matrix and write a repo-tracked index (dti_spectra_index.json) so repeated analyses on the same matrix are skipped.
 - More robust k-selection: eigengap + silhouette sweep, record both candidates and chosen k.
 - Modularity calculation via networkx for the found partition.
 - PDF report combining plots and textual summary (output/report.pdf).
 - If the node count matches the Desikan-Killiany 83 nodes, annotated label outputs are saved.

Usage:
    python3 scripts/run_dti_spectral.py --mat data/DTI-og.mat
    python3 scripts/run_dti_spectral.py --mat data/DTI-sub-01.mat --outdir output/sub-01 --force
"""

from pathlib import Path
import argparse
import numpy as np
from scipy.io import loadmat
import hashlib
import json
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Optional heavy deps installed at runtime if missing

def ensure_package(pkg):
    try:
        __import__(pkg)
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', pkg])

ensure_package('scikit_learn')
ensure_package('networkx')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx

# pythongpu label helper (only used if available)
try:
    from pythongpu.networks.desikan_killiany import labels_for
    _have_labels = True
except Exception:
    _have_labels = False

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mat", default="data/DTI-og.mat", help="path to .mat file (variable 'A')")
parser.add_argument("--var", default="A", help="variable name inside the .mat file")
parser.add_argument("--outdir", default="output/dti_spectral", help="output directory")
parser.add_argument("--k", type=int, default=None, help="force number of spectral clusters (default: eigengap+silhouette heuristic)")
parser.add_argument("--max-k", type=int, default=10, help="max k to consider for silhouette sweep")
parser.add_argument("--index-file", default="dti_spectra_index.json", help="repo-tracked index JSON path")
parser.add_argument("--force", action='store_true', help="force recompute even if matrix is in the index")
args = parser.parse_args()

mat_path = Path(args.mat)
if not mat_path.exists():
    raise SystemExit(f"mat file not found: {mat_path}")

out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)
plots = out / "plots"
plots.mkdir(exist_ok=True)

# Load matrix
mat = loadmat(str(mat_path))
if args.var in mat:
    A = np.asarray(mat[args.var]).astype(np.float64)
else:
    # pick first non-private var
    A = next(v for k,v in mat.items() if not k.startswith('_'))
    A = np.asarray(A).astype(np.float64)

# canonical preprocessing: symmetrise, zero diagonal
A = 0.5 * (A + A.T)
np.fill_diagonal(A, 0.0)
A_w = A.copy()

# fingerprint (hash) of matrix contents + shape
buf = np.ascontiguousarray(A_w, dtype=np.float64).tobytes()
sha = hashlib.sha256(buf + str(A_w.shape).encode()).hexdigest()

index_path = Path(args.index_file)
if index_path.exists():
    idx = json.loads(index_path.read_text())
else:
    idx = {}

if sha in idx and not args.force:
    entry = idx[sha]
    print(f"Matrix already indexed (sha={sha}).\nStored entry: {entry['tag'] if 'tag' in entry else entry.get('filename', '')}")
    print('To recompute, run with --force')
    # write a small pointer to existing outputs
    (out / 'index_pointer.txt').write_text(json.dumps(entry, indent=2))
    raise SystemExit('Existing entry found; aborting (use --force to recompute)')

# Build graph Laplacian
D = np.diag(A_w.sum(axis=1))
L = D - A_w

# normalized Laplacian
deg = A_w.sum(axis=1)
with np.errstate(divide='ignore'):
    inv_sqrt = np.where(deg>0, 1.0/np.sqrt(deg), 0.0)
D_inv_sqrt = np.diag(inv_sqrt)
L_norm = np.eye(A_w.shape[0]) - D_inv_sqrt @ A_w @ D_inv_sqrt

# eigen decomposition
eigvals_A, eigvecs_A = np.linalg.eigh(A_w)
eigvals_L, eigvecs_L = np.linalg.eigh(L)
eigvals_Ln, eigvecs_Ln = np.linalg.eigh(L_norm)

# basic stats
n = A_w.shape[0]
edges = int((A_w>0).sum()//2)
deg_min, deg_med, deg_max = float(deg.min()), int(np.median(deg)), float(deg.max())
alg_conn = float(np.sort(eigvals_L)[1])
spectral_radius = float(np.max(np.abs(eigvals_A)))

# save raw spectra
np.save(out / 'eigvals_A.npy', eigvals_A)
np.save(out / 'eigvecs_A.npy', eigvecs_A)
np.save(out / 'eigvals_L.npy', eigvals_L)
np.save(out / 'eigvecs_L.npy', eigvecs_L)
np.save(out / 'eigvals_Lnorm.npy', eigvals_Ln)
np.save(out / 'eigvecs_Lnorm.npy', eigvecs_Ln)

np.savetxt(out / 'eigvals_A.txt', eigvals_A)
np.savetxt(out / 'eigvals_L.txt', eigvals_L)
np.savetxt(out / 'eigvals_Lnorm.txt', eigvals_Ln)

# plots: adjacency heatmap
plt.figure(figsize=(6,6))
plt.imshow(A_w, cmap='viridis', interpolation='nearest')
plt.title('Adjacency (weighted)')
plt.colorbar()
plt.tight_layout()
plt.savefig(plots / 'adjacency_heatmap.png', dpi=150)
plt.close()

# Laplacian spectrum plot
vals = np.sort(eigvals_L)
plt.figure(figsize=(6,3))
plt.plot(vals, marker='o')
plt.xlabel('index')
plt.ylabel('Laplacian eigenvalue')
plt.title('Laplacian spectrum')
plt.tight_layout()
plt.savefig(plots / 'laplacian_spectrum.png', dpi=150)
plt.close()

# eigengap candidate
m = min(args.max_k+1, len(vals))
diffs = np.diff(vals[:m])
if len(diffs) >= 2:
    gap_idx = int(np.argmax(diffs[1:]) + 1)
else:
    gap_idx = 1
k_eig = max(2, gap_idx+1)

# silhouette sweep for k=2..max_k
k_range = list(range(2, min(args.max_k, n-1)+1))
best_sil = -1.0
best_k_sil = None
sil_scores = {}
for kk in k_range:
    idxs = np.argsort(eigvals_L)[1:kk]
    if idxs.size == 0:
        continue
    emb = eigvecs_L[:, idxs]
    try:
        km = KMeans(n_clusters=kk, random_state=0, n_init=10).fit(emb)
        labels_k = km.labels_
        if emb.shape[0] > kk:
            sc = silhouette_score(emb, labels_k)
        else:
            sc = float('nan')
    except Exception:
        sc = float('nan')
    sil_scores[kk] = sc
    if not np.isnan(sc) and sc > best_sil:
        best_sil = sc
        best_k_sil = kk

# decide k: prefer silhouette if available, otherwise eigengap
if args.k is not None:
    k_chosen = int(args.k)
else:
    if best_k_sil is not None:
        k_chosen = int(best_k_sil)
    else:
        k_chosen = int(k_eig)

# compute final embedding and clustering
idxs_final = np.argsort(eigvals_L)[1:k_chosen]
if idxs_final.size == 0:
    idxs_final = np.argsort(eigvals_L)[1:k_chosen+1]
embedding = eigvecs_L[:, idxs_final]
km_final = KMeans(n_clusters=k_chosen, random_state=0, n_init=10).fit(embedding)
labels = km_final.labels_

# silhouette of chosen
silhouette_chosen = float('nan')
if embedding.shape[0] > k_chosen:
    try:
        silhouette_chosen = float(silhouette_score(embedding, labels))
    except Exception:
        silhouette_chosen = float('nan')

# modularity via networkx
G = nx.from_numpy_array(A_w)
# build communities as list of sets
communities = [set(np.where(labels==c)[0].tolist()) for c in range(k_chosen)]
try:
    modularity = nx.algorithms.community.quality.modularity(G, communities, weight='weight')
except Exception:
    modularity = float('nan')

# save labels and cluster assignments
np.savetxt(out / 'spectral_labels.txt', labels, fmt='%d')
with open(out / 'cluster_assignments.tsv', 'w') as f:
    f.write('node\tcluster\n')
    for i, c in enumerate(labels):
        f.write(f'{i}\t{c}\n')

# if Desikan-Killiany 83 nodes, save labels
if n == 83 and _have_labels:
    lut = labels_for(83)
    with open(out / 'node_labels.tsv', 'w') as f:
        f.write('node\tlabel\n')
        for i, lab in enumerate(lut):
            f.write(f'{i}\t{lab}\n')
    # also save cluster->region summary
    with open(out / 'cluster_regions.tsv', 'w') as f:
        f.write('cluster\tregions\n')
        for c in range(k_chosen):
            nodes = np.where(labels==c)[0].tolist()
            regions = [lut[i] for i in nodes]
            f.write(f"{c}\t{','.join(regions)}\n")

# Fiedler vector plot
fiedler = eigvecs_L[:, np.argsort(eigvals_L)[1]]
plt.figure(figsize=(8,3))
if n == 83 and _have_labels:
    labels_lut = labels_for(83)
    plt.bar(range(n), fiedler)
    plt.xticks(range(n), [l[:6] for l in labels_lut], rotation=90)
else:
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
plt.title(f'Adjacency reordered by spectral clusters (k={k_chosen})')
plt.colorbar()
plt.tight_layout()
plt.savefig(plots / 'adjacency_reordered.png', dpi=150)
plt.close()

# produce PDF report
report_path = out / 'report.pdf'
with PdfPages(report_path) as pdf:
    # page 1: summary text
    fig = plt.figure(figsize=(8.5, 11))
    txt = fig.text(0.01, 0.98, 'DTI spectral diagnostics', fontsize=14, va='top')
    info_lines = [
        f'file: {str(mat_path)}',
        f'sha256: {sha}',
        f'date: {datetime.datetime.now().isoformat()}',
        f'n = {n}',
        f'edges = {edges}',
        f'deg min/med/max = {deg_min}/{deg_med}/{deg_max}',
        f'algebraic_connectivity = {alg_conn:.6f}',
        f'spectral_radius = {spectral_radius:.6f}',
        f'k_eig = {k_eig}',
        f'best_k_sil = {best_k_sil}',
        f'k_chosen = {k_chosen}',
        f'silhouette_chosen = {silhouette_chosen}',
        f'modularity = {modularity}',
    ]
    fig.text(0.01, 0.90, '\n'.join(info_lines), fontsize=10, va='top')
    pdf.savefig(fig)
    plt.close(fig)

    # page: adjacency heatmap
    img = plt.imread(str(plots / 'adjacency_heatmap.png'))
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(img)
    ax.axis('off')
    pdf.savefig(fig)
    plt.close(fig)

    # laplacian spectrum
    img = plt.imread(str(plots / 'laplacian_spectrum.png'))
    fig, ax = plt.subplots(figsize=(8,3))
    ax.imshow(img)
    ax.axis('off')
    pdf.savefig(fig)
    plt.close(fig)

    # embedding
    img = plt.imread(str(plots / 'spectral_embedding.png'))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.imshow(img)
    ax.axis('off')
    pdf.savefig(fig)
    plt.close(fig)

    # reordered adjacency
    img = plt.imread(str(plots / 'adjacency_reordered.png'))
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img)
    ax.axis('off')
    pdf.savefig(fig)
    plt.close(fig)

# update index
outputs_list = []
for p in [out, plots, report_path]:
    try:
        rel = p.resolve().relative_to(Path.cwd())
        outputs_list.append(str(rel))
    except Exception:
        outputs_list.append(str(p.resolve()))

entry = {
    'filename': str(mat_path),
    'sha256': sha,
    'n': int(n),
    'edges': int(edges),
    'deg_min': deg_min,
    'deg_med': deg_med,
    'deg_max': deg_max,
    'algebraic_connectivity': alg_conn,
    'spectral_radius': spectral_radius,
    'k_eig': int(k_eig),
    'best_k_sil': int(best_k_sil) if best_k_sil is not None else None,
    'k_chosen': int(k_chosen),
    'silhouette_chosen': silhouette_chosen,
    'modularity': modularity,
    'outputs': outputs_list,

    'timestamp': datetime.datetime.now().isoformat(),
}
idx[sha] = entry
index_path.write_text(json.dumps(idx, indent=2))

print('Saved outputs and updated index at', index_path)
