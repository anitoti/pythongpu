# Script Manifest — `~/pythongpu`

> **⚠️ SUPERSEDED (2026-07-11).** This is a point-in-time snapshot that predates
> the package restructure. The repository now uses a **flat layout**: the
> importable package `pythongpu/` lives at the repo root (no `src/` prefix), the
> experiment scripts live in `pythongpu/pipeline/`, imports are `from pythongpu.…`,
> and a real `pyproject.toml` exists. Paths and import conventions described below
> are historical. See **[README.md](README.md)** for the current structure and how
> to run everything. Many "reorg risks" flagged here have since been resolved.

Generated 2026-07-10. Covers every `.py` / `.sh` file under `~/pythongpu` (excludes `.venv/`, `nilearn_cache/`, `__pycache__/`). **No `.slurm`/`.sbatch` files exist** — SLURM directives live inline at the top of `run_lorenz_gpu.sh` (`#SBATCH ...`), so that file *is* the cluster submission script. No `pyproject.toml`/`requirements.txt`/`setup.py` exists anywhere — dependency list below is inferred from imports only.

Legend: **Category** groups files by function. **Imports** are first-party (`pythongpu.*`) vs third-party. **Status** flags anything that will break during a reorg.

---

## 1. Simulation core (dynamical systems, GPU integrators)

| Path | Key imports | Notes |
|---|---|---|
| `src/pythongpu/kuramoto.py` | `torch`, `numpy` | `KuramotoSimulator` class, CUDA-capable |
| `src/pythongpu/kuramoto_gnm.py` | `kuramoto` (bare), `gnm_random_graph` (bare), `.kuramoto`, `networkx` | **BROKEN**: mixes bare (`import kuramoto`) and relative (`from .kuramoto import ...`) imports of the same module — the bare imports will fail unless run from inside `src/pythongpu/` with that dir on `PYTHONPATH`. Looks like a scratch/demo file, not wired into the package. |
| `src/pythongpu/rossler.py` | `torch` | `RosslerNetwork` class |
| `rossler.py` (root) | — (no imports; empty/stub) | Root-level stub, likely superseded by `src/pythongpu/rossler.py` — confirm before deleting |
| `src/pythongpu/simulate_network.py` | `torch`, `.utils` | Rössler ODE + `simulate()`, used by `run_sim.py` |
| `src/pythongpu/gnm_random_graph.py` | `networkx`, `torch`, `numpy`, `.utils`, `matplotlib` (Agg) | `generate_gnm_laplacian()` — graph generator, mixes plotting concern in (side-effect: saves a figure) |
| `src/pythongpu/lorenz_basins.py` | `os`, `math`, `numpy`, `torch`, `networkx`, `dataclasses` | Lorenz IC sweep → basin-of-attraction map (fractal dim via box-counting downstream) |
| `src/pythongpu/lorenz_vps_clustering_xcoupled.py` | (see file header) | Lorenz VPS clustering, x-coupled variant. Nimble Brain / Fish lab. |
| `src/pythongpu/lorenz_vps_clustering_z.py` | (see file header) | Lorenz VPS clustering, z-coupled variant |
| `src/pythongpu/rossler_vps_clustering.py` | (see file header) | Rössler VPS clustering |
| `src/pythongpu/kmeans.py` | — (stub/empty) | Referenced conceptually by clustering scripts; verify it's not dead code |
| `src/pythongpu/vps.py` + `scripts/run_vps.py` | `argparse`, `numpy`, `scipy.io`, `torch`, `sklearn.cluster.KMeans`, `pythongpu.vps.main` | VPS = "velocity phase space" (?) clustering entry point |

**⚠️ Cluster-execution break:** `git status` shows `src/pythongpu/lorenz_vps_clustering.py` was **deleted** (presumably split into the `_xcoupled` / `_z` variants above), but both `run_lorenz_gpu.sh` and `sweep_coupling.sh` (below) still invoke the old filename directly. Both SLURM/sweep entry points are currently broken until repointed at `lorenz_vps_clustering_xcoupled.py` or `_z.py` (or a merged replacement).

---

## 2. GPU/Slurm cluster entry points

| Path | Invokes | Notes |
|---|---|---|
| `run_lorenz_gpu.sh` | `python3 lorenz_vps_clustering.py --grid-n 361 --k-clusters 4 --coupling 0.5 --dti-path data/DTI_A.mat --outdir data/` | **SLURM batch script** (`#SBATCH` header: `gpu` partition, 1 GPU, 32G, 2h). Activates `fmri_env/` venv (not `.venv/` — separate env, confirm it still exists on cluster). Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **Broken path** — see §1 warning; also calls script from repo root, not `src/pythongpu/`. |
| `sweep_coupling.sh` | Loops `python ./src/pythongpu/lorenz_vps_clustering.py` over coupling 0.00→1.00 (101 runs) | Plain bash, no SBATCH header — likely run interactively or wrapped by another submission script. Same **broken path** issue, and references `src/pythongpu/` (inconsistent with `run_lorenz_gpu.sh`'s root-relative call). |

**When reorganizing:** both scripts hardcode relative paths (`data/DTI_A.mat`, `data/`, `fmri_env/`) resolved from `$SLURM_SUBMIT_DIR` / cwd at submission time — any module move must keep these paths valid relative to wherever the script is submitted from, or the paths need to become absolute/env-var-driven.

---

## 3. Data preprocessing / ingestion (fMRI, DTI)

| Path | Key imports | Notes |
|---|---|---|
| `src/pythongpu/extract_parcellated_signals.py` | `argparse`, `logging`, `nibabel`, `pandas`, `nilearn.regions.Parcellations` | Extracts ROI time series from NIfTI via Ward/other parcellation |
| `src/pythongpu/fmri_ward_elbow_sweep.py` | `numpy`, `matplotlib`, `nilearn.image`, `nilearn.masking`, `nilearn.regions.Parcellations`, `tqdm` | Elbow-method sweep to pick parcel count (see `pipeline_notes.md` — this is the "(2)" design fork the author was weighing) |
| `src/pythongpu/inspect_data.py` + `scripts/run_inspect.py` | `argparse`, `nibabel`, `numpy`, `pythongpu.inspect_data.main` | NIfTI/data sanity-check CLI |
| `src/pythongpu/correlate_fmri.py` + `scripts/run_correlate_fmri.py` | `argparse`, `nibabel`, `numpy`, `torch`, `pythongpu.correlate_fmri.main` | Builds correlation matrices from fMRI signals |
| `src/pythongpu/fmri_causation_entropy.py` | `argparse`, `logging`, `concurrent.futures`, `numpy`, `pandas`, `seaborn`, `matplotlib`, `scipy.stats.chi2`, `tqdm` | Causal/directed-edge entropy regression — implements the adjacency-matrix design discussed in `pipeline_notes.md`. Mixes analysis + plotting (seaborn/matplotlib) in one file. |
| `src/pythongpu/sparse_brain_sim.py` + `scripts/run_sparse_brain_sim.py` | `argparse`, `csv`, `dataclasses`, `numpy`, `torch`, `pythongpu.sparse_brain_sim.main` | "Memory-efficient sparse brain simulation" — fixes Laplacian diagonal-collision bug; comment notes it's meant to run local (Titan Xp) before scaling to the cluster |
| `src/pythongpu/plot_dti.py` | `matplotlib`, `numpy`, `scipy.io.loadmat` | Loads `.mat` DTI connectome data for plotting — arguably belongs in §4 (Plotting) rather than preprocessing; listed here because it's a DTI *loader* first |

Design notes captured in `src/pythongpu/pipeline_notes.md` (not code, but documents unresolved decisions about parcellation strategy and entropy-based edge construction — worth keeping attached to this group during reorg).

---

## 4. Plotting / visualization

| Path | Key imports | Notes |
|---|---|---|
| `box_counting.py` (root) | (see header) | Box-counting fractal-dimension fit. **Pipeline position: runs after `kuramoto_basins.py`.** Input `/data/kuramoto_basins.npy`, output `/data/box_counting_fit.png`. Hardcoded absolute-looking `/data/` path — verify this is `$CWD/data` or truly `/data` on the cluster. |
| `kuramoto_basins.py` (root) | `os`, `math`, `numpy`, `torch`, `networkx`, `dataclasses` | Root-level version of the Kuramoto basin sweep — likely predates/duplicates a `src/pythongpu` equivalent; no `src/pythongpu/kuramoto_basins.py` exists, so this may be the canonical copy that never got moved into the package. |
| `plot_kuramoto_basins.py` (root) **and** `src/pythongpu/plot_kuramoto_basins.py` | `numpy`, `matplotlib` | **Exact duplicate filenames in two locations** — diff them before consolidating; downstream of `kuramoto_basins.py`, input `/data/kuramoto_basins.npy`, outputs `basin_map_continuous.png` |
| `src/pythongpu/plot_lorenz_basins.py` | `os`, `re`, `glob`, `argparse`, `pathlib`, `datetime`, `numpy` | Replots precomputed `basin_data_{timestamp}.npz` (k-means clustered basin map + boundary) — reads back output of `lorenz_vps_clustering_*` scripts |
| `src/pythongpu/plot_matrix.py` + `scripts/run_plot_matrix.py` | `argparse`, `matplotlib`, `numpy`, `pythongpu.plot_matrix.main` | Generic matrix/heatmap plotter (likely for adjacency/correlation matrices from §3) |

---

## 5. Package scaffolding / shared utilities

| Path | Key imports | Notes |
|---|---|---|
| `src/pythongpu/__init__.py` | `.utils`, `.gnm_random_graph`, `.simulate_network` | Package entry point — only re-exports 3 of the ~25 modules in `src/pythongpu/`; most modules (kuramoto*, lorenz*, fmri*, plot*) are **not** exposed at package level and must be imported by submodule path |
| `src/pythongpu/utils.py` | `pathlib` | `get_laplacian()`, `get_clean_path()` — shared by simulation core and preprocessing |

---

## 6. CLI runners (`scripts/` wrappers + root duplicates)

| Path | Delegates to | Notes |
|---|---|---|
| `run_sim.py` (root) | `pythongpu.gnm_random_graph`, `src.pythongpu.simulate_network` | **Near-duplicate of `scripts/run_sim.py`** — diff shows root version imports `pathlib.Path` (unused-looking) and lacks `--device` `choices=`, `choices=["cpu","cuda"]`; scripts/ version has the stricter arg validation and a shebang. These have drifted — pick one canonical copy. |
| `scripts/run_sim.py` | same as above | See diff note above |
| `scripts/run_inspect.py` | `src.pythongpu.inspect_data.main` | Thin wrapper |
| `scripts/run_correlate_fmri.py` | `src.pythongpu.correlate_fmri.main` | Thin wrapper |
| `scripts/run_plot_matrix.py` | `src.pythongpu.plot_matrix.main` | Thin wrapper |
| `scripts/run_sparse_brain_sim.py` | `src.pythongpu.sparse_brain_sim.main` | Thin wrapper |
| `scripts/run_vps.py` | `src.pythongpu.vps.main` | Thin wrapper |

All six `scripts/*.py` follow the same `from src.pythongpu.X import main` + `if __name__ == "__main__": main()` pattern — these are consistent and reorg-friendly. The two `run_sim.py` copies are the outlier and should be reconciled into this pattern.

**Import-path inconsistency to resolve during reorg:** some files import as `src.pythongpu.X` (absolute from repo root — requires running from repo root or repo root on `PYTHONPATH`), others as `pythongpu.X` (requires the package installed / `src/` on `PYTHONPATH`), and `kuramoto_gnm.py` uses bare `import kuramoto`. A single install mode (e.g. `pip install -e .` with a real `pyproject.toml`, currently absent) would unify these.

---

## 7. Third-party dependency surface (aggregated from all imports above)

`torch`, `numpy`, `networkx`, `matplotlib`, `scipy` (`.io`, `.stats`), `scikit-learn` (`sklearn.cluster.KMeans`), `pandas`, `seaborn`, `tqdm`, `nibabel`, `nilearn` (`.image`, `.masking`, `.regions`). No `requirements.txt`/`pyproject.toml` pins any of these — worth generating one (`pip freeze` from the working `fmri_env`/`.venv`) as part of the reorg so the cluster env is reproducible.

---

## 8. Flagged risks before restructuring

1. **Broken cluster script target**: `lorenz_vps_clustering.py` deleted but still referenced by `run_lorenz_gpu.sh` and `sweep_coupling.sh` — fix before any file moves, or the reorg will "fix" paths for a script that never ran anyway and mask the real bug.
2. **Duplicate `run_sim.py`** (root vs `scripts/`) and **duplicate `plot_kuramoto_basins.py`** (root vs `src/pythongpu/`) — diff and collapse to one location.
3. **Mixed import conventions** (`src.pythongpu.X` / `pythongpu.X` / bare) — standardize via an installable package before moving directories, otherwise every move risks silently breaking a subset of scripts depending on which convention they used.
4. **`kuramoto_gnm.py`** has unresolvable bare imports as-is — confirm whether it's live code or a scratch/demo file before deciding where it belongs.
5. Root-level `rossler.py` is an empty stub shadowing `src/pythongpu/rossler.py` — confirm safe to delete.
6. Hardcoded `/data/` and `data/` paths (box_counting, kuramoto_basins, run_lorenz_gpu.sh) mix repo-relative and possibly cluster-absolute paths — audit each before moving directories so outputs keep landing where downstream plot scripts expect them.

---

## 9. PNG provenance — orphaned root/`data/` figures traced and consolidated (2026-07-11)

All `.py`/`.sh` output figures now go through `get_plot_path(script_name, filename, outdir)` (`pythongpu/utils/__init__.py`), which writes to `<outdir>/derivatives/<script_name>__<filename>`. 17 loose PNGs sitting at the repo root (7 git-tracked, 10 untracked) predated this convention — traced by matching plot titles/axis labels back to the generating script, then moved into `data/derivatives/` with the `<script>__<filename>` naming applied retroactively, and untracked from git (`data/` is gitignored):

| Original root file | Moved to | Traced to |
|---|---|---|
| `basin_boundary.png`, `basin_map_kmeans.png`, `boxcount_loglog.png`, `boxdiv2_synthetic.png`, `elbow_curve.png`, `rewired_adjacency.png` | `lorenz_vps_clustering__*` | `pythongpu/pipeline/lorenz_sweep.py` — plain filenames, title has `Basin Boundary — DTI_A` + `coupling=` |
| `network.png` | `random_graphs__network.png` | `pythongpu/networks/random_graphs.py` |
| `elbow_curve_20260629_130302.png` | `lorenz_vps_clustering__elbow_curve_20260629_130302.png` | live `lorenz_sweep.py` run — title `Elbow Method — Choose Optimal k` (only script with this exact title) |
| `basin_map_kmeans_20260629_130302.png`, `basin_boundary_20260629_130302.png`, `boxcount_loglog_20260629_130302.png` | `plot_lorenz_basins__*_20260629_130302.png` | `pythongpu/visualization/plot_lorenz_basins.py` replotting `basin_data_20260629_130302.npz` — titles omit `coupling=` (script has no access to it, only re-derives `N`/`grid` from the npz) |
| `elbow_curve_20260629_132752.png`, `basin_map_kmeans_20260629_132752.png`, `basin_boundary_20260629_132752.png`, `boxcount_loglog_20260629_132752.png` | `lorenz_vps_clustering__*_20260629_132752.png` | live `lorenz_sweep.py` run — all 4 titles include `coupling=0.5`, so this is a direct run, not a `plot_lorenz_basins.py` replay |
| `elbow_curve_20260629_134733.png`, `basin_map_kmeans_20260629_134733.png` | `lorenz_vps_clustering__*_20260629_134733.png` | live `lorenz_sweep.py` run, `coupling=1.0` `grid=128²` — only 2 of the usual 6 outputs exist, so this run stopped/crashed after the k-means plot, before `basin_boundary` |

Note: `pythongpu/pipeline/lorenz_sweep.py` and its current predecessor `pythongpu/lorenz_vps_clustering.py` (at `HEAD`) both write **plain** filenames only — neither has timestamp logic today. The `_130302`/`_132752`/`_134733` files must come from an intermediate, uncommitted edit of that script that *did* timestamp its outputs; that edit isn't in git history to point to directly, hence attribution above is by title/content matching rather than a line reference.

Four more loose files under `data/` (not `data/derivatives/`) were also unambiguous duplicates of already-identified generators and were consolidated the same way: `data/box_counting_fit.png` → `box_counting_kuramoto__box_counting_fit.png` (`pythongpu/pipeline/box_counting_kuramoto.py`, `suptitle` = `Nimble Brain — Fractal Basin Analysis`), `data/brain_connectivity.png` → `plot_matrix__brain_connectivity.png` (`pythongpu/visualization/plot_matrix.py`, title = `Brain Connectivity Adjacency Matrix Subset`), `data/dti_visual.png` → `plot_dti__dti_visual_legacy.png` (`pythongpu/visualization/plot_dti.py`; `plot_dti__dti_visual.png` already existed in `derivatives/` from a newer run, so this one is suffixed `_legacy`), `data/processed/100307/multifractal_spectrum.png` → `multifractal_analysis__multifractal_spectrum_100307.png` (`pythongpu/processing/multifractal_analysis.py`).

**Update (same day, follow-up pass):** three more stray root-level files were found and relocated: `basin_data.npz`, `basin_data_20260629_130302.npz`, `basin_data_20260629_132752.npz` were sitting at the repo root instead of `data/`, where `plot_lorenz_basins.py`'s `_find_latest_npz()` glob (`basin_data_*.npz`) and default `SAVE_DIR` actually look for them — moved into `data/`. Confirmed via `.npz` contents (`A_dti`/`A_rewired`, 83-node Laplacian, matching `Xg`/`Yg` grid shapes) that they're the exact source data behind the `lorenz_vps_clustering__*`/`plot_lorenz_basins__*` PNGs already sorted above.

**Orphaned family — now attributed, quarantined not renamed:** `basin_boundary.png`, `basin_map_kmeans.png`, `boxcount_loglog.png`, `boxdiv2_synthetic.png`, `elbow_curve.png`, `rewired_adjacency.png`, `basin_map_continuous.png`, `lorenz_basin_map_continuous{,_1,_2}.png`, `lorenz_basin_map_kmeans{,_1,_2}.png` — moved from loose `data/` into `data/legacy_unsourced/`. These use a different node-pair labeling (`Node 73`/`Node 81`, `K_coup=...`) that matches no *currently plotting* script, but tracing `slice_node_x=73, slice_node_y=81, n_nodes=100, coupling=0.0001` in `data/lorenz_basins.npz`'s stored `config` dict against `pythongpu/pipeline/lorenz_basins_sweep.py`'s `Config` dataclass (same field names, same defaults) confirms the *simulation* came from that script. However `lorenz_basins_sweep.py` only writes the `.npz` — it has no `savefig`/`set_title` calls at all. The plotting counterpart was `pythongpu/lorenz_basins.py` (present in the repo as recently as `Jun 29 09:35` per shell history, now `git status` shows it `D` deleted, and empty at `HEAD`) — it was apparently dropped during the reorg without its replotting logic being carried over to `lorenz_basins_sweep.py`. So: attributed to a real, identifiable pipeline stage (`lorenz_basins_sweep.py`'s N=100/Node-73-81 Lorenz sweep), but the exact plotting script no longer exists to confirm a `<script>__` filename prefix against — hence quarantined under `legacy_unsourced/` rather than guess-renamed. If `plot_lorenz_basins.py` (or a new script) grows a code path for this dataset, retroactively rename these using that script's id.

**Heads up — unrelated to this cleanup:** at some point during this session the top-level `pipeline/` directory (and `pipeline_notes.md`) moved to `pythongpu/pipeline/` outside of any edit made here — `git status` now shows those files renamed to the new location. The `pipeline/...` path references in the provenance table above (§9) have been updated to `pythongpu/pipeline/...` accordingly. §1, §2, §4, and §6 above describe the pre-reorg layout as of their original 2026-07-10 generation date and were left as-is (they predate this section and use bare filenames like `run_lorenz_gpu.sh` / `kuramoto_basins.py (root)` rather than `pipeline/`-prefixed paths, so the directory move doesn't directly stale them the same way).
