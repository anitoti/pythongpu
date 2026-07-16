# PythonGPU

GPU-accelerated network dynamical systems, fractal basin analysis, causal
inference, and multifractal analysis for brain data.

## Project structure

All first-party code lives under a single importable package, `pythongpu/`,
at the repository root:

- `pythongpu/oscillators/` - network-coupled dynamical systems (Lorenz, Rössler, Kuramoto, Van der Pol)
- `pythongpu/networks/` - graph builders (empirical connectomes, ER/BA/WS null models)
- `pythongpu/processing/` - analysis (box-counting, basin dimension, chimera classifier, causation entropy, MFDFA)
- `pythongpu/visualization/` - plotting helpers
- `pythongpu/utils/` - shared utilities (Laplacians, output-path helpers)
- `pythongpu/pipeline/` - runnable experiment/sweep scripts that drive the modules above
- `scripts/` - thin path-based entry-point wrappers
- `tests/` - pytest suite
- `data/` - raw data and generated outputs (git-ignored)

Because the `pythongpu` package sits at the repository root, `import pythongpu`
works from the repo root with **no install and no `PYTHONPATH`**. For an
isolated/importable-from-anywhere install, `pip install -e .` (needs
`pip`/`setuptools` new enough for `pyproject.toml` editable installs).

## Running things

Every runnable module can be launched with `python3 -m` from the repo root.
Experiment/sweep scripts live in `pythongpu.pipeline`:

```bash
python3 -m pythongpu.pipeline.inspect_data
python3 -m pythongpu.pipeline.sparse_brain_sim
python3 -m pythongpu.pipeline.run_sim
python3 -m pythongpu.pipeline.lorenz_sweep --grid-n 64 --coupling 0.5 --dti-path data/DTI-og.mat --outdir data/
python3 -m pythongpu.pipeline.run_sweep --system rossler -- --outdir data/rossler/
python3 -m pythongpu.processing.multifractal_analysis
```

The thin wrappers in `scripts/` remain as an equivalent path-based alternative:

```bash
python3 scripts/run_inspect.py
python3 scripts/run_correlate_fmri.py --input data/rfMRI_REST1_LR_hp2000_clean.nii.gz --output data/adjacency_matrix.npy --chunk-size 5000
python3 scripts/run_plot_matrix.py
python3 scripts/run_sparse_brain_sim.py
python3 scripts/run_vps.py
```

If installed via `pip install -e .`, the console commands defined in
`pyproject.toml` are also available (e.g. `pythongpu-vps`, `pythongpu-inspect`).

## Package imports

```python
from pythongpu.oscillators.lorenz import LorenzNetwork
from pythongpu.processing.multifractal_analysis import run
```

## DTI data registration and spectral index (new)

This repository distinguishes raw binary data (kept in `data/` and **git-ignored**) from reproducible derived metadata that is recorded and tracked in the codebase. The workflow below standardises DTI filenames, computes spectral diagnostics once per unique matrix, and records the results in a tracked JSON index so downstream tests and analyses can reliably reuse precomputed results.

1) Register a DTI matrix with a human-readable tag (recommended):

    python3 scripts/register_dti.py --src /path/to/DTI.mat --tag og --move --compute

This will copy the file to `data/DTI-og.mat` (or `data/DTI-<tag>.mat`) and optionally run the spectral diagnostics. Use `--move` to move instead of copy. If the filename already exists and the content differs, a timestamp suffix is appended.

2) Spectral diagnostics and index

Run the diagnostics (the registration script can do this for you), or run directly:

    python3 scripts/run_dti_spectral.py --mat data/DTI-og.mat --outdir output/dti_og

What the diagnostics do:
- Preprocess: symmetrise adjacency, zero diagonal
- Compute: adjacency, Laplacian, normalized Laplacian eigenpairs
- Robust k-selection: eigengap candidate + silhouette sweep
- Spectral clustering (KMeans on Laplacian eigenvectors)
- Quality metrics: silhouette score, modularity (networkx)
- Outputs: eigenvalues/eigenvectors (npy/txt), cluster labels, plots, PDF report
- Indexing: compute SHA256 fingerprint of the matrix and record a structured summary in `dti_spectra_index.json` at the repo root (tracked by git)

The index prevents repeated work: a matrix with the same SHA256 is considered identical and its stored entry is used by downstream tests/pipelines unless recompute is explicitly requested (`--force`).

3) Inspecting results

- `dti_spectra_index.json` lists indexed matrices with their diagnostics and outputs. Use it to drive tests or skip reprocessing when running pipelines on many matrices.

## Tests

```bash
python3 -m pytest
```
