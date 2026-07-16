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

## Tests

```bash
python3 -m pytest
```
