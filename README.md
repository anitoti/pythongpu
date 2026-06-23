# PythonGPU

## Project structure

- `data/` - raw data and generated outputs
- `src/pythongpu/` - reusable package modules and scripts
- `scripts/` - lightweight entry-point wrappers for the package
- `run_sim.py` - root launcher example for the ER graph simulation

## Run scripts

Use the wrapper scripts in `scripts/` to execute the main pipeline components:

```bash
python3 scripts/run_inspect.py
python3 scripts/run_correlate_fmri.py
python3 scripts/run_plot_matrix.py
python3 scripts/run_sparse_brain_sim.py
python3 scripts/run_vps.py
python3 scripts/run_sim.py
```

Each script supports command-line arguments. Example:

```bash
python3 scripts/run_correlate_fmri.py --input data/rfMRI_REST1_LR_hp2000_clean.nii.gz --output data/adjacency_matrix.npy --chunk-size 5000
```

## Package imports

The package source lives under `src/pythongpu` and is importable with:

```python
from src.pythongpu import ...
```

## Notes

- `data/` contains sample files and outputs moved from the repository root.
- `scripts/` provides clean entry points with no hard-coded paths.
