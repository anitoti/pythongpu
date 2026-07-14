# CLAUDE.md

## Project: Fractal Basin Boundaries in Network Dynamics

Clarkson University REU (summer 2026), advised by Dr. Jeremie Fish. Author: anitoti (atotilca@clarkson.edu).

### Goal
Study **basin stability** of (partially) synchronous states in coupled oscillator networks,
with a focus on the **fractal structure** of basin boundaries — the "Nimble Brain" theory
(see Fish et al., "Fractal Basins as a Mechanism for the Nimble Brain").

This corresponds to project (3) from the REU description: develop software to rapidly examine
the fractal basin structure of partially synchronous states, viewed from chosen 2-D planes of
the high-dimensional initial-condition space.

### Core method
1. Sweep a 2-D affine slice of high-dimensional IC space (grid, typically [-9,9] x [-9,9]).
2. Integrate each IC forward (RK4 for continuous systems; map iteration for discrete).
3. Compute a **Vector Pattern State (VPS)** per trajectory to classify its asymptotic state.
4. Cluster VPS features (k-means, with elbow / BIC for choosing k) to label basins.
5. Quantify the basin boundary via **fractal / box-counting dimension**.

### Systems studied
- **Lorenz system** (Python/PyTorch, GPU) — main current work.
- **Coupled Hénon map** (`Coupled_Henon.m`, `Henon_Map.m`).
- **Coupled Rulkov map** (`X_Coupled_Rulkov.m`, `Rulkov_Map.m`) — chemical + electrical coupling.
- **Hindmarsh-Rose** electrical/chemical networks (`HR_ElCh_network.m`, `HR_ElCh_distribution_network.m`).
- **Logistic / delay-logistic maps** (`LogisticMap.m`, `DelayLogisticMap.m`) — toy/reference.

### Key files

Python (current, GPU pipeline — lives under `src/pythongpu/`):
- `lorenz_basins.py` — main driver: samples 4096 ICs, integrates Lorenz, builds basin map,
  streams VPS features via Welford online mean/variance (avoids storing full trajectories / OOM).
- `lorenz_vps_clustering.py` — VPS feature extraction + clustering pipeline; grid/params
  mirror the original MATLAB source exactly.
- `plot_lorenz_basins.py` — replots from saved `lorenz_basins.npz` (continuous norm heatmap
  + k-means basin map).
- `plot_dti.py` — visualizes DTI-derived adjacency matrix (`DTI_A.mat`, 83x83).

MATLAB (original reference implementation):
- Dynamics: `Coupled_Henon.m`, `X_Coupled_Rulkov.m`, `Henon_Map.m`, `Rulkov_Map.m`.
- VPS: `VectorPatternState.m` (scaled norm + tau values between all oscillator pairs).
- Networks: `ER_Adj.m` (Erdos-Renyi), `RewireAdj.m` (edge rewiring), `randcantor.m`.
- Fractal dim: `boxcount.m`, `FindFractalDimension_OfBasin.m` (box-count + polyfit on log-log).
- Clustering: `KmeansBIC.m`, `VarKmeans.m`, `ElbowForKmeans.m`.
- Plotting: `ForPlotting_FractalBasins.m`, `Plotting_New_Zooms.m`, `ForPlottingPNAS_PeaksFigure.m`.

### Conventions & notes
- MATLAB time series shape: `X` is `(T x 2n)` for n oscillators (2 state vars each);
  odd columns = x, even columns = y. Coupling via graph Laplacian `L = diag(sum(A,2)) - A`,
  Kronecker-lifted `LH = kron(L,H)`.
- Python port must match MATLAB grid ranges and transient burn-in (`Tminus`) exactly — this
  is deliberate for cross-validation. Flag any divergence.
- Prefer streaming/online stats over storing full trajectories (memory).
- Reference text: Strogatz, *Nonlinear Dynamics and Chaos* (in project docs).

### When helping on this project
- Keep the Python results reproducible against the MATLAB reference.
- Preserve the exact parameterization unless asked to change it.
- Box-counting dimension is estimated by linear fit of log(N) vs -log(r); watch scaling range.