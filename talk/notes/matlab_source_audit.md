# Audit of the original MATLAB code (`~/matlab_fractalbasin`)

**Paper:** Bollt, Fish, Kumar, Roque dos Santos, Laurienti, *"Fractal Basins
as a Mechanism for the Nimble Brain"*, arXiv:2311.00061 —
https://arxiv.org/abs/2311.00061 (51 pages, 14 figures). Abstract and
author list confirmed to match exactly via direct fetch on 2026-07-23.

**Source:** the folder Dr. Fish provided at the start of the summer,
`~/matlab_fractalbasin/` on this machine. Every file below was read
directly (not inferred from filenames alone). This document is the
result of that read, cross-checked against the current Python port
(`pythongpu/`) wherever a direct comparison was possible.

---

## <span style="color:red">TL;DR — the most important thing in this document</span>

<span style="color:red">**The paper's primary/headline result (Fig. 2, the
DTI-connectome basin map, d_box≈1.8) uses Hindmarsh-Rose, not Lorenz.**</span>
There is no Lorenz code anywhere in this folder. The file that implements
the DTI-network model (`HR_ElCh_network.m`) has a stale comment reading
`%% network of lorenz oscillators`, but its actual equations are
Hindmarsh-Rose, citing Hizanidis et al.'s parameters. This is almost
certainly a copy-paste artifact in Dr. Fish's own code, not evidence the
system is Lorenz — but it may be exactly what led this project to build a
Lorenz-based pipeline as "the" system all summer, when Hindmarsh-Rose was
the one actually used for the paper's main figure.

Four further concrete discrepancies were found between the original code
and this project's Python port (`pythongpu/`), detailed below. None of
these are guesses — each is a direct comparison between the MATLAB source
and the corresponding Python file/line.

---

## 1. The core DTI/Hindmarsh-Rose pipeline (produces Fig. 2 and likely Fig. 3)

### `HR_ElCh_network.m` — the network RHS
```
dx_i/dt = y_i - a*x_i^3 + b*x_i^2 - z_i + Iext - gch*(x_i - Vsyn)*(A * sigmoid(-lambda*(x_i-thetasyn)))
dy_i/dt = c - d*x_i^2 - y_i
dz_i/dt = r*(s*(x_i - p0) - z_i)
minus diffusive coupling: dw/dt -= gelLH * w
```
Two coupling terms, not one:
- **Diffusive/"electrical"** coupling: `gelLH = gel*kron(L,H)`, subtracted
  from the whole state vector.
- **Chemical/sigmoidal synaptic** coupling: `gch*(x_i-Vsyn)*(A @ sigmoid(...))`,
  baked directly into the dx/dt equation, using the adjacency matrix `A`
  directly (not the Laplacian).

Parameters given: `a=1, b=3, c=1, d=5, s=4, p0=-1.6, Iext=3.25, r=0.005,
Vsyn=2, thetasyn=-0.25, lambda=10`, citing "Chimera-like States in Modular
Neural Networks" by Hizanidis et al. for the HR parameter set.

<span style="color:red">**Comment says "network of lorenz oscillators"
— this is wrong/stale, the equations are Hindmarsh-Rose.**</span> Same
stale comment appears in `HR_ElCh_distribution_network.m` too, so it's
consistent, not a one-off typo — looks copy-pasted from an actual Lorenz
template file that isn't in this folder.

**Comparison against `pythongpu/oscillators/hindmarsh_rose.py`:**

| | MATLAB (`HR_ElCh_network.m`) | Python (`HindmarshRoseNetwork`) |
|---|---|---|
| a,b,c,d,s,x_rest,I | 1,3,1,5,4,-1.6,3.25 | 1,3,1,5,4,-1.6,3.2 |
| r | 0.005 | 0.006 |
| **Coupling variable** | <span style="color:red">**Y** (H=[0,0,0;0,1,0;0,0,0], see below)</span> | <span style="color:red">**X** (H=diag[1,0,0], per its own docstring)</span> |
| **Chemical/sigmoidal coupling** | <span style="color:red">**present**</span> | <span style="color:red">**absent entirely**</span> |

The `r` and `I` differences (0.005 vs 0.006, 3.25 vs 3.2) are close enough
to plausibly be an intentional round-number simplification and probably
don't matter much. The coupling-variable and missing-chemical-coupling
differences are not rounding — they're structurally different equations.

### <span style="color:red">Which state variable does the coupling actually act through?</span>

`ForPlottingPNAS_PeaksFigure.m` (the only script in this folder that
actually drives `HR_ElCh_network` on the real DTI connectome, loading
`DTI_A.mat`) sets:
```matlab
H = [0 0 0; 0 1 0; 0 0 0];
```
Given the state vector is interleaved `w = [x1,y1,z1,x2,y2,z2,...]`
(confirmed from `HR_ElCh_network.m`'s own unpacking:
`x = w(1:3:end-2); y = w(2:3:end-1); z = w(3:3:end)`), `H`'s single nonzero
entry at position (2,2) means **`kron(L,H)` couples through Y (the fast
recovery variable), not X (the membrane potential)**.

This project's `HindmarshRoseNetwork` (`pythongpu/oscillators/hindmarsh_rose.py`)
explicitly documents "Diffusive coupling acts through the fast component
only (H = diag[1,0,0])" — i.e. couples through X, matching the Lorenz
convention this whole project has used, not the original HR script's
actual Y-coupling.

**This matters a lot for Act 1.** If the goal is an accurate port that
reproduces Fig. 2/3, the coupling term needs to act on Y, and a chemical/
sigmoidal term needs to exist at all. Neither is true of the current
Python port. This is worth confirming with Dr. Fish before doing the
(nontrivial) rework, in case there's a reason the choice was made
differently on purpose.

### `ForPlottingPNAS_PeaksFigure.m` — the actual driver for the headline plot

Loads `DTI_A.mat` (83-node human connectome — this project's `DTI-og.mat`
is presumably the same or a renamed copy, not independently verified here),
builds `L = diag(sum(A,2)) - A`, sets `gel = 0.5` (electrical coupling,
matches this project's `K`), integrates with `ode45` for `T=2000` seconds
at three coupling configurations side-by-side:
- (a) `gch = 0` — pure electrical coupling (closest to what this project
  has actually been running).
- (b) `gch = 0.015` — adds chemical coupling.
- (c) same as (b), plus the HR parameter `a` drawn from
  `normrnd(1, 1e-1, n, 1)` per node (heterogeneous parameters across
  nodes) via `HR_ElCh_distribution_network.m`.

This script only plots three example time-series panels — it does **not**
sweep a 2-D IC plane or build/cluster VPS vectors. <span style="color:red">
**The actual driver script that sweeps a grid of initial conditions,
computes VPS per point, clusters, and saves the basin map data for the
DTI/HR case is not present anywhere in this folder.**</span> Compare this
to `Henon_ManyPlots.m` (below), which *does* contain that full loop for
Henon — the equivalent script for HR seems to be missing, not just its
output data.

### `VectorPatternState.m` — the VPS itself

```matlab
[cx lagsx] = xcorr(x(:,Vec(jk,1)), x(:,Vec(jk,2)));
f = find(cx == max(cx));
Tau_x(jk) = lagsx(f);
if lagsx(f) > 0
    L(jk) = norm(x(1:end-(lagsx(f)-1), Vec(jk,2)) - x(lagsx(f):end, Vec(jk,1)));
...
```
This is the exact source of the off-by-one already documented in
[`vps_lag_off_by_one.md`](vps_lag_off_by_one.md) — confirmed directly from
source here: the alignment shift is `lagsx(f)-1`, not `lagsx(f)`, for the
truncation length, while the *start* index for the other series uses the
full `lagsx(f)`. Reverified against this project's
`vector_pattern_state_fast` on 2026-07-23: 36/36 pairs, mean ratio 1.956,
matching.

<span style="color:red">**Separate, smaller finding: the `Alpha` input
parameter is documented ("the scaling factor to scale the normed portion
of the VPS against the tau values") but never actually used in the
function body** — `VPS = [Tau_x L]`, no multiplication by `Alpha`
anywhere.</span> In practice this doesn't change any result in this
folder, because every caller (`Henon_ManyPlots.m`, `Rulkov_ManyPlots.m`,
`Hres_Henon_LoadedParams.m`, `RulkovMap_Loaded_In_Version.m`,
`SingleHenon_SlightlyHigherRes.m` — confirmed via grep, all 5 call sites)
passes `Alpha=1`, so the missing multiplication is a no-op either way.
Worth noting as a documentation/implementation mismatch in the original
code, not a numerically consequential bug. This project's own
`vector_pattern_state_fast(..., alpha=1.0)` *does* apply `L * alpha` —
meaning the Python port actually implements what the MATLAB docstring
*promised* rather than what the MATLAB code *does*. Harmless at `alpha=1`,
worth knowing if `alpha != 1` is ever used.

### `boxcount.m` — box-counting dimension

<span style="color:red">**This is a well-known third-party utility (F.
Moisy, MATLAB File Exchange), not code written for this paper.**</span>
Uses power-of-two box sizes only (`r = 1, 2, 4, ..., 2^p`), pads the input
array to the next power of 2 if needed, and merges via logical OR at each
halving (standard dyadic box-counting). Has not been directly compared
line-by-line against this project's `boxcount_2d_gpu` — worth doing if an
"exact reproduction" claim needs to survive scrutiny, since a different
box-counting convention (e.g. non-dyadic box sizes, different padding)
would shift the fitted dimension slightly.

### `KmeansBIC.m` / `VarKmeans.m` — cluster-count selection

```matlab
% VarKmeans.m
Sigma_Squared = sum(SumD) * (1/(N*d));

% KmeansBIC.m
log_likli = sum(C.*log(C)) - N*log(N) - (N*d/2)*log(2*pi*Sig_sqrd) - (d/2)*(N-K);
BIC = log_likli - ((K+K*d)/2)*log(N);
```

The `KmeansBIC` formula (the log-likelihood/BIC expression itself) is
**an exact match** to this project's `KmeansBIC` in
`pythongpu/processing/feature_extraction.py` — verified line-by-line, this
part of the port is accurate.

<span style="color:red">**But the variance term feeding into it isn't.**</span>
MATLAB's `VarKmeans.m` computes `Sigma_Squared = sum(SumD) / (N*d)`. This
project's Python `KmeansBIC` computes `Sig_sqrd = np.sum(SumD) / (N - K)`
inline, instead of calling an equivalent to `VarKmeans` at all — a
different denominator (`N*d` vs `N-K`), which will shift the BIC value
(and therefore potentially the chosen `k`) whenever `d` and `N-K` differ,
which is generically always. This looks like an actual bug introduced
during the port, not a deliberate choice — there's no comment explaining
the change.

### `ElbowForKmeans.m` — how `k` was actually chosen

Not literally a visual "elbow" in the colloquial sense of eyeballing a
bend in a curve. It runs `kmeans(KmeansMat, i)` for `i = 1..100` (not just
up to 8 or so), records both the `KmeansBIC` score and the raw
within-cluster SSE for every `i`, and additionally fits a `log(k)` vs
`log(SSE)` power-law regression. The paper's own text calls this "the
classic elbow method," but the actual artifact computes 100 candidate `k`
values and two different scoring approaches side by side — worth keeping
in mind if trying to reproduce exactly *how* `k=8` was arrived at, since
"classic elbow" undersells what the script actually does.

---

## 2. Null-model / graph utilities

- **`ER_Adj.m`**: Erdos-Renyi random graph generator (`N` nodes, edge
  probability `p`), symmetric adjacency matrix. Straightforward, matches
  this project's `generate_gnm_laplacian`-style utilities in spirit (not
  compared formula-by-formula here).
- **`RewireAdj.m`**: degree-preserving-ish edge rewiring (pick a random
  node, remove one of its edges, add a new edge to a random non-neighbor).
  This is very likely the source this project's `rewire_edges` function
  and the `lorenz_sweep.py` docstring's quoted pseudocode
  (`R1 = randperm(n); ... A(R1,R3) = 1; A(R3,R1) = 1;`) were based on —
  the structure matches closely.

## 3. Other systems on the same DTI network (generality claims — likely Fig. 4)

- **`Henon_ManyPlots.m`**: the one file in this folder that *does* contain
  the full driver loop — loads `DTI_A.mat`, sweeps a tiled 2-D grid of
  initial conditions (an outer coarse tile per `parfor`-independent
  "version," an inner fine grid via `parfor` within each version),
  integrates the network via `Coupled_Henon.m`, computes VPS per IC via
  `VectorPatternState.m`, clusters with `kmeans(KmeansMat, K=3)`, and saves
  `Colors, Xg, Yg, KmeansMat` to a version-numbered `.mat` file. Henon
  couples through **X** (`H = [1 0; 0 0]`), unlike HR's Y-coupling above —
  worth noting these two systems in the same paper couple through
  *different* state variables.
- **`SingleHenon_SlightlyHigherRes.m`**, **`Hres_Henon_LoadedParams.m`**:
  variants of the same Henon pipeline at different resolutions / loading
  previously-saved parameters instead of regenerating them.
- **`Rulkov_ManyPlots.m`**, **`RulkovMap_Loaded_In_Version.m`**,
  **`X_Coupled_Rulkov.m`**, **`Rulkov_Map.m`**: the same architecture for
  the Rulkov map, which (per `X_Coupled_Rulkov.m`) has *both* electrical
  and chemical/sigmoidal coupling terms, much like the HR model. Cites
  Bashkirtseva and Rulkov for the map's parameters.

<span style="color:red">**Every one of these Henon/Rulkov driver scripts
references version-numbered `.mat` data files that are either missing
entirely or present under a different version number than the script
expects**</span> (e.g. `Hres_Henon_LoadedParams.m` wants
`..._VersionNum_12_...`, the file actually present is
`..._VersionNum_1_01-Jun-2023.mat`). None of the actual simulation output
survived in this folder — only the code to produce and later plot it.

## 4. Ambiguous / unclear connection to this paper

- **`MoreTests_Symmetry.m`**: a discrete phase-map (not HR, not Henon, not
  Rulkov) —
  `xtprime = mod(beta*(1-cos(xt))/2 + sigma*A*(1-cos(xt)) + delta, 2*pi)` —
  on a hardcoded 6-node adjacency matrix, testing symmetry via
  `squareform(pdist(...))` and `corr`. <span style="color:red">**Unclear
  which figure, if any, this corresponds to** — possibly exploratory work
  for a different chimera-related question, or a different paper
  entirely. Worth asking Dr. Fish directly.</span>
- **`DelayLogisticMap.m`**, **`OneDirAR.m`**, **`TestSingle.m`**,
  **`TestTimeShifted.m`**, **`ProjectHermite.m`**: small, self-contained
  test/exploration scripts (a delay-coupled logistic map network, a
  one-directional autoregressive process generator + mutual-information
  time-lag test, a single logistic map smoke test, a Hermite-polynomial
  projection utility). None of these reference `DTI_A.mat`, VPS, or
  box-counting. <span style="color:red">**No evidence these connect to
  this paper's figures at all** — most plausibly leftover scratch work
  from unrelated investigations Dr. Fish had in the same working
  directory.</span>

## 5. Plotting-only scripts (all load data not present in this folder)

`Plot_DraftFig.m`, `PlottingElbow.m`, `Plotting_Multiple_K.m`,
`Plotting_New_Zooms.m`, `Plotting_Zoomed_NotDistribution.m`,
`ForPlotting_FractalBasins.m`, `ForPlottingPNAS_PeaksFigure.m` (partially,
see above). Each `load()`s a `PNAS_Paper_*.mat` file and re-clusters
(`kmeans(KmeansMat, K)`, `K=8` hardcoded in two of them, matching the
paper's stated elbow result) or just re-plots already-clustered `Colors`.
<span style="color:red">**None of the referenced `PNAS_Paper_*.mat` files
are present in this folder** — this is the actual basin-map data behind
the published figures, and it isn't here.</span>

---

## Summary table of concrete, checkable gaps

| # | Finding | Severity for "Act 1" |
|---|---|---|
| 1 | Coupling acts on Y in the original, X in the current Python port | High — structurally different dynamics |
| 2 | Chemical/sigmoidal coupling entirely absent from the Python port | High — the headline figure (panel b/c) uses it |
| 3 | VPS lag off-by-one | Already known/documented, see `vps_lag_off_by_one.md` |
| 4 | `KmeansBIC`'s variance term uses a different formula (`N*d` vs `N-K`) | Medium — affects auto-selected `k` |
| 5 | The actual grid-sweep driver script for the DTI/HR case is missing from this folder | Blocks literal re-running of the original code; doesn't block re-deriving it |
| 6 | `boxcount.m` is third-party dyadic box-counting, not independently checked against this project's box-counting | Medium — worth a direct comparison |
| 7 | `Alpha` parameter documented but unused in `VectorPatternState.m` | Low — always called with `Alpha=1`, no-op |
| 8 | Stale "network of lorenz oscillators" comment in both HR files | Cosmetic, but possibly the root cause of this project's Lorenz-first approach |
| 9 | `ElbowForKmeans.m`'s method is more involved than "classic elbow" (100 k values, BIC + SSE + log-log fit) | Low — doesn't change results, changes how to describe the method |
| 10 | `MoreTests_Symmetry.m` and five small scratch scripts have no evident connection to this paper | Low — just needs confirmation they're not relevant |

## Open questions for Dr. Fish

1. Is `HR_ElCh_network.m`'s coupling-through-Y (not X) intentional, or
   was X always meant?
2. Is `gch=0` (pure electrical) or `gch≈0.015-0.03` (with chemical
   coupling) the actual configuration behind Fig. 2's d_box≈1.8 result?
3. Is the `VectorPatternState.m` off-by-one (`|tau|-1` alignment) a
   known/intentional convention, or a bug that's been in the pipeline?
4. Does the missing grid-sweep driver script for the DTI/HR case (the
   equivalent of `Henon_ManyPlots.m` but for HR) exist somewhere else, or
   does it need to be rebuilt from the pieces that are here?
5. Is `MoreTests_Symmetry.m` (and the handful of unrelated-looking scratch
   scripts) relevant to this paper at all?
