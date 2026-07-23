# CLV → Kaplan–Yorke pipeline

How the Kaplan–Yorke dimension gets computed for the coupled-Lorenz DTI
network, and where the numbers in `clv_results_final/` came from.

Code: [`pythongpu/pipeline/clv_diagnostics.py`](../../pythongpu/pipeline/clv_diagnostics.py),
[`pythongpu/pipeline/clv_topology.py`](../../pythongpu/pipeline/clv_topology.py),
driven by [`pythongpu/pipeline/clv_cli.py`](../../pythongpu/pipeline/clv_cli.py).

## 1. System

N=83 Lorenz oscillators coupled through the DTI connectome Laplacian `L`
(`data/DTI-og.mat`), diffusive coupling on X/Y/Z or sigmoidal, strength
`coupling`. Flat state vector of size n=3N: `[X(1..N), Y(1..N), Z(1..N)]`.
Standard Lorenz RHS per node plus `-coupling * L` (or a state-dependent
sech² block for `sigmoidal` mode) in the Jacobian.

## 2. Ginelli forward pass

RK4-integrate the state **and** an `m`-dimensional tangent basis
`T ∈ R^{n×m}` using the analytic Jacobian `J(x(t))`:

```
ẋ = f(x)
Ṫ = J(x) T
```

Every `qr_interval` steps, orthonormalize:

```
T = Q R          (QR decomposition, R upper-triangular)
```

Store `Q` and `R` at each slice, then continue integrating with `T ← Q`
(this is what keeps the basis from collapsing onto the single expanding
direction over long integrations — the standard Benettin trick).

## 3. Ginelli backward pass

CLVs are *covariant* (dynamically invariant, not merely orthonormal like
the QR vectors). Recover them by solving the triangular recursion
backward from the last stored slice, `C = I`:

```
R_k C_{k-1} = C_k         (solve for C_{k-1}, upper-triangular solve)
```

normalizing columns of `C_{k-1}` after each solve. Then at every stored
time:

```
V_k = Q_k C_k
```

`V_k` are the covariant Lyapunov vectors.

## 4. Lyapunov spectrum

The diagonal of each stored `R` records the log-growth of each direction
over one QR interval `τ = qr_interval · dt`. Average `log|R_ii|` over all
QR steps (first 10% discarded as transient) and divide by elapsed time:

```
λ_i = ⟨ log|R_ii| ⟩ / τ
```

Exponents come out already sorted descending (QR preserves ordering):
λ₁ ≥ λ₂ ≥ … ≥ λ_m.

## 5. Kaplan–Yorke (Lyapunov) dimension

Interpolates where the cumulative exponent sum crosses zero:

```
D_KY = k + ( Σ_{i=1}^{k} λ_i ) / |λ_{k+1}|
```

where `k` is the largest index such that the partial sum
`Σ_{i≤k} λ_i ≥ 0`.

Edge cases handled in [`kaplan_yorke_dimension()`](../../pythongpu/pipeline/clv_diagnostics.py):

- `λ₁ ≤ 0` → D_KY = 0 (fixed point / fully contracting).
- Cumulative sum still ≥ 0 after **all** m computed exponents → no k+1 term
  exists to divide by. Code reports this as a **ceiling**, D_KY = m
  (`kaplan_yorke_is_ceiling = True`), not a real interpolated value — it
  means you haven't computed enough CLVs to find where the sum actually
  turns negative, and D_KY should be re-run with larger `--m`.

## 6. Riddling diagnostic (side product of the same CLVs)

For the leading K CLVs at each stored time, normalize and take pairwise
`|cos(angle)|` via the Gram matrix, convert to angles, keep the minimum:

```
θ_min(t) = min_{i≠j} arccos( |v̂_i(t) · v̂_j(t)| )
```

2-means cluster the `θ_min(t)` series. A bimodal split (populated
low-angle + high-angle clusters, centroids separated by
`separation_thresh`) is the Ott/Alexander riddling fingerprint
("Intermingled basins in coupled Lorenz systems", arXiv:1111.5581):
long stretches locked near the synchronization manifold, punctuated by
transverse bursts.

## Did the ceiling ever actually trigger?

Yes — in 3 of the 4 sweep points in `clv_results_final/` (`m=83` computed
each time):

| coupling | n_positive / m | ceiling? | D_KY reported | λ_max |
|---|---|---|---|---|
| 0.05 | 83/83 | **yes** | `>= 83` | 0.475 |
| 0.10 | 74/83 | **yes** | `>= 83` | 0.526 |
| 0.15 | 34/83 | **yes** | `>= 83` | 0.600 |
| 0.20 | 14/83 | no | 45.40 | 0.423 |

Note K=0.10 and K=0.15 hit the ceiling even though a chunk of individual
exponents are already negative (74/83, 34/83 positive) — the *cumulative*
sum stayed non-negative all the way out to i=83, so there was no k+1 to
interpolate against with only 83 CLVs computed. Only K=0.20 (14/83
positive) resolved to a genuine, non-ceiling D_KY.

Practical upshot: for K ≤ 0.15 the reported `D_KY=83` numbers are not
real dimension estimates — they're a floor telling you the attractor
dimension is at least as large as everything you computed. Resolving
them for real needs `--m` well above 83 (which for n=3·83=249 means CLVs
covering nearly the whole tangent space) or working with a reduced/
projected coupling scheme where the true dimension is smaller.
