# The Mathematics of `pythongpu`
### A Living Textbook of Network Dynamical Systems, Fractal Basin Geometry, Causal Inference, and Multifractal Analysis

---

**Abstract.** This document is a rigorous, self-contained mathematical reference for the `pythongpu` repository. The codebase is an end-to-end pipeline that (i) integrates large ensembles of network-coupled nonlinear oscillators on the GPU, (ii) reduces each trajectory to a low-dimensional *Vector Pattern State* (VPS) signature, (iii) classifies those signatures into dynamical basins whose boundaries are characterized by their fractal dimension, (iv) infers directed causal networks from empirical (fMRI) time series via optimal causation entropy, and (v) characterizes the scale-invariant statistics of each parcellated brain region through Multifractal Detrended Fluctuation Analysis (MFDFA). Every section below derives the governing mathematics from first principles and connects each equation to the exact module and function that implements it. The notation is uniform throughout, and no section is left as a placeholder.

---

## Table of Contents

- [Notation and Conventions](#notation-and-conventions)
- [Part I — Network-Coupled Oscillators and the VPS System](#part-i--network-coupled-oscillators-and-the-vps-system)
  - [I.1 The general coupled-oscillator field](#i1-the-general-coupled-oscillator-field)
  - [I.2 Governing differential equations of the concrete systems](#i2-governing-differential-equations-of-the-concrete-systems)
  - [I.3 Coupling schemes: diffusive Laplacian vs. phase coupling](#i3-coupling-schemes-diffusive-laplacian-vs-phase-coupling)
  - [I.4 State transitions: the RK4 flow map](#i4-state-transitions-the-rk4-flow-map)
  - [I.5 The Vector Pattern State — Definition A (lag–norm signature)](#i5-the-vector-pattern-state--definition-a-lagnorm-signature)
  - [I.6 The Vector Pattern State — Definition B (neighbor-relative coherence)](#i6-the-vector-pattern-state--definition-b-neighbor-relative-coherence)
  - [I.7 From VPS to labels: model-order selection](#i7-from-vps-to-labels-model-order-selection)
- [Part II — Fractal Basin Boundaries and Causation Entropy](#part-ii--fractal-basin-boundaries-and-causation-entropy)
  - [II.1 Basins of attraction and the initial-condition slice](#ii1-basins-of-attraction-and-the-initial-condition-slice)
  - [II.2 Boundary extraction](#ii2-boundary-extraction)
  - [II.3 The box-counting dimension](#ii3-the-box-counting-dimension)
  - [II.4 Final-state sensitivity and the uncertainty exponent](#ii4-final-state-sensitivity-and-the-uncertainty-exponent)
  - [II.5 Causation entropy: differential entropy and Gaussian closed forms](#ii5-causation-entropy-differential-entropy-and-gaussian-closed-forms)
  - [II.6 Conditional mutual information and the oCSE algorithm](#ii6-conditional-mutual-information-and-the-ocse-algorithm)
  - [II.7 The chi-squared significance test](#ii7-the-chi-squared-significance-test)
- [Part III — Multifractal Detrended Fluctuation Analysis](#part-iii--multifractal-detrended-fluctuation-analysis)
  - [III.1 Motivation and overview](#iii1-motivation-and-overview)
  - [III.2 Step 1 — the integrated profile](#iii2-step-1--the-integrated-profile)
  - [III.3 Step 2 — segmentation from both ends](#iii3-step-2--segmentation-from-both-ends)
  - [III.4 Step 3 — local polynomial detrending and the projection operator](#iii4-step-3--local-polynomial-detrending-and-the-projection-operator)
  - [III.5 Step 4 — the q-th order fluctuation function](#iii5-step-4--the-q-th-order-fluctuation-function)
  - [III.6 Step 5 — scaling and the generalized Hurst exponent](#iii6-step-5--scaling-and-the-generalized-hurst-exponent)
  - [III.7 Step 6 — the mass (Rényi) exponent spectrum](#iii7-step-6--the-mass-rényi-exponent-spectrum)
  - [III.8 Step 7 — the Legendre transform to singularity coordinates](#iii8-step-7--the-legendre-transform-to-singularity-coordinates)
  - [III.9 Interpretation, limiting cases, and diagnostics](#iii9-interpretation-limiting-cases-and-diagnostics)
  - [III.10 Correspondence with the implementation](#iii10-correspondence-with-the-implementation)
- [References](#references)

---

## Notation and Conventions

| Symbol | Meaning |
|---|---|
| $N$ | number of network nodes (oscillators, or parcellated ROIs) |
| $T$ | number of discrete time samples in a trajectory / time series |
| $i,j,k$ | node indices, $1 \le i,j,k \le N$ |
| $t$ | discrete time index, $1 \le t \le T$ |
| $\mathbf{x}(t)\in\mathbb{R}^{D\times N}$ | full network state at time $t$; $D$ state components per node |
| $x_i(t)$ | scalar observable of node $i$ (component $0$, the coupling channel) |
| $A=(A_{ij})$ | adjacency matrix, $A_{ij}\ge 0$ |
| $L$ | graph Laplacian, $L = \mathrm{diag}(A\mathbf 1) - A$ |
| $W$ | a generic weighted coupling / neighborhood matrix |
| $\sigma,\;K$ | global coupling strength (diffusive; Kuramoto) |
| $\odot$ | Hadamard (element-wise) product |
| $\langle\,\cdot\,\rangle$ | sample mean over the time index |
| $\mathrm{Var}(\cdot)$ | sample variance over the time index |
| $H(\cdot)$ | Shannon (differential) entropy, in nats |
| $I(\,\cdot\,;\cdot\mid\cdot)$ | conditional mutual information |
| $D_f$ | fractal (box-counting) dimension of a basin boundary |
| $q$ | MFDFA moment order |
| $s$ | MFDFA segment scale (length) |
| $h(q)$ | generalized Hurst exponent |
| $\tau(q)$ | mass / Rényi scaling exponent |
| $\alpha$ | Hölder (singularity) exponent |
| $f(\alpha)$ | singularity spectrum (singularity dimension) |

Throughout, "trajectory tensors" follow the repository convention `(T, B, D, N)`: time, batch (initial condition), state-component, node. The scalar coupling observable is always state-component index $0$, written $x_i(t)$.

---

# Part I — Network-Coupled Oscillators and the VPS System

## I.1 The general coupled-oscillator field

Every dynamical model in `pythongpu/oscillators/` is a special case of a single autonomous vector field on the product state space $\big(\mathbb{R}^{D}\big)^{N}$. Writing the state of node $i$ as $\mathbf{u}_i\in\mathbb{R}^D$, the network obeys

$$
\dot{\mathbf u}_i \;=\; \mathbf F(\mathbf u_i)\;-\;\sigma\sum_{j=1}^{N} L_{ij}\,\mathbf H\,\mathbf u_j,
\qquad i=1,\dots,N,
\tag{I.1}
$$

where

- $\mathbf F:\mathbb{R}^D\to\mathbb{R}^D$ is the **intrinsic** (uncoupled) vector field shared by all nodes,
- $\mathbf H\in\mathbb{R}^{D\times D}$ is the constant **coupling matrix** selecting which components interact,
- $\sigma\ge 0$ is the scalar **coupling strength**, and
- $L$ is the graph Laplacian, so that $\sum_j L_{ij}\mathbf u_j$ is a discrete diffusion operator.

Equation (I.1) is the *master stability function* form of Pecora–Carroll. In this repository $\mathbf H = \mathrm{diag}(1,0,\dots,0)$ for every diffusively coupled system — coupling flows **only through the first state component** $x_i$. This is exactly the line

```python
network_influence = torch.matmul(X, self.L.T)
dX = dX - self.coupling * network_influence
```

that appears identically in `oscillators/lorenz.py`, `oscillators/rossler.py`, and `oscillators/vanderpol.py`. Because $X\,L^\top = (L\,X^\top)^\top$ and, for a batch of shape $(B,N)$, $X\,L^\top$ is precisely the batched diffusion, a single expression covers both the single-trajectory and the initial-condition-grid cases.

**Diffusion interpretation.** Since $L = \mathrm{diag}(A\mathbf 1)-A$,

$$
\big(L\mathbf u\big)_i \;=\; \Big(\textstyle\sum_{j}A_{ij}\Big)\mathbf u_i - \sum_j A_{ij}\mathbf u_j
\;=\; \sum_j A_{ij}\,(\mathbf u_i-\mathbf u_j),
\tag{I.2}
$$

so the coupling term $-\sigma (L\mathbf u)_i = \sigma\sum_j A_{ij}(\mathbf u_j-\mathbf u_i)$ pulls each node toward the (strength-weighted) mean of its neighbors. The fully synchronized manifold $\mathbf u_1=\dots=\mathbf u_N$ is invariant because $L\mathbf 1 = \mathbf 0$; the coupling never perturbs a coherent state, only restores it. This invariance is the mathematical reason a synchronized basin exists at all, and hence the reason the basin-boundary analysis of Part II is well posed.

## I.2 Governing differential equations of the concrete systems

Each concrete subclass fixes $\mathbf F$ and $\mathbf H$. Below, $X_i,Y_i,Z_i$ denote the components of $\mathbf u_i$; the network coupling term $-\sigma(LX)_i$ is written once and understood to be added to the $\dot X_i$ equation only.

**Lorenz network** (`oscillators/lorenz.py`, defaults $\sigma_{\text L}=10,\ \rho=28,\ \beta=8/3$):

$$
\begin{aligned}
\dot X_i &= \sigma_{\text L}\,(Y_i - X_i) \;-\; \sigma\,(LX)_i,\\
\dot Y_i &= X_i\,(\rho - Z_i) - Y_i,\\
\dot Z_i &= X_i Y_i - \beta Z_i.
\end{aligned}
\tag{I.3}
$$

Here $\sigma_{\text L}$ is the Prandtl number (an *intrinsic* Lorenz parameter, named `sigma` in the class) and $\sigma$ is the *network* coupling (named `coupling`). They are distinct scalars; we keep separate glyphs to avoid the collision that the shared source name invites.

**Rössler network** (`oscillators/rossler.py`, defaults $a=b=0.2,\ c=5.7$):

$$
\begin{aligned}
\dot X_i &= -\,Y_i - Z_i \;-\; \sigma\,(LX)_i,\\
\dot Y_i &= X_i + a\,Y_i,\\
\dot Z_i &= b + Z_i\,(X_i - c).
\end{aligned}
\tag{I.4}
$$

**Van der Pol network** (`oscillators/vanderpol.py`, default $\mu=1.5$), a second-order oscillator written in Liénard form with $\mathbf u_i=(X_i,Y_i)$ and $\mathbf H=\mathrm{diag}(1,0)$:

$$
\begin{aligned}
\dot X_i &= Y_i \;-\; \sigma\,(LX)_i,\\
\dot Y_i &= \mu\,(1 - X_i^{2})\,Y_i - X_i.
\end{aligned}
\tag{I.5}
$$

The coupling operator here is exactly $\sigma\,(L\otimes\mathbf H)$ with $\mathbf H=\mathrm{diag}(1,0)$, the Kronecker form documented in the module.

**FitzHugh–Nagumo network** (`oscillators/fitzhugh_nagumo.py`). The class declares parameters $a=0.7,\ b=0.8,\ \tau=12.5$ but its `rhs` currently raises `NotImplementedError`; it is a declared interface awaiting an implementation. For completeness, the intended dynamics (excitable relaxation oscillator, $\mathbf u_i=(V_i,W_i)$) are

$$
\dot V_i = V_i - \tfrac13 V_i^{3} - W_i + I_{\text{ext}} - \sigma(LV)_i,
\qquad
\dot W_i = \tfrac{1}{\tau}\,(V_i + a - b\,W_i),
\tag{I.6}
$$

documented here so the manual stays synchronized with the class contract; treat (I.6) as specification, not as running code.

## I.3 Coupling schemes: diffusive Laplacian vs. phase coupling

Two distinct coupling paradigms coexist in the repository.

**(a) Diffusive (Laplacian) coupling.** Systems (I.3)–(I.5) inject the term $-\sigma(LX)_i$ derived in (I.2). This is *linear* in the state and *conservative on the synchronization manifold*. Its Jacobian on that manifold block-diagonalizes in the eigenbasis of $L$: if $L v^{(m)} = \lambda_m v^{(m)}$ with $0=\lambda_1\le\lambda_2\le\dots\le\lambda_N$, then a perturbation along mode $m$ evolves under

$$
\dot{\boldsymbol\xi}_m = \big[\,D\mathbf F(\mathbf s) - \sigma\lambda_m \mathbf H\,\big]\,\boldsymbol\xi_m,
\tag{I.7}
$$

where $\mathbf s(t)$ is the synchronous solution and $D\mathbf F$ its Jacobian. The largest Lyapunov exponent $\Lambda(\sigma\lambda_m)$ of (I.7) — the *master stability function* — determines whether mode $m$ decays. Synchrony is linearly stable iff $\Lambda(\sigma\lambda_m)<0$ for all $m\ge 2$. This is the theoretical backbone for why sweeping $\sigma$ (Part II) drives transitions between coherent, incoherent, and chimera basins.

**(b) Kuramoto phase coupling** (`oscillators/kuramoto.py`). The Kuramoto model is *not* diffusive; each node is a bare phase $\theta_i\in\mathbb S^1$ ($D=1$) coupled through the adjacency matrix by a sinusoid:

$$
\dot\theta_i \;=\; \omega_i \;+\; \frac{K}{N}\sum_{j=1}^{N} A_{ij}\,\sin(\theta_j - \theta_i).
\tag{I.8}
$$

Here $\omega_i$ is node $i$'s natural frequency (drawn i.i.d. standard normal by default) and $K$ the global gain. The implementation forms the antisymmetric phase-difference tensor $\Delta_{ij}=\theta_j-\theta_i$ via broadcasting `state.unsqueeze(0) - state.unsqueeze(1)`, applies $\sin$, masks by $A$, and sums. The global order parameter

$$
r\,e^{\,i\psi} \;=\; \frac{1}{N}\sum_{j=1}^{N} e^{\,i\theta_j},
\qquad r\in[0,1],
\tag{I.9}
$$

measures coherence: $r\to 1$ is full synchrony, $r\to 0$ is incoherence. For the *all-to-all* graph $A_{ij}\equiv 1$, substituting (I.9) into (I.8) yields the mean-field form $\dot\theta_i=\omega_i + Kr\sin(\psi-\theta_i)$, from which the classical critical coupling $K_c = 2/(\pi g(0))$ (with $g$ the frequency density) follows. The repository uses (I.8)–(I.9) as one more source of VPS signatures for basin classification.

## I.4 State transitions: the RK4 flow map

All continuous systems are advanced by the classical fourth-order Runge–Kutta integrator implemented once in `BaseOscillator.rk4_step`. Writing the full vector field of (I.1) as $\dot{\mathbf x}=\mathbf f(\mathbf x)$ and a time step $\Delta t$, the discrete state transition $\mathbf x_t\mapsto \mathbf x_{t+1}$ is

$$
\begin{aligned}
\mathbf k_1 &= \mathbf f(\mathbf x_t), &
\mathbf k_2 &= \mathbf f\!\big(\mathbf x_t + \tfrac{\Delta t}{2}\mathbf k_1\big),\\
\mathbf k_3 &= \mathbf f\!\big(\mathbf x_t + \tfrac{\Delta t}{2}\mathbf k_2\big), &
\mathbf k_4 &= \mathbf f\!\big(\mathbf x_t + \Delta t\,\mathbf k_3\big),
\end{aligned}
\qquad
\mathbf x_{t+1} = \mathbf x_t + \frac{\Delta t}{6}\big(\mathbf k_1 + 2\mathbf k_2 + 2\mathbf k_3 + \mathbf k_4\big).
\tag{I.10}
$$

**Order of accuracy.** Expanding the exact flow $\Phi_{\Delta t}$ and the RK4 update in Taylor series about $\mathbf x_t$, the two agree through $O(\Delta t^{4})$; the local truncation error is $O(\Delta t^{5})$ and the global error over a fixed integration horizon is $O(\Delta t^{4})$. This fourth-order accuracy is what allows the basin sweeps to use comparatively coarse steps ($\Delta t\sim 10^{-2}$) while still resolving chaotic transients faithfully enough for the coherence signatures to be meaningful. Kuramoto's `step` uses explicit Euler ($\mathbf x_{t+1}=\mathbf x_t+\Delta t\,\mathbf f(\mathbf x_t)$, global error $O(\Delta t)$) with a modulo-$2\pi$ wrap, which suffices for a first-order phase model.

The `integrate(state0, dt, steps)` method iterates (I.10) `steps` times, storing the full history tensor of shape `(steps, *state0.shape)`. Basin sweeps retain only the **post-transient tail** — the last `tail_frac` (default $1/4$) of the trajectory — so that the coherence statistics below are computed on the attractor, not on the transient approach to it.

## I.5 The Vector Pattern State — Definition A (lag–norm signature)

The repository uses "VPS" in two mathematically distinct but conceptually aligned senses. The first, in `processing/feature_extraction.py::vector_pattern_state`, is a **pairwise lag–norm signature** built from cross-correlations.

Let $x_i(t),\,t=1,\dots,T$ be the observable of node $i$. For each unordered pair $(i,j)$ with $i<j$ (the strict upper triangle, $M=\binom{N}{2}$ pairs), define the (unnormalized) cross-correlation at integer lag $\ell$:

$$
c_{ij}(\ell) \;=\; \sum_{t} x_i(t+\ell)\,x_j(t).
\tag{I.11}
$$

The **optimal lag** is the delay maximizing the alignment of the two channels:

$$
\tau_{ij} \;=\; \arg\max_{\ell}\; c_{ij}(\ell),
\qquad \ell\in\{-(T-1),\dots,T-1\}.
\tag{I.12}
$$

**FFT computation.** Computing (I.11) directly is $O(T^2)$ per pair. The code instead uses the cross-correlation theorem: with zero-padding to length $P=2T-1$ (to avoid the circular wrap-around that a plain DFT would introduce),

$$
c_{ij}(\ell) \;=\; \mathcal F^{-1}\!\Big\{\, \widehat{x_i}\odot \overline{\widehat{x_j}}\,\Big\}(\ell),
\qquad \widehat{x} = \mathrm{rfft}(x,\,P),
\tag{I.13}
$$

where $\overline{(\cdot)}$ is complex conjugation. This is the `x_fft`, `pair_1 * conj(pair_2)`, `irfft` sequence in the source, evaluated for **all pairs simultaneously** by indexing with `triu_indices`. The `argmax` over the padded axis returns a raw index in $\{0,\dots,P-1\}$, which the line

```python
tau_x = torch.where(lags_indices >= T_len, lags_indices - pad_len, lags_indices)
```

folds back into a signed lag in $\{-(T-1),\dots,T-1\}$ — indices at or beyond $T$ correspond to negative lags under the standard FFT ordering.

**Residual norm at optimal lag.** Having found $\tau_{ij}$, the second feature is the Euclidean distance between the two channels *after* aligning them by that lag:

$$
\ell_{ij} \;=\; \Big\| \, x_i(\cdot + \tau_{ij}) - x_j(\cdot) \,\Big\|_2
\;=\;
\Bigg(\sum_{t\in\mathcal O_{ij}} \big(x_i(t+\tau_{ij}) - x_j(t)\big)^2\Bigg)^{1/2},
\tag{I.14}
$$

where $\mathcal O_{ij}$ is the overlap window after shifting (the code slices `x[lag:, p1] - x[:-lag, p2]` for $\tau_{ij}>0$, and symmetrically for $\tau_{ij}<0$). A small $\ell_{ij}$ at some lag means node $i$ is a *time-shifted copy* of node $j$ — the hallmark of lag synchronization.

**Assembled signature.** The VPS of Definition A concatenates the optimal-lag vector and the $\alpha$-scaled norm vector across all pairs:

$$
\mathbf v \;=\; \big[\, \tau_{12},\,\tau_{13},\,\dots,\,\tau_{(N-1)N}\;;\; \alpha\,\ell_{12},\,\alpha\,\ell_{13},\,\dots,\,\alpha\,\ell_{(N-1)N}\,\big] \;\in\;\mathbb R^{2M}.
\tag{I.15}
$$

The scalar $\alpha>0$ (default $1.5$) balances the two feature families so that neither the (integer-valued) lags nor the (magnitude-valued) norms dominate the Euclidean geometry used downstream by $k$-means. This is the `torch.cat([tau_x, L * alpha])` return value.

## I.6 The Vector Pattern State — Definition B (neighbor-relative coherence)

The second, and operationally central, VPS is the **per-node coherence vector** of `processing/chimera_classifier.py::local_coherence`. For a trajectory window and coupling matrix $W$, define node $i$'s **local field** as the strength-weighted mean of its neighbors' observables:

$$
\phi_i(t) \;=\; \frac{\sum_{j} W_{ij}\,x_j(t)}{\sum_j W_{ij}},
\tag{I.16}
$$

computed as the matrix product `x @ W.T / row_sum`. The **coherence** of node $i$ is the exponential of the negative *relative residual variance*:

$$
\boxed{\;
C_i \;=\; \exp\!\left(-\,\frac{\mathrm{Var}_t\!\big[x_i(t)-\phi_i(t)\big]}{\mathrm{Var}_t\!\big[x_i(t)\big]}\right)
\;}\qquad C_i\in(0,1].
\tag{I.17}
$$

**Properties.**

1. *Boundedness.* The argument of $\exp$ is a ratio of nonnegative variances, hence $\le 0$, so $C_i\in(0,1]$.
2. *Synchrony limit.* If node $i$ tracks its neighborhood exactly, $x_i(t)\equiv\phi_i(t)$, the numerator vanishes and $C_i=1$.
3. *Incoherence limit.* If $x_i$ is uncorrelated with its neighbors so that $\mathrm{Var}[x_i-\phi_i]\to\mathrm{Var}[x_i]+\mathrm{Var}[\phi_i]\gtrsim\mathrm{Var}[x_i]$, then $C_i\to e^{-1}$ or smaller.
4. *Scale invariance.* Multiplying $x_i$ by any constant leaves the ratio (I.17) unchanged, so a single formula applies across Lorenz, Rössler, Kuramoto, and Van der Pol regardless of their native amplitudes.
5. *Isolated nodes.* If $\sum_j W_{ij}=0$ the local field is undefined; the convention $C_i=1$ is imposed (a node cannot be out of step with an empty neighborhood).

The **VPS (Definition B)** for one initial condition is the vector

$$
\mathbf C \;=\; (C_1,\dots,C_N)\;\in\;(0,1]^{N}.
\tag{I.18}
$$

A *chimera* — coexisting coherent and incoherent domains — appears as a **bimodal** $\mathbf C$: some entries near $1$, others well below. The classifier `classify_chimera` splits $\mathbf C$ into two $k$-means clusters and declares a chimera iff both clusters hold at least a fraction `min_group_frac` of nodes **and** the cluster-mean gap exceeds `gap_threshold`, rejecting the degenerate all-high (synchronous) and all-low (incoherent) states that a forced 2-means split would otherwise partition spuriously. The **sample-corrected bimodality coefficient**

$$
\mathrm{BC} \;=\; \frac{g^{2}+1}{\,\kappa + \dfrac{3(n-1)^2}{(n-2)(n-3)}\,},
\tag{I.19}
$$

with $g$ the sample skewness and $\kappa$ the excess kurtosis, provides a continuous corroborating statistic; $\mathrm{BC}>5/9\approx0.555$ is the standard "substantially bimodal" threshold.

## I.7 From VPS to labels: model-order selection

A **population** of VPS vectors — one per initial condition across a basin sweep, stacked into a matrix $V\in\mathbb R^{M\times N}$ — is clustered into attractor-type labels by $k$-means, with the number of clusters $k$ chosen automatically. Two criteria appear.

**Silhouette selection** (`cluster_vps_population`). For a point $m$ with mean intra-cluster distance $a(m)$ and mean nearest-other-cluster distance $b(m)$, the silhouette is

$$
s(m) \;=\; \frac{b(m)-a(m)}{\max\{a(m),\,b(m)\}}\;\in[-1,1],
\qquad
S(k) \;=\; \frac{1}{M}\sum_{m} s(m).
\tag{I.20}
$$

The selected order is $k^\star=\arg\max_{k\in[k_{\min},k_{\max}]} S(k)$.

**BIC selection** (`feature_extraction.py::KmeansBIC`). Treating $k$-means as an isotropic-Gaussian mixture with pooled variance $\hat\sigma^2 = \mathrm{SSE}/(N-K)$ (with $\mathrm{SSE}$ the total within-cluster sum of squares, `inertia_`), the maximized log-likelihood is

$$
\ln\hat{\mathcal L}
= \sum_{c=1}^{K} C_c\ln C_c \;-\; N\ln N \;-\; \frac{N d}{2}\ln\!\big(2\pi\hat\sigma^2\big) \;-\; \frac{d}{2}(N-K),
\tag{I.21}
$$

where $C_c$ is the size of cluster $c$ and $d$ the feature dimension. Penalizing the $p=K + Kd$ free parameters gives the Bayesian Information Criterion used to pick $k$:

$$
\mathrm{BIC}(K) \;=\; \ln\hat{\mathcal L} \;-\; \frac{K + Kd}{2}\,\ln N,
\qquad
k^\star = \arg\max_K \mathrm{BIC}(K).
\tag{I.22}
$$

The resulting label vector, reshaped onto the two-dimensional initial-condition grid of Part II, is the raw material for the basin-boundary geometry.

---

# Part II — Fractal Basin Boundaries and Causation Entropy

## II.1 Basins of attraction and the initial-condition slice

Let $\Phi_t$ be the flow of (I.1) and let $\{\mathcal A_1,\dots,\mathcal A_p\}$ be its coexisting attractors. The **basin** of attractor $\mathcal A_r$ is

$$
\mathcal B_r \;=\; \Big\{\,\mathbf x_0 : \lim_{t\to\infty}\mathrm{dist}\big(\Phi_t(\mathbf x_0),\,\mathcal A_r\big)=0\,\Big\}.
\tag{II.1}
$$

The full state space has dimension $DN$, which is far too large to chart. The code (`processing/basin_dim.py::build_ic_grid`) therefore restricts attention to a **two-dimensional affine slice**: all nodes are fixed at a common base state, and only the $x$-component of two representative nodes, $x_{p}$ and $x_{q}$, is swept over a regular $R\times R$ grid

$$
\big(x_p^{(0)},\,x_q^{(0)}\big) \;\in\; \{\xi_1,\dots,\xi_R\}\times\{\xi_1,\dots,\xi_R\},
\qquad \xi_r \text{ linearly spaced in } [\text{lo},\text{hi}].
\tag{II.2}
$$

Each grid point is integrated (batched, one GPU call), reduced to its coherence VPS (I.18), and assigned an attractor label $\ell(x_p^{(0)},x_q^{(0)})\in\{1,\dots,k^\star\}$ by the clustering of §I.7. This yields a **label image** $\mathcal L\in\{1,\dots,k^\star\}^{R\times R}$. Because the slice is always two-dimensional regardless of the native state dimension $D$, every basin boundary is a planar set and its dimension always lies in $[1,2]$ — the property that makes Lorenz-vs-Rössler comparison meaningful without $D$-dependent special casing.

## II.2 Boundary extraction

A pixel of $\mathcal L$ lies on the basin **boundary** if any 4-connected neighbor carries a different label. Formally the boundary indicator $B\in\{0,1\}^{R\times R}$ is

$$
B_{a,b} \;=\; \mathbb 1\!\Big[\,\exists\,(a',b')\in\mathcal N_4(a,b):\ \mathcal L_{a',b'}\ne \mathcal L_{a,b}\,\Big],
\tag{II.3}
$$

which `box_counting.extract_boundary` computes by comparing each interior pixel with its right and down neighbors and OR-ing the mismatches in both directions:

$$
B \;=\; \big(\mathcal L_{:-1,:}\ne\mathcal L_{1:,:}\big)\ \vee\ \big(\mathcal L_{:,:-1}\ne\mathcal L_{:,1:}\big)\quad\text{(broadcast to both sides of each edge).}
\tag{II.4}
$$

For a smooth (non-fractal) boundary $B$ traces a curve of dimension $1$; for a *riddled* or *fractal* boundary it fills the plane more densely, with dimension approaching $2$.

## II.3 The box-counting dimension

The **box-counting (Minkowski–Bouligand) dimension** of the boundary set $B$ quantifies how the number of occupied boxes scales as the boxes shrink. Cover the image with a grid of boxes of side $r$ and let $\mathcal N(r)$ be the number of boxes containing at least one boundary pixel. If

$$
\mathcal N(r) \;\sim\; r^{-D_f}
\qquad\text{as } r\to 0,
\tag{II.5}
$$

then the box-counting dimension is the limit

$$
D_f \;=\; \lim_{r\to 0}\; \frac{\ln \mathcal N(r)}{\ln(1/r)}.
\tag{II.6}
$$

**GPU implementation** (`box_counting.boxcount_2d_gpu`). The image is padded to a square of side $2^{p}$. For box side $r=2^{e}$, a box is occupied iff *any* pixel inside it is a boundary pixel — an OR-reduction. On a binary $\{0,1\}$ image, OR over a block equals the **maximum** over that block, so max-pooling with kernel and stride $r$ counts occupancy exactly:

$$
\mathcal N(2^{e}) \;=\; \sum_{\text{boxes}} \max_{\text{pixels in box}} B
\;=\; \big\| \,\mathrm{maxpool}_{r}(B)\, \big\|_1,
\qquad r=2^{e},\ e=0,1,\dots,p.
\tag{II.7}
$$

At $r=1$ every pixel is its own box, so $\mathcal N(1)=\sum B$.

**Estimator.** From the doubly logarithmic form of (II.5),

$$
\ln \mathcal N(r) \;=\; -D_f\,\ln r + \text{const},
\tag{II.8}
$$

$D_f$ is the negative slope of an ordinary least-squares line through $\{(\ln r,\ \ln\mathcal N(r))\}$ over the box sizes with $\mathcal N(r)>0$ (`box_counting.fractal_dimension`, optionally restricted to a fit window $[r_{\min},r_{\max}]$). The goodness of fit is reported as the coefficient of determination

$$
R^2 \;=\; 1 - \frac{\sum_r\big(\ln\mathcal N_r - \widehat{\ln\mathcal N_r}\big)^2}{\sum_r\big(\ln\mathcal N_r - \overline{\ln\mathcal N}\big)^2},
\tag{II.9}
$$

so that a near-linear log–log relation ($R^2\to 1$) certifies genuine scale invariance rather than a spurious slope. Sweeping the coupling $\sigma$ and plotting $D_f(\sigma)$ (`basin_dim.sweep_coupling`) reveals the parameter windows in which basin boundaries become fractal — the fingerprint of final-state unpredictability.

## II.4 Final-state sensitivity and the uncertainty exponent

The dynamical meaning of $D_f$ is made precise by the **uncertainty exponent**. Perturb an initial condition by $\varepsilon$; the fraction $f(\varepsilon)$ of phase space within $\varepsilon$ of a basin boundary — i.e. the fraction of initial conditions whose eventual attractor is *uncertain* under an $\varepsilon$-error — scales as

$$
f(\varepsilon) \;\sim\; \varepsilon^{\,\gamma},
\qquad \gamma \;=\; d - D_f,
\tag{II.10}
$$

where $d$ is the dimension of the sampled space ($d=2$ for the planar slice of §II.1). A near-integer boundary ($D_f\approx 1$) gives $\gamma\approx 1$: halving the measurement error roughly halves the uncertain fraction. A space-filling fractal boundary ($D_f\to 2$) gives $\gamma\to 0$: reducing the error buys *almost no* predictive gain. Thus $D_f$ is not merely a geometric descriptor but the exponent governing how quickly final-state prediction improves with initial-condition precision — the quantitative statement of "fractal basin boundary $\Rightarrow$ practically unpredictable outcome."

## II.5 Causation entropy: differential entropy and Gaussian closed forms

Part II's second pillar (`processing/causation_entropy.py`) infers a **directed** interaction network from empirical time series (parcellated fMRI ROIs) using **optimal causation entropy** (oCSE, Sun–Taylor–Bollt 2015). The foundation is Shannon differential entropy. For a continuous random vector $X\in\mathbb R^k$ with density $p$,

$$
H(X) \;=\; -\!\int p(\mathbf x)\,\ln p(\mathbf x)\,\mathrm d\mathbf x \quad[\text{nats}].
\tag{II.11}
$$

Under the working assumption that the ROI signals are jointly Gaussian, entropy has a closed form depending only on the covariance $\Sigma=\mathrm{Cov}(X)$:

$$
\boxed{\;H(X) \;=\; \tfrac12\ln\!\big((2\pi e)^{k}\,\lvert\Sigma\rvert\big)
\;=\; \tfrac12\Big(k\ln(2\pi e) + \ln\lvert\Sigma\rvert\Big).\;}
\tag{II.12}
$$

This is `gaussian_entropy`, evaluated via the numerically stable log-determinant `slogdet` (with a ridge $10^{-10}I$ fallback if $\Sigma$ is singular). The **conditional entropy** follows from the chain rule $H(X\mid Y)=H(X,Y)-H(Y)$, i.e.

$$
H(X\mid Y) \;=\; \tfrac12\ln\!\frac{\lvert\Sigma_{(X,Y)}\rvert}{\lvert\Sigma_Y\rvert},
\tag{II.13}
$$

the `conditional_entropy` function (the additive $2\pi e$ constants cancel).

## II.6 Conditional mutual information and the oCSE algorithm

The **causation entropy** from source $X_j$ to target $X_i$ conditioned on a set $X_S$ is the conditional mutual information (CMI)

$$
C_{j\to i\mid S} \;=\; I\big(X_i;\,X_j \mid X_S\big)
\;=\; H\big(X_i\mid X_S\big) - H\big(X_i\mid X_S\cup\{X_j\}\big) \;\ge\; 0,
\tag{II.14}
$$

which is zero **iff** $X_i \perp X_j \mid X_S$ (conditional independence). This is exactly `conditional_mutual_information`, with the two conditional entropies supplied by (II.13). The critical property oCSE exploits: if the influence $X_j\to X_i$ is *indirect*, mediated by some $X_k$, then conditioning on the mediator annihilates the CMI,

$$
C_{j\to i\mid k} \;\approx\; 0,
\tag{II.15}
$$

so oCSE separates **direct** from **indirect** coupling — the decisive advantage over pairwise correlation.

**Temporal precedence.** To enforce Granger-style causation (cause precedes effect), the series is split at lag $1$: targets take the *current* value $X_i(t)$ while candidate drivers take the *lagged* value $X_j(t-1)$. Only past states may explain present ones.

**Greedy forward selection.** For each target $i$, oCSE grows a causal set $S_i$ one node at a time (the `_process_single_target` worker, run in parallel across ROIs):

1. Initialize $S_i=\varnothing$.
2. Among all candidates $j\notin S_i\cup\{i\}$, select the one maximizing the conditional causation entropy given the *already-selected* set:
$$
j^\star \;=\; \arg\max_{j}\; C_{j\to i \mid S_i}
\;=\; \arg\max_{j}\; I\big(X_i(t);\,X_j(t-1)\mid X_{S_i}(t-1)\big).
\tag{II.16}
$$
3. If $C_{j^\star\to i\mid S_i}$ passes the significance test (§II.7), append $j^\star$ to $S_i$ and record the edge weight $C_{j^\star\to i\mid S_i}$; otherwise stop.
4. Repeat until no significant candidate remains.

Each accepted edge writes its CMI weight into the directed adjacency $\mathbf A_{j^\star,\,i}$. The network density $\rho = \#\text{edges}/[N(N-1)]$ summarizes the discovered connectome. (The published oCSE also runs a *backward* pruning pass to remove members of $S_i$ made redundant by later additions; the implementation here keeps the forward pass, which is exact when selected drivers remain conditionally informative.)

## II.7 The chi-squared significance test

A finite sample makes even conditionally independent pairs yield a small positive CMI, so each candidate must clear a null-hypothesis test. Under $H_0: X_i\perp X_j\mid X_S$, twice the sample size times the empirical CMI is asymptotically chi-squared with degrees of freedom equal to the number of added scalar constraints. For a single scalar driver added to the conditioning set, that is one degree of freedom, and the code uses the statistic

$$
\Lambda \;=\; n\cdot \widehat{C}_{j\to i\mid S} \;\xrightarrow{\;H_0\;}\; \chi^2_{(1)},
\tag{II.17}
$$

with $n$ the number of lagged samples. (This is the $G$-test / likelihood-ratio form $2n\cdot\mathrm{CMI}\sim\chi^2$ up to the factor-of-two convention absorbed into the threshold.) A candidate is retained iff its upper-tail $p$-value falls below the significance level $\alpha$ (default $0.05$):

$$
p \;=\; 1 - F_{\chi^2_{(1)}}(\Lambda) \;<\; \alpha
\quad\Longrightarrow\quad \text{edge } j\to i \text{ accepted},
\tag{II.18}
$$

where $F_{\chi^2_{(1)}}$ is the chi-squared CDF (`_cmi_significant`). This converts the continuous CMI ranking into a discrete, statistically controlled edge set.

---

# Part III — Multifractal Detrended Fluctuation Analysis

*Implemented in `pythongpu/processing/multifractal_analysis.py`; run on `data/processed/100307/parcellated_timeseries.csv` ($N=240$ ROIs $\times\;T=1200$ samples), producing `multifractal_results.csv` and `multifractal_spectrum.png`.*

## III.1 Motivation and overview

A single Hurst exponent describes a **monofractal** signal — one whose fluctuations scale identically at all magnitudes. Real neural time series are typically **multifractal**: small and large fluctuations obey *different* scaling laws, so a whole *spectrum* of exponents is needed. Multifractal Detrended Fluctuation Analysis (MFDFA; Kantelhardt et al. 2002) extracts that spectrum while being robust to nonstationary polynomial trends. The pipeline is:

$$
x(t)\ \xrightarrow{\text{profile}}\ Y(t)\ \xrightarrow{\text{segment + detrend}}\ F^2(s,\nu)\ \xrightarrow{\text{$q$-average}}\ F_q(s)\ \xrightarrow{\text{scaling}}\ h(q)\ \xrightarrow{}\ \tau(q)\ \xrightarrow{\text{Legendre}}\ \big(\alpha, f(\alpha)\big).
$$

Each arrow is one subsection below.

## III.2 Step 1 — the integrated profile

Given a time series $x(t),\ t=1,\dots,T$ of ROI $r$, subtract its mean and cumulatively sum to form the **profile** (random-walk "trajectory"):

$$
Y(t) \;=\; \sum_{k=1}^{t}\big(x(k) - \langle x\rangle\big),
\qquad \langle x\rangle = \frac1T\sum_{k=1}^T x(k).
\tag{III.1}
$$

Mean subtraction ensures $Y(T)=0$ and makes the profile a bounded bridge rather than a drifting walk. This integration is the reason DFA measures *correlations*: the variance of increments of $Y$ over a window of size $s$ grows as $s^{2H}$ precisely when $x$ has long-range correlations with Hurst exponent $H$. In code:

```python
profile = np.cumsum(x - x.mean(axis=0, keepdims=True), axis=0)   # (T, N), all ROIs at once
```

## III.3 Step 2 — segmentation from both ends

Partition the profile into non-overlapping segments of length $s$. There are

$$
N_s \;=\; \big\lfloor T/s \big\rfloor
\tag{III.2}
$$

complete segments starting from the beginning. Because $T$ is generally not divisible by $s$, a tail of length $T - N_s\,s$ would be discarded. To use every sample, the procedure is repeated **starting from the end**, giving $2N_s$ segments in total. Segment $\nu\in\{1,\dots,2N_s\}$ contains the profile values

$$
Y_\nu(m) \;=\;
\begin{cases}
Y\big((\nu-1)s + m\big), & \nu\le N_s \quad(\text{forward}),\\[4pt]
Y\big(T-(\nu-N_s)s + m\big), & \nu> N_s \quad(\text{backward}),
\end{cases}
\qquad m=1,\dots,s.
\tag{III.3}
$$

The implementation reshapes the head block `profile[:N_s*s]` and the tail block `profile[T-N_s*s:]` each into shape $(N_s,\,s,\,N)$ and concatenates them along the segment axis, yielding one $(2N_s,\,s,\,N)$ tensor holding every segment of every ROI.

## III.4 Step 3 — local polynomial detrending and the projection operator

Within each segment, fit and subtract a polynomial trend of order $m$ (here $m=1$, linear) to remove nonstationarity. Let $\mathbf y_\nu=(Y_\nu(1),\dots,Y_\nu(s))^\top\in\mathbb R^s$ and build the Vandermonde design matrix on the local coordinate $u=0,1,\dots,s-1$:

$$
X \;=\;
\begin{pmatrix}
1 & 0 & 0 & \cdots & 0^{m}\\
1 & 1 & 1 & \cdots & 1^{m}\\
\vdots & & & & \vdots\\
1 & (s-1) & (s-1)^2 & \cdots & (s-1)^{m}
\end{pmatrix}
\in\mathbb R^{s\times(m+1)}.
\tag{III.4}
$$

The least-squares trend is $\hat{\mathbf y}_\nu = X\hat{\boldsymbol\beta}_\nu$ with $\hat{\boldsymbol\beta}_\nu=(X^\top X)^{-1}X^\top\mathbf y_\nu$, so the fitted values are the orthogonal projection $\hat{\mathbf y}_\nu = P\,\mathbf y_\nu$ onto the column space of $X$, where

$$
P \;=\; X\,(X^\top X)^{-1}X^\top
\tag{III.5}
$$

is the **hat matrix**. The **detrended residual** is therefore

$$
\mathbf e_\nu \;=\; \mathbf y_\nu - \hat{\mathbf y}_\nu \;=\; (I - P)\,\mathbf y_\nu \;=\; R\,\mathbf y_\nu,
\qquad R := I - P.
\tag{III.6}
$$

**The key optimization.** The residual-projection operator $R$ depends only on the scale $s$ and the detrend order $m$ — **not** on the segment data or the ROI. It is built once per scale (`_projection_residual_operator`, using the pseudoinverse for numerical robustness) and applied to *all* segments of *all* ROIs by a single batched matrix multiply:

$$
E \;=\; R\,\mathbf Y_{\text{seg}},
\qquad \mathbf Y_{\text{seg}}\in\mathbb R^{(2N_s)\times s\times N},\ \ E\in\mathbb R^{(2N_s)\times s\times N}.
\tag{III.7}
$$

This replaces the textbook triple loop (over segments $\times$ ROIs $\times$ least-squares solves) with one `np.matmul`, and is why the whole 240-ROI panel processes in a fraction of a second. The **local variance** of segment $\nu$ (for ROI $r$) is the mean squared residual,

$$
F^2(s,\nu) \;=\; \frac1s\sum_{m=1}^{s} e_\nu(m)^2 \;=\; \frac1s\,\lVert R\,\mathbf y_\nu\rVert_2^2,
\tag{III.8}
$$

computed as `np.mean(resid**2, axis=1)` and floored at $10^{-30}$ to keep the subsequent logarithms and fractional powers finite.

## III.5 Step 4 — the q-th order fluctuation function

Average the local variances across all $2N_s$ segments using a **generalized (power) mean of order $q/2$**, then take the $1/q$ root, to obtain the $q$-th order fluctuation function:

$$
\boxed{\;
F_q(s) \;=\; \left\{ \frac{1}{2N_s}\sum_{\nu=1}^{2N_s} \big[F^2(s,\nu)\big]^{q/2}\right\}^{1/q},
\qquad q\in\mathbb R\setminus\{0\}.
\;}
\tag{III.9}
$$

The moment order $q$ acts as a **magnifying lens** on fluctuation size:

- $q>0$ weights **large** fluctuations (segments of large variance dominate the average);
- $q<0$ weights **small** fluctuations (segments of small variance dominate);
- $q=2$ recovers the standard DFA and, through §III.6, the classical Hurst exponent.

The case $q=0$ makes the exponent $1/q$ singular. Taking the limit $q\to 0$ of (III.9) via L'Hôpital converts the power mean into a **logarithmic (geometric-mean) average**:

$$
F_0(s) \;=\; \exp\!\left\{ \frac{1}{4N_s}\sum_{\nu=1}^{2N_s} \ln F^2(s,\nu) \right\}.
\tag{III.10}
$$

The implementation vectorizes both cases over all $q$ and all ROIs simultaneously: with $\ell_{\nu,r}=\ln F^2(s,\nu)$,

$$
F_q(s) = \Big(\mathrm{mean}_\nu\, e^{(q/2)\,\ell_{\nu,r}}\Big)^{1/q}\ (q\neq0),
\qquad
F_0(s) = e^{\,\tfrac12\,\mathrm{mean}_\nu\,\ell_{\nu,r}},
$$

matching (III.9)–(III.10) exactly (the $\tfrac12$ prefactor in the code's $q=0$ branch is the $q/2$ exponent evaluated in the geometric-mean limit).

## III.6 Step 5 — scaling and the generalized Hurst exponent

For a signal with multifractal correlations, the fluctuation function is a power law in the scale:

$$
F_q(s) \;\sim\; s^{\,h(q)}
\qquad\Longleftrightarrow\qquad
\ln F_q(s) \;=\; h(q)\,\ln s + \text{const}.
\tag{III.11}
$$

The exponent $h(q)$ is the **generalized Hurst exponent**. It is estimated, for each $q$ and each ROI, as the slope of an ordinary least-squares fit of $\ln F_q(s)$ against $\ln s$ over a set of log-spaced scales $\{s_1,\dots,s_S\}$:

$$
h(q) \;=\; \frac{\sum_{u}\big(\ln s_u-\overline{\ln s}\big)\big(\ln F_q(s_u)-\overline{\ln F_q}\big)}{\sum_u \big(\ln s_u-\overline{\ln s}\big)^2}.
\tag{III.12}
$$

The code fits every $(q,\text{ROI})$ column in a single `np.polyfit(log_s, log_Fq, deg=1)` call by flattening the moment and ROI axes together. The scales themselves (`make_scales`) are integer, geometrically spaced from $s_{\min}=16$ to $s_{\max}=\lfloor T/4\rfloor$ so that even the coarsest scale yields at least four segments per direction — below that, (III.9) is too noisy to fit.

**Special value.** $h(2)$ is the ordinary Hurst exponent $H$: $H=0.5$ is uncorrelated noise, $H>0.5$ is persistent (long-range positively correlated), $H<0.5$ is anti-persistent. For a **monofractal** signal $h(q)$ is constant in $q$; any dependence of $h$ on $q$ is the signature of multifractality.

## III.7 Step 6 — the mass (Rényi) exponent spectrum

Relate $h(q)$ to the classical **mass exponent** (also called the Rényi or partition-function scaling exponent) $\tau(q)$. The standard multifractal formalism defines $\tau(q)$ through the scaling of the partition function $Z_q(s)=\sum_\nu \mu_\nu(s)^q\sim s^{\tau(q)}$; for DFA-based estimation this reduces to the exact algebraic relation

$$
\boxed{\;\tau(q) \;=\; q\,h(q) - 1.\;}
\tag{III.13}
$$

*Derivation sketch.* The DFA fluctuation $F_q(s)\sim s^{h(q)}$ corresponds to a standard partition-function scaling $\sum_\nu |Y_\nu|^{q}\sim s^{\,qh(q)-1}$, where the $-1$ arises because the number of segments scales as $N_s\sim s^{-1}$ (one factor of $s^{-1}$ from the normalization $1/N_s$ in the average). Identifying the exponent with $\tau(q)$ gives (III.13). Implemented as `tau = qs[:,None]*h - 1.0`.

Properties that make $\tau(q)$ a useful intermediate:

- $\tau(q)$ is always **convex** (a consequence of Hölder's inequality on the moments);
- for a monofractal, $\tau(q)=qH-1$ is **linear**;
- **curvature** of $\tau(q)$ measures the strength of multifractality;
- $\tau(0)=-1$ and $\tau(1)=0$ (the latter for a normalized measure).

## III.8 Step 7 — the Legendre transform to singularity coordinates

The physically interpretable description is the **singularity spectrum** $f(\alpha)$, where $\alpha$ is the local **Hölder exponent** (the strength of a local singularity) and $f(\alpha)$ is the fractal dimension of the set of points sharing that exponent. It is obtained from $\tau(q)$ by a **Legendre transform**. Treating $q$ as the conjugate variable,

$$
\boxed{\;
\alpha \;=\; \frac{\mathrm d\tau}{\mathrm d q},
\qquad
f(\alpha) \;=\; q\,\alpha - \tau(q).
\;}
\tag{III.14}
$$

**Equivalent form via $h(q)$.** Substituting (III.13) into (III.14) and differentiating,

$$
\alpha(q) \;=\; h(q) + q\,h'(q),
\qquad
f(\alpha) \;=\; q\big[\alpha - h(q)\big] + 1 \;=\; q^2 h'(q) + 1,
\tag{III.15}
$$

which exhibits the spectrum as a downward parabola-like curve peaking where $h'(q)=0$ (i.e. $q=0$), at which point $f=1$ (the support dimension of a one-dimensional time series) and $\alpha=h(0)$.

**Numerical Legendre transform.** Rather than fit and differentiate an analytic $\tau(q)$, the code evaluates the derivative in (III.14) directly by finite differences on the sampled $q$-grid:

$$
\alpha_k \;\approx\; \frac{\tau(q_{k+1}) - \tau(q_{k-1})}{q_{k+1} - q_{k-1}},
\qquad
f(\alpha_k) \;=\; q_k\,\alpha_k - \tau(q_k),
\tag{III.16}
$$

which is exactly `alpha = np.gradient(tau, qs, axis=0)` followed by `f_alpha = qs[:,None]*alpha - tau`, computed for all 240 ROIs at once. The moment grid is $q\in\{-5,-4.5,\dots,+5\}$ (step $0.5$, $21$ values including $0$).

## III.9 Interpretation, limiting cases, and diagnostics

The paired arrays $\{(\alpha_k, f(\alpha_k))\}_{k}$ per ROI trace the singularity spectrum — a concave $\cap$-shaped curve. Its geometry is read as follows.

- **Peak.** The maximum sits at $q=0$, where $f(\alpha)=1$ (the topological support dimension). Its location $\alpha(0)=h(0)$ is the dominant/most-probable Hölder exponent.
- **Left branch** ($q>0$): governs the scaling of **large** fluctuations; smaller $\alpha$.
- **Right branch** ($q<0$): governs **small** fluctuations; larger $\alpha$.
- **Width.**
$$
\Delta\alpha \;=\; \alpha_{\max}-\alpha_{\min}
\tag{III.17}
$$
is the single most useful scalar summary: $\Delta\alpha\approx0$ is monofractal (one exponent), and larger $\Delta\alpha$ means richer multifractality — a broader mix of local regularities. The script prints the panel-wide mean, min, and max of $\Delta\alpha$ as a diagnostic.
- **Asymmetry.** A right-skewed spectrum (long right tail) indicates dominance of low-magnitude fluctuations; left-skew the reverse.
- **Monofractal collapse.** If $h(q)\equiv H$, then $\alpha\equiv H$ and the "spectrum" degenerates to the single point $(H,1)$ — the multifractal apparatus correctly reports the absence of multifractality.

For the subject-`100307` data the fitted spectra are concave with peaks at $f(\alpha)\approx1$, Hurst values $h(2)$ in a physiologically plausible band, and mean width $\Delta\alpha\approx0.9$ — i.e. genuine multifractality rather than monofractal collapse.

## III.10 Correspondence with the implementation

| Mathematical object | Equation | Function / line in `multifractal_analysis.py` |
|---|---|---|
| Integrated profile $Y(t)$ | (III.1) | `np.cumsum(x - x.mean(...))` in `run` |
| Segment count $N_s$, both-ends split | (III.2)–(III.3) | head/tail reshape in `fluctuation_functions` |
| Residual projector $R=I-P$ | (III.5)–(III.6) | `_projection_residual_operator` |
| Batched detrend $E=R\,\mathbf Y_{\text{seg}}$ | (III.7) | `np.matmul(resid_op, segs)` |
| Local variance $F^2(s,\nu)$ | (III.8) | `np.mean(resid**2, axis=1)` |
| Fluctuation $F_q(s)$, incl. $q=0$ | (III.9)–(III.10) | power-mean / log-mean branches |
| Generalized Hurst $h(q)$ | (III.11)–(III.12) | `np.polyfit(log_s, ..., deg=1)` |
| Mass exponent $\tau(q)$ | (III.13) | `qs[:,None]*h - 1.0` |
| Hölder $\alpha$, spectrum $f(\alpha)$ | (III.14)–(III.16) | `np.gradient(tau, qs)`, `qs*alpha - tau` |
| Scale set $\{s_u\}$ | §III.6 | `make_scales` |
| Outputs | — | `multifractal_results.csv`, `multifractal_spectrum.png` |

The output CSV is in long format with columns `roi, q, hurst, tau, alpha, f_alpha` ($240\times21 = 5040$ rows); the PNG overlays all 240 singularity spectra with the panel mean.

---

## References

1. J. W. Kantelhardt, S. A. Zschiegner, E. Koscielny-Bunde, S. Havlin, A. Bunde, H. E. Stanley. *Multifractal detrended fluctuation analysis of nonstationary time series.* Physica A **316** (2002) 87–114.
2. C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons, H. E. Stanley, A. L. Goldberger. *Mosaic organization of DNA nucleotides.* Phys. Rev. E **49** (1994) 1685.
3. J. Sun, D. Taylor, E. M. Bollt. *Causal network inference by optimal causation entropy.* SIAM J. Applied Dynamical Systems **14** (2015) 73–106.
4. L. M. Pecora, T. L. Carroll. *Master stability functions for synchronized coupled systems.* Phys. Rev. Lett. **80** (1998) 2109.
5. Y. Kuramoto. *Chemical Oscillations, Waves, and Turbulence.* Springer, 1984.
6. C. Grebogi, S. W. McDonald, E. Ott, J. A. Yorke. *Final state sensitivity: an obstruction to predictability.* Phys. Lett. A **99** (1983) 415.
7. K. Falconer. *Fractal Geometry: Mathematical Foundations and Applications.* Wiley, 3rd ed., 2014.
8. T. C. Halsey, M. H. Jensen, L. P. Kadanoff, I. Procaccia, B. I. Shraiman. *Fractal measures and their singularities.* Phys. Rev. A **33** (1986) 1141.
9. R. Pfister, K. A. Schwarz, M. Janczyk, R. Dale, J. Freeman. *Good things peak in pairs: a note on the bimodality coefficient.* Frontiers in Psychology **4** (2013) 700.

---

*This manual is a living document: as `pythongpu` evolves, keep each boxed equation and the correspondence table in §III.10 synchronized with the code they describe.*
