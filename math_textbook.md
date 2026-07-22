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
    - [I.5a The alignment convention — and a discrepancy with the reference implementation](#i5a-the-alignment-convention--and-a-discrepancy-with-the-reference-implementation)
    - [I.5b The lag–norm confound](#i5b-the-lagnorm-confound)
    - [I.5c Cost: the FFT is not the bottleneck](#i5c-cost-the-fft-is-not-the-bottleneck)
  - [I.6 The Vector Pattern State — Definition B (neighbor-relative coherence)](#i6-the-vector-pattern-state--definition-b-neighbor-relative-coherence)
  - [I.6a The Vector Pattern State — Definition C (streaming surrogate), and what it is not](#i6a-the-vector-pattern-state--definition-c-streaming-surrogate-and-what-it-is-not)
  - [I.6b The alternative-norm question: $L^1$ and cosine distance for the $\ell$ feature](#i6b-the-alternative-norm-question-l1-and-cosine-distance-for-the-ell-feature)
  - [I.7 From VPS to labels: model-order selection](#i7-from-vps-to-labels-model-order-selection)
    - [I.7a Consensus selection: elbow, BIC, silhouette, and the structure-vs-noise null guard](#i7a-consensus-selection-elbow-bic-silhouette-and-the-structure-vs-noise-null-guard)
    - [I.7b Emergent attractor counts: DBSCAN with an auto-selected radius](#i7b-emergent-attractor-counts-dbscan-with-an-auto-selected-radius)
  - [I.8 Lobe-locking: an exact, clustering-free basin label](#i8-lobe-locking-an-exact-clustering-free-basin-label)
    - [I.8a The mechanism](#i8a-the-mechanism)
    - [I.8b The label](#i8b-the-label)
    - [I.8c Validity: the label is only as good as the locking](#i8c-validity-the-label-is-only-as-good-as-the-locking)
    - [I.8d Consequence for basin geometry](#i8d-consequence-for-basin-geometry)
- [Part II — Fractal Basin Boundaries and Causation Entropy](#part-ii--fractal-basin-boundaries-and-causation-entropy)
  - [II.1 Basins of attraction and the initial-condition slice](#ii1-basins-of-attraction-and-the-initial-condition-slice)
  - [II.2 Boundary extraction](#ii2-boundary-extraction)
  - [II.3 The box-counting dimension](#ii3-the-box-counting-dimension)
  - [II.4 Final-state sensitivity and the uncertainty exponent](#ii4-final-state-sensitivity-and-the-uncertainty-exponent)
  - [II.4a Basin entropy and the Daza Wada-suspect criterion](#ii4a-basin-entropy-and-the-daza-wada-suspect-criterion)
  - [II.4b The grid (dilation) Wada-bounds test](#ii4b-the-grid-dilation-wada-bounds-test)
  - [II.4c Control-theoretic perturbation sensitivity: the $P_{\text{flip}}(\delta)$ test](#ii4c-control-theoretic-perturbation-sensitivity-the-p_textflipdelta-test)
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
- [Part IV — Covariant Lyapunov Vectors, Riddling, and Spectral Graph Theory](#part-iv--covariant-lyapunov-vectors-riddling-and-spectral-graph-theory)
  - [IV.1 Entropic regression: the backward elimination pass](#iv1-entropic-regression-the-backward-elimination-pass)
  - [IV.2 Structural–functional fusion](#iv2-structuralfunctional-fusion)
  - [IV.3 Covariant Lyapunov vectors and the Ginelli algorithm](#iv3-covariant-lyapunov-vectors-and-the-ginelli-algorithm)
  - [IV.4 The Lyapunov spectrum from the R-diagonals](#iv4-the-lyapunov-spectrum-from-the-r-diagonals--dimension-counting-part-1)
  - [IV.5 The Kaplan–Yorke dimension](#iv5-the-kaplanyorke-dimension--dimension-counting-part-2)
  - [IV.6 Transversality angles and riddled basins](#iv6-transversality-angles-and-riddled-basins)
    - [IV.6a The transverse Lyapunov exponent and the riddling criterion](#iv6a-the-transverse-lyapunov-exponent-and-the-riddling-criterion)
    - [IV.6b The CLV-angle proxy and the k-means detector](#iv6b-the-clv-angle-proxy-and-the-k-means-detector)
  - [IV.7 Spectral graph theory: the Laplacian eigenbasis, synchronizability, and random-graph baselines](#iv7-spectral-graph-theory-the-laplacian-eigenbasis-synchronizability-and-random-graph-baselines)
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

**Hindmarsh–Rose network** (`oscillators/hindmarsh_rose.py`, defaults $a=1,\ b=3,\ c=1,\ d=5,\ r=0.006,\ s=4,\ x_{\text{rest}}=-1.6,\ I=3.2$), a slow–fast neuron model with $\mathbf u_i=(X_i,Y_i,Z_i)$ and $\mathbf H=\mathrm{diag}(1,0,0)$:

$$
\begin{aligned}
\dot X_i &= Y_i - a\,X_i^{3} + b\,X_i^{2} - Z_i + I \;-\; \sigma\,(LX)_i,\\
\dot Y_i &= c - d\,X_i^{2} - Y_i,\\
\dot Z_i &= r\big(s\,(X_i - x_{\text{rest}}) - Z_i\big).
\end{aligned}
\tag{I.5a}
$$

$X_i$ is the fast membrane potential, $Y_i$ the fast recovery current, and $Z_i$ a slow adaptation variable whose small rate constant $r\ll1$ separates the two time scales: on the fast $(X,Y)$ plane the system spikes, while $Z$ integrates the spiking history over the much longer time scale $1/r\approx167$ and organizes it into bursts. This time-scale separation has a direct consequence for basin sweeps: with the default parameters the network needs an integration horizon far longer than the $t_{\max}=30$ window sufficient for the other three systems before trajectories on the affine 2-slice diverge into distinguishable long-time behavior — at short horizons the coherence field saturates and every basin sweep collapses to one trivial partition, which is why the coupled-network experiments in Part II default to Lorenz rather than Hindmarsh–Rose for the topology and node-switching studies.

**Chua's-circuit network** (`oscillators/chua.py`, canonical double-scroll parameters $\alpha=15.6,\ \beta=28,\ m_0=-8/7,\ m_1=-5/7$), a third autonomous chaotic exemplar with $\mathbf u_i=(X_i,Y_i,Z_i)$ and $\mathbf H=\mathrm{diag}(1,0,0)$:

$$
\begin{aligned}
\dot X_i &= \alpha\big(Y_i - X_i - f(X_i)\big) \;-\; \sigma\,(LX)_i,\\
\dot Y_i &= X_i - Y_i + Z_i,\\
\dot Z_i &= -\beta\,Y_i,
\end{aligned}
\tag{I.5b}
$$

where $f$ is the piecewise-linear **Chua-diode** characteristic

$$
f(X) \;=\; m_1 X + \tfrac12(m_0-m_1)\big(\lvert X+1\rvert - \lvert X-1\rvert\big),
\tag{I.5c}
$$

a three-segment odd-symmetric nonlinear resistor with inner slope $m_0$ (for $\lvert X\rvert<1$) and outer slope $m_1$ (for $\lvert X\rvert>1$). At the canonical parameters the uncoupled unit exhibits the double-scroll attractor: a single bounded chaotic trajectory whose $X$-coordinate visits both signs while remaining confined, the two "scrolls" corresponding to the two outer linear regions of $f$.

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

### I.5a The alignment convention — and a discrepancy with the reference implementation

Definition A is a port of `VectorPatternState.m` (Fish, 2023), the implementation behind Bollt et al. (2023). The port **does not reproduce it numerically**, and the reason is a one-sample convention difference in (I.14).

MATLAB's `xcorr` returns the lag axis `lagsx` $\in\{-(T-1),\dots,T-1\}$, so $\tau_{ij}=$ `lagsx(f)` is a **physical lag**. The reference then forms the residual as

```matlab
L(jk) = norm( x(1:end-(lag-1), j) - x(lag:end, i) );   % lag > 0
```

Because MATLAB indexes from $1$, the slice `x(lag:end)` begins at sample $\tau_{ij}$, i.e. it applies a shift of $\tau_{ij}-1$. The physical lag is being used as a **1-based index**. At $\tau_{ij}=1$ both slices are the full series and *no shift is applied at all*, despite a detected lag of one. The port instead shifts by $\tau_{ij}$, as (I.14) states. Writing $s$ for the applied shift, the two conventions are

$$
s_{\text{ref}} \;=\; \max\big(\lvert\tau_{ij}\rvert-1,\,0\big),
\qquad
s_{\text{port}} \;=\; \lvert\tau_{ij}\rvert .
\tag{I.15a}
$$

Since $\tau_{ij}$ maximizes the alignment (I.12), evaluating the residual at any other shift can only worsen it. The reference's systematic under-shift therefore **inflates** $\ell_{ij}$. Measured on the reference test matrix `Example_A_3.mat`: $\ell^{\text{ref}}_{ij}\ge\ell^{\text{port}}_{ij}$ on **36/36 pairs**, mean ratio $\mathbf{1.96}$. The lags $\tau_{ij}$ themselves are unaffected by the convention, and agree exactly on that input; they disagree on `Example_A_1/2` (max difference $2$), a separate issue traceable to tie-breaking — MATLAB's `find(cx == max(cx))` returns *every* maximizer while `argmax` returns the first.

Which convention is *correct* and which is *faithful* are different questions, so `vector_pattern_state_fast` exposes both via `alignment='corrected'|'matlab'`; the discrepancy is then a measurement rather than an argument.

### I.5b The lag–norm confound

Equation (I.14) sums over the overlap window $\mathcal O_{ij}$, whose length is $T-\lvert\tau_{ij}\rvert$, and $\lVert\cdot\rVert_2$ is **not** normalized by that length. Hence

$$
\mathbb E\big[\ell_{ij}^2\big] \;\propto\; \big(T-\lvert\tau_{ij}\rvert\big)
\tag{I.15b}
$$

for residuals of fixed per-sample scale: $\ell_{ij}$ shrinks mechanically as $\lvert\tau_{ij}\rvert$ grows, for purely combinatorial reasons unrelated to the dynamics. **The two halves of the signature (I.15) are therefore not independent** — $\ell$ carries a deterministic imprint of $\tau$ — yet they are concatenated and handed to a Euclidean $k$-means as if they were. A length-normalized residual $\ell_{ij}/\sqrt{T-\lvert\tau_{ij}\rvert}$, or a fixed common window, would remove the confound; neither the reference nor the port does this.

### I.5c Cost: the FFT is not the bottleneck

The batched form (I.13) reduces the lag search to $O(T\log T)$ for *all* $M$ pairs at once. The residual (I.14), however, is evaluated in the original implementation by looping over every candidate lag value,

```python
for lag_val in range(-T_len + 1, T_len):   # 2T-1 iterations
```

which is $2T-1$ host-side iterations ($19{,}999$ at the production $T=10^4$) and dominates the runtime, negating the FFT's advantage. Benchmarked at $N=83$ ($M=3403$ pairs) against a serial CPU reference of the same algorithm, the looped GPU implementation attains only $3.05\times$ ($T=256$) to $6.54\times$ ($T=2048$), and is $0.83\times$ — *slower than serial* — at $T=128$.

The loop is removable. Each pair has its own $\tau_{ij}$, so rather than iterating over candidate lags one builds per-pair aligned index grids and gathers once:

$$
\mathcal I^{a}_{t,ij} = t + s_{ij}\mathbb 1[\tau_{ij}>0],\qquad
\mathcal I^{b}_{t,ij} = t + s_{ij}\mathbb 1[\tau_{ij}<0],\qquad
\text{valid}_{t,ij} = \mathbb 1\big[\mathcal I^{a}_{t,ij}<T\big]\wedge\mathbb 1\big[\mathcal I^{b}_{t,ij}<T\big],
\tag{I.15c}
$$

after which $\ell_{ij}=\lVert(x_{\mathcal I^a}-x_{\mathcal I^b})\odot\text{valid}\rVert_2$ along the time axis. This is `vector_pattern_state_fast`, and it attains $233\times$–$375\times$ over the same serial reference (a further $\sim57\times$ over the looped version) while matching it to float32 precision ($\max\lvert\Delta\ell\rvert=1.9\times10^{-6}$).

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

## I.6a The Vector Pattern State — Definition C (streaming surrogate), and what it is not

Definitions A and B both require the **whole trajectory** to be resident: A cross-correlates $x_i$ against $x_j$ over all lags, B needs the time series to form variances. For a basin sweep this is fatal. At grid $64^2$ with $N=83$ and $T=10^4$ the trajectory tensor is $O(T\cdot B\cdot N\cdot 3)\approx41$ GB, and the production grid is $361^2$.

`pipeline/lorenz_sweep.py::run_sweep_streaming` therefore accumulates pairwise statistics **online** (Welford), never storing a trajectory, at $O(B\cdot M)$ memory. For each pair it forms

$$
\tilde\tau_{ij} \;=\; \frac{\big\langle\,\lvert X_i-X_j\rvert\,\big\rangle_t}{\sqrt{\mathrm{Var}_t\big[\lvert X_i-X_j\rvert\big]}},
\qquad
\tilde\ell_{ij} \;=\; \big\langle\,\lVert \mathbf x_i-\mathbf x_j\rVert_2\,\big\rangle_t ,
\tag{I.18a}
$$

each block independently $z$-scored across the batch before concatenation (the two live on incommensurable scales, and raw concatenation lets $\tilde\ell$ dominate every Euclidean distance).

**This is not Definition A, and the distinction is not cosmetic.** $\tilde\tau$ is a coefficient of variation of the *instantaneous* separation — a dimensionless reciprocal noise-to-signal ratio. It is **not a time lag**; no cross-correlation is computed and no lag is searched. Likewise $\tilde\ell$ is a mean instantaneous distance, with **no alignment** applied. Under a pure time shift $x_j(t)=x_i(t-\Delta)$ — perfect lag synchronization, which Definition A registers as $\tau_{ij}=\Delta,\ \ell_{ij}\approx0$ — the surrogate reports a large $\tilde\ell$ and an unremarkable $\tilde\tau$. The surrogate is therefore **blind to phase-lagged synchrony**, which is precisely the structure the VPS was introduced to detect (Bollt et al. 2023) and the defining signature of a chimera.

The surrogate is a legitimate coherence feature in its own right; it is simply a *different* one. Every basin-sweep result in this repository was computed from (I.18a), not from (I.15) — a memory constraint silently substituting one quantity for another is a failure mode worth stating explicitly, because nothing downstream announces it.

## I.6b The alternative-norm question: $L^1$ and cosine distance for the $\ell$ feature

Bollt et al. (2023) [26] fix their pairwise similarity as an $L^2$ (Euclidean) measure and say so explicitly, flagging alternative norms as unexplored future work — a gap their own Discussion names but does not close. Definition C's $\tilde\ell_{ij}$ of (I.18a) is exactly this: $\lVert\mathbf x_i-\mathbf x_j\rVert_2$, time-averaged. Two alternatives replace only this one norm, leaving $\tilde\tau_{ij}$ and the streaming/Welford machinery of §I.6a untouched:

$$
\tilde\ell^{(1)}_{ij} \;=\; \big\langle\, \lVert \mathbf x_i - \mathbf x_j\rVert_1 \,\big\rangle_t
\;=\; \big\langle\, \lvert X_i-X_j\rvert+\lvert Y_i-Y_j\rvert+\lvert Z_i-Z_j\rvert \,\big\rangle_t,
\tag{I.18b}
$$

$$
\tilde\ell^{(\cos)}_{ij} \;=\; \Big\langle\, 1 - \frac{\mathbf x_i(t)\cdot\mathbf x_j(t)}{\lVert\mathbf x_i(t)\rVert\,\lVert\mathbf x_j(t)\rVert} \,\Big\rangle_t .
\tag{I.18c}
$$

These are not interchangeable rescalings of the same quantity. $\tilde\ell^{(1)}$ measures separation magnitude like $\tilde\ell_{ij}$ but weights each coordinate axis linearly rather than quadratically, so it is comparatively less dominated by whichever axis happens to have the largest instantaneous spread — for the Lorenz system, typically $Y$ or $Z$ during a lobe excursion. $\tilde\ell^{(\cos)}$ is qualitatively different in kind: it discards magnitude entirely and measures only whether $\mathbf x_i(t)$ and $\mathbf x_j(t)$ point the *same direction* in phase space, so two nodes deep in the same lobe but at different radii from the fixed point score as coherent, whereas $\tilde\ell_{ij}$ and $\tilde\ell^{(1)}$ would report them as separated.

**Implementation** (`pipeline/lorenz_sweep.py::run_sweep_streaming`, parameter `norm ∈ {"l2","l1","cosine"}`, threaded through `pipeline/lorenz_fine_coupling_sweep.py` as `--vps-norm`). The branch sits at the one line that forms $\tilde\ell_{ij}$ each step: `l2`/`l1` operate on the pairwise difference `x0[:,i,:] - x0[:,j,:]` exactly as (I.18a)/(I.18b) require, while `cosine` reads the raw state slices `x0[:,i,:]`, `x0[:,j,:]` directly, since (I.18c) needs the two vectors separately, not their difference. Default `norm="l2"` reproduces prior output bit-for-bit.

**Preliminary finding.** A smoke-scale comparison (grid $24^2$, $K=0.5$, $t_{\max}=10$ — far below production length, reported only as a sanity check that the branch does something, not as a result) gave $D_f=1.669$ ($L^2$), $1.671$ ($L^1$), $1.486$ (cosine): the two magnitude-based norms agree closely, while the direction-based norm differs by a margin well outside plausible sampling noise at this scale. The clustering-free lobe-sign label (I.28) was, as it must be, identical across all three ($\gamma_{\text{sign}}=0.311$ in every run) — confirming the norm swap touches only the VPS/$k$-means path of §I.7, not the independent labeling of §I.8. A production-resolution rerun of all three norms on the same $K$, node pair, and grid is the actual experiment; this is only evidence the plumbing is correct and the effect is worth chasing.

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

### I.7a Consensus selection: elbow, BIC, silhouette, and the structure-vs-noise null guard

`processing/basin_clustering.py::select_optimal_clusters` supersedes a once hand-transcribed constant ("k=8 was found optimal") with an auditable sweep over $k\in[k_{\min},k_{\max}]$ combining three criteria on a $z$-scored, optionally PCA-reduced feature matrix $X$.

**Elbow (Kneedle).** The within-cluster sum of squares $W(k)=\mathrm{SSE}(k)$ (`inertia_`) is monotone decreasing; its knee is located by the **Kneedle** chord-distance heuristic (Satopää et al. 2011). Both axes are min–max normalized to $x_k,y_k\in[0,1]$, and the knee is the point of maximal perpendicular distance from the chord joining the curve's endpoints $(x_0,y_0)\to(x_1,y_1)$:

$$
d(k) \;=\; \frac{\big\lvert (y_1-y_0)x_k - (x_1-x_0)y_k + x_1y_0 - y_1x_0\big\rvert}{\sqrt{(y_1-y_0)^2+(x_1-x_0)^2}},
\qquad
k_{\text{elbow}} \;=\; \arg\max_k d(k).
\tag{I.23}
$$

The identical Kneedle formula (I.23) is reused verbatim by `pipeline/attractor_id.py` to locate the knee of the sorted $k$-distance graph in §I.7b — a single scale-free knee-finder serves both an inertia curve and a nearest-neighbor-distance curve.

**Gaussian-mixture BIC.** Distinct from the isotropic pooled-variance BIC of (I.21)–(I.22), here a **full-covariance** Gaussian mixture with $k$ components is fit at each candidate $k$ and scored by `sklearn`'s canonical
$\mathrm{BIC}(k) = -2\ln\hat{\mathcal L}(k) + p(k)\ln n$, with $p(k)$ the free-parameter count of a full-covariance $k$-component mixture in $d$ dimensions; $k_{\text{bic}}=\arg\min_k\mathrm{BIC}(k)$. Allowing full covariance (rather than the isotropic assumption of §I.7) guards against the elbow's tendency to over-segment anisotropic clusters.

**Silhouette.** As in (I.20), $k_{\text{silhouette}}=\arg\max_k S(k)$, subsampled to a fixed budget for tractability on wide VPS blocks.

**Consensus.** The reported cluster count is the rounded median of the three optima,
$$
k^\star \;=\; \mathrm{round}\big(\mathrm{median}(k_{\text{elbow}},\,k_{\text{bic}},\,k_{\text{silhouette}})\big),
\tag{I.24}
$$
clipped to $[k_{\min},k_{\max}]$ — a majority-vote-like estimate that is robust to any single criterion's idiosyncratic bias, with all three curves retained for audit.

**Structure-vs-noise null guard.** A near-degenerate feature blob (e.g. a synchronized regime with no real basin structure) can still yield a spurious elbow/BIC optimum $>1$. To prevent fabricating basins out of noise, the best silhouette achievable on the real features is compared against the same statistic computed on a **structureless reference** of matched shape. The features are declared **structured** — and clustering trusted — only if the *effect size* clears a margin:

$$
\boxed{\;
\Delta \;=\; s^\star_{\text{real}} \;-\; \mathrm{P}_{95}\big(s^\star_{Z}\big) \;\ge\; \Delta_{\min}
\;}
\tag{I.25}
$$

with $Z$ the reference cloud. If (I.25) fails, $k^\star$ is forced to $1$ — the count is measured, not tuned, and a flat feature landscape is reported honestly as "no attractors distinguishable" rather than an arbitrary split.

Each ingredient of (I.25) exists because a simpler version of the test **failed on real data**, and the failures are instructive.

**(i) The reference must not inherit the real marginals.** The original null permuted each feature column independently, destroying joint structure while preserving every marginal. On outlier-heavy features this is catastrophic: $k$-means maximizes the silhouette by isolating a handful of extreme points, and it can do so equally well on the *shuffled* matrix, because shuffling preserves exactly the tails that make the isolation possible. Measured on the $K\in[0.45,0.65]$ sweep, the shuffle null reached $s^\star_{Z}\approx0.72$–$0.94$ where a null must sit near $0$ (it correctly gave $0.069$ at $K=0$, where the features are genuinely featureless). "Structure" was then being declared on margins as thin as $0.018$ between two numbers both $\approx0.94$. The reference is now drawn from a **single covariance-matched Gaussian**,
$$
Z \sim \mathcal N\!\big(\hat\mu,\ \hat\Sigma\big),
\qquad \hat\mu = \mathrm{mean}(X),\quad \hat\Sigma = \mathrm{cov}(X),
\tag{I.25a}
$$
which is *unimodal by construction* yet reproduces the real elliptical spread — the correct null for the question actually being asked ("one blob, or several?"). It is well calibrated where it matters: on unimodal data it reproduces the data's own score ($\Delta\approx0.00$ on both a Gaussian blob and a heavy-tailed $t_{1.2}$ blob), while genuine three-cluster structure clears it by $\Delta\approx+0.38$. A uniform bounding-box reference (the Tibshirani gap-statistic convention) is retained as an option but is shape-mismatched — a box is easier to beat than an ellipsoid.

**(ii) Only balanced partitions may count.** A split whose smallest cluster holds a handful of points is an outlier split, not a basin, and scores a near-perfect silhouette while conveying nothing. The maximization in $s^\star$ is therefore restricted to partitions satisfying
$$
\min_c \frac{\lvert\{i : \text{label}_i = c\}\rvert}{n} \;\ge\; \rho_{\min}
\tag{I.25b}
$$
(default $\rho_{\min}=0.05$). If no $k$ admits a balanced partition, the features are unstructured by definition.

**(iii) The margin must exceed the statistic's own scatter.** $s^\star$ is itself a random variable, and on heavy-tailed features its run-to-run scatter is substantial: on one fixed configuration ($n=3000$, $d=60$, $t_{1.2}$) it moved $0.172\to0.076$ **on the random seed alone**. A margin of $0.05$ sits *inside* that scatter, and duly produced a false "structured" verdict at $\Delta=+0.053$ that vanished on reseeding. A threshold crossable by reseeding is not a threshold; hence $\Delta_{\min}=0.15$ by default, comfortably below the $\approx+0.4$ of genuine structure and well above the noise.

The general lesson is worth stating once: **an estimator that cannot abstain is not evidence.** The elbow (I.23) always returns a knee, $k$-means always returns $k$ groups, and a silhouette test with an inflated null always finds structure. Each must be given an explicit way to answer "nothing here," and each such mechanism must itself be validated against data whose answer is already known.

### I.7b Emergent attractor counts: DBSCAN with an auto-selected radius

`pipeline/attractor_id.py` addresses a structural limitation of every $k$-means-based scheme above: $k$-means always returns exactly $k$ nonempty groups, so it can never report "there is only one attractor here" as a *discovered* fact rather than an edge case of a sweep. The alternative is density-based clustering, whose cluster count is an **output**, not an input.

**Descriptors.** Each initial condition is reduced to an *attractor-invariant* summary — either the coherence VPS $\mathbf C$ of (I.18) already stored from a completed sweep, or, when integrating from scratch, the per-node long-time moments of the fast coordinate over the recorded (post-transient) window,
$$
\bar X_i = \langle X_i\rangle,\qquad
\sigma_{X_i} = \sqrt{\langle X_i^2\rangle-\bar X_i^2},\qquad
\overline{\lvert X_i\rvert} = \langle\lvert X_i\rvert\rangle,
\tag{I.26}
$$
concatenated into a $3N$-vector per initial condition. Being long-time averages, all of (I.26) are (to sampling error) constant along a single trajectory once it has settled onto its attractor, exactly as required of an attractor-invariant descriptor.

**Auto-radius via the $k$-distance knee.** After standardizing and (if wide) PCA-reducing the descriptor matrix, DBSCAN's neighborhood radius $\varepsilon$ is not hand-set but read off the data: for each point, the distance to its $m$-th nearest neighbor ($m=$ `min_samples`) is computed, these distances are sorted ascending into the **$k$-distance graph**, and $\varepsilon_0$ is set to the value at the graph's knee — located by the same Kneedle formula (I.23), applied to the sorted-distance curve instead of an inertia curve. This is the standard DBSCAN radius-selection heuristic (Ester et al. 1996; Kneedle: Satopää et al. 2011).

**Radius-sensitivity scan.** DBSCAN is run at $\varepsilon\in\{0.7\varepsilon_0,\ 1.0\varepsilon_0,\ 1.4\varepsilon_0\}$; the attractor count is the number of non-noise clusters at $\varepsilon_0$, and the estimate is flagged **radius-stable** only if all three scales agree. An emergent count is trustworthy exactly when it is insensitive to the arbitrary choice of neighborhood scale — the analogue, for cluster *count*, of the $R^2$ goodness-of-fit check on $D_f$ in (II.9).

**Plateau vs. continuum.** Three points are too coarse to settle the question they are asked. Applied to the DTI-coupled Lorenz sweep, the scan returned counts of $1,1,39,4,6,3,11$ across $K$ with **not one** radius-stable verdict, and individual scans as violent as $64\to6\to2$ (and $46\to4\to9$, non-monotone). Such numbers are artifacts of where the knee happened to land.

The decisive test is the *shape* of the count-versus-radius curve. $n$ well-separated attractors hold the same count over a **broad** range of $\varepsilon$, because real density gaps do not care where the threshold is put; a continuum — one connected cloud of varying density — fragments monotonically with no flat region. `scan_radius` therefore sweeps $\varepsilon$ over $[\,0.25\varepsilon_0,\ 4\varepsilon_0\,]$ geometrically and reports the longest run of a constant count, measured in **radius decades** so the verdict is independent of sampling density:

$$
\mathcal P \;=\; \max\Big\{\ \log_{10}\tfrac{\varepsilon_b}{\varepsilon_a}\ :\ n(\varepsilon)\ \text{constant on}\ [\varepsilon_a,\varepsilon_b]\ \Big\},
\qquad \mathcal P \ge 0.3 \;\Rightarrow\; \text{report the count}.
\tag{I.26a}
$$

Validated on ground truth: three separated groups yield a plateau at $n=3$ spanning $0.75$ decades; a single blob yields a plateau at $n=1$ spanning $0.79$ decades — the correct answer in both cases. Absent a plateau, **no number is reported at all**.

**A structural blind spot.** Even so, density clustering has a failure mode that the plateau test cannot repair: it reports the number of *well-separated* groups. If a system possesses combinatorially many attractors whose descriptors densely fill a region — as §I.8 establishes is the case here — the descriptor cloud is one connected component, DBSCAN returns $n=1$, and no radius exhibits a plateau. **The verdict "one attractor" and the verdict "a continuum of attractors" are indistinguishable to this method.** The count of §I.8b, which never clusters at all, exists to resolve exactly this ambiguity.

## I.8 Lobe-locking: an exact, clustering-free basin label

Every construction in §I.7 estimates a label by clustering a continuous feature vector, and each inherits the pathologies catalogued there. For the Laplacian-coupled Lorenz network the estimation is unnecessary: the label is **discrete and directly observable**.

### I.8a The mechanism

The Lorenz attractor is symmetric under $(X,Y,Z)\mapsto(-X,-Y,Z)$ and carries two wings, centered on the fixed points $C^{\pm}=\big(\pm\sqrt{\beta(\rho-1)},\,\pm\sqrt{\beta(\rho-1)},\,\rho-1\big)$, i.e. $X\approx\pm7.8$ at the standard parameters. An **ergodic** trajectory visits both wings, so by symmetry its long-time mean satisfies $\langle X_i\rangle\to0$. A trajectory **locked** to a single wing instead yields $\langle X_i\rangle\approx\pm7.8$. The order parameter

$$
\Lambda \;=\; \frac{1}{N}\sum_{i=1}^{N}\big\lvert\langle X_i\rangle\big\rvert
\tag{I.27}
$$

therefore separates the two regimes cleanly, and the locked fraction is $\mathbb P\big[\lvert\langle X_i\rangle\rvert>\theta\big]$ with $\theta=4$ (any cut well inside $(0,7.8)$ serves).

Measured on a deterministic $32^2$ slice ($1024$ initial conditions, transient $100$, window $600$):

| $K$ | $\Lambda$ | locked |
|---|---|---|
| $0.00$ | $0.026$ | $0.0\%$ |
| $0.05$ | $1.202$ | $11.0\%$ |
| $0.08$ | $5.318$ | $66.0\%$ |
| $0.12$ | $6.847$ | $87.8\%$ |
| $0.20$ | $7.291$ | $95.1\%$ |
| $0.50$ | $6.934$ | $97.8\%$ |

Uncoupled, every node is ergodic and $\Lambda\to0$; coupled, the network pins its nodes to individual wings. The distribution of $\langle X_i\rangle$ is sharply **bimodal** at $\pm7$ once locked (at $K=0.5$, only $0.3\%$ of nodes lie in $\lvert\langle X\rangle\rvert<1$). The transition is a clean sigmoid with midpoint $K\approx0.07$.

### I.8b The label

Once locked, the network's asymptotic state is fully specified by *which* wing each node occupies:

$$
\boxed{\;
\mathbf b \;=\; \Big(\operatorname{sign}\langle X_1\rangle,\ \dots,\ \operatorname{sign}\langle X_N\rangle\Big)\ \in\ \{-1,+1\}^{N}
\;}
\tag{I.28}
$$

— an exact, discrete, $N$-bit label admitting up to $2^{83}\approx10^{25}$ values. It requires **no clustering, no $k$, no elbow, no silhouette, no null test**: the entire apparatus of §I.7, and every failure mode it guards against, is bypassed. The count of distinct realized $\mathbf b$ is then a direct measurement rather than an estimate.

At $K=0.5$, $1024$ initial conditions realize **$1010$ distinct patterns**, the largest basin holding $4$ ($0.4\%$); the mean Hamming distance between patterns is $41.5$ of $83$ bits — exactly the separation of *random* bit strings, so grid neighbours reach statistically independent patterns. The sampling is **saturated**: the true count is $\ge1010$ and bounded only by $B$.

This resolves the blind spot of §I.7b. The descriptors of $\sim10^3$ densely-packed attractors form one connected cloud, so DBSCAN reports $n=1$ with no plateau — a **false negative**, not a discovery of monostability. It also explains the pathologies of §I.7a: no $k$ is correct when the true count is astronomical, so the elbow's $6$–$8$ and the silhouette's $2$ are both meaningless, and neither the consensus nor any repair of it could have been right.

### I.8c Validity: the label is only as good as the locking

Definition (I.28) thresholds at zero, so where $\langle X_i\rangle\approx0$ the sign is decided by sampling noise. The label must therefore be validated, not assumed. Computing $\mathbf b$ over two **disjoint** windows of the same trajectories ($t\in[100,700]$ vs $[700,1300]$) gives:

| $K$ | locked | bits agreeing | exact $\mathbf b$ |
|---|---|---|---|
| $0.01$ | $0.0\%$ | $\mathbf{50.2\%}$ | $0.0\%$ |
| $0.06$ | $32.8\%$ | $91.1\%$ | $3.6\%$ |
| $0.10$ | $80.7\%$ | $98.0\%$ | $30.1\%$ |
| $0.20$ | $95.1\%$ | $98.7\%$ | $63.5\%$ |
| $0.50$ | $97.8\%$ | $\mathbf{99.9\%}$ | $94.6\%$ |

The $K=0.01$ row is the **negative control and it passes**: a $50.2\%$ coin flip exactly where the theory says the signs are noise. Agreement then tracks the locking monotonically to $99.9\%$. **The label is precisely as trustworthy as $\Lambda$ is large** — decisive at $K=0.5$, meaningless at $K=0.01$.

Note that the two right-hand columns measure different things and can disagree while both are correct: with $N=83$ independent bits, $0.98^{83}\approx0.19$, so $98\%$ per-bit agreement *necessarily* implies only $\sim30\%$ of patterns reproduce exactly. At $K=0.10$ roughly $1.5$ nodes per initial condition flicker; the pattern is mostly, but not wholly, permanent.

Two caveats bound the present evidence. First, $\Lambda$ measured on the second window **exceeds** the first at every coupling ($+26\%$ at $K=0.06$, still rising at $K=0.20$): the system is *still consolidating* at $t=1300$, most strongly near the onset — the critical slowing down expected near a bifurcation, and a warning that transients near $K\approx0.07$ far exceed the $100$ discarded here. Second, the natural refinement is to restrict (I.28) to the locked subset $\{i:\lvert\langle X_i\rangle\rvert>\theta\}$, since the unlocked minority supplies the flicker.

### I.8d Consequence for basin geometry

The riddling of Part II is then not an artifact but a prediction. With $\gtrsim10^3$ interleaved basins whose neighbours are uncorrelated, essentially every pixel of a basin map borders a differently-labelled pixel; the boundary is **space-filling**, so $D_f\to d$ and $\alpha=d-D_f\to0$ by (II.9) and (II.13). The observed $D_f\approx1.89$–$1.96$, $\alpha\approx0.04$, flat in $K$, is exactly this — and "flat in $K$" is expected, since locking (and hence riddling) persists across the whole swept range.

The basin *sizes* order the regimes. The largest basin holds $0.1\%$ of initial conditions below the onset (sign noise), rises to $9.3\%$ at $K\approx0.12$, and falls to $0.4\%$ by $K=0.5$. The reference coupling of Bollt et al. (`gel = 0.5`) thus sits deep in the riddled regime, where individual basins are too finely intermingled to map; the only window in which basins have appreciable measure is $K\approx0.08$–$0.12$ — where, per §I.8c, the labels are not yet fully permanent.

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

> **Cross-reference — the $\gamma\to0$ limit is not automatically a discovery.** Both estimators below read the exponent off a *labelled* image $\mathcal L$, so both inherit whatever produced the labels. A labelling in which neighbouring pixels are uncorrelated — whether because the basins are genuinely riddled, or because a clusterer was handed featureless data and returned $k$ groups anyway (§I.7a) — makes *every* pixel $\varepsilon$-uncertain at every radius, giving $f(\varepsilon)\approx\text{const}$, $\gamma\to0$, $D_f\to d$. **The exciting result and the null result are numerically identical**, and no goodness-of-fit on (II.10a) distinguishes them: both fit beautifully.
>
> The measured $D_f\approx1.89$–$1.96$, $\gamma\approx0.04$, *flat in $K$*, on the DTI-coupled Lorenz sweep is exactly this ambiguity. It is resolved not here but in §I.8: the labels are real, the basins genuinely number $\gtrsim10^3$ with uncorrelated neighbours, and $\gamma\to0$ is therefore a **prediction** of the lobe-locking mechanism (§I.8d) rather than an artifact. The resolution required a clustering-free label (I.28) and controls against known answers — *not* a better fit. Read $\gamma\to0$ as a question, and answer it upstream of the box count.

**Numerical estimator** (`box_counting.uncertainty_exponent`). On the discrete label image $\mathcal L$, a pixel is **$\varepsilon$-uncertain** if some $L^\infty$ perturbation of radius up to $\varepsilon$ pixels can move it into a different basin — i.e. if the $(2\varepsilon+1)\times(2\varepsilon+1)$ neighborhood spans more than one integer label. Since labels are integers, this indicator is exactly a max/min-filter mismatch:
$$
U_\varepsilon \;=\; \mathbb 1\!\big[\,\mathrm{maxfilter}_{2\varepsilon+1}(\mathcal L) \;\ne\; \mathrm{minfilter}_{2\varepsilon+1}(\mathcal L)\,\big],
\qquad
f(\varepsilon) \;=\; \langle U_\varepsilon\rangle,
\tag{II.10a}
$$
evaluated over a radius ladder $\varepsilon\in\{1,2,3,4,6,8,12,16\}$ pixels. Radii with a saturated ($f=1$) or empty ($f=0$) response carry no scaling information and are dropped; $\gamma$ is the OLS slope of $\ln f(\varepsilon)$ against $\ln\varepsilon$ over the remaining radii, exactly as in (II.8)–(II.9). Because $\gamma$ is a *scaling slope* rather than a raw pixel tally, it is invariant to the grid resolution used to sample $\Sigma$ — refining the grid rescales $\varepsilon$ uniformly and leaves the fitted slope unchanged — making $D_f=d-\gamma$ from (II.10) a grid-independent cross-check of the box-counting $D_f$ of §II.3, which *is* resolution-dependent through its pixelation. Agreement between the two estimators (e.g. $\gamma=0.259\Rightarrow D_f=1.741$ against a box-counting fit of $D_f=1.748$ on the same slice) certifies that the fractal signature is a property of the flow, not an artifact of one particular numerical method.

**Structural universality case study.** Cloning the degree sequence of the empirical `data/DTI-og.mat` connectome (83 nodes, 850 edges) via the configuration model (stub-matching; `networkx.configuration_model`) — which reproduces the exact degree sequence while destroying every higher-order structural feature — and comparing $\gamma$, $D_f$ against Erdős–Rényi and Barabási–Albert nulls at matched edge count (Lorenz network, $K=0.25$) gave $\gamma=0.245\pm0.068$ (configuration model), $0.172\pm0.029$ (ER), $0.198\pm0.083$ (BA): all three overlap within one standard deviation (cross-family spread $0.072$). The interpretation is that, for this substrate and coupling, basin fractality is set essentially by the **degree distribution** alone rather than by finer topological detail — a structurally universal result, with the $D_f=2-\gamma$ identity of (II.10) independently reproduced by both the uncertainty-exponent and box-counting estimators in every realization.

## II.4a Basin entropy and the Daza Wada-suspect criterion

A second, entropy-based route to quantifying basin-boundary complexity — complementary to the geometric $D_f$ of §II.3 — comes from Daza et al.'s **basin entropy** (`pipeline/universality_sweep.py::basin_entropy`). Cover the labeled slice $\mathcal L$ with disjoint $\varepsilon\times\varepsilon$ boxes (padding the image with a massless sentinel label so every box is full-sized). Within box $i$, let $p_{i,j}$ be the empirical fraction of the box's pixels carrying basin label $j$; the box's **Gibbs entropy** is
$$
S_i \;=\; -\sum_j p_{i,j}\,\ln p_{i,j}.
\tag{II.10b}
$$
Two averages of (II.10b) are reported:
$$
S_b \;=\; \big\langle S_i\big\rangle_{\text{all occupied boxes}}
\qquad\text{(basin entropy)},
\qquad\qquad
S_{bb} \;=\; \big\langle S_i\big\rangle_{\text{boundary boxes only}}
\qquad\text{(boundary basin entropy)},
\tag{II.10c}
$$
where a **boundary box** is one whose interior meets $\ge2$ distinct basin labels. $S_b$ summarizes uncertainty over the whole slice; $S_{bb}$ isolates it to the boundary region, so it is the more sensitive fractality diagnostic — a smooth 1-D boundary contributes only thin strips of mixed boxes to the average, while a space-filling boundary makes nearly every box a boundary box.

**Wada-suspect criterion.** With the natural-log convention, a boundary box that only ever straddles **two** basins has entropy bounded above by the two-outcome maximum $\ln 2$ (attained at the uniform $50/50$ split). Consequently
$$
\boxed{\; S_{bb} \;>\; \ln 2 \;\;\Longrightarrow\;\; \text{Wada-suspect} \;}
\tag{II.10d}
$$
is a **sufficient** (not necessary) condition for the boundary to have the **Wada property** — every boundary point borders *all* coexisting basins, not just two — since exceeding $\ln 2$ requires boundary boxes to routinely contain three or more basins with enough evenness to push the average past the two-basin ceiling. `basin_entropy` returns $S_b$, $S_{bb}$, the occupied/boundary box counts, and the boolean `wada_suspect` flag used throughout the coupling-sweep and topology-comparison experiments of `universality_sweep.py`.

## II.4b The grid (dilation) Wada-bounds test

A second, purely combinatorial certificate of the Wada property comes from the **grid (neighborhood) form** of the Daza et al. Wada test (`analyze_wada.py::daza_wada`), independent of the entropy functional of §II.4a. For each basin label $b$, form the binary indicator $I_b=\mathbb 1[\mathcal L=b]$ and morphologically dilate it by an $8$-connected structuring element of radius $r$ (default $r=1$). The per-pixel **coverage**
$$
C(x) \;=\; \sum_b \mathrm{dilate}_r\big(I_b\big)(x)
\tag{II.10e}
$$
counts how many distinct basins have a representative within radius $r$ of $x$. Background/masked pixels are excluded so they never inflate a boundary count. Three nested sets follow:
$$
\text{boundary} = \{x: C(x)\ge2\},\qquad
\text{Wada} = \{x: C(x)\ge3\},\qquad
\text{strict} = \{x: C(x)\ge n_{\text{basins}}\}\ (n_{\text{basins}}\ge3),
\tag{II.10f}
$$
i.e. a **Wada point** borders at least three basins, and a **strict** point borders *every* coexisting basin. The reported summary statistic, the **Wada-boundary coverage fraction**
$$
\mathrm{frac}_{\text{Wada}} \;=\; \frac{\lvert\text{Wada}\rvert}{\lvert\text{boundary}\rvert} \;\in\;[0,1],
\tag{II.10g}
$$
measures what proportion of the boundary is a genuine triple(-or-more)-junction rather than a simple two-basin edge; $\mathrm{frac}_{\text{Wada}}\approx1$ (with the strict fraction also near $1$) is the discrete analogue of declaring the whole boundary Wada. On the real coupling-sweep basin maps (5 coexisting basins, near-space-filling $D_f\approx1.999$), this test finds $\approx97\%$ Wada-boundary coverage but only $\approx2\%$ strict coverage — almost every boundary point borders three-or-more basins, but rarely all five at once — a finer-grained picture than the box-entropy sufficient condition (II.10d) alone provides, since (II.10e)–(II.10g) directly counts basin memberships rather than inferring them from an entropy bound.

## II.4c Control-theoretic perturbation sensitivity: the $P_{\text{flip}}(\delta)$ test

Riddled-basin theory makes a specific, falsifiable geometric claim, independent of any box-counting or entropy statistic: near a riddled point, *every* neighborhood — no matter how small — already contains points belonging to a different basin [19,20]. Bollt et al. (2023) [26] lean on exactly this property to argue that switching between coexisting network states requires only a "vanishingly small" perturbation, framing it as the dynamical mechanism behind rapid task-switching in the brain — but the paper stops at the geometric argument; no perturbation is actually injected and no switch is actually observed. This is the gap `pipeline/perturbation_sensitivity.py` closes.

**Definition.** Fix a base initial condition $\mathbf x_0$ and its asymptotic label $\mathbf b(\mathbf x_0)$ from (I.28). For a perturbation magnitude $\delta$ and a random unit direction $\hat{\mathbf u}$ in the full $3N$-dimensional state space, define the **flip indicator**

$$
F(\mathbf x_0,\delta,\hat{\mathbf u}) \;=\; \mathbb 1\!\big[\, \mathbf b(\mathbf x_0+\delta\hat{\mathbf u}) \;\ne\; \mathbf b(\mathbf x_0) \,\big],
\tag{II.10h}
$$

and the **flip probability** at scale $\delta$, averaged over $n_{\text{pts}}$ base points and $n_{\text{dir}}$ directions per point:

$$
\boxed{\;
P_{\text{flip}}(\delta) \;=\; \Big\langle\, F(\mathbf x_0,\delta,\hat{\mathbf u}) \,\Big\rangle_{\mathbf x_0,\,\hat{\mathbf u}}
\;}
\tag{II.10i}
$$

**The riddling signature is a statement about the $\delta\to0$ limit of (II.10i).** For a point strictly interior to an ordinary (non-riddled) basin, a finite-radius neighborhood is entirely one basin, so $P_{\text{flip}}(\delta)\to0$ as $\delta\to0$ — some minimum kick is required to reach the boundary. For a riddled point, no such neighborhood exists at any radius, so $P_{\text{flip}}(\delta)$ stays bounded away from $0$ even as $\delta\to$ machine precision. The two regimes are distinguished by the *shape* of the curve, not a single number: a decaying $P_{\text{flip}}(\delta)$ is an ordinary boundary; a curve that flattens toward a nonzero floor as $\delta$ shrinks is the riddling claim, made operational.

**Implementation.** Base points are drawn either uniformly on the initial-condition slice or, more informatively, from the clustering-free lobe-sign boundary/interior of §I.8 at a fixed reference coupling $K_{\text{ref}}$ (`--base-ic-mode boundary|interior`, located by `build_sign_slice`, which reuses (I.28) rather than the VPS/$k$-means path so the sampling itself carries no clustering artifact). For each $(\mathbf x_0,\delta,\hat{\mathbf u})$ triple, the perturbed and unperturbed states are integrated with the same batched `rk4_step_batched` used throughout Part I — the full $(n_{\text{pts}}\times n_{\text{dir}}\times n_\delta)$ grid is one flat batch, so this is embarrassingly parallel over exactly the same axis as a basin sweep. $\delta$ is swept geometrically, $\delta\in[10^{-8},10^{-1}]$, to resolve the small-$\delta$ asymptote against the finite-$\delta$ crossover.

**Status.** As of this writing the production sweep ($K\in\{0,0.1,0.5\}$ — uncoupled control, onset region, measured riddled regime — full $t_{\text{transient}}=100$, $t_{\max}=500$ integration) is running; the $P_{\text{flip}}(\delta)$ curves are the first direct experimental test of the control-theoretic claim in [26], on this system or any other reported in the literature to date.

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

Each accepted edge writes its CMI weight into the directed adjacency $\mathbf A_{j^\star,\,i}$. The network density $\rho = \#\text{edges}/[N(N-1)]$ summarizes the discovered connectome. The published method (Sun–Taylor–Bollt; and, in the neurological setting, Fish–Bollt) also runs a *backward* pruning pass that removes members of $S_i$ made redundant by later additions — the step that turns forward oCSE into full **entropic regression**. That backward pass is implemented in `processing/entropic_regression.py` and derived in §IV.1.

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

# Part IV — Covariant Lyapunov Vectors, Riddling, and Spectral Graph Theory

*Implemented in `pythongpu/pipeline/clv_diagnostics.py`, `clv_topology.py`, `clv_cli.py`, `baseline_models.py`, and `processing/entropic_regression.py`. Part IV takes the directed connectome of Part II and the coupled-oscillator field of Part I and answers two questions the earlier parts set up but did not close: (i) which reconstructed edges are genuinely **direct**, and (ii) what is the fine-grained stability geometry of the synchronization manifold — is it **riddled**? The spectral-graph-theory section (§IV.7) then explains, from the Laplacian eigenvalues alone, why the empirical connectome and its random-graph nulls sit where they do.*

## IV.1 Entropic regression: the backward elimination pass

Section II.6 built the directed connectome by *forward* greedy selection. Forward selection is order-dependent: a driver $X_j$ admitted early can be rendered redundant by a later admission $X_k$, yet the forward sweep never revisits it. **Entropic regression** (ER; Fish–DeWitt–AlMomani–Laurienti–Bollt 2021) closes this with a *backward* pass — the discriminating step for accurate reconstruction.

Given the forward-aggregated causal set $S_i$, backward elimination re-tests every retained driver *conditioned on all the others*:

$$
C_{j\to i \mid S_i\setminus\{j\}} \;=\; I\big(X_i(t);\,X_j(t-1)\,\big|\,X_{S_i\setminus\{j\}}(t-1)\big),
\qquad j\in S_i,
\tag{IV.1}
$$

and drops $j$ whenever this falls below the §II.7 significance threshold. Removing one edge can expose another as redundant, so the pass iterates to a fixed point (the `_backward_pass` loop in `entropic_regression.py`). Survivors are re-weighted with their conditional-on-the-rest value — the honest direct-influence strength, not the forward pick-order value.

The mechanism is the annihilation identity (II.15): if $X_j\to X_i$ is indirect through some $X_k\in S_i$, then $C_{j\to i\mid S_i\setminus\{j\}}\approx 0$ once $X_k$ is in the conditioning set, and the edge is cut. Forward-only oCSE keeps such an edge whenever it was admitted *before* its mediator; forward+backward removes it.

$$
\boxed{\;\text{ER} \;=\; \underbrace{\text{oCSE forward selection}}_{\S\text{II.6}}\;\oplus\;\underbrace{\text{backward elimination to a fixed point}}_{(\text{IV.1})},\qquad \mathbf A_{j,i}=C_{j\to i\mid S_i\setminus\{j\}}.\;}
\tag{IV.2}
$$

The unit test `test_entropic_regression_kills_indirect_edge` verifies this on a synthetic chain $X_0\to X_1\to X_2$: ER recovers $0\to1$ and $1\to2$ and deletes the indirect $0\to2$ that a forward-only sweep leaves behind.

## IV.2 Structural–functional fusion

The ER edges of (IV.2) describe *information flow*; DTI tractography (`processing/tractography.py`) gives the *physical* white-matter wiring $S$. Fish et al.'s premise is that flow is carried by wiring, so a functional edge with no structural substrate is suspect. `fuse_structural_functional` offers two combinations:

$$
\text{gate:}\quad \tilde A_{ij} = A_{ij}\,\mathbf 1[\,S_{ij}>0\,],
\qquad\qquad
\text{weight:}\quad \tilde A_{ij} = A_{ij}\,\frac{S_{ij}}{\sum_k S_{ik}}.
\tag{IV.3}
$$

The gate is a hard anatomical mask; the weighted form is a soft prior that attenuates (rather than deletes) functionally-inferred edges in proportion to their structural support. Direction is inherited entirely from the functional side; $S$ is symmetric.

## IV.3 Covariant Lyapunov vectors and the Ginelli algorithm

To probe stability we linearize the flow $\dot{\mathbf x}=\mathbf f(\mathbf x)$ of Part I about a trajectory $\mathbf x(t)$. A tangent perturbation $\mathbf v$ obeys the **variational equation**

$$
\dot{\mathbf v} \;=\; J(\mathbf x(t))\,\mathbf v,
\qquad J = D\mathbf f,
\tag{IV.4}
$$

integrated for $m$ tangent vectors alongside the state by the same RK4 kernel (`_rk4_step` co-advances `state` and the $n\times m$ tangent block, with $J$ supplied analytically — the vectorised block Jacobian of §I.2/`clv_topology.lorenz_clv_closures`).

**Oseledets' theorem** guarantees Lyapunov exponents $\lambda_1\ge\lambda_2\ge\dots$ and a covariant splitting of tangent space into subspaces $E_i(\mathbf x)$ that are *invariant* under the linearized flow and grow at rate $\lambda_i$. The **covariant Lyapunov vectors** (CLVs) $\mathbf v_i(t)$ span these subspaces:

$$
M(t,s)\,\mathbf v_i(s) \;=\; \frac{\lVert M(t,s)\,\mathbf v_i(s)\rVert}{\lVert \mathbf v_i(s)\rVert}\,\mathbf v_i(t),
\qquad
\lambda_i=\lim_{t\to\infty}\frac1t\ln\frac{\lVert M(t,0)\mathbf v_i(0)\rVert}{\lVert\mathbf v_i(0)\rVert},
\tag{IV.5}
$$

where $M(t,s)$ is the linear propagator. Unlike the orthonormal Gram–Schmidt vectors produced by plain QR, CLVs are generally **non-orthogonal**, norm-independent, invariant under time reversal, and are the physically meaningful directions of growth — which is precisely why they, not the QR vectors, are used to measure transversality in §IV.6.

**Benettin QR (forward pass).** Direct integration of (IV.4) collapses every tangent vector onto the leading direction, so `run_forward` re-orthonormalizes every `qr_interval` steps by a QR factorization $V = QR$, storing $Q$ (float32, CPU) and the upper-triangular $R$ (float16, GPU — the VRAM-saving choice) at each of the $P$ QR times. The columns of $Q$ are the Gram–Schmidt vectors; the $\ln R_{ii}$ accumulate the exponents (§IV.4).

**Ginelli backward pass.** CLVs are recovered by expressing them in the stored $Q$-bases, $\mathbf v = Q\,\mathbf c$, and propagating the coefficient matrix $C$ *backward*. From $V_{k+1}=V_kR_{k+1}$ one gets the recursion

$$
C_{k} \;=\; \frac{R_{k+1}^{-1}\,C_{k+1}}{\big\lVert R_{k+1}^{-1}\,C_{k+1}\big\rVert_{\text{col}}},
\qquad \text{CLV at step }k:\;\; V^{\text{cov}}_k = Q_k\,C_k,
\tag{IV.6}
$$

started from an arbitrary upper-triangular $C_P$ and iterated to $k=1$; the upper-triangular structure makes it contract onto the covariant coefficients exponentially fast. This is `run_backward_reconstruct`: it solves the upper-triangular systems $R\,C_{\text{prev}} = C$ (`_solve_triangular`), normalizes columns, and returns $Q_kC_k$. The whole two-pass procedure is the Ginelli et al. (2007) algorithm.

## IV.4 The Lyapunov spectrum from the R-diagonals — "dimension counting," part 1

The diagonal of each stored $R^{(k)}$ records how much each orthonormal direction stretched over one QR interval $\tau=\texttt{qr\_interval}\cdot\Delta t$. Averaging their logs over the run gives the spectrum (`lyapunov_spectrum`):

$$
\boxed{\;\lambda_i \;=\; \frac{1}{P'\tau}\sum_{k=k_0+1}^{P}\ln\big|R^{(k)}_{ii}\big|,\;}
\qquad P' = P-k_0,
\tag{IV.7}
$$

after discarding the first fraction $k_0/P$ of QR steps (default $0.1$) so the tangent basis has aligned with the true Oseledets directions. Verified against a fabricated $R$-stream in `test_lyapunov_spectrum_from_R_diagonals` (recovers $[0.5,0,-1.0]$) and, physically, by the sign count: $\#\{\lambda_i>0\}\ge 1$ is the operational definition of chaos.

## IV.5 The Kaplan–Yorke dimension — "dimension counting," part 2

Order the spectrum $\lambda_1\ge\lambda_2\ge\dots$ and let $k$ be the largest index whose partial sum is still non-negative, $\sum_{i=1}^{k}\lambda_i\ge0>\sum_{i=1}^{k+1}\lambda_i$. The **Kaplan–Yorke (Lyapunov) dimension** interpolates the fractional level set where the cumulative expansion rate crosses zero:

$$
\boxed{\;D_{\mathrm{KY}} \;=\; k \;+\; \frac{\sum_{i=1}^{k}\lambda_i}{\lvert\lambda_{k+1}\rvert}.\;}
\tag{IV.8}
$$

The Kaplan–Yorke conjecture identifies $D_{\mathrm{KY}}$ with the information dimension $D_1$ of the attractor for typical systems, so (IV.8) is a cheap fractal-dimension estimate obtained *for free* from the CLV forward pass. `kaplan_yorke_dimension` implements it, verified against the textbook Lorenz value $D_{\mathrm{KY}}=2.062$ (`test_kaplan_yorke_lorenz_value`). Two boundary cases matter in practice:

- **Fixed point / full contraction** ($\lambda_1<0$): the formula returns $D_{\mathrm{KY}}=0$.
- **Ceiling** ($\sum_{i=1}^m\lambda_i\ge 0$ for *all* $m$ computed CLVs): the zero-crossing lies beyond the integrated subspace, so (IV.8) can only report $D_{\mathrm{KY}}\ge m$ — a lower bound, not the true dimension. `clv_cli`/`clv_topology` flag this as `kaplan_yorke_is_ceiling`. On the $N=83$ DTI–Lorenz network at coupling $0.1$, even $m=40$ CLVs hit the ceiling: the regime is *hyperchaotic* (dozens of positive exponents), and resolving $D_{\mathrm{KY}}$ needs of order the full $3N$ spectrum.

## IV.6 Transversality angles and riddled basins

### IV.6a The transverse Lyapunov exponent and the riddling criterion

The synchronization manifold $\mathcal M=\{\mathbf u_1=\dots=\mathbf u_N\}$ is invariant under the diffusively coupled field (I.1); on it lives a chaotic attractor $\mathbf A$. Its stability *within* $\mathcal M$ is set by the internal exponents, but its stability *as a subset of the full space* is governed by the largest **transverse Lyapunov exponent** $\lambda_\perp$ — the growth rate of perturbations pointing off $\mathcal M$, i.e. the leading exponent of (IV.4) restricted to the transverse subspace (the modes $m\ge2$ of the master-stability decomposition I.7).

A basin is **riddled** (Alexander–Yorke–You–Kan 1992; Ott–Sommerer 1994) when

$$
\lambda_\perp < 0 \quad\text{(attracting on average)}
\qquad\text{but}\qquad
\exists\ \text{orbits in }\mathbf A\ \text{with finite-time }\lambda_\perp(t,\Delta)>0,
\tag{IV.9}
$$

i.e. the manifold attracts *on average* yet contains transversely-repelling unstable periodic orbits, so the finite-time transverse exponent fluctuates positive. The consequence is dramatic: every neighborhood of $\mathbf A$ contains a positive-measure set of points that are *ejected* to another attractor. The basin is shot through with holes at every scale — arbitrarily close to a point that synchronizes is a point that flies away. This is the "why it gets messy" phenomenon, and it is exactly the intermingled-basin structure reported for coupled Lorenz systems (Ott et al.).

### IV.6b The CLV-angle proxy and the k-means detector

Condition (IV.9) has a geometric fingerprint in the CLVs: as the finite-time $\lambda_\perp$ swings positive, the covariant directions tangent to $\mathcal M$ and those transverse to it approach **tangency** — the hallmark of local loss of hyperbolicity (homoclinic tangencies; Yang–Radons, Takeuchi et al.). `compute_transversality_angles` measures this by taking the leading $K$ CLVs $\{\mathbf v_i(t)\}$, normalizing them, forming the Gram matrix of absolute cosines $G_{ij}=\lvert\hat{\mathbf v}_i^{\top}\hat{\mathbf v}_j\rvert$, and recording the **minimum pairwise angle**

$$
\theta_{\min}(t) \;=\; \min_{i\ne j}\ \arccos\big(\lvert\hat{\mathbf v}_i(t)^{\top}\hat{\mathbf v}_j(t)\rvert\big).
\tag{IV.10}
$$

Under (IV.9) the series $\theta_{\min}(t)$ is **bimodal**: long stretches at moderate angle (the manifold locally hyperbolic) punctuated by near-zero-angle **tangency bursts** (transverse instability). `detect_riddling_kmeans` fits $2$-means to $\{\theta_{\min}(t)\}$ and returns

$$
\text{RIDDLED}\quad\Longleftrightarrow\quad
\min_c \pi_c \ge \pi_{\min}\ \ \text{and}\ \ \lvert\mu_{\text{hi}}-\mu_{\text{lo}}\rvert \ge \delta,
\tag{IV.11}
$$

both clusters populated (population fraction $\pi_c\ge\pi_{\min}=0.02$) and their centroids separated by at least $\delta=0.15$ rad; otherwise SYNCHRONISED. The reported `burst_fraction` is the mass of the high-angle cluster — a scalar proxy for how "leaky" the basin is. This is an *operational* detector for the tangency signature of (IV.9), not a literal sign test on $\lambda_\perp$; it is validated on synthetic bimodal vs. unimodal series in `test_riddling_detector_bimodal_vs_unimodal`.

## IV.7 Spectral graph theory: the Laplacian eigenbasis, synchronizability, and random-graph baselines

Everything above runs on a network whose only fingerprint, at the linear level, is the spectrum of its Laplacian $L=D-A$ (I.68). Because $L$ is symmetric positive-semidefinite, its eigenvalues are real and ordered

$$
0=\lambda_1\le\lambda_2\le\dots\le\lambda_N,
\tag{IV.12}
$$

with the multiplicity of $0$ equal to the number of connected components (the all-ones vector $\mathbf 1$ always spans a null direction, since $L\mathbf 1=0$). Two eigenvalues carry the dynamics:

- **$\lambda_2$ — the algebraic connectivity (Fiedler value).** The smallest *nonzero* eigenvalue quantifies how hard the graph is to disconnect; its eigenvector (the Fiedler vector) gives the spectral bipartition. Larger $\lambda_2$ ⇒ better-connected, faster consensus/diffusion.
- **$\lambda_N$ — the spectral radius.** Bounded by the maximum degree, $\lambda_N \le 2\,k_{\max}$ and $\lambda_N\ge k_{\max}+1$, so hubs inflate it.

**The synchronizability eigenratio.** Recall the master-stability decomposition (I.7): a transverse mode $m$ perturbing the synchronous state $\mathbf s(t)$ evolves under $\dot{\boldsymbol\xi}_m=[D\mathbf F(\mathbf s)-\sigma\lambda_m\mathbf H]\boldsymbol\xi_m$, stable iff the master stability function $\Lambda(\sigma\lambda_m)<0$. For the common case where $\Lambda<0$ on a bounded interval $(\alpha_1,\alpha_2)$, **all** transverse modes $m\ge2$ are simultaneously stable iff $\sigma\lambda_2>\alpha_1$ and $\sigma\lambda_N<\alpha_2$, which is feasible for some $\sigma$ precisely when

$$
\boxed{\;R \;=\; \frac{\lambda_N}{\lambda_2} \;<\; \frac{\alpha_2}{\alpha_1}.\;}
\tag{IV.13}
$$

The **eigenratio** $R=\lambda_N/\lambda_2$ is thus an intrinsic, coupling-free measure of a graph's synchronizability: *smaller $R$ ⇒ wider stable coupling window ⇒ easier to synchronize*. This is exactly `baseline_models._laplacian_spectral_summary`.

**The random-graph baselines.** `random_graphs.match_baselines_from_adjacency` builds two nulls with the empirical connectome's node count $N$ and (binarized) edge count $E$:

- **Erdős–Rényi $G(n,m)$** (`gnm_random_graph`): edges thrown uniformly at random. Degrees are Binomial$(N-1,p)$ with $p=2E/[N(N-1)]$, hence homogeneous: the degree heterogeneity $\kappa=\langle k^2\rangle/\langle k\rangle^2\to1$. Its Laplacian spectrum concentrates ($\lambda_2\approx np-\!\sqrt{2np\ln n}$, $\lambda_N\approx np+\!\sqrt{2np\ln n}$), so $R\to1^+$: **ER graphs are near-optimal synchronizers.**
- **Barabási–Albert** (`barabasi_albert_graph`, $m\approx E/N$): preferential attachment yields a scale-free degree law $P(k)\sim k^{-3}$ with hubs. Hubs push $\lambda_N$ up (via $k_{\max}$) while $\lambda_2$ stays modest, so $\kappa\gg1$ and $R$ is substantially larger than ER's — the *paradox of heterogeneity*: scale-free hubs, for all their efficiency, *hurt* synchronizability.

**What the comparison shows.** Running the identical CLV pipeline (§IV.3–IV.6) on all three networks (`baseline_models.py`) puts topology on equal footing — same $N$, same $E$, unit weights — so only the *shape* differs. An illustrative short CPU run on the $N=83$ connectome ($E=850$, coupling $0.1$) gave:

| network | $\lambda_2$ | $\lambda_N$ | $R=\lambda_N/\lambda_2$ | $\kappa$ (het.) |
|---|---|---|---|---|
| Empirical (DTI) | $1.92$ | $45.3$ | $\mathbf{23.7}$ | $1.18$ |
| Erdős–Rényi | $9.64$ | $32.8$ | $3.4$ | $1.04$ |
| Barabási–Albert | $7.58$ | $47.3$ | $6.2$ | $1.24$ |

The empirical connectome's eigenratio ($\approx 23.7$) is $\sim7\times$ the ER null and $\sim4\times$ the BA null: the brain's modular, hierarchical wiring is a markedly **poorer** synchronizer than either random baseline with the same edge budget. That low $\lambda_2$ (sluggish global consensus) with high $\lambda_N$ (fast local hub modes) is precisely the regime (IV.13) *cannot* keep globally stable — so the network is pushed off the synchronization manifold into the transverse-instability / riddled dynamics diagnosed in §IV.6, rather than collapsing to clean global sync. The brain's *shape*, read straight off two Laplacian eigenvalues, predicts the mess.

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
10. J. L. Hindmarsh, R. M. Rose. *A model of neuronal bursting using three coupled first order differential equations.* Proc. R. Soc. Lond. B **221** (1984) 87–102.
11. L. O. Chua. *The genesis of Chua's circuit.* Archiv für Elektronik und Übertragungstechnik **46** (1992) 250–257.
12. A. Daza, A. Wagemakers, B. Georgeot, D. Guéry-Odelin, M. A. F. Sanjuán. *Basin entropy: a new tool to analyze uncertainty in dynamical systems.* Scientific Reports **6** (2016) 31416.
13. A. Daza, A. Wagemakers, M. A. F. Sanjuán. *A grid algorithm to identify basins of attraction and the Wada property.* Chaos **28** (2018) 093117.
14. M. Ester, H.-P. Kriegel, J. Sander, X. Xu. *A density-based algorithm for discovering clusters in large spatial databases with noise (DBSCAN).* Proc. KDD **96** (1996) 226–231.
15. V. Satopää, J. Albrecht, D. Irwin, B. Raghavan. *Finding a "Kneedle" in a haystack: detecting knee points in system behavior.* Proc. ICDCS Workshops (2011) 166–171.
16. J. Fish, A. DeWitt, A. A. R. AlMomani, P. J. Laurienti, E. M. Bollt. *Entropic regression with neurologically motivated applications.* Chaos **31** (2021) 113105.
17. F. Ginelli, P. Poggi, A. Turchi, H. Chaté, R. Livi, A. Politi. *Characterizing dynamics with covariant Lyapunov vectors.* Phys. Rev. Lett. **99** (2007) 130601.
18. J. Kaplan, J. A. Yorke. *Chaotic behavior of multidimensional difference equations.* In *Functional Differential Equations and Approximation of Fixed Points*, Lecture Notes in Mathematics **730**, Springer (1979) 204–227.
19. J. C. Alexander, J. A. Yorke, Z. You, I. Kan. *Riddled basins.* Int. J. Bifurcation and Chaos **2** (1992) 795–813.
20. E. Ott, J. C. Sommerer. *Blowout bifurcations: the occurrence of riddled basins and on-off intermittency.* Phys. Lett. A **188** (1994) 39–47.
21. K.-A. Takeuchi, H.-L. Yang, F. Ginelli, G. Radons, H. Chaté. *Hyperbolic decoupling of tangent space and effective dimension of dissipative systems.* Phys. Rev. E **84** (2011) 046214.
22. P. Erdős, A. Rényi. *On the evolution of random graphs.* Publ. Math. Inst. Hung. Acad. Sci. **5** (1960) 17–61.
23. A.-L. Barabási, R. Albert. *Emergence of scaling in random networks.* Science **286** (1999) 509–512.
24. M. Fiedler. *Algebraic connectivity of graphs.* Czechoslovak Mathematical Journal **23** (1973) 298–305.
25. F. R. K. Chung. *Spectral Graph Theory.* CBMS Regional Conference Series in Mathematics **92**, American Mathematical Society, 1997.
26. E. M. Bollt, J. Fish, A. Kumar, C. Roque dos Santos, P. J. Laurienti. *Fractal Basins as a Mechanism for the Nimble Brain.* arXiv:2311.00061 (2023) — the project's motivating paper: the Vector Pattern State (Definition A, §I.5), riddled/fractal chimera basins on the DTI connectome, and the $L^2$-by-fiat and control-theoretic gaps addressed in §I.6b and §II.4c.

---

*This manual is a living document: as `pythongpu` evolves, keep each boxed equation and the correspondence table in §III.10 synchronized with the code they describe.*
