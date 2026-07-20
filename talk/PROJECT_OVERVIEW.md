# Project Overview — Rapidly Examining Fractal Basin Structure

**Anna Totilca · Clarkson University HPC REU 2026 · Advisor: Dr. Jeremie Fish**
**Prepared for the Wednesday meeting / RAPS practice · 2026-07-20**

---

## 0. The project scope (from the REU listing)

> *(3) It has been recently observed that the basin structure of partially
> synchronous states is very rich, and apparently in many cases leads to a
> fractal structure. In this project we would develop software to **rapidly
> examine this structure** given various potential **planes to view the basin
> from** (the basin is generally very high dimensional, so it is easiest to
> choose a plane from which to view the basin structure).*

Everything below maps onto that one sentence. Three phrases in it became the
three axes of the summer:

| Scope phrase | What it became |
|---|---|
| **"rapidly examine"** | The HPC deliverable: a GPU port of the paper's MATLAB — 130,321 initial conditions integrated *at once* per coupling, on ACRES. |
| **"planes to view the basin from"** | The state space is 249-dimensional (83 regions × 3). A *basin map* is a 2-D **plane** (slice) of initial conditions, colored by which attractor each one reaches. The software takes a chosen plane and resolves it. |
| **"fractal structure"** | Measuring the crinkliness (box-counting dimension `D_f`, basin entropy) of the boundaries *on that plane* — and, critically, checking whether a fractal reading is real or a measurement artifact. |

The one-line summary of the summer: **I built the fast plane-scanning software the
project asked for, then discovered that making it fast was the easy half — the hard
half was proving the fast thing still computed the right thing.** Both real
scientific findings came out of that second half.

---

## 1. The science this software serves

The underlying paper is my advisor's:

> Bollt, Fish, Kumar, Roque dos Santos & Laurienti, *Fractal Basins as a
> Mechanism for the Nimble Brain* (arXiv:2311.00061).

**The idea.** A brain must be *stable* (hold a thought against noise) and
*nimble* (drop it the instant something matters). Those demands fight. The
paper's proposed resolution: the resolution is built into the **geometry** of
the state space. Picture a marble on a landscape of valleys — each valley a
brain state.

- **Smooth ridgelines** between valleys ⇒ you need a big shove to switch states.
- **Fractal (infinitely crinkly) ridgelines** ⇒ almost wherever you stand, a
  boundary is a hair away. A whisper moves you. **Crinkliness *is* nimbleness.**

The same geometry governs any coupled-oscillator system that must be reliable
*and* switchable (power grids staying phase-locked; hard limits on
predictability). The math doesn't know it's in a skull.

---

## 2. The model and the "plane" we view

- **Nodes.** Each of `N = 83` brain regions is a chaotic **Lorenz** oscillator
  (`σ=10, ρ=28, β=8/3`).
- **Wiring.** Regions are coupled through a real human **DTI** connectome
  (83 nodes, 850 edges, density 0.25, diameter 4). From adjacency `A` we build
  the graph Laplacian `L = diag(A·1) − A` and couple the x-components:

  `ẋ_i = f(x_i) − K · Σ_j L_ij H x_j`,  with `H = e₁e₁ᵀ`, and `K` the one knob.

- **Why the Laplacian matters.** `L·1 = 0`, so full synchrony is *always* an
  exact solution. The scientific question is whether anything *else* coexists —
  i.e. whether the basin structure is rich.
- **The plane.** The full state space is `83 × 3 = 249`-dimensional. We cannot
  see a 249-D basin, so we pick a 2-D **plane of initial conditions** (our
  working slice varies 2 coordinates and holds the other ~81 nodes nearly fixed),
  sweep a grid over it, integrate each grid point, and color by outcome. That
  colored grid **is** the basin map — the "view from a plane" the scope calls for.

---

## 3. What we measure on a plane, and the trap built into it

- **Basin map.** Color a grid of initial conditions on the chosen plane by which
  attractor they reach.
- **Box-counting dimension `D_0`.** Cover the boundary with boxes of side `ε`;
  if `N(ε) ~ ε^(−D)`, then `D_0 = lim log N(ε) / log(1/ε)`. Smooth ⇒ `D=1`;
  area-filling ⇒ `D=2`; fractal in between.
- **Uncertainty exponent (Grebogi–McDonald–Ott–Yorke, 1983).** `D = d − α`
  (`d=2` on our slice). `10×` more precision cuts final-state error only by
  `10^α`. As `α → 0`, extra precision buys *nothing* — a hard limit on
  prediction, not an engineering gap.
- **Basin entropy (Daza et al., 2016).** `S_bb > ln 2` is a *sufficient*
  theorem-backed condition for a fractal boundary — stronger than a slope fit.

**The trap (this drives the whole project).** `D → 2` has **two** explanations:
1. an intricately interwoven boundary — the exciting one; or
2. a **meaningless labelling**, where every pixel borders another color so the
   "boundary" trivially fills the plane.

Both give `D ≈ 2` with a beautiful `R²`. **`D` alone cannot tell them apart.**
Holding onto that ambiguity is what eventually produced the real result.

---

## 4. The HPC deliverable — "rapidly examine"

**The bottleneck.** Resolving a fractal boundary needs a *huge* grid of initial
conditions, each an independent simulation. Embarrassingly parallel — but the
original MATLAB runs them one at a time.

**The expensive kernel: the VPS (Vector Pattern State).** Each run is labeled by
*how every pair of regions moves together*. For `N=83` that is `C(83,2) = 3403`
pairs, so `VPS ∈ ℝ^6806`. For each pair `(i,j)`, from the cross-correlation:
- `τ_ij` = the **time lag** at the correlation peak — how far out of phase they are;
- `L_ij` = the norm of their difference **after aligning by that lag**.

The lag is the whole point: it detects **phase-lagged synchrony**, the signature
of a chimera. MATLAB computes it with a `for` loop over all 3403 pairs, calling
`xcorr` on each.

**The port.**
- **Wiener–Khinchin**: correlation → product in Fourier space,
  `O(T²) → O(T log T)`.
- **Batching**: all 3403 pairs in one kernel — the loop becomes a tensor axis.
- **Batched RK4**: all initial conditions on the plane integrated simultaneously.
- **SLURM job arrays**: one coupling `K` per GPU on ACRES.
- **Scale reached**: `361² = 130,321` initial conditions per coupling, 6806
  features each, ≈ 1 GPU-hour per `K`, ~20 couplings.

**The port was slow — and the FFT wasn't why.** My first port never cleared `7×`
over a serial CPU reference (and at `T=128` the looped version was actually
*slower* than serial, `0.83×`). Profiling found the culprit: after the FFT
parallelized `τ`, a **Python loop over every candidate lag** (`2T−1 = 19,999`
iterations at `T=10⁴`) threw the win away. The fix — build per-pair aligned index
grids and gather once, no loop — took it to **375×** over the serial reference
(~57× over my own first port), verified correct to float32 precision
(`max|ΔL| = 1.9×10⁻⁶`).

> **Benchmark note.** The comparator is a *serial CPU reference* — a line-by-line
> transcription of the MATLAB, same algorithm — so it measures parallelization,
> not Python-vs-MATLAB language overhead. A true MATLAB-vs-GPU wall-clock number
> is still **pending** (no MATLAB license on our machines).

---

## 5. The two things checking my own work revealed

### 5a. Does the port reproduce the paper? I tested. No.

I transcribed `VectorPatternState.m` line-by-line into a reference and compared
on the paper's own test matrices. `τ` matches on some inputs, differs on others
(tie-breaking); **`L` disagrees on *every* input.** The cause is an **off-by-one
in the *original* MATLAB**: for `lag > 0` it aligns by `lag−1`, so at a detected
lag of 1 it applies *no shift at all*. The under-shift mis-aligns every pair and
**inflates `L`**: `L_matlab ≥ L_corrected` on **36/36** pairs, mean ratio **1.96**.

So which is right? To *replicate* the paper I must reproduce the original; to be
*correct* I must not. The code now ships **both** paths — it is a measurement, not
an argument. **This needs a conversation with Dr. Fish, not a unilateral fix.**

### 5b. The memory wall silently changed the science.

The true VPS needs the **whole trajectory** (`T × N`) per initial condition to
cross-correlate. At a `64²` grid that is ≈ **41 GB** — it doesn't fit. So the
production sweep uses a **streaming** (Welford, single-pass) approximation that
never stores a trajectory. But the streaming surrogate is *not* the VPS:

| paper / port | streaming surrogate |
|---|---|
| `τ` = time lag | `τ` = mean\|ΔX\| / std\|ΔX\| |
| `L` after alignment | `L` = instantaneous mean |

**There is no lag anywhere** — and the lag is exactly what detects the
phase-lagged synchrony the VPS exists to find. **Every result this summer came
from the streaming path.** A resource constraint forced an approximation, and the
approximation quietly stopped computing the quantity the paper defines. Nobody
noticed for weeks — including me. That is the real HPC lesson: *the substitution
you make to fit in memory can change the science underneath you.*

---

## 6. What the pipeline first reported — and the anomalies

Across the sweep `K ∈ [0.45, 0.65]`: `D_f ≈ 1.89–1.96`, `R² > 0.99`, so
`α = 2 − D_f ≈ 0.04` — near-total final-state sensitivity. **Exactly the result
the project wanted.** Two things itched:

1. `D_f` was **flat in `K`** — a real bifurcation should move it.
2. The automatic basin count said **6–8**; the silhouette criterion said **2**.

And the basin map at `K=0.65` was pure static — no connected regions at any
scale. *A basin map with no connected regions is not obviously a basin map.* This
is where the slide-3 ambiguity detonated: **we had measured `D → 2`**, which means
*either* a genuinely interwoven boundary *or* a meaningless labelling.

**Why I suspected an artifact.** `k`-means *always* returns `k` groups; hand it
structureless data and it returns `k` confident, meaningless basins. If the labels
are noise, every pixel is a boundary pixel ⇒ `D_f → 2`, `α → 0` — *exactly* what
we saw. And the basin count came from the elbow method, which never abstains when
there is no elbow. Everything downstream of clustering inherits the disease.

---

## 7. The methodological turn — validate the ruler, then measure

I needed an observable that requires **no clustering at all**: per-node long-time
average `⟨x_i⟩`. It's a time average, so it's constant on an attractor.

**Validate the ruler on a known answer first.** At `K=0` the network is
uncoupled: every node has one ergodic chaotic attractor, so long-time averages
*must* converge to a common value. My metric passes: `K=0` tracks `1/√T` as
ergodicity demands, while `K=0.5` stays **flat** over `32×` more averaging and
`320×` more burn-in (20 → 6400). Both innocent explanations are excluded:
- **Unconverged averages?** No — `K=0` converges under the identical metric.
- **Long transients?** No — still flat under 320× more burn-in.

Ergodicity forbids IC-dependent long-time averages on *one* attractor. **So there
are many.**

---

## 8. The finding — coupling locks nodes onto lobes

A Lorenz attractor has two wings. Long-time `⟨X⟩` is `0` if a trajectory visits
both, and `±7.8` if it stays **locked** on one.

- **Uncoupled:** every node ergodic, `⟨X⟩ → 0`.
- **Coupled:** **97.8%** of nodes pinned to one wing at `±7`.

So the network's attractor is a **pattern of ± lobes across 83 nodes**, and the
basin label is **exact and clustering-free**:

> **label = sign⟨X_i⟩** — 83 bits, discrete. No `k`, no elbow, no silhouette.
> Up to `2^83 ≈ 10^25` candidate attractors.

And they're **permanent**: same ICs, two *disjoint* time windows → **99.9%** of
bits agree; **94.6%** of ICs reproduce their exact 83-bit pattern.

**Where does locking switch on?** A deterministic `32²` slice (1024 ICs per
coupling) shows a **sigmoid bifurcation at `K ≈ 0.07`** — *below* the window
anyone had swept. The order parameter climbs from `0.026` (ergodic) to `7.29`
(vs a single-wing value of 7.8). The largest basin holds `0.1%` of ICs below onset
(sign noise), peaks at **`9.3%` near `K ≈ 0.12`**, and falls to `0.4%` by `K=0.2`.

**Practical payoff for the plane-scanning software:** the paper's coupling
`gel = 0.5` sits **deep in the riddled regime** — basins too finely intermingled
for *any* plane to show connected structure. The only window where basins have
appreciable size (where a plane view is legible) is **`K ≈ 0.08–0.12`**. This
tells the next user *which plane, at which coupling, is even worth rendering.*

---

## 9. The result — the basins really are mingled

Counting patterns exactly on the lobe labels:

| `K` | distinct patterns | biggest basin |
|---|---|---|
| 0.20 | 1015 / 1024 ICs | 4 ICs (0.4%) |
| 0.50 | 1010 / 1024 ICs | 4 ICs (0.4%) |

Nearly every initial condition is its own attractor — we're **sampling-saturated**
(true count ≥ 1010). Mean Hamming distance between patterns is **41.5 of 83 bits**
— exactly the distance between *random* strings.

So `D_f ≈ 1.96` **was right, for the right reason**: boundaries among ~1000+
interleaved basins genuinely *are* space-filling; `α → 0` is real; the static map
is what **riddled** basins look like. Every anomaly resolves: flat `D_f` (locking
persists across `K`), the failure of every cluster-count method (the true count is
astronomical, not 8). And it **supports the paper** — Bollt–Fish report
"intricately *mingled* basin boundaries" at `gel = 0.5` (our `K=0.5`); we
**confirm it, and add the mechanism the paper doesn't name.**

---

## 10. What I got wrong (and why it worked out)

Four hypotheses I proposed and then **killed with a control**:

1. The sync manifold is stable, so everything collapses to sync — *No sync at any
   `K`.*
2. IC jitter is manufacturing the static — *Zero jitter: identical result.*
3. The whole `D_f ≈ 1.9` is a clustering artifact — *It's real riddling.*
4. My own structure test was correct — *It had three separate bugs.*

The one that stings: I wrote the "`D → 2` is ambiguous" warning early, then spent
days failing to apply it to my *own* conclusion. Every wrong hypothesis died to a
control against a system whose answer I already knew — the only reason the real
mechanism surfaced.

> **The most dangerous bug is the one that makes your result look better.**

---

## 11. Deliverables

Repository: `github.com/anitoti/pythongpu` (package `pythongpu/`).

- **GPU port** of the paper's MATLAB: batched RK4, batched FFT cross-correlation,
  SLURM job arrays for ACRES (`pythongpu/pipeline/`, `scripts/`).
- **oscillators/** Lorenz, Rössler, Kuramoto, Hindmarsh–Rose, Van der Pol,
  FitzHugh–Nagumo, Chua.
- **networks/** DTI connectome + ER / Watts–Strogatz / Barabási–Albert null
  models; DTI spectral-diagnostics index (`dti_spectra_index.json`).
- **diagnostics** (structure gate, emergent attractor count, radius-plateau
  test) — each validated on known-answer data.
- **replication harness** — GPU VPS vs a literal MATLAB transcription on the
  paper's own test matrices (ships both `lag` and `lag−1` alignments).
- **`pytest` suite.** The pipeline now *argues back*: it refuses to report a
  `D_f` it cannot support.

---

## 12. Next / open threads

**Finish the port (the HPC thread):**
- Resolve the `lag` vs `lag−1` alignment with Dr. Fish — replicate, or correct?
- Vectorize the residual `L` loop (still `2T−1 = 19,999` Python iterations at
  `T=10⁴`).
- Beat the 41 GB memory wall (chunking / fp16) so the **real** VPS runs at 130k ICs.
- Produce a measured MATLAB-vs-GPU benchmark.

**Finish the science:**
- `S_bb > ln 2` on the *exact* lobe labels — a theorem, not a slope fit.
- **Null models** (already wired): is riddling a property of *brain* wiring, or of
  any dense graph? DTI vs Barabási–Albert scale-free (n=83).
- **CLV diagnostics** (Ginelli's algorithm, implemented): covariant-Lyapunov-vector
  transversality angles as a precursor to riddling; parallel SLURM sweep over
  `K ∈ [0.05, 0.20]`, DTI vs null model. *In progress — not yet in the RAPS deck.*
- **Other viewing planes** — our slice holds 81 of 83 nodes nearly fixed; the
  scope's "various potential planes" is still largely unexplored.

---

### One-paragraph version (for the top of an email)

I built the GPU software the project asked for — it resolves a chosen 2-D plane of
a 249-dimensional basin by integrating up to 130,321 initial conditions at once on
ACRES, ~375× over a serial reference after I profiled out a hidden Python loop.
Then I checked it, and two things surfaced: the port doesn't reproduce the paper's
VPS (an off-by-one in the *original* code inflates a key quantity ~2×), and a
memory wall had silently swapped the VPS for an approximation that drops the lag
entirely. Building clustering-free diagnostics — and validating them on a
known-answer control — turned up the mechanism: coupling locks each Lorenz node
onto one wing, so the network has ~10²⁵ candidate attractors and genuinely mingled
basins. That confirms the paper at its own coupling and adds the mechanism it
doesn't name — and it tells us *which* coupling window (`K ≈ 0.08–0.12`) is even
worth viewing a plane in.
