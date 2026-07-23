# RAPS Practice Script — *Rapidly Examining Fractal Basin Structure on the GPU*

**Anna Totilca · Clarkson HPC REU 2026 · Advisor: Dr. Jeremie Fish**
Deck: `talk/summer_talk.tex` (13 slides). Actual slot: **15 min, questions after.**
This script is budgeted at **~13:20** to leave real margin — a 15-minute slot with
a hard stop is not the place to run long.

**How to use this:** the *italic cues* are delivery notes (don't say them). The plain
text is roughly what to say — say it in your own words, don't memorize verbatim. The
running clock on the right tells you if you're on pace. If you hit a slide and the
clock is behind, cut to the **bold** sentence and move on.

> ✅ **Every slide has a real figure, including the two added tonight.** Slide 7's
> basin map is `data/derivatives/lorenz_basins_n73_n81_K0.6500.npz` (8312 distinct
> 83-bit lobe patterns among 9216 ICs) — genuine static, not a mock-up. Slides 10-11
> are built from real production data too (HR true-VPS/surrogate comparison, CLV
> sweeps for both Lorenz and Hindmarsh-Rose, and a pre-existing null-model CLV run).

**What changed from the 16-slide version:** slides 5+6 (port + speedup) merged into
one; slides 8+9 (memory wall + false positive) merged into one; slides 10+11 (ruler
validation + mechanism) merged into one; the onset-curve slide and "What I Got Wrong"
slide were cut entirely to make room; two new slides were added (a second dynamical
system, and a null-model control) since those are now real, finished results, not
future work.

---

## Pacing table

| # | Slide | Budget | Clock ends |
|---|---|---|---|
| 1 | Title | 0:15 | 0:15 |
| 2 | The Science | 1:00 | 1:15 |
| 3 | The System and the Viewing Plane | 0:55 | 2:10 |
| 4 | Measuring Crinkliness + landmine | 0:55 | 3:05 |
| 5 | Porting the VPS: 3403 Loops to One Gather | 1:45 | 4:50 |
| 6 | Did I Reproduce It? | 0:55 | 5:45 |
| 7 | The Memory Wall + the False Positive | 1:30 | 7:15 |
| 8 | Validate the Ruler, Find the Mechanism | 1:40 | 8:55 |
| 9 | The Result | 1:00 | 9:55 |
| 10 | A Second System: Hindmarsh-Rose | 1:15 | 11:10 |
| 11 | Is Riddling Connectome-Specific? | 1:10 | 12:20 |
| 12 | Summary & Next | 0:55 | 13:15 |
| 13 | Thank you | 0:05 | 13:20 |

---

## 1 — Title · *(0:15)*

Hi, I'm Anna. My REU was an HPC project: take the MATLAB behind a published brain
paper and make it **rapidly scan basin structure on a GPU**. I'll give you the
science, what I built, how fast it got — and then the two things I found by checking
my own work that matter more than the speedup.

*(Move on quickly — don't linger on the title.)*

---

## 2 — The Science · *(1:00)*

Your brain has to do two opposite things. It has to hold a thought **steadily**,
against noise — and it has to **drop it instantly** when something matters. Stability
and sensitivity fight each other.

*(Point at the cartoon.)* Picture a marble on a landscape of valleys — each valley is
a brain state. If the ridgelines between valleys are **smooth**, you need a big shove
to switch states. But if they're **fractal** — infinitely crinkly — then almost
wherever the marble sits, an edge is a hair away, and a whisper moves you.

**So crinkliness is nimbleness.** And there's no controller flipping the switch — the
resolution is built into the *geometry*. This is my advisor's paper, and my whole
project is its code.

---

## 3 — The System and the Viewing Plane · *(0:55)*

Here's the actual system. Eighty-three brain regions, each a chaotic **Lorenz
oscillator**, wired together by a real human **DTI** scan through the graph
Laplacian. One knob: the coupling strength **K**.

The catch for *my* project is dimension. The state space is 83 times 3 — **249
dimensions**. You can't look at a 249-dimensional basin. So the task is exactly what
the project title says: **pick a 2-D plane of initial conditions, sweep a grid over
it, integrate every point, and color it by where it ends up.** That colored grid is
the basin map — the view from a plane.

*(Tap the Laplacian box.)* One quick thing: the Laplacian's rows sum to zero, so
**full synchrony is always a solution.** The whole question is whether anything
*else* survives alongside it.

---

## 4 — Measuring Crinkliness + the landmine · *(0:55)*

To measure crinkliness we use the **box-counting dimension**: cover the boundary with
little boxes and see how the count scales. Smooth boundary gives dimension 1,
space-filling gives 2, fractal is in between. And there's a theorem — Grebogi and
co-authors — that ties the dimension to *predictability*: as it approaches 2, extra
precision buys you almost nothing. That's a hard limit, not an engineering problem.

*(Slow down. Point at the red box.)* Now hold onto this, because **it detonates later
in the talk.** A dimension near 2 has *two* explanations. One: a genuinely
interwoven, fractal boundary — the exciting answer. Two: **a meaningless labelling**,
where every pixel just borders a different color, so the "boundary" trivially fills
the plane. **Both give you dimension 2 with a beautiful fit. The number alone cannot
tell them apart.** Remember that.

---

## 5 — Porting the VPS: 3403 Loops to One Gather · *(1:45)* — **HPC centerpiece**

So here's my job. Resolving a fractal boundary needs a *huge* grid of initial
conditions — each one an independent simulation. That's embarrassingly parallel, but
the original MATLAB runs them one at a time.

The expensive kernel is this thing called the **Vector Pattern State**, the VPS. It
labels each run by how *every pair* of regions moves together — for 83 nodes that's
3,403 pairs. For each pair it computes two numbers: **tau**, the time lag between
them, and **L**, how different they are once you line them up by that lag. **The lag
is the whole point** — it's what detects phase-lagged synchrony, the chimera
signature. Keep the word "lag" in mind, it pays off in a few slides.

MATLAB does this with a for-loop over all 3,403 pairs. I turned that into **one
batched FFT** — Wiener–Khinchin turns correlation into multiplication. My first port
was still **slow** though — never cleared about seven times faster than a plain
serial CPU version, and at small sizes was actually *slower* than serial. So I
profiled it, and the bottleneck was **not** the FFT — the FFT was fine. The problem
was that *after* the FFT found the lag, I had a **Python loop over every possible lag
value**, twenty thousand iterations, that threw the whole parallel win away.

The fix: each pair has its own lag, so build **per-pair aligned index grids and
gather once, no loop at all.** That took it from three times faster to **375 times
faster** than serial, verified to float32 precision against the reference. Plus
batched RK4 and SLURM job arrays, one coupling per GPU — on ACRES that's **130,000
initial conditions per coupling at once.**

**The one-liner: the FFT was never the problem. The loop after it was.**

*(If behind schedule, compress the profiling detail and land on the 375x number.)*

---

## 6 — Did I Actually Reproduce It? · *(0:55)*

A port is only a port if the numbers match. So I transcribed the original MATLAB
line-by-line and compared, on the paper's own test matrices. *(Point at the plot.)*
The lag matches most of the time. But **L disagrees on every single input** — every
point sits above the diagonal.

The cause is an **off-by-one — in the *original* code.** For a lag of 1, MATLAB's
array slices end up applying *no shift at all*, even though it detected a lag. That
under-shift systematically **inflates L** — it's bigger in the original on 36 out of
36 pairs, by about a factor of two.

So which one is right? To *replicate* the paper I have to reproduce the original; to
be *correct* I have to fix it. I can't decide that unilaterally — so **my code ships
both**, and it becomes a measurement instead of an argument. This is a conversation
for Dr. Fish.

---

## 7 — The Memory Wall + the False Positive · *(1:30)*

Then I hit a wall. The *real* VPS needs the entire trajectory of every simulation in
memory to cross-correlate it. At even a modest grid, that's **41 gigabytes** — it
doesn't fit. So production quietly switched to a **streaming approximation** that
never stores a trajectory. Clever for memory — *but* *(point at the table)* the
streaming version isn't the VPS. It redefines both tau and L, and crucially **there's
no lag anywhere in it.** And the lag was the entire reason the VPS exists. **Every
result I got this summer came from that streaming path** — a resource constraint
silently changed the science underneath me, and nobody noticed for weeks, including
me.

And here's what it produced: at first, **exactly the result the project wanted** —
dimension between 1.9 and 1.96, gorgeous fits. But two things itched. The dimension
was **flat** as I changed the coupling — a real bifurcation should move it — and the
automatic basin count and silhouette score flatly disagreed. *(Point at the map — the
static.)* The basin map was **pure static.** No connected regions at any zoom.

*(This is the callback — say it deliberately.)* Remember slide 4? Dimension-near-2
means *either* a real fractal boundary *or* a meaningless labelling. **We had measured
dimension-near-2** — and k-means will *always* hand back k confident groups, even from
noise. So I genuinely thought I'd found an artifact.

---

## 8 — Validate the Ruler, Find the Mechanism · *(1:40)*

To settle it I needed a measurement that uses **no clustering at all.** I used each
node's long-time average — constant on an attractor. But here's the methodological
move I want you to take away: **before trusting the ruler, I checked it on a system
whose answer I already knew.** At zero coupling, one chaotic attractor per node, so
the averages *must* converge together — and they do, falling off like
one-over-root-T exactly as ergodicity demands. The coupled curve stays **flat**, even
under 320 times more burn-in. Ergodicity forbids that on *one* attractor. **Which
means there are many.**

And here's *why*. A Lorenz attractor has two wings. A node's long-time average is
**zero if it visits both, plus-or-minus 8 if it locks onto one.** *(Point at the
histogram.)* Uncoupled, everything sits at zero. Coupled, **98% of nodes are pinned
to a single wing.** So the network's attractor is a **pattern of plus/minus signs
across 83 nodes** — the basin label is *exact*, 83 bits, no clustering, no k, no
elbow. Up to **two-to-the-83 possible attractors**, and they're **permanent**: two
separate time windows, 99.9% of bits agree. This is the finding that makes every
clustering headache in this talk disappear.

---

## 9 — The Result: Basins Are Genuinely Mingled · *(1:00)*

So I counted the attractors exactly, using those sign patterns. *(Point at the
table.)* At coupling 0.2 and 0.5, essentially **every initial condition is its own
attractor** — over a thousand distinct patterns out of 1,024 samples. We're
sampling-saturated: the real number is even bigger. And the typical distance between
two patterns is **exactly what you'd get from random strings.**

Which means — *(beat)* — the dimension near 1.96 was **right all along, for the right
reason.** Boundaries between a thousand interleaved basins genuinely *are*
space-filling. The static map is what **riddled basins actually look like.**

And I want to be clear: this **supports** the paper. Bollt and Fish describe
"intricately mingled" boundaries at exactly this coupling. **We confirm it — and we
add the mechanism the paper didn't name.**

---

## 10 — A Second System: Hindmarsh-Rose · *(1:15)*

Everything so far is one system, Lorenz. So I asked the obvious next question: does
any of this hold up on a *different* chaotic model? I ported the whole pipeline —
true VPS, streaming surrogate, the exact lobe-locking label, and the CLV method too —
to **Hindmarsh-Rose**, the other canonical bursting-neuron model in this literature.

*(Point at the left figure.)* On Lorenz, the true VPS and the streaming surrogate
**agreed** — 0.067 apart in fractal dimension. On Hindmarsh-Rose, **they disagree** —
0.284 apart, and it's not just the number, the *shape* is different: the surrogate
drops sharply between two couplings and flattens, the true VPS declines smoothly the
whole way.

*(Point at the right figure.)* The CLV method, meanwhile, agrees with the surrogate
on HR — both say riddled — and HR's Kaplan-Yorke dimension resolves cleanly every
time, where Lorenz's hyperchaotic network hit a ceiling.

**The takeaway, and say this plainly: the Lorenz validation does not transfer.** A
substitution that happened to agree on one system is not a general license to trust
it on the next. Each system needs its own check — that's not a caveat, that's the
finding.

---

## 11 — Is Riddling Connectome-Specific, or Just Generic? · *(1:10)*

One more control, because a real human connectome is only interesting if it's doing
something a generic network wouldn't. The CLV method reads the flow's own Jacobian —
no VPS, no clustering at all — so I ran it on the real DTI connectome *and* on a
size-matched null model, a random scale-free graph with the same 83 nodes but
completely unrelated wiring.

*(Point at the figure.)* Both say riddled, at every coupling tested. But the real
connectome's riddling signature is **far noisier** — including a sharp dip at one
coupling that the null model never shows at all. And only the real connectome ever
resolves a finite Kaplan-Yorke dimension; the null model stays at the ceiling
throughout.

**Read this honestly, don't overclaim it:** riddling itself isn't connectome-specific
— a generic graph riddles too. What might be specific is *how* the real network's
riddling structure organizes underneath that. That's an open question, not something
I've settled tonight.

---

## 12 — Summary and What's Next · *(0:55)*

So, in one breath: I **built the fast plane-scanner** — 130,000 simulations at once,
375 times over serial, and I **quantified the distributed-computing claim** instead
of asserting it, 1.34 to 3.38 times faster, growing with grid size. I **tested my own
port** and found it doesn't match the source — possibly a bug in the *published*
code. I found the **mechanism**: coupling locks nodes onto lobes, giving a
thousand-plus permanent, mingled basins. I **beat the memory wall** — the true VPS
now runs at production scale, and agrees with the surrogate on Lorenz but
**disagrees** on Hindmarsh-Rose. And CLV, an independent method with no VPS and no
clustering, says riddled on Lorenz, HR, *and* a null-model graph — but only the real
connectome ever resolves a finite dimension.

Next up: settle the lag question with Dr. Fish; figure out *why* HR's validation
didn't transfer; the production-scale norm sweep and Rössler extension are running
now.

**The takeaway: a fast pipeline that can't fail isn't a measurement. The speedup was
the easy part — checking that the fast thing still computed the *right* thing, on
more than one system, is where the summer actually went.**

---

## 13 — Thank you · *(0:05)*

Thank you — I'm happy to take questions.

---

## Anticipated questions (prep, not spoken)

- **"Is the off-by-one really a bug in the published paper?"** → I'd frame it as: my
  transcription reproduces the original's behavior, and that behavior applies no shift
  at lag 1. Whether it's a bug or an intended convention is exactly the conversation I
  want to have with Dr. Fish — my code supports both so we can decide with data.
- **"Why Lorenz nodes for a brain model?"** → It's the model in the paper — chaotic
  units on an empirical connectome. The point isn't neural realism; it's the geometry
  of partially-synchronous basins, which is model-independent.
- **"2^83 attractors — are they all real, or sampling noise?"** → The count is bounded
  below by what we *observe* (≥1010 distinct), and the labels are permanent across
  disjoint time windows (99.9% bit agreement), so they're not noise. The true count is
  saturated by our sampling, not by the dynamics.
- **"MATLAB-vs-GPU speedup?"** → Honest answer: pending. There's no MATLAB license on
  our machines, so I benchmarked against a serial CPU transcription of the same
  algorithm — that isolates parallelization from language overhead. A true
  MATLAB-vs-GPU number is on the to-do list.
- **"What does 'streaming isn't the VPS' mean for your results?"** → The lobe-locking
  finding doesn't depend on the VPS at all — it uses sign of the long-time average, no
  correlation, no lag. So that result stands regardless. The VPS question matters for
  reproducing the paper's *specific* chimera diagnostic, which is future work.
- **"Do you have the perturbation results yet?"** → Honest answer: running as of this
  talk, not in hand. It's a direct test of the paper's claim that a vanishingly small
  perturbation near a riddled point should flip the outcome — I'm not going to put a
  number on the slide I haven't measured yet. What I *do* have is the smoke test that
  caught my own bug first: an "uncoupled control" that turned out invalid, because K=0
  labels are already a coin flip before any perturbation touches them. Fixed design
  (boundary vs. interior at the same coupling) is what's running now.
- **"Why does the surrogate disagree with the true VPS on Hindmarsh-Rose but not
  Lorenz?"** → Honest answer: not resolved yet, flagged as an open question on slide
  10. Leading candidate: HR's slow adaptation variable organizes bursting on a much
  longer timescale than the surrogate's running-mean window can track, so it may be
  averaging over structure the true lag-search can still see. Could also be that the
  true VPS's lag search is finding genuinely different (tau, L) pairs in bursting
  dynamics than in continuously-chaotic ones. Both are testable, neither is tested yet.
- **"If riddling isn't connectome-specific, why does the connectome matter at all?"**
  → Riddling as a *phenomenon* looks generic to any similarly-sized network — that's
  the honest read of slide 11. What's still open is whether the real connectome's
  *specific pattern* of riddling (which nodes lock together, in what combinations)
  reflects real anatomy, versus just being one riddled configuration among many
  equally-likely ones. That's a different, harder question than "does it riddle,"
  and it's not settled by this comparison.

---

## CLV / Kaplan–Yorke backup (for slide 11 and Q&A)

Full math and provenance: [`talk/notes/clv_kaplan_yorke_pipeline.md`](notes/clv_kaplan_yorke_pipeline.md).
Cross-check against the true-VPS/surrogate/lobe-locking comparison:
[`talk/notes/meeting_outline_2026-07-22.md`](notes/meeting_outline_2026-07-22.md).

**One-breath version, if asked:** CLV/Kaplan-Yorke is a third, independent
riddling check that doesn't use any basin-labeling scheme at all — it
looks at the covariant Lyapunov vectors' transversality angles directly.
It agrees with the VPS-surrogate and lobe-locking results: RIDDLED at
every coupling tested.

**How it's built:** Ginelli two-pass algorithm on the coupled-Lorenz
Jacobian — forward RK4 + periodic QR to get orthonormal-but-not-covariant
directions and their stretching factors (`T = QR`), then a backward
triangular solve (`R_k C_{k-1} = C_k`) to recover the true covariant
vectors `V_k = Q_k C_k`. The `R` diagonals give the Lyapunov exponents;
Kaplan-Yorke interpolates where the cumulative exponent sum crosses zero:
`D_KY = k + (Σ_{i≤k} λ_i) / |λ_{k+1}|`.

**The honest caveat, if pressed:** the sweep only resolved a real D_KY at
one of four couplings. At K=0.05/0.10/0.15 the cumulative sum was still
positive after all 83 computed CLVs, so the code reports a **ceiling**
(`D_KY >= 83`, not a number) rather than a fabricated value — meaning the
true dimension there is *at least* 83 out of 249 tangent dimensions, and
resolving it for real needs many more CLVs. Only K=0.20 (14/83 positive
exponents) gave a resolved D_KY of 45.4. The burst-fraction/riddling
verdict itself, unlike D_KY, doesn't need the ceiling resolved — that's
why the RIDDLED call stands at every coupling even though the dimension
number only exists for one of them.

| coupling | n_positive / m | ceiling? | D_KY reported |
|---|---|---|---|
| 0.05 | 83/83 | yes | `>= 83` |
| 0.10 | 74/83 | yes | `>= 83` |
| 0.15 | 34/83 | yes | `>= 83` |
| 0.20 | 14/83 | no | 45.40 |

*(If asked "why didn't you just raise --m" — n=3·83=249 is the hard
ceiling; going much past 83 CLVs starts approaching the full tangent
space and gets expensive fast. This is future work, not done tonight.)*

---

## "What I got wrong" backup (cut from the 16-slide deck for time, Q&A only)

Was its own slide. Good material if asked "what was the hardest part" or "what
mistakes did you make": four hypotheses proposed and killed with a control each —
(1) everything collapses to sync, killed: no sync at any K; (2) IC jitter
manufactured the static map, killed: zero jitter gives an identical result; (3) the
whole D_f≈1.9 result is a clustering artifact, killed: it's real riddling; (4) my
own structure test was correct, killed: it had three separate bugs. The one that
stings, if asked for a personal reflection: the "D→2 is ambiguous" warning is on
slide 4, and it still took days to apply it to my own conclusion. Every one of these
died to a control against a system whose answer was already known — that's the only
reason the real mechanism ever surfaced. Closing line if used: **"The most
dangerous bug is the one that makes your result look better."**

---

## Onset-curve backup (cut from the 16-slide deck for time, Q&A only)

Was its own slide ("Where Does the Locking Switch On?"). Real finding, worth having
ready if asked "does the locking always happen, or is there a transition?": swept a
deterministic 32² slice, 1024 ICs per coupling. Locking is a clean **sigmoid
bifurcation at K≈0.07** — below the window anyone had swept before. The largest
basin is ~0.1% of ICs below onset, peaks at 9.3% near K≈0.12, falls to 0.4% by
K=0.2. Practical punchline: the paper's own coupling (gel=0.5) sits deep in the
riddled regime, basins too finely intermingled to map on any plane — the only
legible window for a basin-plane figure is roughly K≈0.08–0.12.
`data/derivatives/onset_curve.png` if it comes up.

---

## Perturbation-test bug + basin reconnaissance backup (Q&A only, not in the 13 slides)

Full detail and both figures: [`talk/notes/meeting_outline_2026-07-22.md`](notes/meeting_outline_2026-07-22.md).

**The invalid-control catch, one breath:** Built the perturbation-sensitivity
test with K=0 as an "uncoupled control," and the control curve came back
looking just as sensitive as the riddled K=0.5 case — because at K=0 there's
zero lobe-locking, so the basin label there is already a coin flip before any
perturbation touches it. Not a control, just noise. Fix: compare boundary vs.
interior points at the *same* coupling instead of comparing across couplings.
Smoke-tested at 96×96, now resubmitted at production scale.

*(`data/derivatives/perturbation_sensitivity_raw.png` — P_flip(δ) for
K=0.0/0.1/0.5; the K=0.0 curve, labeled INVALID, is already saturating to
P_flip≈1 within two decades of δ and tracks the riddled K=0.5 curve almost
exactly. This is the plot that made the bug visible, not the final result.)*

**Basin reconnaissance near K=0.5, one breath:** Before committing to K=0.5 as
the perturbation test's reference slice, ran three basin-map sweeps at
K=0.45/0.475/0.50 on the same (73, 81) node pair the true-VPS ladder uses.
D_f came back flat — 1.8671 → 1.8671 → 1.8673, R²≈0.999 — and 94–95% of ICs
are their own distinct 83-bit lobe pattern at all three couplings. Another
independent confirmation that the riddled regime is already fully
established well before K=0.5, consistent with everything else in the talk.

*(`data/derivatives/basin_maps_K045_050.png`, built by
`scripts/plot_basin_maps_k045_050.py` — three basin planes plus the flat
D_f-vs-K line.)*

---

## Delivery reminders

- The three callbacks are the spine: **plant** the ambiguity (slide 4), **detonate**
  it (slide 7), **resolve** it (slides 8–9). Land each one.
- Say the word **"lag"** on slide 5 — it pays off on slide 7's memory-wall reveal.
- Slides 5 and 8 are the two you must not rush; they each carry two merged ideas.
  Everything else can flex.
- Slides 10–11 (HR, null model) are new and carry real findings — don't compress
  these to make time; if you're behind schedule, cut from slide 2-4's framing
  instead, since 10-11 are the least likely material for this audience to have
  seen before.
- If you're over time going into 12, skip straight to the takeaway line and stop —
  don't try to list every bullet.
