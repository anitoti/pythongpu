# RAPS Practice Script — *Rapidly Examining Fractal Basin Structure on the GPU*

**Anna Totilca · Clarkson HPC REU 2026 · Advisor: Dr. Jeremie Fish**
Deck: `talk/summer_talk.tex` (16 slides). Target: **≤ 15 min.** This script is
budgeted at **~13.5 min** to leave room for pauses, laughs, and questions.

**How to use this:** the *italic cues* are delivery notes (don't say them). The plain
text is roughly what to say — say it in your own words, don't memorize verbatim. The
running clock on the right tells you if you're on pace. If you hit a slide and the
clock is behind, cut to the **bold** sentence and move on.

> ✅ **All 16 slides have real figures.** Slide 9's basin map is generated from the
> measured `data/derivatives/lorenz_basins_n73_n81_K0.6500.npz` run (8312 distinct
> 83-bit lobe patterns among 9216 ICs) — genuine static, not a mock-up.

---

## Pacing table

| # | Slide | Budget | Clock ends |
|---|---|---|---|
| 1 | Title | 0:15 | 0:15 |
| 2 | The Science | 1:00 | 1:15 |
| 3 | The System & the Plane | 0:55 | 2:10 |
| 4 | Measuring Crinkliness + landmine | 0:55 | 3:05 |
| 5 | My Project: port the VPS | 1:00 | 4:05 |
| 6 | The Port Was Slow (HPC core) | 1:15 | 5:20 |
| 7 | Did I Reproduce It? | 0:55 | 6:15 |
| 8 | The Memory Wall | 0:55 | 7:10 |
| 9 | Pipeline Reported + Detonates | 0:55 | 8:05 |
| 10 | Validate the Ruler | 0:55 | 9:00 |
| 11 | The Mechanism | 1:00 | 10:00 |
| 12 | Where Locking Switches On | 0:55 | 10:55 |
| 13 | The Result | 1:00 | 11:55 |
| 14 | What I Got Wrong | 0:50 | 12:45 |
| 15 | Summary & Next | 0:45 | 13:30 |
| 16 | Thank you | 0:05 | 13:35 |

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

## 3 — The System & the Plane · *(0:55)*

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

## 5 — My Project: port the VPS · *(1:00)*

So here's my job. Resolving a fractal boundary needs a *huge* grid of initial
conditions — each one an independent simulation. That's embarrassingly parallel, but
the original MATLAB runs them one at a time.

And the expensive kernel is this thing called the **Vector Pattern State**, the VPS.
It labels each run by how *every pair* of regions moves together — for 83 nodes
that's 3,403 pairs. For each pair it computes two numbers: **tau**, the time lag
between them, and **L**, how different they are once you line them up by that lag.
**The lag is the whole point** — it's what detects phase-lagged synchrony, the
chimera signature. Keep the word "lag" in mind.

*(Gesture to the code.)* MATLAB does this with a for-loop over all 3,403 pairs. I
turned that into **one batched FFT** — Wiener–Khinchin turns correlation into
multiplication — plus batched RK4 for the integration, and SLURM job arrays, one
coupling per GPU. On ACRES that's **130,000 initial conditions per coupling at once.**

---

## 6 — The Port Was Slow — and the FFT Wasn't Why · *(1:15)* — **HPC centerpiece**

Here's the honest part. My first port was **slow** — it never cleared about seven
times faster than a plain serial CPU version. At small sizes it was actually *slower*
than serial. For an HPC project, that's a failing grade.

*(Point at the code snippet.)* So I profiled it. And the bottleneck was **not** the
FFT. The FFT was fine. The problem was that *after* the FFT found the lag, I had a
**Python loop over every possible lag value** — twenty thousand iterations — that
threw the whole parallel win away.

The fix was to realize each pair has its own lag, so I build **per-pair aligned index
grids and gather once, with no loop at all.** That took it from about three times
faster to **375 times faster** than serial — and I verified it matches the reference
to float32 precision, so it's not fast *and wrong*.

**The one-liner: the FFT was never the problem. The loop after it was.** That's the
whole HPC lesson on this slide — the speedup lives in the part you didn't think to
look at.

*(If behind schedule, stop here and go to slide 7.)*

---

## 7 — Did I Actually Reproduce It? · *(0:55)*

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

## 8 — The Memory Wall · *(0:55)*

Then I hit a wall. The *real* VPS needs the entire trajectory of every simulation in
memory to cross-correlate it. At even a modest grid, that's **41 gigabytes** — it
doesn't fit.

So the production runs quietly switched to a **streaming approximation** that never
stores a trajectory. Clever for memory — *but* *(point at the table)* the streaming
version isn't the VPS. It redefines both tau and L, and crucially **there's no lag
anywhere in it.** And the lag was the entire reason the VPS exists.

*(Beat.)* **Every result I got this summer came from that streaming path.** A resource
constraint silently changed the science underneath me — and nobody noticed for weeks,
including me. *That's* the HPC lesson people don't put on slides.

---

## 9 — What the Pipeline Reported — and the Ambiguity Detonates · *(0:55)*

And at first the pipeline gave us **exactly the result the project wanted**:
dimension between 1.9 and 1.96, gorgeous fits, near-total sensitivity. *(Short pause.)*

But two things itched. The dimension was **flat** as I changed the coupling — a real
bifurcation should move it. And the automatic basin count and the silhouette score
flatly disagreed. *(Point at the map — the static.)* And the basin map was **pure
static.** No connected regions at any zoom.

*(This is the callback — say it deliberately.)* Remember slide 4? Dimension-near-2
means *either* a real fractal boundary *or* a meaningless labelling. **We had measured
dimension-near-2** — and k-means will *always* hand back k confident groups, even from
noise. So I genuinely thought I'd found an artifact.

---

## 10 — Validate the Ruler Before Trusting the Measurement · *(0:55)*

To settle it I needed a measurement that uses **no clustering at all.** I used each
node's long-time average — a time average, so it's constant on an attractor.

But here's the methodological move I want you to take away: **before trusting the
ruler, I checked it on a system whose answer I already knew.** At zero coupling the
network is uncoupled — one chaotic attractor per node — so the averages *must*
converge together. *(Point at the plot.)* And they do: the zero-coupling curve falls
off like one-over-root-T, exactly as ergodicity demands. The coupled curve stays
**flat** — even under 320 times more burn-in.

So it's not unconverged, and it's not a transient. Ergodicity forbids that on *one*
attractor. **Which means there are many.**

---

## 11 — The Mechanism: Coupling Locks Nodes onto Lobes · *(1:00)*

And here's *why* there are many. A Lorenz attractor has two wings. A node's long-time
average is **zero if it visits both wings, and about plus-or-minus 8 if it locks onto
one.** *(Point at the histogram.)* Uncoupled, everything sits at zero. Coupled,
**98% of nodes are pinned to a single wing.**

So the network's attractor is just a **pattern of plus and minus signs across 83
nodes.** And that means the basin label is *exact* — it's 83 bits, no clustering, no
k, no elbow, nothing to argue about. That gives up to **two-to-the-83 — about
ten-to-the-25 — possible attractors.**

And they're **permanent**: take the same initial conditions in two separate time
windows, and 99.9% of the bits agree. This is the finding — and notice it makes every
clustering headache in this talk simply disappear.

---

## 12 — Where Does the Locking Switch On? · *(0:55)* — **the RAPS figure**

*(This is the figure that's genuinely new — own it.)* Nobody had asked *when* the
locking turns on. So I swept it. It's a clean **sigmoid bifurcation at a coupling of
about 0.07** — *below* the window anyone had looked at.

And the practical payoff for a plane-scanning tool: the largest basin is basically
zero below onset, peaks around **9% near coupling 0.12**, and then collapses again.
The paper's own coupling, 0.5, sits **deep in the riddled regime** — the basins are
too finely intermingled for *any* plane to show structure. **The only coupling window
where a basin plane is even legible is roughly 0.08 to 0.12.** That tells the next
person exactly where to point the software.

---

## 13 — The Result: The Basins Really Are Mingled · *(1:00)*

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

## 14 — What I Got Wrong · *(0:50)*

I want to show you the wrong turns, because they're the most useful part. I proposed
four hypotheses and **killed every one with a control**: that it all collapses to
sync — no; that jitter made the static — no; that the whole thing was a clustering
artifact — no, it's real; and that my own test was correct — it had three bugs.

*(Point at the red box.)* The one that stings: I put the "dimension-2 is ambiguous"
warning on slide 4, and then spent *days* failing to apply it to my own conclusion.
Every one of these died to a control against a system whose answer I already knew —
and that's the *only* reason the real mechanism ever surfaced.

**The most dangerous bug is the one that makes your result look better.**

---

## 15 — Summary & Next · *(0:45)*

So, in one breath: I **built the fast plane-scanner** — 130,000 simulations at once,
375 times over serial. I **tested my own port** and found it doesn't match the source
— possibly a bug in the *published* code. I found a **memory wall** that had silently
swapped out the real computation. And I built **validated diagnostics** that turned up
the mechanism: coupling locks nodes onto lobes, giving a thousand-plus permanent,
mingled basins.

Next up: settle the lag question with Dr. Fish, beat the memory wall so the *real*
VPS runs at full scale, and explore other viewing planes.

**The takeaway: a fast pipeline that can't fail isn't a measurement. The speedup was
the easy part — checking that the fast thing still computed the *right* thing is where
the summer actually went.**

---

## 16 — Thank you · *(0:05)*

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

---

## Delivery reminders

- The three callbacks are the spine: **plant** the ambiguity (slide 4), **detonate**
  it (slide 9), **resolve** it (slides 11–13). Land each one.
- Say the word **"lag"** on slides 5, 8 — it pays off the memory-wall slide.
- Slides 6 and 11 are the two you must not rush. Everything else can flex.
- If you're over time at slide 13, compress 14 to just the last bold line and skip
  straight to the takeaway on 15.
