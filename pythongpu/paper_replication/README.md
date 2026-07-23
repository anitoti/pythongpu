# Sorting this project by "Acts"

This is a map of where things live, organized by the three-part story this
project actually tells — not a proposal to move the whole codebase (that's
riskier than it's worth this close to the deadline, and most of Acts 2/3
are already validated and running production jobs). New Act-1 work goes in
this `pythongpu/paper_replication/` package; everything else stays where
it is, indexed here.

## Act 1 — Accurate reproduction of the original MATLAB

**Goal:** match `~/matlab_fractalbasin`'s actual equations and pipeline as
closely as possible, so the other two Acts are well-motivated by a port
that's actually correct, not just fast.

| What | Where |
|---|---|
| Audit of every file in the original MATLAB, gaps found | [`talk/notes/matlab_source_audit.md`](../../talk/notes/matlab_source_audit.md) |
| The VPS lag off-by-one, reverified | [`talk/notes/vps_lag_off_by_one.md`](../../talk/notes/vps_lag_off_by_one.md) |
| True (lag-based) VPS, both alignment conventions | [`pythongpu/processing/feature_extraction.py`](../processing/feature_extraction.py) — `vector_pattern_state_fast(..., alignment="matlab"\|"corrected")` |
| **Exact Hindmarsh-Rose physics** (Y-coupling + chemical/sigmoidal term, matching `HR_ElCh_network.m` line for line) | [`hindmarsh_rose_exact.py`](hindmarsh_rose_exact.py) — **new tonight, replaces the simplified X-coupled HR model for Act-1 purposes** |
| Seeded null-model graphs (BA/GNM/WS), matched to the real DTI connectome's size | [`scripts/seeded_random_graph_sweep.py`](../../scripts/seeded_random_graph_sweep.py) |
| True-VPS production run, `matlab` alignment (exact paper reproduction, not the `corrected` fix) | ACRES job 4556214, `data/derivatives/true_vps_matlab_c{K}/` |

**Still missing / open, per the audit:**
- `KmeansBIC`'s variance term uses a different denominator than the
  original (`N-K` vs. the original's `N*d`) — not yet fixed.
- `boxcount.m` (third-party, dyadic box sizes) hasn't been directly
  compared against this project's `boxcount_2d_gpu`.
- The actual grid-sweep driver for the DTI/HR case (integrate → VPS →
  cluster → save, the equivalent of `Henon_ManyPlots.m` but for HR) isn't
  in the given MATLAB folder and hasn't been rebuilt yet.
- The paper's other systems/graphs (6-node synthetic network without
  symmetries, 10-node Kuramoto with random edge removal, Henon on the DTI
  network) haven't been ported to Act 1 yet — only the null-model graph
  generation is done so far.

## Act 2 — The streaming surrogate, and why it was necessary

**Goal:** the memory-forced approximation that everything this summer
actually ran on, and the validation work asking whether it was trustworthy.

| What | Where |
|---|---|
| Streaming VPS surrogate (Welford, no lag search) | `pythongpu/pipeline/*_sweep.py` — `run_sweep_streaming` in each system's file |
| True-VPS-vs-surrogate validation (Lorenz: agree; HR: disagree) | `talk/notes/meeting_outline_2026-07-22.md`, `data/derivatives/*_true_vps_vs_surrogate_comparison.png` |
| Exact clustering-free lobe-locking label | same `*_sweep.py` files — `sign(mean_x)` |
| Formal ergodicity/permanence validation | [`pythongpu/processing/ergodicity_validation.py`](../processing/ergodicity_validation.py) |

## Act 3 — Escaping the 2D plane: CLV, Kaplan-Yorke, null-model comparison

**Goal:** methods that don't need a 2-D IC-slice or any clustering at all.

| What | Where |
|---|---|
| CLV / Kaplan-Yorke pipeline (Ginelli algorithm, Jacobian-based) | [`pythongpu/pipeline/clv_topology.py`](../pipeline/clv_topology.py), `clv_cli.py`, `hr_clv_cli.py` |
| DTI vs. null-model (BA) riddling comparison | `scripts/plot_dti_vs_null_clv.py`, `data/derivatives/dti_vs_null_clv_comparison.png` |
| HR CLV results (fully resolved D_KY, unlike Lorenz's ceiling) | `output/hr_clv_results/`, `data/derivatives/hr_vps_surrogate_vs_clv_comparison.png` |

---

Note on why Lorenz, not confusion: Lorenz was chosen deliberately as a
non-brain chaotic system for methodology development, independent of
`HR_ElCh_network.m`'s stale "network of lorenz oscillators" comment in the
original MATLAB (that comment is a coincidence, not the cause — confirmed
directly with the project owner). Hindmarsh-Rose is being added properly
(Act 1, exact physics) *and* was already added in a simplified form for
Acts 2/3 (`pythongpu/oscillators/hindmarsh_rose.py`) — these are two
different HR models serving two different goals, not a duplicate.
