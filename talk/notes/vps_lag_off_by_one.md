# The VPS lag off-by-one: what happened, and reverification

## The bug

The paper's Vector Pattern State computes, for every pair of nodes `(i, j)`:

1. **`tau_ij`**: the time lag at the peak of the cross-correlation between
   node `i` and node `j`'s time series.
2. **`L_ij`**: the norm of the residual `x_i(t + tau) - x_j(t)`, computed
   *after* shifting the two signals into alignment by that lag.

The paper's own reference MATLAB (`VectorPatternState.m`) implements the
alignment as:

```matlab
% MATLAB, lag > 0:
norm( x(1:end-(lag-1), j) - x(lag:end, i) )
```

This shifts by `lag - 1`, not `lag`. At `lag = 1` this applies **no shift at
all** — both slices are the full series — even though the code just detected
a lag of 1 sample. A direct transcription of the *intended* alignment
(`|tau|`, not `|tau|-1`) is:

```python
# PyTorch, lag > 0:
x[lag:, i] - x[:-lag, j]
```

These two are not a rounding-error apart — the under-shift systematically
**inflates** every residual, because it leaves the pair mis-aligned by
exactly one sample.

## How it was found

Transcribed the paper's VPS line-by-line and compared both alignment
conventions against the paper's own worked example, `Example_A_3.mat`
(source: `~/matlab_fractalbasin`, the original MATLAB project this repo
ports). Computed `L` both ways for all 36 node pairs in that matrix and
plotted one against the other (`talk/figs/alignment_bias.png`,
`talk/make_figs.py:fig_alignment`).

**Result:** all 36 of 36 pairs land on the same side of the agreement line —
the `|tau|-1` alignment is larger every time, by a mean ratio of 1.96×.
Every point above the diagonal (not scattered around it) is what makes this
a systematic off-by-one rather than noise: a random discrepancy would
straddle the line.

## Reverification (2026-07-23)

`talk/make_figs.py`'s `fig_alignment()` uses its own small, standalone
`vps()` implementation to build the figure — it does not call the
project's actual production code. Before trusting the 36/36 claim for the
deck, reran the same comparison using the real pipeline function,
`vector_pattern_state_fast` in
[`pythongpu/processing/feature_extraction.py`](../../pythongpu/processing/feature_extraction.py),
which has an explicit `alignment` parameter (`"corrected"` vs `"matlab"`)
for exactly this purpose:

```python
from pythongpu.processing.feature_extraction import vector_pattern_state_fast
vps_corrected = vector_pattern_state_fast(x, alignment="corrected")
vps_matlab    = vector_pattern_state_fast(x, alignment="matlab")
```

Result on the same `Example_A_3.mat` matrix:

| check | result |
|---|---|
| pairs where `L_matlab >= L_corrected` | **36 / 36** |
| mean ratio `L_matlab / L_corrected` | **1.956** |
| max \|L_matlab - L_corrected\| | 0.707 |
| pairs where `tau` itself differs between the two modes | **0 / 36** |

This matches the standalone figure script's claim (1.96×) using the actual
production code instead of a one-off reimplementation — the finding holds.
The last row is an important sanity check: `tau` (which lag was detected)
is identical either way, since the alignment convention only changes how
far the signals are *shifted*, not which lag the cross-correlation found.
If `tau` had differed between the two modes, that would indicate a bug
unrelated to the off-by-one being investigated.

## Which one is "correct"?

Two different, both legitimate goals point in different directions:

- **To reproduce the paper's published numbers exactly**, use
  `alignment="matlab"` — it deliberately replicates the reference
  implementation's exact behavior, off-by-one included. This is what
  "an accurate port" means if the goal is matching what was published.
- **To compute what the lag is actually supposed to measure**, use
  `alignment="corrected"` — the under-shift is very likely an indexing bug
  (0- vs 1-indexing, or a dropped first-step convention) in the original
  MATLAB, not an intentional choice.

The code ships both, gated by the `alignment` parameter, specifically so
this doesn't need to be decided unilaterally — the code produces a
measurement either way, not an argument for one over the other. This is
exactly the question to raise with Dr. Fish: is `|tau|-1` in the reference
implementation deliberate, or a bug that's been sitting in the published
pipeline?

## What's been run at production scale, and what hasn't

As of 2026-07-23:

- **`corrected`** alignment: run at production scale on real DTI-Lorenz
  data (4 couplings, grid 96²) — this is the run validated against the
  streaming surrogate (AGREE, max |ΔD_f| = 0.067; see
  `talk/notes/meeting_outline_2026-07-22.md`).
- **`matlab`** alignment (the literal paper-reproduction mode): validated
  only against the static `Example_A_3.mat` test matrix above, until
  tonight. A matching production-scale run (same 4 couplings, same node
  pair, same grid) was submitted to ACRES as job `4556214` to give an
  "exact reproduction" data point at real scale, not just the toy example.
  Check `data/derivatives/true_vps_matlab_c{K}/` for results once it
  finishes.
