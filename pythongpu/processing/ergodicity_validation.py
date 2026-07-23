"""
System-agnostic, clustering-free validation of "how many attractors are there."

Formalizes the check this project used to catch the k-means "meaningless
labelling" artifact (k-means always returns k confident groups, even from
pure noise -- see lobe-locking-is-the-mechanism / basin-fractal-regime memos)
into two quantitative tests instead of an eyeballed plot:

  1. Ergodic scaling exponent: on a single attractor, the across-IC spread of
     the per-IC long-time average of an observable must shrink like 1/sqrt(T)
     (standard ergodic-theorem / CLT scaling). Fit log(spread) vs log(T) and
     test the slope against -0.5, instead of just "does the curve look flat."
  2. Formal permanence: sign(<x_i>) compared across two genuinely DISJOINT
     integration windows [0, T/2) and [T/2, T), upgraded with a z-score so a
     sign flip near the sign boundary (consistent with measurement noise) is
     distinguished from a statistically genuine flip.

STATISTICAL NOTES, read before trusting the scaling-fit p-values -- TWO
separate issues were found and fixed/worked around while building this:

1. The spread at each checkpoint T_k must come from statistically
   INDEPENDENT initial conditions, not from re-measuring the SAME growing
   trajectory at increasing prefixes. Nested prefixes share early data, so
   their residuals are strongly autocorrelated -- an OLS regression through
   them understates the true uncertainty. Fixed by using DISJOINT
   SUB-BATCHES of initial conditions per checkpoint (`checkpoint_batch_split`)
   -- different ICs' trajectories are legitimately independent draws.

2. Even after fixing (1), testing the fitted slope against the exact
   asymptotic value -0.5 is the WRONG test: at finite T the slope has a
   real, expected finite-size bias toward 0 (confirmed empirically here --
   quadrupling T moved a measured -0.4626 to -0.4919, converging exactly as
   expected), while the regression's standard error shrinks *faster* than
   that bias vanishes as the fit gets cleaner. So a literal z-test against
   -0.5 will eventually reject the TRUE ergodic case too, purely from
   having enough precision -- it is not a usable pass/fail criterion by
   itself. The correct test compares the fitted slope AT THE COUPLING OF
   INTEREST against the fitted slope AT A KNOWN-TRIVIAL CONTROL (e.g. K=0),
   both measured with the same T budget so finite-T bias affects them
   equally and cancels in the comparison -- see `compare_scaling_to_control`.
   Use that, not `consistent_with_single_attractor` in isolation, to decide
   whether a coupling's scaling looks like the control's or looks flat.

Reusable across systems: only requires an `rk4_step_fn(x, L_gpu, p) -> x'`
with the (B, N, 3) state convention every system in this repo already shares
(lorenz_sweep.py, hr_sweep.py, rossler_sweep.py all have this exact
signature), plus `p.steps_record`. Nothing Lorenz-specific.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch


@dataclass
class ScalingFitResult:
    slope: float
    slope_stderr: float
    intercept: float
    r_squared: float
    z_vs_ergodic: float             # (slope - (-0.5)) / slope_stderr
    consistent_with_ergodic: bool   # |z| < 2 -> can't reject slope=-0.5


@dataclass
class ControlComparisonResult:
    z_vs_control: float           # (slope_test - slope_control) / combined_stderr
    matches_control: bool         # |z| < 2 -> scaling looks like the control's
    slope_test: float
    slope_control: float


def compare_scaling_to_control(result_test: dict, result_control: dict) -> ControlComparisonResult:
    """
    The test that actually answers "is this coupling's scaling different
    from a known-trivial baseline," per the module docstring's note (2):
    compares two `validate_ergodicity(...)` summaries computed at the SAME
    T budget (same `n_scaling_checkpoints`, same `p.steps_record`) so
    finite-T bias is shared and cancels, rather than testing either slope
    against the idealized asymptotic -0.5.
    """
    s_t, se_t = result_test["scaling_exponent"], result_test["scaling_exponent_stderr"]
    s_c, se_c = result_control["scaling_exponent"], result_control["scaling_exponent_stderr"]
    combined_se = (se_t**2 + se_c**2) ** 0.5
    z = (s_t - s_c) / combined_se if combined_se > 0 else float("inf")
    return ControlComparisonResult(
        z_vs_control=float(z), matches_control=bool(abs(z) < 2.0),
        slope_test=s_t, slope_control=s_c,
    )


@dataclass
class PermanenceResult:
    agreement_fraction: float          # naive sign(<x>) agreement, window1 vs window2
    genuine_flip_fraction: float       # sign differs AND |z| > 2 (real, not noise)
    marginal_flip_fraction: float      # sign differs but |z| <= 2 (near the boundary)


@torch.no_grad()
def _welford_integrate(
    x0: torch.Tensor, L_gpu: torch.Tensor, p, rk4_step_fn, device, n_steps: int,
    component: int = 0, split_at: int | None = None,
):
    """
    Integrate a batch for n_steps, accumulating a single running Welford
    mean/M2 of x[...,component] per (IC, node). If `split_at` is given, ALSO
    accumulate a second, independent Welford pass that starts fresh once
    step > split_at (giving genuinely disjoint window-2 statistics without a
    second integration).
    """
    B, N, _ = x0.shape
    x = x0
    count = 0
    mean = torch.zeros(B, N, device=device)
    M2 = torch.zeros(B, N, device=device)

    mean_pre = M2_pre = None
    count2 = 0
    mean2 = torch.zeros(B, N, device=device)
    M2_2 = torch.zeros(B, N, device=device)

    for step in range(n_steps):
        x = rk4_step_fn(x, L_gpu, p)
        val = x[..., component]
        count += 1
        delta = val - mean
        mean = mean + delta / count
        M2 = M2 + delta * (val - mean)

        if split_at is not None:
            if step + 1 == split_at:
                mean_pre, M2_pre = mean.clone(), M2.clone()
            if step + 1 > split_at:
                count2 += 1
                delta2 = val - mean2
                mean2 = mean2 + delta2 / count2
                M2_2 = M2_2 + delta2 * (val - mean2)

    if split_at is None:
        return mean, M2
    return mean, M2, mean_pre, M2_pre, mean2, M2_2, count2


def checkpoint_batch_split(B: int, n_checkpoints: int) -> list[slice]:
    """Split a batch of B ICs into n_checkpoints disjoint, near-equal-size
    groups -- one per scaling checkpoint, so each checkpoint's spread
    estimate comes from statistically independent initial conditions rather
    than a re-measurement of the same trajectories at a longer prefix."""
    edges = np.linspace(0, B, n_checkpoints + 1).astype(int)
    return [slice(edges[i], edges[i + 1]) for i in range(n_checkpoints)]


@torch.no_grad()
def validate_ergodicity(
    x0            : torch.Tensor,
    L_gpu         : torch.Tensor,
    p,
    rk4_step_fn   : Callable[[torch.Tensor, torch.Tensor, object], torch.Tensor],
    device        : torch.device,
    n_scaling_checkpoints: int = 5,
    component     : int = 0,
) -> dict:
    """
    Full clustering-free validation for one coupling value.

    Scaling-exponent fit: splits the IC batch into `n_scaling_checkpoints`
    disjoint groups and integrates group k for T_k = T_max / 2^(n-1-k)
    steps -- independent ICs at each checkpoint, so the log-log regression's
    standard error is trustworthy (see module docstring).

    Permanence check: uses the LONGEST-integrated group (T_max), split into
    two genuinely disjoint halves [0, T_max/2) and [T_max/2, T_max) via a
    second Welford accumulator carried alongside the first in the same pass.

    Returns a JSON-serialisable summary dict, same convention as
    clv_topology.run_clv_topology.
    """
    T_max = p.steps_record
    checkpoints = sorted(set(
        T_max // (2**k) for k in range(n_scaling_checkpoints - 1, -1, -1)
        if T_max // (2**k) > 1
    ))
    n = len(checkpoints)
    B_total = x0.shape[0]
    groups = checkpoint_batch_split(B_total, n)

    spreads = []
    permanence = None
    for k, (T_k, group) in enumerate(zip(checkpoints, groups)):
        x0_k = x0[group]
        is_longest = (k == n - 1)
        if is_longest:
            mean, M2, mean1, M2_1, mean2, M2_2, count2 = _welford_integrate(
                x0_k, L_gpu, p, rk4_step_fn, device, T_k,
                component=component, split_at=T_k // 2)
            permanence = _formal_permanence_check(
                mean1, M2_1, T_k // 2, mean2, M2_2, count2)
        else:
            mean, M2 = _welford_integrate(
                x0_k, L_gpu, p, rk4_step_fn, device, T_k, component=component)
        spreads.append(float(mean.std(dim=0).mean().item()))

    scaling = _fit_ergodic_scaling(np.array(spreads), np.array(checkpoints, dtype=float))

    return dict(
        scaling_exponent=scaling.slope,
        scaling_exponent_stderr=scaling.slope_stderr,
        scaling_fit_r_squared=scaling.r_squared,
        z_vs_ergodic=scaling.z_vs_ergodic,
        consistent_with_single_attractor=scaling.consistent_with_ergodic,
        checkpoints=checkpoints,
        ics_per_checkpoint=[g.stop - g.start for g in groups],
        permanence_agreement_fraction=permanence.agreement_fraction,
        permanence_genuine_flip_fraction=permanence.genuine_flip_fraction,
        permanence_marginal_flip_fraction=permanence.marginal_flip_fraction,
    )


def _fit_ergodic_scaling(spread: np.ndarray, checkpoints: np.ndarray) -> ScalingFitResult:
    spread = np.clip(spread, 1e-300, None)   # guard log(0)
    log_T = np.log(checkpoints)
    log_s = np.log(spread)
    n = len(log_T)
    slope, intercept = np.polyfit(log_T, log_s, 1)

    fitted = slope * log_T + intercept
    resid = log_s - fitted
    dof = max(n - 2, 1)
    s_err2 = float((resid**2).sum() / dof)
    sxx = float(((log_T - log_T.mean())**2).sum())
    slope_stderr = float(np.sqrt(s_err2 / sxx)) if sxx > 0 else float("inf")

    ss_res = float((resid**2).sum())
    ss_tot = float(((log_s - log_s.mean())**2).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    z = (slope - (-0.5)) / slope_stderr if slope_stderr > 0 else float("inf")
    return ScalingFitResult(
        slope=float(slope), slope_stderr=slope_stderr, intercept=float(intercept),
        r_squared=r_squared, z_vs_ergodic=float(z),
        consistent_with_ergodic=bool(abs(z) < 2.0),
    )


def _formal_permanence_check(
    mean_1: torch.Tensor, M2_1: torch.Tensor, T1: int,
    mean_2: torch.Tensor, M2_2: torch.Tensor, T2: int,
) -> PermanenceResult:
    """
    Upgrades "does sign(<x_i>) agree across two disjoint windows" from a raw
    percentage into a statistically-aware one: a sign flip where both
    estimates are within ~2 combined standard errors of zero is a marginal
    boundary case (consistent with measurement noise, not a real difference),
    while a sign flip far outside that band is a genuine change of attractor.
    """
    sign1 = torch.sign(mean_1)
    sign2 = torch.sign(mean_2)
    agree = (sign1 == sign2)

    var1 = M2_1 / max(T1 - 1, 1)
    var2 = M2_2 / max(T2 - 1, 1)
    sem1 = torch.sqrt(var1 / T1)
    sem2 = torch.sqrt(var2 / T2)
    combined_sem = torch.sqrt(sem1**2 + sem2**2) + 1e-12
    z = (mean_1 - mean_2) / combined_sem

    disagree = ~agree
    genuine_flip = disagree & (z.abs() > 2.0)
    marginal_flip = disagree & (z.abs() <= 2.0)

    total = float(agree.numel())
    return PermanenceResult(
        agreement_fraction=float(agree.sum()) / total,
        genuine_flip_fraction=float(genuine_flip.sum()) / total,
        marginal_flip_fraction=float(marginal_flip.sum()) / total,
    )
