"""
Multifractal Detrended Fluctuation Analysis (MFDFA) of parcellated fMRI
timeseries.

Implements the Kantelhardt et al. (2002) MFDFA estimator and its Legendre
transform to the singularity spectrum f(alpha) vs alpha (Holder exponent).

The whole 240-ROI panel is processed at once: for every scale the segments
of *all* ROIs are stacked into a single (n_segments, scale, n_roi) tensor and
detrended with one matrix multiply against a precomputed projection operator.
There is no Python loop over ROIs or over segments -- only a short loop over
the handful of scales and moment orders q. On a 1200 x 240 panel this turns a
minutes-long triple loop into a fraction of a second.

Algorithm (Kantelhardt et al., Physica A 316 (2002) 87-114):
  1. Profile    Y(i) = cumsum(x - <x>)
  2. For scale s split Y into Ns = floor(N/s) non-overlapping segments from
     both ends (2*Ns segments), least-squares detrend each with an order-m
     polynomial, take the segment variance F2(s, v).
  3. Fluctuation Fq(s) = ( mean_v F2(s,v)^{q/2} )^{1/q}   (q != 0)
                       = exp( 0.5 * mean_v ln F2(s,v) )    (q  = 0)
  4. Fq(s) ~ s^{h(q)}  ->  generalized Hurst exponent h(q) from a log-log fit.
  5. tau(q) = q*h(q) - 1,  alpha = d tau / d q,  f(alpha) = q*alpha - tau(q).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNG without a display server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed/100307")
INPUT_CSV = DATA_DIR / "parcellated_timeseries.csv"
OUTPUT_CSV = DATA_DIR / "multifractal_results.csv"
OUTPUT_PNG = DATA_DIR / "multifractal_spectrum.png"


def make_scales(n: int, s_min: int = 16, s_max: int | None = None,
                n_scales: int = 20) -> np.ndarray:
    """Integer, log-spaced segment scales in [s_min, s_max].

    Upper bound defaults to N/4 so that even the largest scale yields >= 4
    segments per direction -- below that the fluctuation estimate is too noisy
    to fit reliably.
    """
    if s_max is None:
        s_max = n // 4
    scales = np.unique(
        np.round(np.geomspace(s_min, s_max, n_scales)).astype(int)
    )
    return scales[scales >= s_min]


def _projection_residual_operator(s: int, order: int) -> np.ndarray:
    """(I - X (X^T X)^-1 X^T) for the order-`order` Vandermonde design on a
    length-`s` segment. Left-multiplying a segment by this returns the
    detrending residual; it is identical for every segment at this scale, so
    it is built once and reused across all segments and all ROIs.
    """
    t = np.arange(s, dtype=float)
    x = np.vander(t, order + 1, increasing=True)      # (s, order+1)
    hat = x @ np.linalg.pinv(x)                        # projection onto trend
    return np.eye(s) - hat                             # residual projector


def fluctuation_functions(
    profile: np.ndarray,
    scales: np.ndarray,
    qs: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Fq(s) for every scale, moment q and ROI.

    Parameters
    ----------
    profile : (T, R) cumulative, mean-removed profile for R ROIs.
    scales  : (S,)   segment lengths.
    qs      : (Q,)   moment orders (may include 0).
    order   : polynomial detrend order (1 = linear).

    Returns
    -------
    fq : (S, Q, R) fluctuation function.
    """
    n, n_roi = profile.shape
    fq = np.empty((scales.size, qs.size, n_roi), dtype=float)
    q_col = qs[:, None]  # (Q, 1) for broadcasting against ROI axis

    for si, s in enumerate(scales):
        n_seg = n // s
        resid_op = _projection_residual_operator(s, order)

        # Forward from the start and backward from the end -> 2*n_seg segments,
        # so the tail of the series is not discarded (Kantelhardt step 2).
        head = profile[: n_seg * s].reshape(n_seg, s, n_roi)
        tail = profile[n - n_seg * s:].reshape(n_seg, s, n_roi)
        segs = np.concatenate([head, tail], axis=0)      # (2*n_seg, s, R)

        # One matmul detrends every segment of every ROI at this scale.
        resid = np.matmul(resid_op, segs)                # (2*n_seg, s, R)
        f2 = np.mean(resid ** 2, axis=1)                 # (2*n_seg, R) variance
        f2 = np.maximum(f2, 1e-30)                       # guard log/neg powers

        # q-order fluctuation, averaged over segments. q=0 uses the log form.
        log_f2 = np.log(f2)                              # (2*n_seg, R)
        nonzero = qs != 0
        fq_s = np.empty((qs.size, n_roi), dtype=float)
        # q != 0: broadcast (Qnz,1,1) against (1, 2*n_seg, R)
        q_nz = qs[nonzero][:, None, None]
        powered = np.exp((q_nz / 2.0) * log_f2[None, :, :])
        fq_s[nonzero] = np.mean(powered, axis=1) ** (1.0 / q_col[nonzero])
        # q == 0
        if (~nonzero).any():
            fq_s[~nonzero] = np.exp(0.5 * np.mean(log_f2, axis=0))
        fq[si] = fq_s

    return fq


def singularity_spectrum(
    fq: np.ndarray,
    scales: np.ndarray,
    qs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Legendre transform of the multifractal scaling exponents.

    Returns dict of (Q, R) arrays: h (generalized Hurst), tau, alpha (Holder
    exponent) and f_alpha (singularity dimension).
    """
    log_s = np.log(scales)
    log_fq = np.log(fq)                                  # (S, Q, R)

    # h(q, roi) = slope of log Fq vs log s. polyfit fits every (q, roi) column
    # in one call by flattening the (Q, R) trailing axes.
    s_flat = log_fq.reshape(scales.size, -1)             # (S, Q*R)
    slopes = np.polyfit(log_s, s_flat, deg=1)[0]         # (Q*R,)
    h = slopes.reshape(qs.size, fq.shape[2])             # (Q, R)

    tau = qs[:, None] * h - 1.0                          # (Q, R)
    # alpha = d tau / d q  (numerical Legendre transform along the q axis)
    alpha = np.gradient(tau, qs, axis=0)
    f_alpha = qs[:, None] * alpha - tau

    return {"h": h, "tau": tau, "alpha": alpha, "f_alpha": f_alpha}


def run(
    input_csv: Path = INPUT_CSV,
    output_csv: Path = OUTPUT_CSV,
    output_png: Path = OUTPUT_PNG,
    q_min: float = -5.0,
    q_max: float = 5.0,
    q_step: float = 0.5,
    order: int = 1,
) -> pd.DataFrame:
    """End-to-end MFDFA: load, compute spectra for all ROIs, save CSV + plot."""
    data = pd.read_csv(input_csv)
    x = data.to_numpy(dtype=float)                       # (T, R)
    roi_ids = data.columns.to_numpy()
    n, n_roi = x.shape

    # Profile: cumulative sum of the mean-removed signal, per ROI.
    profile = np.cumsum(x - x.mean(axis=0, keepdims=True), axis=0)

    qs = np.round(np.arange(q_min, q_max + q_step / 2, q_step), 6)
    scales = make_scales(n)

    fq = fluctuation_functions(profile, scales, qs, order=order)
    spec = singularity_spectrum(fq, scales, qs)

    # Long-format table: one row per (ROI, q).
    n_q = qs.size
    df = pd.DataFrame(
        {
            "roi": np.repeat(roi_ids, n_q),
            "q": np.tile(qs, n_roi),
            "hurst": spec["h"].T.reshape(-1),
            "tau": spec["tau"].T.reshape(-1),
            "alpha": spec["alpha"].T.reshape(-1),
            "f_alpha": spec["f_alpha"].T.reshape(-1),
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    _plot_spectrum(spec["alpha"], spec["f_alpha"], output_png)

    width = spec["alpha"].max(axis=0) - spec["alpha"].min(axis=0)
    print(f"Processed {n_roi} ROIs x {n} timepoints.")
    print(f"Scales ({scales.size}): {scales.min()}..{scales.max()}; "
          f"q in [{q_min}, {q_max}] step {q_step}.")
    print(f"Spectrum width (alpha_max - alpha_min): "
          f"mean={width.mean():.3f}, min={width.min():.3f}, "
          f"max={width.max():.3f}.")
    print(f"Wrote {output_csv} and {output_png}.")
    return df


def _plot_spectrum(alpha: np.ndarray, f_alpha: np.ndarray,
                   output_png: Path) -> None:
    """Overlay every ROI's f(alpha) curve plus the panel mean."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(alpha, f_alpha, color="steelblue", alpha=0.15, linewidth=0.8)
    ax.plot(alpha.mean(axis=1), f_alpha.mean(axis=1), color="crimson",
            linewidth=2.5, label="mean over ROIs")
    ax.set_xlabel(r"Holder exponent $\alpha$")
    ax.set_ylabel(r"singularity dimension $f(\alpha)$")
    ax.set_title("MFDFA singularity spectrum -- 240 ROIs (subject 100307)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
