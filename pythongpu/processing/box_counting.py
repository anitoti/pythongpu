"""
Box-counting / fractal-dimension utilities deduped from the four
Lorenz/Rössler VPS-clustering pipeline scripts. boxcount_2d_gpu, boxdiv2,
and extract_boundary were byte-identical across all four (confirmed via
AST diff before extraction). fractal_dimension differed only in that
rossler_vps_clustering.py applies an extra fit-range filter
(mask = (n>0) & (r>=2) & (r<=90)) that the Lorenz variants don't — that's
now an optional r_min/r_max kwarg (default None = unfiltered, matching
the Lorenz variants' original behavior exactly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, minimum_filter


def boxcount_2d_gpu(
    boundary : np.ndarray,
    device   : torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated 2-D box-counting via max-pooling.

    max_pool2d on binary {0,1} image = OR-pooling:
    a box is occupied iff any pixel inside equals 1.
    Vectorised equivalent of MATLAB boxcount.m 2x2 OR-reduction:
    [Page 37, full_.m_script.pdf:
     "c(i,j)=(c(i,j)||c(i+siz2,j)||c(i,j+siz2)||c(i+siz2,j+siz2))"
     "p = log(width)/log(2); % nbre of generations"
     "f2 = fit(log(BxR2'),log(BxN2'),'poly1')"]

    Returns
    -------
    r : box sizes  [1, 2, 4, …, 2^p]
    n : box counts matching r
    """
    H, W  = boundary.shape
    p_exp = int(np.ceil(np.log2(max(H, W))))
    sz    = 2 ** p_exp

    # Pad to square power-of-two; shape for max_pool2d: (1, 1, sz, sz)
    c = torch.zeros((1, 1, sz, sz), dtype=torch.float32, device=device)
    c[0, 0, :H, :W] = torch.tensor(boundary.astype(np.float32))

    r_list, n_list = [], []
    for exp in range(p_exp + 1):
        r_val = 2 ** exp
        if r_val == 1:
            # r=1: every pixel is its own box
            # [Page 37, full_.m_script.pdf: "n(p+1) = sum(c(:));"]
            n_list.append(int(c.sum().item()))
        else:
            # max over {0,1} image = OR = "any pixel occupied in this box"
            pooled = F.max_pool2d(c, kernel_size=r_val, stride=r_val)
            n_list.append(int(pooled.sum().item()))
        r_list.append(r_val)

    return np.array(r_list, dtype=np.int64), np.array(n_list, dtype=np.int64)


def fractal_dimension(
    r     : np.ndarray,
    n     : np.ndarray,
    r_min : int | None = None,
    r_max : int | None = None,
) -> tuple[float, float]:
    """
    Estimate D_f via log-log linear fit (poly1).
    [Page 37, full_.m_script.pdf: "f2 = fit(log(BxR2'),log(BxN2'),'poly1')"]

    N(r) ~ r^{-D_f}  =>  D_f = -slope of log(N) vs log(r)

    r_min/r_max optionally restrict the fit to a box-size range
    (rossler_vps_clustering.py used r_min=2, r_max=90; the Lorenz
    variants used the full unfiltered range — leave both None for that).
    """
    mask = n > 0
    if r_min is not None:
        mask &= r >= r_min
    if r_max is not None:
        mask &= r <= r_max
    log_r     = np.log(r[mask].astype(float))
    log_n     = np.log(n[mask].astype(float))
    coeffs    = np.polyfit(log_r, log_n, deg=1)
    residuals = log_n - np.polyval(coeffs, log_r)
    ss_res    = np.sum(residuals ** 2)
    ss_tot    = np.sum((log_n - log_n.mean()) ** 2)
    r_sq      = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(-coeffs[0]), float(r_sq)


@dataclass
class UncertaintyExponent:
    """
    Grebogi–McDonald–Ott–Yorke uncertainty-exponent estimate on a labelled
    initial-condition slice.

    gamma      : the scaling exponent γ of the ε-uncertain fraction f(ε) ∼ ε^γ.
    r_squared  : coefficient of determination of the log f vs log ε fit.
    D_f        : the boundary dimension recovered from the codimension identity
                 D_f = d − γ  (d = embedding dimension of the slice). This is a
                 grid-independent cross-check of the box-counting D_f: γ is a
                 scaling slope and so does not inherit the discretisation bias
                 of a raw pixel count.
    d          : embedding dimension of the slice (2 for a 2-D IC slice).
    radii      : the perturbation radii ε (pixels) that carried scaling info.
    f          : the ε-uncertain fraction at each retained radius.
    """
    gamma: float | None
    r_squared: float | None
    D_f: float | None
    d: int
    radii: np.ndarray
    f: np.ndarray


def uncertainty_exponent(
    labels : np.ndarray,
    radii  : tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16),
    d      : int = 2,
) -> UncertaintyExponent:
    """
    Estimate the uncertainty exponent γ of a basin partition and, from it, the
    boundary dimension via  D_f = d − γ.

    An initial condition is ε-uncertain if an L∞ perturbation of up to ε pixels
    can move it into a different basin — i.e. if the (2ε+1)² neighbourhood spans
    more than one integer label. Since the labels are integers, that is exactly
    ``maximum_filter(labels, 2ε+1) ≠ minimum_filter(labels, 2ε+1)``. The fraction
    of such points obeys the Grebogi–McDonald–Ott–Yorke scaling law
    f(ε) ∼ ε^γ, so γ is the slope of log f against log ε over the radii where
    0 < f < 1 (a saturated f = 1 or empty f = 0 carries no scaling information).

    Because γ is a *scaling exponent* rather than a box tally, it is independent
    of the grid resolution used to sample the slice — refining the grid rescales
    ε uniformly and leaves the slope unchanged. The dual dimension D_f = d − γ is
    therefore a grid-independent counterpart to :func:`fractal_dimension`.

    Returns an :class:`UncertaintyExponent`; ``gamma`` (and ``D_f``) are None when
    fewer than two radii fall in the informative 0 < f < 1 band, e.g. a slice
    that collapses to a single basin.
    """
    lab = np.asarray(labels).astype(np.int32)
    eps_used, f_used = [], []
    for eps in radii:
        size = 2 * int(eps) + 1
        hi = maximum_filter(lab, size=size, mode="nearest")
        lo = minimum_filter(lab, size=size, mode="nearest")
        frac = float((hi != lo).mean())
        if 0.0 < frac < 1.0:
            eps_used.append(float(eps))
            f_used.append(frac)
    eps_arr = np.array(eps_used)
    f_arr = np.array(f_used)
    if len(eps_used) < 2:
        return UncertaintyExponent(None, None, None, d, eps_arr, f_arr)

    log_e = np.log(eps_arr)
    log_f = np.log(f_arr)
    slope, intercept = np.polyfit(log_e, log_f, deg=1)
    resid = log_f - (slope * log_e + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((log_f - log_f.mean()) ** 2))
    r_sq = 1.0 - ss_res / (ss_tot + 1e-12)
    gamma = float(slope)
    return UncertaintyExponent(gamma, float(r_sq), float(d - gamma), d,
                               eps_arr, f_arr)


def boxdiv2(c: np.ndarray, p: float) -> np.ndarray:
    """
    Recursive 2-D fractal subdivision.
    [Page 37, full_.m_script.pdf:
     "siz2 = round(siz/2);
      c(1:siz2,1:siz2) = c(1:siz2,1:siz2) & (rand<p);
      if c(1,1)  c(1:siz2,1:siz2) = boxdiv2(c(1:siz2,1:siz2),p);  end"]
    """
    siz = c.shape[0]
    if siz == 1:
        c[0, 0] = True
        return c
    siz2 = round(siz / 2)

    c[:siz2, :siz2] = c[:siz2, :siz2] & (np.random.rand() < p)
    if c[0, 0]:
        c[:siz2, :siz2] = boxdiv2(c[:siz2, :siz2], p)

    c[siz2:, :siz2] = c[siz2:, :siz2] & (np.random.rand() < p)
    if c[siz2, 0]:
        c[siz2:, :siz2] = boxdiv2(c[siz2:, :siz2], p)

    c[:siz2, siz2:] = c[:siz2, siz2:] & (np.random.rand() < p)
    if c[0, siz2]:
        c[:siz2, siz2:] = boxdiv2(c[:siz2, siz2:], p)

    c[siz2:, siz2:] = c[siz2:, siz2:] & (np.random.rand() < p)
    if c[siz2, siz2]:
        c[siz2:, siz2:] = boxdiv2(c[siz2:, siz2:], p)

    return c


def extract_boundary(labels: np.ndarray) -> np.ndarray:
    """
    4-connected boundary pixels — True where any neighbour differs.
    [Page 37, full_.m_script.pdf:
     "F2 = getframe(gcf); [X2,Map2] = frame2im(F2); [BxN2,BxR2] = boxcount(X2)"]
    """
    b = np.zeros_like(labels, dtype=bool)
    b[:-1, :] |= (labels[:-1, :] != labels[1:,  :])
    b[1:,  :] |= (labels[:-1, :] != labels[1:,  :])
    b[:, :-1] |= (labels[:, :-1] != labels[:, 1:])
    b[:, 1:]  |= (labels[:, :-1] != labels[:, 1:])
    return b
