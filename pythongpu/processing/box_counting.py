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

import numpy as np
import torch
import torch.nn.functional as F


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
