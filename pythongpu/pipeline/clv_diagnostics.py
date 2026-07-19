"""CLV diagnostics implementing the Ginelli algorithm.

The core object is :class:`CLVCalculator`, which runs the standard two-pass
Ginelli construction for covariant Lyapunov vectors (CLVs):

* Forward pass: integrate the nonlinear state and tangent dynamics with a
  Runge-Kutta 4 (RK4) stepper. Every ``qr_interval`` steps, orthonormalize the
  tangent basis with QR and store the factors. Writing ``T_k = Q_k R_k`` keeps
  the expanding directions numerically stable while preserving the triangular
  coupling needed by the backward pass.
* Backward pass: recover the CLV coordinates by solving the triangular system
  ``R_k C_{k-1} = C_k`` from the final QR slice back to the first. The actual
  CLVs are then ``V_k = Q_k C_k``. This is the Ginelli recursion: QR captures
  the Oseledets subspaces forward in time, and triangular back-substitution
  reconstructs the covariant directions that evolve with the dynamics.

The module also exposes diagnostics for Lyapunov spectra, Kaplan-Yorke
dimension, and transversality-angle time series used to detect riddling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch


def _ensure_dir(path: str) -> None:
    """Create ``path`` if needed."""
    os.makedirs(path, exist_ok=True)


@dataclass
class CLVCalculator:
    rhs_fn: Callable[[torch.Tensor], torch.Tensor]
    jac_fn: Callable[[torch.Tensor], torch.Tensor]
    n: int
    dt: float = 0.01
    device: Optional[torch.device] = None
    qr_interval: int = 10
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Use float32 for stable computations; R buffers will be stored as float16 to save VRAM
        self.compute_dtype = self.dtype
        self.storage_dtype = torch.float16

    def _rk4_step(self, state: torch.Tensor, tangents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance the nonlinear state and tangent equations by one RK4 step.

        Args:
            state: Current state vector with shape ``(n,)``.
            tangents: Tangent basis with shape ``(n, m)``.

        Returns:
            A tuple ``(new_state, new_tangents)`` after one RK4 step.
        """
        dt = self.dt

        # k1
        k1_state = self.rhs_fn(state)
        J1 = self.jac_fn(state)  # (n, n)
        k1_v = J1.matmul(tangents)

        # stage 2
        s2 = state + 0.5 * dt * k1_state
        v2 = tangents + 0.5 * dt * k1_v
        k2_state = self.rhs_fn(s2)
        J2 = self.jac_fn(s2)
        k2_v = J2.matmul(v2)

        # stage 3
        s3 = state + 0.5 * dt * k2_state
        v3 = tangents + 0.5 * dt * k2_v
        k3_state = self.rhs_fn(s3)
        J3 = self.jac_fn(s3)
        k3_v = J3.matmul(v3)

        # stage 4
        s4 = state + dt * k3_state
        v4 = tangents + dt * k3_v
        k4_state = self.rhs_fn(s4)
        J4 = self.jac_fn(s4)
        k4_v = J4.matmul(v4)

        new_state = state + (dt / 6.0) * (k1_state + 2.0 * k2_state + 2.0 * k3_state + k4_state)
        new_tangents = tangents + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        return new_state, new_tangents

    def run_forward(
        self,
        initial_state: torch.Tensor,
        total_steps: int,
        m: int,
        init_tangents: Optional[torch.Tensor] = None,
    ) -> Tuple[List[np.ndarray], List[torch.Tensor], List[int]]:
        """Run the Ginelli forward pass and store QR factors.

        Args:
            initial_state: Initial condition for the nonlinear system.
            total_steps: Number of RK4 steps to integrate.
            m: Number of tangent directions to propagate.
            init_tangents: Optional custom initial tangent basis.

        Returns:
            ``(Q_list, R_half_list, qr_steps)`` where ``Q_list`` contains CPU
            ``float32`` NumPy arrays, ``R_half_list`` contains device-resident
            half-precision triangular factors, and ``qr_steps`` stores the step
            indices at which QR was performed.
        """
        device = self.device
        n = self.n

        state = initial_state.to(device=device, dtype=self.compute_dtype).clone()

        if init_tangents is None:
            # initialize orthonormal basis for m tangent vectors
            if m > n:
                raise ValueError('m must be <= n')
            T0 = torch.zeros((n, m), device=device, dtype=self.compute_dtype)
            for i in range(m):
                T0[i, i] = 1.0
        else:
            T0 = init_tangents.to(device=device, dtype=self.compute_dtype).clone()
            if T0.shape != (n, m):
                raise ValueError('init_tangents must have shape (n,m)')

        tangents = T0

        Q_list: List[np.ndarray] = []
        R_half_list: List[torch.Tensor] = []
        qr_steps: List[int] = []

        step = 0
        while step < total_steps:
            # take one RK4 step for state and tangents
            state, tangents = self._rk4_step(state, tangents)
            step += 1

            if step % self.qr_interval == 0:
                # Orthonormalize tangent vectors via QR (compute in float32 for stability)
                # Ensure tangents are on device and in compute dtype
                T = tangents.to(device=device, dtype=self.compute_dtype)
                # QR decomposition: T = Q R
                # Use mode='reduced' to get (n,m) and (m,m)
                # torch.linalg.qr returns Q,R
                Q, R = torch.linalg.qr(T, mode='reduced')

                # Convert Q to CPU numpy (float32) for long-term storage
                Q_cpu = Q.detach().to('cpu').numpy().astype(np.float32, copy=False)
                Q_list.append(Q_cpu)

                # Store R on GPU in float16 to reduce VRAM usage
                R_half = R.detach().to(device=device).to(self.storage_dtype)
                R_half_list.append(R_half)

                qr_steps.append(step)

                # Replace tangents with orthonormal Q to continue
                tangents = Q

        return Q_list, R_half_list, qr_steps

    def _solve_triangular(self, R: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Solve an upper-triangular linear system for the Ginelli recursion.

        Args:
            R: Upper-triangular factor from QR, shape ``(m, m)``.
            B: Right-hand side with shape ``(m, m)`` or ``(m, k)``.

        Returns:
            The solution ``X`` to ``R X = B``.
        """
        try:
            # PyTorch >=1.9
            X = torch.linalg.solve_triangular(R, B, upper=True)
        except AttributeError:
            # older API: triangular_solve(b, a)
            X = torch.triangular_solve(B, R, upper=True)[0]
        return X

    def run_backward_reconstruct(
        self,
        Q_list: List[np.ndarray],
        R_half_list: List[torch.Tensor],
        qr_steps: List[int],
        leading_m: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Run the Ginelli backward pass and reconstruct CLVs.

        The backward recursion starts from ``C = I`` at the final QR slice and
        repeatedly solves ``R_k C_{k-1} = C_k``. Normalizing the columns after
        each solve keeps the coordinates well-scaled while preserving the
        covariant directions. The resulting CLVs are recovered by multiplying
        the stored QR basis with the reconstructed coordinate matrices:
        ``V_k = Q_k C_k``.

        Args:
            Q_list: QR ``Q`` factors stored during the forward pass.
            R_half_list: QR ``R`` factors stored during the forward pass.
            qr_steps: Step indices corresponding to each QR slice.
            leading_m: If provided, only reconstruct the leading ``leading_m``
                CLVs.

        Returns:
            A list of CLV arrays with shape ``(n, leading_m)``.
        """
        device = self.device
        m = R_half_list[0].shape[0]
        if leading_m is None:
            leading_m = m
        if not (1 <= leading_m <= m):
            raise ValueError('leading_m must be in 1..m')

        num = len(R_half_list)
        # C matrices will be kept in float32 on device for numeric stability
        C = torch.eye(m, dtype=self.compute_dtype, device=device)

        # We'll reconstruct C at each QR index going backward; store matrix copies for each Q
        C_list: List[torch.Tensor] = [None] * num  # type: ignore

        # Backward recursion: C_{n-1} = solve(R_n, C_n), then normalize columns
        for i in range(num - 1, -1, -1):
            R_half = R_half_list[i]
            # cast R to compute dtype (float32) for stable solve
            R = R_half.to(dtype=self.compute_dtype, device=device)

            # Solve R X = C for X
            C_prev = self._solve_triangular(R, C)

            # Column-wise normalization (Euclidean norm)
            # compute norms along rows? C_prev is (m,m) where columns correspond to CLV coordinates in Q-basis
            col_norms = torch.linalg.norm(C_prev, dim=0)
            # Avoid div by zero
            col_norms = torch.where(col_norms == 0.0, torch.ones_like(col_norms), col_norms)
            C_prev = C_prev / col_norms

            # Save C_prev for corresponding Q (this C_prev corresponds to time just before QR step)
            C_list[i] = C_prev.clone()

            # Set C for next iteration
            C = C_prev

        # Now compute CLVs at each stored time: CLV_n = Q_n @ C_n
        clv_list: List[np.ndarray] = []
        for i, Q_cpu in enumerate(Q_list):
            Q_t = torch.from_numpy(Q_cpu).to(device=device, dtype=self.compute_dtype)
            Cn = C_list[i][:, :leading_m]  # (m, leading_m)
            clvs = Q_t.matmul(Cn)  # (n, leading_m)
            # move to CPU numpy float32 for downstream analysis/storage
            clv_list.append(clvs.detach().to('cpu').numpy().astype(np.float32, copy=False))

        return clv_list

    def compute_transversality_angles(
        self,
        clv_list: List[np.ndarray],
        K: int = 10,
        out_prefix: str = 'output/clv_angles_K',
    ) -> np.ndarray:
        """Compute minimum pairwise transversality angles for the leading CLVs.

        For each stored time, the first ``K`` CLVs are normalized and their
        absolute pairwise dot products are converted to angles via ``arccos``.
        The minimum angle acts as a simple transversality diagnostic: small
        angles indicate near-tangencies among the leading covariant directions.

        Args:
            clv_list: CLV arrays returned by :meth:`run_backward_reconstruct`.
            K: Number of leading CLVs to include in the angle diagnostic.
            out_prefix: Output prefix for the ``.npy`` file.

        Returns:
            A one-dimensional NumPy array of minimum angles in radians.
        """
        _ensure_dir(os.path.dirname(out_prefix) or '.')
        m_available = clv_list[0].shape[1]
        K_use = min(K, m_available)
        times = len(clv_list)
        min_angles = np.zeros(times, dtype=np.float32)

        for t_idx, clvs in enumerate(clv_list):
            # clvs shape (n, m)
            # take first K_use columns
            A = clvs[:, :K_use]
            # normalize each vector
            norms = np.linalg.norm(A, axis=0)
            norms = np.where(norms == 0, 1.0, norms)
            A_norm = A / norms
            # compute pairwise dot products (absolute)
            # Gram matrix
            G = np.abs(A_norm.T @ A_norm)
            # clamp numerical noise
            G = np.clip(G, -1.0, 1.0)
            # ignore diagonal by setting to 0
            np.fill_diagonal(G, 0.0)
            # convert dot to angles
            # ensure values within [-1,1]
            angles = np.arccos(np.clip(G, -1.0, 1.0))
            # take minimum positive angle across pairs
            if angles.size == 0:
                min_angles[t_idx] = 0.0
            else:
                min_val = float(np.min(angles))
                min_angles[t_idx] = min_val

        out_fname = f"{out_prefix}{K_use}.npy"
        np.save(out_fname, min_angles)
        return min_angles


# =============================================================================
# TOPOLOGICAL DIAGNOSTICS OF THE CHAOTIC DYNAMICS
# =============================================================================
# Two additions requested for the HCP/VPS pipeline, both derived from data the
# Ginelli forward/backward passes above already produce:
#
#   1. lyapunov_spectrum() + kaplan_yorke_dimension()
#      "Dimension counting." The diagonal of each stored R matrix records how
#      much each orthonormal direction stretched (log|R_ii| > 0) or shrank
#      (< 0) over one QR interval. Averaging log|R_ii| over the whole run and
#      dividing by the elapsed time gives the Lyapunov exponents λ_1 ≥ … ≥ λ_m.
#      The Kaplan–Yorke (Lyapunov) dimension interpolates where the cumulative
#      sum of exponents crosses zero:
#          D_KY = k + (Σ_{i≤k} λ_i) / |λ_{k+1}|
#      with k the largest index keeping the partial sum non-negative. This is
#      the fractal dimension of the attractor — the "how many numbers you need
#      to pin down the wiggle" count.
#
#   2. detect_riddling_kmeans()
#      Riddling signature (Ott/Alexander; "Intermingled basins in coupled
#      Lorenz systems", arXiv:1111.5581): the transverse Lyapunov exponent is
#      negative *on average* yet has positive finite-time fluctuations, so the
#      minimum-transversality-angle time series is bimodal — long stretches
#      near the synchronisation manifold (small angle) punctuated by transverse
#      bursts (large angle). A 2-means split of the angle series that yields two
#      well-separated, both-populated clusters is the operational fingerprint of
#      (locally) riddled / intermingled basin structure.
# =============================================================================


def lyapunov_spectrum(
    R_half_list: List[torch.Tensor],
    qr_interval: int,
    dt: float,
    discard_frac: float = 0.1,
) -> np.ndarray:
    """Lyapunov exponents from the stored R diagonals of the forward pass.

    λ_i = < log|R_ii| > / (qr_interval * dt), averaged over QR steps after an
    initial transient (``discard_frac`` of the steps) so the tangent basis has
    aligned with the true Oseledets directions.

    Returns an array of m exponents in descending order (they already come out
    ordered because QR keeps the leading directions leading).
    """
    if not R_half_list:
        raise ValueError("R_half_list is empty — run run_forward first.")
    m = R_half_list[0].shape[0]
    n_qr = len(R_half_list)
    start = int(discard_frac * n_qr)
    tau = qr_interval * dt

    log_growth = np.zeros(m, dtype=np.float64)
    count = 0
    for R_half in R_half_list[start:]:
        R = R_half.to(dtype=torch.float32).detach().cpu().numpy()
        diag = np.abs(np.diag(R))
        # guard against zero/denormal diagonal entries from float16 storage
        diag = np.where(diag < 1e-12, 1e-12, diag)
        log_growth += np.log(diag)
        count += 1
    if count == 0:
        raise ValueError("No QR steps left after discarding transient; lower discard_frac.")
    return log_growth / (count * tau)


def kaplan_yorke_dimension(exponents: np.ndarray) -> float:
    """Kaplan–Yorke (Lyapunov) dimension from an ordered exponent spectrum.

    D_KY = k + (Σ_{i=1..k} λ_i) / |λ_{k+1}|, where k is the largest index whose
    cumulative exponent sum is still ≥ 0. Returns 0.0 if even λ_1 < 0 (a fixed
    point / fully contracting), and m (all exponents, no interpolation) if the
    whole spectrum sums non-negative.
    """
    lam = np.sort(np.asarray(exponents, dtype=np.float64))[::-1]
    csum = np.cumsum(lam)
    m = len(lam)
    if lam[0] <= 0:
        return 0.0
    # largest k with cumulative sum >= 0
    k = int(np.searchsorted(-csum, 0.0, side="right"))  # first index where csum < 0
    if k >= m:
        return float(m)
    if lam[k] == 0:
        return float(k)
    return float(k) + csum[k - 1] / abs(lam[k])


def detect_riddling_kmeans(
    min_angles: np.ndarray,
    n_clusters: int = 2,
    seed: int = 0,
    separation_thresh: float = 0.15,
    min_pop_frac: float = 0.02,
) -> dict:
    """k-means on the minimum-transversality-angle series to flag riddling.

    A riddled/intermingled regime shows a *bimodal* angle distribution: a dense
    low-angle cluster (locked to the synchronisation manifold) and a sparse but
    populated high-angle cluster (transverse bursts). We 2-means the angles and
    call it RIDDLED when both clusters are populated (each ≥ ``min_pop_frac`` of
    samples) and their centroids differ by ≥ ``separation_thresh`` radians.

    Returns a dict with the verdict, centroids, populations, and the
    burst_fraction (fraction of time spent in the high-angle cluster) — a proxy
    for how "leaky" the basin is.
    """
    from sklearn.cluster import KMeans

    x = np.asarray(min_angles, dtype=np.float64).reshape(-1, 1)
    n = x.shape[0]
    if n < n_clusters:
        return {"verdict": "INSUFFICIENT_DATA", "n_samples": n}

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(x)
    centers = km.cluster_centers_.ravel()
    labels = km.labels_
    order = np.argsort(centers)  # ascending centroid
    centers_sorted = centers[order]
    pops = np.array([(labels == c).sum() for c in range(n_clusters)], dtype=np.float64)
    pops_sorted = pops[order] / n

    low_c, high_c = centers_sorted[0], centers_sorted[-1]
    separation = high_c - low_c
    both_populated = bool(pops_sorted.min() >= min_pop_frac)
    well_separated = bool(separation >= separation_thresh)

    high_cluster = order[-1]
    burst_fraction = float((labels == high_cluster).sum() / n)

    riddled = both_populated and well_separated
    return {
        "verdict": "RIDDLED" if riddled else "SYNCHRONISED",
        "centroids_rad": centers_sorted.tolist(),
        "cluster_pop_frac": pops_sorted.tolist(),
        "centroid_separation_rad": separation,
        "burst_fraction": burst_fraction,
        "n_samples": int(n),
    }


if __name__ == '__main__':
    # Lightweight smoke demonstration using a synthetic Lorenz-like 83D system if executed directly.
    # This is not a full test; users should integrate with the project's Lorenz DTI model provided elsewhere.
    print('CLVCalculator module loaded. Import and use CLVCalculator with project rhs/jac functions.')
