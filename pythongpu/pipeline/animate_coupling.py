# ============================================================
#  Coupling-Sweep Basin Animation — GPU RK4 → MP4
#  Project : Nimble Brain (REU @ Clarkson)
#  Purpose : Sweep the global Kuramoto coupling K over [0.00, 0.90]
#            on a 256x256 slice of initial conditions, classify the
#            order-parameter field into synchrony-level basins, and
#            compile the per-K basin frames into a smooth MP4.
#
#  Notes
#  -----
#  * High-performance: the 256x256 = 65,536 initial conditions are
#    integrated on the GPU in memory-bounded chunks so peak VRAM stays
#    flat regardless of grid resolution.
#  * Fully-connected mean-field topology puts the synchronization
#    transition at Kc ~ 1.596 * omega_scale ~ 0.56, inside the sweep.
#    As K climbs the basins merge; at the top of the range the grid is
#    fully synced (a single basin), so extract_boundary() returns an
#    all-False map whose flat index vector is empty.
#  * Hotfix: any task whose boundary vector is empty is *dropped*
#    safely (no box-counting, no boundary overlay) instead of crashing
#    the downstream fit on a zero-length array. This is what guards
#    task 10 (K=0.90), whose fully-synced basin has no boundary.
#
#  Output : data/derivatives/basin_coupling_sweep.mp4
# ============================================================

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ── Make the flat `pythongpu` package importable when this file is run
#    by path from pythongpu/pipeline/ (repo root is parents[2]). The
#    package is not pip-installed, so a path invocation needs the repo
#    root on sys.path for `import pythongpu` to resolve. ──
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless: render straight to a pixel buffer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import cv2  # MP4 muxing without a system ffmpeg (bundled mp4v encoder)

from pythongpu.pipeline.kuramoto_sweep import (
    kuramoto_rhs,
    rk4_step,
    order_parameter,
    build_ic_grid,
    Config as SweepConfig,
)
from pythongpu.processing.box_counting import (
    extract_boundary,
    boxcount_2d_gpu,
    fractal_dimension,
)


# ── 1. CONFIG ────────────────────────────────────────────────
@dataclass
class AnimConfig:
    # Dynamics / topology
    n_nodes:      int   = 100
    omega_scale:  float = 0.35   # Kc ~ 1.596 * omega_scale ~ 0.56 (mean-field)
    graph_seed:   int   = 7
    omega_seed:   int   = 11
    baseline_seed: int  = 123    # spread phases for the non-swept nodes
    tmax:         float = 25.0
    dt:           float = 0.02

    # IC-grid slice
    grid_n:       int   = 256    # 256 x 256 initial conditions

    # Coupling sweep: K = 0.00, 0.09, ..., 0.90  (11 tasks, index 0..10)
    k_start:      float = 0.00
    k_stop:       float = 0.90
    k_step:       float = 0.09

    # Basin classification: synchrony-level bins on R in [0, 1]
    r_bins: tuple[float, ...] = (0.30, 0.60, 0.90)

    # Video
    interp_frames: int  = 10     # tween frames inserted between keyframes
    fps:           int  = 12
    dpi:           int  = 120

    # GPU
    chunk:        int   = 16384  # ICs integrated per GPU chunk
    device:       str   = "cuda" if torch.cuda.is_available() else "cpu"

    out_path:     Path  = _REPO_ROOT / "data" / "derivatives" / "basin_coupling_sweep.mp4"


# ── 2. GRAPH ─────────────────────────────────────────────────
def build_complete_adjacency(cfg: AnimConfig, device: torch.device) -> torch.Tensor:
    """Fully-connected, unweighted, no self-loops (mean-field Kuramoto)."""
    A = torch.ones((cfg.n_nodes, cfg.n_nodes), dtype=torch.float32, device=device)
    A.fill_diagonal_(0.0)
    return A


# ── 3. GPU INTEGRATION (chunked) ─────────────────────────────
def integrate_order_parameter(
    theta0: torch.Tensor,
    omega:  torch.Tensor,
    A:      torch.Tensor,
    K:      float,
    cfg:    AnimConfig,
) -> np.ndarray:
    """
    Integrate every initial condition to t=tmax with RK4 and return the
    final Kuramoto order parameter R for each, as a (grid_n, grid_n) map.

    ICs are processed in chunks of cfg.chunk rows so peak VRAM is bounded
    by (chunk, N, N) rather than (grid_n^2, N, N).
    """
    steps = int(round(cfg.tmax / cfg.dt))
    B = theta0.shape[0]
    two_pi = 2.0 * np.pi
    R_out = torch.empty(B, dtype=torch.float32, device=theta0.device)

    with torch.no_grad():
        for lo in range(0, B, cfg.chunk):
            hi = min(lo + cfg.chunk, B)
            theta = theta0[lo:hi].clone()
            for _ in range(steps):
                theta = rk4_step(theta, cfg.dt, omega, A, K)
                # wrap to [-pi, pi] to keep long integrations numerically sane
                theta = torch.remainder(theta + np.pi, two_pi) - np.pi
            R_out[lo:hi] = order_parameter(theta)

    R = R_out.cpu().numpy().reshape(cfg.grid_n, cfg.grid_n)
    return R


# ── 4. CLASSIFY → BASINS → BOUNDARY (with hotfix) ────────────
def classify_and_boundary(R: np.ndarray, cfg: AnimConfig):
    """
    Bin the continuous order-parameter field into synchrony-level basins
    and extract their 4-connected boundary.

    Returns (labels, boundary, boundary_idx). `boundary_idx` is the flat
    index vector of boundary pixels; an *empty* vector means the field
    collapsed to a single basin (fully synced / fully incoherent) — the
    caller drops that task rather than box-counting an empty boundary.
    """
    labels = np.digitize(R, bins=list(cfg.r_bins)).astype(np.int32)
    boundary = extract_boundary(labels)
    boundary_idx = np.flatnonzero(boundary)
    return labels, boundary, boundary_idx


def fractal_dim_or_none(boundary: np.ndarray, device: torch.device):
    """Box-count fractal dimension of a non-empty boundary, else None."""
    r, n = boxcount_2d_gpu(boundary, device)
    if not np.any(n > 0):
        return None
    D_f, _ = fractal_dimension(r, n)
    return D_f


# ── 5. RENDER ONE FRAME ──────────────────────────────────────
def render_frame(fig, canvas, R, boundary, K, D_f, cfg: AnimConfig) -> np.ndarray:
    """Draw the R heat-map (+ boundary overlay if present) → BGR uint8."""
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(
        R, origin="lower", cmap="turbo", vmin=0.0, vmax=1.0,
        extent=[-np.pi, np.pi, -np.pi, np.pi], interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Order parameter  R")

    if boundary is not None and boundary.any():
        # overlay boundary pixels as a translucent black mask
        ov = np.zeros((*boundary.shape, 4), dtype=np.float32)
        ov[boundary, 3] = 0.85
        ax.imshow(
            ov, origin="lower",
            extent=[-np.pi, np.pi, -np.pi, np.pi], interpolation="nearest",
        )
        sub = f"basin boundary  |  D_f = {D_f:.3f}" if D_f is not None else "basin boundary"
    else:
        sub = "single basin — boundary dropped"

    ax.set_xlabel(r"$\theta_0$  (node 0 phase)")
    ax.set_ylabel(r"$\theta_1$  (node 1 phase)")
    ax.set_title(f"Kuramoto basin sweep    K = {K:0.3f}\n{sub}")

    fig.tight_layout()
    canvas.draw()

    rgba = np.asarray(canvas.buffer_rgba())
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    return np.ascontiguousarray(bgr)


# ── 6. MAIN PIPELINE ─────────────────────────────────────────
def main() -> int:
    cfg = AnimConfig()
    device = torch.device(cfg.device)
    print(f"[device]  {device}")

    # Coupling tasks: 0.00 .. 0.90 inclusive.
    n_tasks = int(round((cfg.k_stop - cfg.k_start) / cfg.k_step)) + 1
    K_values = np.round(cfg.k_start + cfg.k_step * np.arange(n_tasks), 2)
    print(f"[sweep]   {n_tasks} tasks  K = {K_values.tolist()}")

    # Topology, natural frequencies, IC grid (reused across every task).
    A = build_complete_adjacency(cfg, device)
    gen = torch.Generator(device="cpu").manual_seed(cfg.omega_seed)
    omega = (torch.randn(cfg.n_nodes, generator=gen) * cfg.omega_scale).to(device)

    grid_cfg = SweepConfig(
        n_nodes=cfg.n_nodes, sweep_points=cfg.grid_n, device=cfg.device
    )
    theta0, _, _ = build_ic_grid(grid_cfg)  # (grid_n^2, N) on device
    theta0 = theta0.clone()
    # The IC slice varies only nodes 0 & 1. Leaving the other N-2 nodes at
    # theta=0 pins the global order parameter near 1 (98 aligned phases),
    # washing out the coupling dependence. Seed them with a fixed spread
    # baseline instead so R genuinely tracks the sync transition across K.
    bgen = torch.Generator(device="cpu").manual_seed(cfg.baseline_seed)
    baseline = ((torch.rand(cfg.n_nodes, generator=bgen) * 2.0 - 1.0) * np.pi).to(device)
    theta0[:, 2:] = baseline[2:].unsqueeze(0)
    print(f"[grid]    {cfg.grid_n}x{cfg.grid_n} = {theta0.shape[0]} ICs, "
          f"N={cfg.n_nodes}, steps={int(round(cfg.tmax / cfg.dt))}")

    # ── Task loop: integrate → R field → classify → (hotfix) boundary ──
    R_fields: list[np.ndarray] = []
    dropped: list[int] = []
    for task, K in enumerate(K_values):
        t0 = time.time()
        R = integrate_order_parameter(theta0, omega, A, float(K), cfg)
        _, _, boundary_idx = classify_and_boundary(R, cfg)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        if boundary_idx.size == 0:
            # HOTFIX: single-basin field → empty boundary vector. Drop it
            # so the downstream fit never sees a zero-length array.
            dropped.append(task)
            print(f"[task {task:>2}]  K={K:0.2f}  R=[{R.min():.3f},{R.max():.3f}]  "
                  f"{dt:5.1f}s  -> empty boundary vector, DROPPED (hotfix)")
        else:
            print(f"[task {task:>2}]  K={K:0.2f}  R=[{R.min():.3f},{R.max():.3f}]  "
                  f"{dt:5.1f}s  boundary px={boundary_idx.size}")
        R_fields.append(R)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Build the smooth frame sequence (keyframes + tweened fields) ──
    # Tweening the R field between successive K keyframes yields a smooth
    # coupling morph; boundary + fractal dim are recomputed per frame so
    # the hotfix applies to every rendered frame, not just the keyframes.
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.5, 6.5), dpi=cfg.dpi)
    canvas = FigureCanvasAgg(fig)

    writer = None
    n_written = 0
    for i in range(len(R_fields)):
        segment = [(R_fields[i], float(K_values[i]))]
        if i + 1 < len(R_fields):
            for j in range(1, cfg.interp_frames + 1):
                a = j / (cfg.interp_frames + 1)
                R_mix = (1.0 - a) * R_fields[i] + a * R_fields[i + 1]
                K_mix = (1.0 - a) * K_values[i] + a * K_values[i + 1]
                segment.append((R_mix, float(K_mix)))

        for R_frame, K_frame in segment:
            _, boundary, boundary_idx = classify_and_boundary(R_frame, cfg)
            if boundary_idx.size == 0:
                boundary, D_f = None, None          # hotfix: no overlay/fit
            else:
                D_f = fractal_dim_or_none(boundary, device)
            frame = render_frame(fig, canvas, R_frame, boundary, K_frame, D_f, cfg)

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(cfg.out_path), fourcc, cfg.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"cv2 could not open writer for {cfg.out_path}")
                print(f"[video]   {w}x{h} @ {cfg.fps} fps -> {cfg.out_path}")
            writer.write(frame)
            n_written += 1

    if writer is not None:
        writer.release()
    plt.close(fig)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ok = cfg.out_path.exists() and cfg.out_path.stat().st_size > 0
    print(f"[done]    wrote {n_written} frames, "
          f"dropped tasks {dropped or '[]'} (empty boundary)")
    print(f"[out]     {cfg.out_path}  "
          f"({cfg.out_path.stat().st_size / 1e6:.2f} MB)" if ok else "[out]     FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
