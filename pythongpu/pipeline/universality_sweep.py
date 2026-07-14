# ============================================================
#  Universality Sweep — Cross-System Basin Morphometry
#  Project : Nimble Brain (REU @ Clarkson)
#  Purpose : A multi-system testing suite that probes the STRUCTURAL
#            UNIVERSALITY of basin fractalisation across four canonical
#            vector fields — Hindmarsh–Rose, Lorenz, Rössler, Van der Pol —
#            and across three topological null models — Erdős–Rényi,
#            Watts–Strogatz, Barabási–Albert.
#
#  Formalism
#  ---------
#  For a diffusively-coupled ensemble  ẋ = F(x) - K (L ⊗ H) x  we sample a
#  2-simplex-free affine 2-slice Σ ⊂ phase space (the initial-condition
#  manifold) at resolution 256×256 and integrate every point of Σ to its
#  ω-limit. A scale-invariant order functional r ∈ (0,1] (neighbour-relative
#  coherence) partitions Σ into a finite family of basins {B_j}. Over that
#  partition we evaluate two measure-theoretic descriptors of unpredictability
#  (Daza et al., Sci. Rep. 2016):
#
#      Basin Entropy            S_b   — the mean Gibbs entropy per ε-box,
#                                       a global measure of final-state
#                                       unpredictability.
#      Boundary Basin Entropy   S_bb  — the same functional restricted to
#                                       boxes straddling ≥2 basins.
#
#  The inequality  S_bb > log 2  is a SUFFICIENT condition for a fractal
#  (measure-non-trivial) basin boundary; where it holds we flag the slice as
#  Wada-suspect. The box-counting codimension D_f of the boundary set gives an
#  independent, geometric corroboration of the same phenomenon.
#
#  Experiments (see main())
#  ------------------------
#    1. coupling  — sweep K ∈ [0, 0.9]; tabulate S_b, S_bb, Wada flag, D_f
#                   for all four systems.
#    2. switch    — hold K fixed; animate the driver→target pair migrating
#                   along the degree spectrum of a scale-free graph, from a
#                   leaf-node coupling to a high-degree hub coupling → MP4.
#    3. topology  — hold (system, K) fixed; compare D_f across ER / WS / BA
#                   null models to test structural universality.
#
#  Output : data/derivatives/universality_*.{npz,png,mp4}
# ============================================================

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Make the flat `pythongpu` package importable when this file is run by
#    path from pythongpu/pipeline/ (repo root is parents[2]). Mirrors the
#    shim already used by pythongpu/pipeline/animate_coupling.py. ──
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # headless: render straight to a pixel buffer
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

import cv2  # MP4 muxing without a system ffmpeg (bundled mp4v encoder)  # noqa: E402
import networkx as nx  # noqa: E402

from pythongpu.oscillators.hindmarsh_rose import HindmarshRoseNetwork  # noqa: E402
from pythongpu.oscillators.lorenz import LorenzNetwork  # noqa: E402
from pythongpu.oscillators.rossler import RosslerNetwork  # noqa: E402
from pythongpu.oscillators.vanderpol import VanDerPolNetwork  # noqa: E402
from pythongpu.processing.box_counting import (  # noqa: E402
    boxcount_2d_gpu,
    extract_boundary,
    fractal_dimension,
)
from pythongpu.utils import get_laplacian  # noqa: E402


# ── 1. SYSTEM REGISTRY ───────────────────────────────────────
@dataclass(frozen=True)
class VectorField:
    """
    A single exemplar of the cross-system family: its oscillator class, the
    quiescent base point of the state manifold (every non-swept coordinate is
    pinned here), and the coordinate window over which the 2-slice Σ sweeps
    the fast component of the driver / target nodes.
    """
    name: str
    cls: type
    base_state: tuple[float, ...]      # (D,) pin for the non-swept coordinates
    slice_bounds: tuple[float, float]  # window of the swept fast coordinate


SYSTEMS: dict[str, VectorField] = {
    "hindmarsh_rose": VectorField(
        "hindmarsh_rose", HindmarshRoseNetwork, (-1.0, 0.0, 3.0), (-3.0, 3.0)),
    "lorenz": VectorField(
        "lorenz", LorenzNetwork, (1.0, 1.0, 1.0), (-15.0, 15.0)),
    "rossler": VectorField(
        "rossler", RosslerNetwork, (0.1, 0.1, 0.1), (-8.0, 8.0)),
    "vanderpol": VectorField(
        "vanderpol", VanDerPolNetwork, (0.1, 0.1), (-3.0, 3.0)),
}


# ── 2. CONFIG ────────────────────────────────────────────────
@dataclass
class SweepConfig:
    # Initial-condition slice Σ
    grid_n: int = 256                 # 256×256 = 65,536 sampled ICs

    # Ensemble / topology
    n_nodes: int = 64
    graph_seed: int = 7
    er_edges: int = 256               # Erdős–Rényi edge count (G(n,M))
    ws_k: int = 8                     # Watts–Strogatz ring degree (even)
    ws_p: float = 0.15                # Watts–Strogatz rewiring probability
    ba_m: int = 4                     # Barabási–Albert attachment degree

    # Integration
    dt: float = 0.02
    tmax: float = 30.0
    tail_frac: float = 0.30           # fraction of the tail used for r
    chunk: int = 8192                 # ICs integrated per GPU chunk

    # Coupling ladder  K = 0.00, 0.09, …, 0.90
    k_start: float = 0.00
    k_stop: float = 0.90
    k_step: float = 0.09

    # Heterogeneous substrate: fixed spread applied to the non-swept nodes'
    # fast coordinate so the pinned bulk is not degenerate and the coupling
    # term has non-trivial structure to route (cf. animate_coupling.py).
    baseline_spread_frac: float = 0.15
    baseline_seed: int = 123

    # Basin partition: level sets of the coherence functional r ∈ (0,1]
    r_bins: tuple[float, ...] = (0.25, 0.50, 0.75)

    # Basin-entropy box scale ε (Daza covering)
    box_eps: int = 8

    # Node-switching animation (K held in the pre-synchronisation window where
    # the basin partition is still non-trivial — past each system's coupling
    # threshold the ensemble synchronises and the slice collapses to one basin)
    switch_system: str = "hindmarsh_rose"
    switch_k: float = 0.12
    switch_frames: int = 12           # degree-ordered target nodes sampled
    fps: int = 6
    dpi: int = 120

    # GPU
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir: Path = field(default_factory=lambda: _REPO_ROOT / "data" / "derivatives")

    def coupling_ladder(self) -> np.ndarray:
        n = int(round((self.k_stop - self.k_start) / self.k_step)) + 1
        return np.round(self.k_start + self.k_step * np.arange(n), 2)


# ── 3. TOPOLOGY (null models) ────────────────────────────────
def build_topology(kind: str, cfg: SweepConfig, device: torch.device):
    """
    Realise one topological null model as (A, L, degree):

        A       — (N,N) symmetric binary adjacency (numpy)
        L       — (N,N) combinatorial Laplacian D − A (torch, on device)
        degree  — (N,) degree sequence (numpy)

    kind ∈ {"er", "ws", "ba"} selects Erdős–Rényi G(n,M), Watts–Strogatz
    small-world, or Barabási–Albert scale-free respectively. Edge budgets are
    matched as closely as each generator's native parametrisation allows so
    the comparison isolates *topology* rather than raw connection density.
    """
    if kind == "er":
        g = nx.gnm_random_graph(cfg.n_nodes, cfg.er_edges, seed=cfg.graph_seed)
    elif kind == "ws":
        g = nx.watts_strogatz_graph(cfg.n_nodes, cfg.ws_k, cfg.ws_p, seed=cfg.graph_seed)
    elif kind == "ba":
        g = nx.barabasi_albert_graph(cfg.n_nodes, cfg.ba_m, seed=cfg.graph_seed)
    else:
        raise ValueError(f"unknown topology kind {kind!r}")

    A = nx.to_numpy_array(g, dtype=np.float32)
    np.fill_diagonal(A, 0.0)
    L = get_laplacian(A, norm=None, device=device)
    degree = A.sum(axis=1)
    return A, L, degree


# ── 4. INITIAL-CONDITION SLICE Σ ─────────────────────────────
def build_ic_grid(vf: VectorField, node_x: int, node_y: int, n_nodes: int,
                  grid_n: int, device: torch.device, baseline: np.ndarray | None = None):
    """
    Construct the batched (B, D, N) initial-condition ensemble spanning the
    affine 2-slice Σ: the fast (index-0) coordinate of the driver node
    `node_x` and the target node `node_y` are swept over slice_bounds²; every
    other coordinate is pinned at vf.base_state (with an optional fixed
    per-node `baseline` offset added to the fast coordinate so the pinned bulk
    is a heterogeneous substrate rather than a degenerate constant). B = grid_n².
    """
    D = len(vf.base_state)
    lo, hi = vf.slice_bounds
    ax = np.linspace(lo, hi, grid_n, dtype=np.float32)
    Xg, Yg = np.meshgrid(ax, ax)
    B = grid_n * grid_n

    state0 = torch.zeros((B, D, n_nodes), dtype=torch.float32, device=device)
    for d, value in enumerate(vf.base_state):
        state0[:, d, :] = value
    if baseline is not None:
        state0[:, 0, :] += torch.as_tensor(baseline, dtype=torch.float32, device=device)
    state0[:, 0, node_x] = torch.tensor(Xg.ravel(), device=device)
    state0[:, 0, node_y] = torch.tensor(Yg.ravel(), device=device)
    return state0, Xg, Yg


# ── 5. CHUNKED GPU INTEGRATION → COHERENCE FIELD ─────────────
def _batched_coherence_matrix(tail: torch.Tensor, W: torch.Tensor,
                              row_sum: torch.Tensor, has_nbr: torch.Tensor) -> torch.Tensor:
    """
    Vectorised per-node neighbour-relative coherence for a whole chunk at once.

    tail    : (T, C, N) fast-coordinate history of C initial conditions.
    W       : (N, N) adjacency; row_sum, has_nbr precomputed from W.
    Returns : (C, N) coherence matrix whose (c, i) entry is

        r_i = exp( -Var_t[x_i - Σ_j W_ij x_j / Σ_j W_ij] / Var_t[x_i] ) ∈ (0,1],

    the same measure chimera_classifier.local_coherence computes per node,
    here evaluated in a single batched pass. Isolated / motionless nodes map
    to 1 (a node cannot be out of step with an empty neighbourhood).
    """
    tail = torch.nan_to_num(tail, nan=0.0, posinf=0.0, neginf=0.0)
    rs = row_sum.clamp_min(1.0)
    local_field = torch.matmul(tail, W.T) / rs           # (T, C, N)
    deviation = tail - local_field
    var_dev = deviation.var(dim=0)                        # (C, N)
    var_own = tail.var(dim=0)                             # (C, N)

    coherence = torch.exp(-var_dev / (var_own + 1e-9))
    valid = has_nbr.unsqueeze(0) & (var_own > 1e-9)
    coherence = torch.where(valid, coherence, torch.ones_like(coherence))
    return coherence                                     # (C, N)


def integrate_coherence_field(net, state0: torch.Tensor, W: torch.Tensor,
                              weights: torch.Tensor, cfg: SweepConfig,
                              device: torch.device) -> np.ndarray:
    """
    Flow every initial condition of Σ to t = tmax with RK4 and return the
    terminal coherence field r as a (grid_n, grid_n) map.

    r is the `weights`-weighted mean of the per-node coherence — pass a local
    patch indicator (the driver, the target, and their neighbourhoods) so the
    order functional stays sensitive to the two swept nodes rather than being
    diluted by the pinned bulk of the network.

    ICs are streamed in chunks of cfg.chunk so peak VRAM is bounded by the
    tail buffer (T_tail, chunk, N) rather than the full (B, D, N, T) history —
    this is what keeps a 256×256 slice tractable at any node count.
    """
    steps = int(round(cfg.tmax / cfg.dt))
    tail_steps = max(2, int(steps * cfg.tail_frac))
    rec_from = steps - tail_steps
    B, _, N = state0.shape

    row_sum = W.sum(dim=1)
    has_nbr = row_sum > 0
    w = weights / weights.sum().clamp_min(1e-9)

    r_out = torch.empty(B, dtype=torch.float32, device=device)
    with torch.no_grad():
        for lo in range(0, B, cfg.chunk):
            hi = min(lo + cfg.chunk, B)
            s = state0[lo:hi].clone()                     # (C, D, N)
            C = hi - lo
            tail = torch.empty((tail_steps, C, N), dtype=torch.float32, device=device)
            for t in range(steps):
                s = net.rk4_step(s, cfg.dt)
                if t >= rec_from:
                    tail[t - rec_from] = s[:, 0, :]       # fast coordinate
            coh = _batched_coherence_matrix(tail, W, row_sum, has_nbr)  # (C, N)
            r_out[lo:hi] = torch.matmul(coh, w)           # weighted patch mean

    return r_out.cpu().numpy().reshape(cfg.grid_n, cfg.grid_n)


# ── 6. BASIN PARTITION + ENTROPY FUNCTIONALS ─────────────────
def classify_basins(r_field: np.ndarray, r_bins: tuple[float, ...]) -> np.ndarray:
    """
    Partition Σ into basins as the level sets of the coherence functional r:
    a strictly-increasing bin ladder maps r → an integer basin label. This is
    the finite colouring on which the entropy functionals are evaluated.
    """
    return np.digitize(r_field, bins=list(r_bins)).astype(np.int32)


def basin_entropy(labels: np.ndarray, eps: int) -> dict:
    """
    Daza basin-entropy functionals over an ε-box covering of the labelled
    slice.

    Cover Σ with disjoint ε×ε boxes. Within box i, with empirical basin
    frequencies p_{i,j}, the Gibbs entropy is  S_i = -Σ_j p_{i,j} ln p_{i,j}.
    Then

        S_b  = ⟨S_i⟩ over ALL boxes            (basin entropy)
        S_bb = ⟨S_i⟩ over BOUNDARY boxes only  (boundary basin entropy),

    a boundary box being one whose interior meets ≥2 basins. The natural-log
    convention makes  S_bb > ln 2  the sufficient fractal-boundary (Wada-
    suspect) criterion. Returns S_b, S_bb, the box counts, and the fraction of
    boxes on the boundary.
    """
    H, W = labels.shape
    n_labels = int(labels.max()) + 1

    # Pad to a whole number of ε-boxes with a sentinel that carries no mass.
    pad_h = (-H) % eps
    pad_w = (-W) % eps
    padded = np.full((H + pad_h, W + pad_w), -1, dtype=np.int64)
    padded[:H, :W] = labels

    nby, nbx = padded.shape[0] // eps, padded.shape[1] // eps
    boxes = (padded.reshape(nby, eps, nbx, eps)
                   .transpose(0, 2, 1, 3)
                   .reshape(nby * nbx, eps * eps))

    # Per-box basin occupancy (sentinel −1 contributes to no label column).
    counts = np.zeros((boxes.shape[0], n_labels), dtype=np.float64)
    for v in range(n_labels):
        counts[:, v] = (boxes == v).sum(axis=1)
    totals = counts.sum(axis=1)
    occupied = totals > 0

    p = np.zeros_like(counts)
    p[occupied] = counts[occupied] / totals[occupied, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(p), 0.0)
    box_entropy = -plogp.sum(axis=1)

    n_colors = (counts > 0).sum(axis=1)
    boundary_box = occupied & (n_colors > 1)

    s_b = float(box_entropy[occupied].mean()) if occupied.any() else 0.0
    s_bb = float(box_entropy[boundary_box].mean()) if boundary_box.any() else 0.0
    return {
        "S_b": s_b,
        "S_bb": s_bb,
        "n_boxes": int(occupied.sum()),
        "n_boundary_boxes": int(boundary_box.sum()),
        "boundary_fraction": float(boundary_box.sum() / max(1, occupied.sum())),
        "wada_suspect": bool(s_bb > math.log(2.0)),
    }


def boundary_fractal_dim(labels: np.ndarray, device: torch.device):
    """
    Box-counting codimension D_f of the inter-basin boundary set. Returns
    (D_f, R²) or (None, None) when the slice collapses to a single basin and
    the boundary set is empty (nothing to fit).
    """
    boundary = extract_boundary(labels)
    if not boundary.any():
        return None, None
    r, n = boxcount_2d_gpu(boundary, device)
    if not np.any(n > 0):
        return None, None
    d_f, r_sq = fractal_dimension(r, n)
    return d_f, r_sq


# ── 7. ONE BASIN OBSERVATION ─────────────────────────────────
@dataclass
class BasinObservation:
    system: str
    topology: str
    coupling: float
    node_x: int
    node_y: int
    r_field: np.ndarray
    labels: np.ndarray
    S_b: float
    S_bb: float
    wada_suspect: bool
    boundary_fraction: float
    D_f: float | None
    r_squared: float | None
    n_basins: int


def observe_basin(vf: VectorField, A: np.ndarray, L: torch.Tensor, coupling: float,
                  node_x: int, node_y: int, cfg: SweepConfig,
                  device: torch.device) -> BasinObservation:
    """
    Full morphometry of one basin slice: flow Σ, partition into basins, and
    evaluate the entropy functionals plus the boundary codimension.
    """
    net = vf.cls(L=L, coupling=coupling, device=device)
    W = torch.as_tensor(A, dtype=torch.float32, device=device)
    N = A.shape[0]

    # Fixed heterogeneous baseline for the non-swept nodes' fast coordinate.
    lo, hi = vf.slice_bounds
    amp = cfg.baseline_spread_frac * (hi - lo)
    rng = np.random.default_rng(cfg.baseline_seed)
    baseline = (rng.uniform(-1.0, 1.0, size=N) * amp).astype(np.float32)
    baseline[node_x] = 0.0   # swept columns are overwritten anyway
    baseline[node_y] = 0.0

    # Local patch: reduce the coherence functional over the driver, the target
    # and their neighbourhoods so it tracks the two swept nodes' fate.
    patch = np.zeros(N, dtype=np.float32)
    patch[node_x] = patch[node_y] = 1.0
    patch[A[node_x] > 0] = 1.0
    patch[A[node_y] > 0] = 1.0
    weights = torch.as_tensor(patch, dtype=torch.float32, device=device)

    state0, _, _ = build_ic_grid(vf, node_x, node_y, N, cfg.grid_n, device, baseline)
    r_field = integrate_coherence_field(net, state0, W, weights, cfg, device)
    labels = classify_basins(r_field, cfg.r_bins)

    ent = basin_entropy(labels, cfg.box_eps)
    d_f, r_sq = boundary_fractal_dim(labels, device)

    return BasinObservation(
        system=vf.name, topology="", coupling=float(coupling),
        node_x=node_x, node_y=node_y, r_field=r_field, labels=labels,
        S_b=ent["S_b"], S_bb=ent["S_bb"], wada_suspect=ent["wada_suspect"],
        boundary_fraction=ent["boundary_fraction"], D_f=d_f, r_squared=r_sq,
        n_basins=int(labels.max()) + 1,
    )


# ── 8. EXPERIMENT 1 — COUPLING SWEEP (all four systems) ──────
def experiment_coupling(cfg: SweepConfig, device: torch.device,
                        systems: list[str]) -> dict:
    """
    Sweep K ∈ [0, 0.9] for each vector field on a shared Erdős–Rényi
    substrate and tabulate the entropy descriptors and boundary codimension.
    Produces a per-system S_b / S_bb / D_f(K) figure and an .npz record.
    """
    A, L, degree = build_topology("er", cfg, device)
    # Driver = a representative interior node; target = the maximal-degree hub.
    node_x = int(np.argsort(degree)[len(degree) // 2])
    node_y = int(np.argmax(degree))
    ladder = cfg.coupling_ladder()
    print(f"[coupling] N={cfg.n_nodes} ER(M={cfg.er_edges})  grid={cfg.grid_n}²  "
          f"driver={node_x} target={node_y}  K={ladder.tolist()}")

    results: dict[str, list[BasinObservation]] = {}
    for name in systems:
        vf = SYSTEMS[name]
        row: list[BasinObservation] = []
        for K in ladder:
            t0 = time.time()
            obs = observe_basin(vf, A, L, float(K), node_x, node_y, cfg, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            wada = "WADA" if obs.wada_suspect else "----"
            df = f"{obs.D_f:.3f}" if obs.D_f is not None else "  -  "
            print(f"  [{name:>14}] K={K:0.2f}  S_b={obs.S_b:.3f}  "
                  f"S_bb={obs.S_bb:.3f} {wada}  D_f={df}  "
                  f"basins={obs.n_basins}  {time.time()-t0:5.1f}s")
            row.append(obs)
        results[name] = row

    _plot_coupling(results, ladder, cfg)
    _save_coupling_npz(results, ladder, cfg)
    return results


def _plot_coupling(results, ladder, cfg: SweepConfig):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=cfg.dpi)
    wada = math.log(2.0)
    for name, row in results.items():
        sb = [o.S_b for o in row]
        sbb = [o.S_bb for o in row]
        df = [o.D_f if o.D_f is not None else np.nan for o in row]
        axes[0].plot(ladder, sb, marker="o", label=name)
        axes[1].plot(ladder, sbb, marker="o", label=name)
        axes[2].plot(ladder, df, marker="o", label=name)
    axes[0].set_title("Basin entropy  $S_b(K)$")
    axes[1].set_title("Boundary basin entropy  $S_{bb}(K)$")
    axes[1].axhline(wada, ls="--", color="k", lw=1)
    axes[1].text(ladder[0], wada, r"  $\ln 2$ (Wada)", va="bottom", fontsize=8)
    axes[2].set_title("Boundary codimension  $D_f(K)$")
    for ax in axes:
        ax.set_xlabel("coupling  $K$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Cross-system basin morphometry over the coupling ladder")
    fig.tight_layout()
    out = cfg.out_dir / "universality_coupling.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[coupling] figure -> {out}")


def _save_coupling_npz(results, ladder, cfg: SweepConfig):
    payload = {"K": ladder}
    for name, row in results.items():
        payload[f"{name}__S_b"] = np.array([o.S_b for o in row])
        payload[f"{name}__S_bb"] = np.array([o.S_bb for o in row])
        payload[f"{name}__D_f"] = np.array(
            [o.D_f if o.D_f is not None else np.nan for o in row])
        payload[f"{name}__wada"] = np.array([o.wada_suspect for o in row])
    out = cfg.out_dir / "universality_coupling.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    print(f"[coupling] record -> {out}")


# ── 9. EXPERIMENT 2 — NODE-SWITCHING ANIMATION ───────────────
def _render_switch_frame(fig, canvas, obs: BasinObservation, degree_val: float,
                         rank: int, total: int, cfg: SweepConfig) -> np.ndarray:
    """Draw the coherence field + basin boundary for one target node → BGR."""
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    lo, hi = SYSTEMS[obs.system].slice_bounds
    im = ax.imshow(obs.r_field, origin="lower", cmap="turbo", vmin=0.0, vmax=1.0,
                   extent=[lo, hi, lo, hi], interpolation="bilinear")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("coherence functional  $r$")

    boundary = extract_boundary(obs.labels)
    if boundary.any():
        ov = np.zeros((*boundary.shape, 4), dtype=np.float32)
        ov[boundary, 3] = 0.85
        ax.imshow(ov, origin="lower", extent=[lo, hi, lo, hi], interpolation="nearest")

    wada = "  •  Wada-suspect ($S_{bb}>\\ln 2$)" if obs.wada_suspect else ""
    df = f"{obs.D_f:.3f}" if obs.D_f is not None else "n/a"
    ax.set_xlabel(f"$x_0$ (driver node {obs.node_x})")
    ax.set_ylabel(f"$x_0$ (target node {obs.node_y})")
    ax.set_title(
        f"Node-switching sweep  [{obs.system}]   K={obs.coupling:0.2f}\n"
        f"target rank {rank+1}/{total}  •  deg(target)={int(degree_val)}   "
        f"$S_b$={obs.S_b:.3f}  $S_{{bb}}$={obs.S_bb:.3f}  $D_f$={df}{wada}")
    fig.tight_layout()
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return np.ascontiguousarray(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))


def experiment_switch(cfg: SweepConfig, device: torch.device) -> Path:
    """
    Hold K fixed and migrate the driver→target coupling pair along the degree
    spectrum of a Barabási–Albert scale-free substrate: the target node walks
    from a minimal-degree leaf to the maximal-degree hub while the driver stays
    a fixed low-degree anchor. Each configuration's basin slice becomes one MP4
    keyframe, so the film shows how attaching the perturbation to progressively
    more central nodes reshapes the basin geometry.
    """
    vf = SYSTEMS[cfg.switch_system]
    A, L, degree = build_topology("ba", cfg, device)
    order = np.argsort(degree)                 # ascending: leaf → hub
    driver = int(order[0])                      # a minimal-degree anchor
    # Sample switch_frames target nodes evenly across the degree ranking.
    picks = np.linspace(0, len(order) - 1, cfg.switch_frames).round().astype(int)
    targets = [int(order[i]) for i in picks]
    print(f"[switch]  {vf.name}  BA(m={cfg.ba_m})  K={cfg.switch_k}  "
          f"driver={driver}(deg {int(degree[driver])})  "
          f"targets deg {int(degree[targets[0]])}→{int(degree[targets[-1]])}")

    fig = plt.figure(figsize=(7.5, 6.5), dpi=cfg.dpi)
    canvas = FigureCanvasAgg(fig)
    out = cfg.out_dir / f"universality_switch_{vf.name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for rank, tgt in enumerate(targets):
        if tgt == driver:                       # keep the 2-slice non-degenerate
            continue
        t0 = time.time()
        obs = observe_basin(vf, A, L, cfg.switch_k, driver, tgt, cfg, device)
        frame = _render_switch_frame(fig, canvas, obs, degree[tgt], rank,
                                     len(targets), cfg)
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out), fourcc, cfg.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"cv2 could not open writer for {out}")
        writer.write(frame)
        wada = "WADA" if obs.wada_suspect else "----"
        print(f"  [frame {rank+1:>2}/{len(targets)}] target={tgt} "
              f"deg={int(degree[tgt])}  S_bb={obs.S_bb:.3f} {wada}  "
              f"{time.time()-t0:5.1f}s")
    if writer is not None:
        writer.release()
    plt.close(fig)
    ok = out.exists() and out.stat().st_size > 0
    print(f"[switch]  MP4 -> {out}  ({out.stat().st_size/1e6:.2f} MB)" if ok
          else "[switch]  FAILED")
    return out


# ── 10. EXPERIMENT 3 — TOPOLOGICAL UNIVERSALITY ──────────────
def experiment_topology(cfg: SweepConfig, device: torch.device) -> dict:
    """
    Hold (system, K) fixed and recompute the basin morphometry on each of the
    three null models — Erdős–Rényi, Watts–Strogatz, Barabási–Albert — to test
    whether the boundary codimension D_f (and the entropy descriptors) are
    invariant under the choice of random-graph ensemble. Structural
    universality would manifest as a D_f that is insensitive to topology.
    """
    vf = SYSTEMS[cfg.switch_system]
    K = cfg.switch_k
    print(f"[topology] {vf.name}  K={K}  N={cfg.n_nodes}  grid={cfg.grid_n}²")
    summary = {}
    for kind, label in (("er", "Erdős–Rényi"), ("ws", "Watts–Strogatz"),
                        ("ba", "Barabási–Albert")):
        A, L, degree = build_topology(kind, cfg, device)
        node_x = int(np.argsort(degree)[len(degree) // 2])
        node_y = int(np.argmax(degree))
        t0 = time.time()
        obs = observe_basin(vf, A, L, K, node_x, node_y, cfg, device)
        obs.topology = label
        summary[kind] = obs
        edges = int(A.sum() // 2)
        df = f"{obs.D_f:.3f}" if obs.D_f is not None else "n/a"
        print(f"  [{label:>16}] edges={edges}  ⟨k⟩={2*edges/cfg.n_nodes:.1f}  "
              f"S_b={obs.S_b:.3f}  S_bb={obs.S_bb:.3f}  D_f={df}  "
              f"{time.time()-t0:5.1f}s")

    _plot_topology(summary, vf.name, K, cfg)
    _print_topology_summary(summary)
    return summary


def _plot_topology(summary, system, K, cfg: SweepConfig):
    labels = [o.topology for o in summary.values()]
    df = [o.D_f if o.D_f is not None else 0.0 for o in summary.values()]
    sb = [o.S_b for o in summary.values()]
    sbb = [o.S_bb for o in summary.values()]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=cfg.dpi)
    ax.bar(x - 0.25, df, 0.25, label="$D_f$ (codimension)")
    ax.bar(x, sb, 0.25, label="$S_b$")
    ax.bar(x + 0.25, sbb, 0.25, label="$S_{bb}$")
    ax.axhline(math.log(2.0), ls="--", color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"Structural universality  [{system}]  K={K}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = cfg.out_dir / f"universality_topology_{system}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[topology] figure -> {out}")


def _print_topology_summary(summary):
    dfs = [o.D_f for o in summary.values() if o.D_f is not None]
    spread = (max(dfs) - min(dfs)) if dfs else float("nan")
    print("  ── comparative summary ──────────────────────────────")
    print(f"  {'topology':>16} {'D_f':>8} {'S_b':>8} {'S_bb':>8} {'Wada':>6}")
    for o in summary.values():
        df = f"{o.D_f:.3f}" if o.D_f is not None else "  n/a "
        print(f"  {o.topology:>16} {df:>8} {o.S_b:8.3f} {o.S_bb:8.3f} "
              f"{'yes' if o.wada_suspect else 'no':>6}")
    print(f"  cross-topology D_f spread = {spread:.3f}  "
          f"({'universal (small spread)' if spread == spread and spread < 0.15 else 'topology-dependent'})")


# ── 11. CLI ──────────────────────────────────────────────────
def _apply_smoke(cfg: SweepConfig) -> SweepConfig:
    """Shrink every dimension for a fast end-to-end correctness pass."""
    cfg.grid_n = 48
    cfg.n_nodes = 24
    cfg.er_edges = 60
    cfg.ws_k = 6
    cfg.ba_m = 3
    cfg.tmax = 12.0
    cfg.chunk = 4096
    cfg.k_start, cfg.k_stop, cfg.k_step = 0.0, 0.9, 0.3
    cfg.switch_frames = 5
    cfg.box_eps = 4
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", nargs="?", default="all",
                    choices=["all", "coupling", "switch", "topology"])
    ap.add_argument("--systems", default=",".join(SYSTEMS),
                    help="comma-separated subset of the vector-field registry")
    ap.add_argument("--grid-n", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--switch-system", default=None, choices=list(SYSTEMS),
                    help="vector field driving the switch/topology experiments")
    ap.add_argument("--switch-k", type=float, default=None,
                    help="fixed coupling for the switch/topology experiments")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end pass for testing")
    args = ap.parse_args(argv)

    cfg = SweepConfig()
    if args.smoke:
        cfg = _apply_smoke(cfg)
    if args.grid_n is not None:
        cfg.grid_n = args.grid_n
    if args.device is not None:
        cfg.device = args.device
    if args.switch_system is not None:
        cfg.switch_system = args.switch_system
    if args.switch_k is not None:
        cfg.switch_k = args.switch_k
    device = torch.device(cfg.device)
    systems = [s for s in args.systems.split(",") if s in SYSTEMS]
    print(f"[device]  {device}   systems={systems}")

    if args.experiment in ("all", "coupling"):
        experiment_coupling(cfg, device, systems)
    if args.experiment in ("all", "switch"):
        experiment_switch(cfg, device)
    if args.experiment in ("all", "topology"):
        experiment_topology(cfg, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
