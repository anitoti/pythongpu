#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  analyze_wada.py  —  Daza Wada-bounds test over a coupling sweep
#
#  For every  basin_data*.npz  in a coupling-sweep directory this script runs
#  the grid (neighbourhood) form of the Daza et al. Wada test on the basin
#  labelling and quantifies how much of the inter-basin boundary is *Wada*
#  (i.e. simultaneously borders three or more basins).
#
#  Method (scipy.ndimage).  Let  L  be the H×W integer basin map (background /
#  non-converged cells carry the sentinel −1, matching the convention in
#  universality_sweep.basin_entropy).  For each basin label b we form the
#  indicator  I_b = [L == b]  and dilate it by a radius-r structuring element
#  (8-connected, r = 1 by default).  The per-pixel *coverage*
#
#        C(x) = Σ_b  dilate(I_b)(x)
#
#  counts how many distinct basins lie within r pixels of x.  Then
#
#        boundary  = { C ≥ 2 }          (pixel borders ≥2 basins)
#        wada      = { C ≥ 3 }          (pixel borders ≥3 basins  →  Wada point)
#        strict    = { C ≥ n_basins }   (pixel borders EVERY basin)
#
#  The reported Wada-boundary coverage fraction is  |wada| / |boundary|; a basin
#  is Daza-Wada when  |strict| / |boundary| ≈ 1  (a boundary every point of
#  which touches all basins).  Background cells never count as a basin, so
#  masked / unconverged initial conditions do not inflate the boundary.
#
#  Outputs (per sweep) land in  wada_results/ :
#    • <stem>_wada.png     two-panel: basin map + Wada overlay | coverage(K),D_f(K)
#    • wada_coverage_vs_K.png   standalone sweep summary
#    • wada_summary.csv / .npz  the tabulated per-K descriptors
#
#  The D_f(K) curve is read from each file's stored `fractal_dim` (the boundary
#  box-counting codimension already computed by the sweep) so no re-fitting is
#  needed.
#
#  Usage:
#    python3 analyze_wada.py [--data-dir DIR] [--out DIR] [--radius R]
#                            [--background N] [--wada-thresh F]
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")  # headless: no display on the cluster / login node
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ── path resolution ──────────────────────────────────────────────────────────
# The sweep is authored against the cluster mount  /mnt/home/atotilca/...  which
# on a dev/login box is the same tree under  /home/atotilca/...  .  Try the
# requested path first, then the /mnt↔/home twin, then the in-repo data dir.
_DEFAULT_DATA_DIR = "/mnt/home/atotilca/pythongpu/data/coupling_sweep/"
_REPO_ROOT = Path(__file__).resolve().parent


def resolve_data_dir(requested: str) -> Path:
    """Return the first existing candidate directory that holds basin_data*.npz."""
    cand: list[Path] = []
    r = Path(requested)
    cand.append(r)
    # /mnt/home/... ↔ /home/...  (cluster mount vs. local view)
    s = str(r)
    if s.startswith("/mnt/home/"):
        cand.append(Path(s.replace("/mnt/home/", "/home/", 1)))
    elif s.startswith("/home/"):
        cand.append(Path("/mnt" + s))
    # in-repo fallbacks
    cand.append(_REPO_ROOT / "data" / "coupling_sweep")
    cand.append(_REPO_ROOT / "data")

    seen: set[str] = set()
    for c in cand:
        cs = str(c)
        if cs in seen:
            continue
        seen.add(cs)
        if c.is_dir() and sorted(c.glob("basin_data*.npz")):
            return c
    # nothing matched — report what we tried
    tried = "\n  ".join(str(c) for c in cand)
    sys.exit(f"[analyze_wada] no basin_data*.npz found. Tried:\n  {tried}")


# ── K extraction ─────────────────────────────────────────────────────────────
_K_PATTERNS = [
    re.compile(r"[Kk]\s*[=_]?\s*(-?\d+(?:\.\d+)?)"),   # K=0.30, K_0.3, k0.6
    re.compile(r"[Kk](\d+)p(\d+)"),                     # k0p30  ->  0.30
]


def extract_K(npz, path: Path, index: int) -> tuple[float, str]:
    """Best-effort coupling value for one file. Returns (K, provenance)."""
    # 1) explicit key inside the archive
    for key in ("K", "coupling", "k"):
        if key in getattr(npz, "files", []):
            try:
                return float(np.asarray(npz[key]).ravel()[0]), "npz-key"
            except Exception:
                pass
    # 2) parse from filename, then parent directory name
    for text in (path.stem, path.parent.name, str(path.parent)):
        for pat in _K_PATTERNS:
            m = pat.search(text)
            if m:
                if m.re is _K_PATTERNS[1]:
                    return float(f"{m.group(1)}.{m.group(2)}"), "path"
                return float(m.group(1)), "path"
    # 3) fall back to file order
    return float(index), "index"


# ── Daza grid Wada test ──────────────────────────────────────────────────────
def background_mask(labels: np.ndarray, background: int) -> np.ndarray:
    """True where a cell is masked/background and must be excluded from a basin."""
    bg = np.zeros(labels.shape, dtype=bool)
    if np.issubdtype(labels.dtype, np.floating):
        bg |= ~np.isfinite(labels)
    bg |= labels < 0            # sentinel convention: negatives are background
    bg |= labels == background  # explicit sentinel (default −1, already covered)
    return bg


def daza_wada(labels: np.ndarray, background: int, radius: int) -> dict:
    """
    Grid Wada-bounds test. Returns coverage count, the wada/boundary/strict
    masks and their scalar summaries. `labels` may carry a background sentinel.
    """
    bg = background_mask(labels, background)
    basins = [int(b) for b in np.unique(labels[~bg])] if (~bg).any() else []
    n_basins = len(basins)

    struct = ndimage.generate_binary_structure(2, 2)  # 8-connected Moore nbhd
    coverage = np.zeros(labels.shape, dtype=np.int32)
    for b in basins:
        ind = (labels == b) & ~bg
        if radius > 0:
            ind = ndimage.binary_dilation(ind, structure=struct, iterations=radius)
        coverage += ind.astype(np.int32)

    coverage[bg] = 0  # background is never part of any boundary

    boundary = coverage >= 2
    wada = coverage >= 3
    strict = coverage >= max(3, n_basins) if n_basins >= 3 else np.zeros_like(wada)

    n_bnd = int(boundary.sum())
    frac = float(wada.sum()) / n_bnd if n_bnd else float("nan")
    frac_strict = float(strict.sum()) / n_bnd if n_bnd else float("nan")

    return {
        "n_basins": n_basins,
        "coverage": coverage,
        "bg_mask": bg,
        "boundary_mask": boundary,
        "wada_mask": wada,
        "strict_mask": strict,
        "n_boundary_px": n_bnd,
        "n_wada_px": int(wada.sum()),
        "wada_fraction": frac,
        "strict_fraction": frac_strict,
    }


# ── per-file loading ─────────────────────────────────────────────────────────
def read_fractal_dim(npz) -> float:
    """Boundary codimension D_f already stored by the sweep, or NaN."""
    if "fractal_dim" in getattr(npz, "files", []):
        try:
            return float(np.asarray(npz["fractal_dim"]).ravel()[0])
        except Exception:
            pass
    return float("nan")


def compute_basin_entropy(labels: np.ndarray, eps: int) -> dict:
    """
    Boundary basin entropy S_bb via the Daza box covering, reusing the
    implementation in universality_sweep so the two tools cannot drift apart.

    NOTE the two criteria answer DIFFERENT questions and are reported side by
    side rather than conflated:
      * dilation test  -> strict_fraction ~ 1  : the boundary is Wada (every
        boundary point borders every basin). Strictly stronger.
      * S_bb > ln 2    -> wada_suspect        : a SUFFICIENT condition for a
        FRACTAL boundary. It does not by itself establish Wada.
    A slice can satisfy S_bb > ln 2 and not be Wada.
    """
    # Imported lazily: universality_sweep pulls in torch, which we do not want
    # to require for a pure-numpy label analysis unless the user asks for it.
    from pythongpu.pipeline.universality_sweep import basin_entropy
    return basin_entropy(labels, eps)


def load_record(path: Path, index: int, args) -> dict | None:
    """Load one basin_data npz and run the Daza test. Only touches small keys
    (never the multi-GB `vectors` array, thanks to npz lazy loading)."""
    npz = np.load(path, allow_pickle=False, mmap_mode="r")
    if "labels" not in npz.files:
        print(f"[skip] {path.name}: no 'labels' array")
        return None
    labels = np.asarray(npz["labels"])
    K, prov = extract_K(npz, path, index)
    res = daza_wada(labels, args.background, args.radius)
    ent = compute_basin_entropy(labels, args.entropy_eps)

    # ── trust gates ─────────────────────────────────────────────────────────
    # (1) Degeneracy, label-only and free. If essentially EVERY occupied box is
    #     a boundary box, the labelling is salt-and-pepper: the "boundary" fills
    #     the slice, S_b collapses onto S_bb, D_f -> 2, and S_bb > ln 2 fires for
    #     a partition of noise. Such a slice reports a spectacular false positive
    #     unless it is quarantined.
    degenerate = ent["boundary_fraction"] >= args.trust_boundary_frac
    # (2) Full VPS structure gate (opt-in: needs the multi-GB `vectors`).
    structured = None
    if args.gate:
        from pythongpu.pipeline.vps_diagnostics import diagnose
        try:
            d = diagnose(path, args.gate_k_max, args.gate_n_sub,
                         args.gate_n_eig, args.gate_null_reps,
                         np.random.default_rng(0))
            structured = bool(d["structured"])
        except KeyError:
            structured = None      # npz predates feature persistence
    trusted = (not degenerate) and (structured is not False)

    # state-space extent for the basin map, if the IC grid is stored
    extent = None
    if "Xg" in npz.files and "Yg" in npz.files:
        Xg, Yg = np.asarray(npz["Xg"]), np.asarray(npz["Yg"])
        extent = [float(Xg.min()), float(Xg.max()),
                  float(Yg.min()), float(Yg.max())]

    rec = dict(res)
    rec.update(
        path=path, stem=path.stem, K=K, K_provenance=prov,
        labels=labels, extent=extent, D_f=read_fractal_dim(npz),
        S_b=ent["S_b"], S_bb=ent["S_bb"],
        entropy_fractal=ent["wada_suspect"],       # S_bb > ln 2
        entropy_boundary_fraction=ent["boundary_fraction"],
        n_entropy_boxes=ent["n_boxes"],
        degenerate=degenerate, structured=structured, trusted=trusted,
    )
    return rec


# ── plotting ─────────────────────────────────────────────────────────────────
_WADA_CMAP = ListedColormap(["#d62728"])  # crimson overlay for Wada cells


def _draw_basin_panel(ax, rec):
    labels = rec["labels"]
    vmax = int(labels[~rec["bg_mask"]].max()) if (~rec["bg_mask"]).any() else 1
    disp = np.ma.masked_where(rec["bg_mask"], labels)
    base = plt.get_cmap("tab20").copy()
    base.set_bad("0.85")  # masked/background → light grey
    ax.imshow(disp, origin="lower", cmap=base, vmin=0, vmax=max(vmax, 1),
              extent=rec["extent"], interpolation="nearest", aspect="auto")
    overlay = np.ma.masked_where(~rec["wada_mask"], rec["wada_mask"])
    ax.imshow(overlay, origin="lower", cmap=_WADA_CMAP, vmin=0, vmax=1,
              extent=rec["extent"], interpolation="nearest", aspect="auto",
              alpha=0.9)
    ax.set_title(f"Basin map + Wada boundary   $K={rec['K']:g}$\n"
                 f"{rec['n_basins']} basins · "
                 f"Wada coverage = {rec['wada_fraction']*100:.1f}%")
    if rec["extent"] is not None:
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$y_0$")
    else:
        ax.set_xlabel("grid column")
        ax.set_ylabel("grid row")
    ax.plot([], [], "s", color="#d62728", label="Wada boundary")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def _draw_sweep_panel(ax, Ks, fracs, Dfs, mark_K=None):
    l1, = ax.plot(Ks, fracs, "-o", color="C0", label="Wada coverage")
    ax.set_xlabel("coupling  $K$")
    ax.set_ylabel("Wada-boundary coverage fraction", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_ylim(-0.02, 1.02)

    ax2 = ax.twinx()
    l2, = ax2.plot(Ks, Dfs, "-s", color="C3", label="$D_f(K)$")
    ax2.set_ylabel("boundary codimension  $D_f$", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    if mark_K is not None:
        ax.axvline(mark_K, ls="--", lw=1, color="0.4")

    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="best", fontsize=8)
    ax.set_title("Wada coverage & $D_f$ over the coupling ladder")


def render_pair_figure(rec, Ks, fracs, Dfs, out_dir: Path, dpi: int):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=dpi)
    _draw_basin_panel(axes[0], rec)
    _draw_sweep_panel(axes[1], Ks, fracs, Dfs, mark_K=rec["K"])
    fig.tight_layout()
    out = out_dir / f"{rec['stem']}_wada.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def render_sweep_figure(Ks, fracs, strict, Dfs, out_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=dpi)
    l1, = ax.plot(Ks, fracs, "-o", color="C0", label="Wada coverage ($C\\geq3$)")
    l2, = ax.plot(Ks, strict, "--^", color="C1", label="strict Wada ($C\\geq n$)")
    ax.set_xlabel("coupling  $K$")
    ax.set_ylabel("boundary coverage fraction", color="k")
    ax.set_ylim(-0.02, 1.02)
    ax2 = ax.twinx()
    l3, = ax2.plot(Ks, Dfs, "-s", color="C3", label="$D_f(K)$")
    ax2.set_ylabel("boundary codimension  $D_f$", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    handles = [l1, l2, l3]
    ax.legend(handles, [h.get_label() for h in handles], loc="best", fontsize=8)
    ax.set_title("Daza Wada-boundary coverage vs. coupling")
    fig.tight_layout()
    out = out_dir / "wada_coverage_vs_K.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ── driver ───────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="Daza Wada-bounds test over a coupling sweep.")
    ap.add_argument("--data-dir", default=_DEFAULT_DATA_DIR,
                    help="directory of basin_data*.npz files")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <data-dir>/wada_results)")
    ap.add_argument("--radius", type=int, default=1,
                    help="dilation radius of the neighbourhood test (pixels)")
    ap.add_argument("--background", type=int, default=-1,
                    help="basin-label sentinel for masked/background cells")
    ap.add_argument("--wada-thresh", type=float, default=0.95,
                    help="strict-coverage fraction above which a slice is flagged Wada")
    ap.add_argument("--entropy-eps", type=int, default=5,
                    help="box side (px) for the Daza basin-entropy covering used "
                         "to evaluate the S_bb > ln 2 fractal-boundary criterion")
    ap.add_argument("--trust-boundary-frac", type=float, default=0.95,
                    help="quarantine a slice as degenerate when at least this "
                         "fraction of occupied boxes are boundary boxes — i.e. "
                         "salt-and-pepper labels, for which S_bb > ln 2 and "
                         "D_f -> 2 are artifacts, not geometry")
    ap.add_argument("--gate", action="store_true",
                    help="additionally run the full VPS structure gate on each "
                         "slice's saved `vectors` (null test); slices with no "
                         "separable structure are quarantined. Costs a multi-GB "
                         "load per file.")
    ap.add_argument("--gate-null-reps", type=int, default=10)
    ap.add_argument("--gate-n-sub", type=int, default=6000)
    ap.add_argument("--gate-n-eig", type=int, default=1500)
    ap.add_argument("--gate-k-max", type=int, default=12)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args(argv)

    data_dir = resolve_data_dir(args.data_dir)
    files = sorted(data_dir.glob("basin_data*.npz"))
    print(f"[analyze_wada] data-dir : {data_dir}")
    print(f"[analyze_wada] found    : {len(files)} basin_data*.npz")

    out_dir = Path(args.out) if args.out else (data_dir / "wada_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, f in enumerate(files):
        rec = load_record(f, i, args)
        if rec is None:
            continue
        records.append(rec)
        print(f"  {f.name:<40} K={rec['K']:<6g}({rec['K_provenance']:>5}) "
              f"basins={rec['n_basins']} "
              f"wada={rec['wada_fraction']*100:5.1f}% "
              f"strict={rec['strict_fraction']*100:5.1f}% "
              f"D_f={rec['D_f']:.3f}  "
              f"S_bb={rec['S_bb']:.3f}"
              f"{' >ln2 FRACTAL' if rec['entropy_fractal'] else '  <=ln2'}"
              f"  [{'WADA' if rec['strict_fraction'] >= args.wada_thresh else 'not-Wada'}]"
              + ("" if rec["trusted"] else
                 f"  ** UNTRUSTED: {'degenerate/salt-and-pepper' if rec['degenerate'] else 'no VPS structure'} **"))

    if not records:
        sys.exit("[analyze_wada] no usable basin maps (no 'labels' arrays).")

    records.sort(key=lambda r: r["K"])
    Ks = np.array([r["K"] for r in records], float)
    fracs = np.array([r["wada_fraction"] for r in records], float)
    strict = np.array([r["strict_fraction"] for r in records], float)
    Dfs = np.array([r["D_f"] for r in records], float)

    # per-K two-panel figures
    for r in records:
        p = render_pair_figure(r, Ks, fracs, Dfs, out_dir, args.dpi)
        print(f"  wrote {p.name}")

    # sweep summary figure
    sp = render_sweep_figure(Ks, fracs, strict, Dfs, out_dir, args.dpi)
    print(f"  wrote {sp.name}")

    # tabulated summary: CSV + NPZ
    csv_path = out_dir / "wada_summary.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "K", "K_provenance", "n_basins", "n_boundary_px",
                    "n_wada_px", "wada_fraction", "strict_fraction", "D_f",
                    "is_wada", "S_b", "S_bb", "S_bb_gt_ln2", "agree",
                    "boundary_box_fraction", "degenerate", "structured",
                    "trusted"])
        for r in records:
            is_wada = int(r["strict_fraction"] >= args.wada_thresh)
            sbb_frac = int(r["entropy_fractal"])
            w.writerow([r["stem"], f"{r['K']:g}", r["K_provenance"], r["n_basins"],
                        r["n_boundary_px"], r["n_wada_px"],
                        f"{r['wada_fraction']:.6f}", f"{r['strict_fraction']:.6f}",
                        f"{r['D_f']:.6f}", is_wada,
                        f"{r['S_b']:.6f}", f"{r['S_bb']:.6f}", sbb_frac,
                        # Wada implies a fractal boundary, so a Wada slice that
                        # fails S_bb > ln 2 is an inconsistency worth inspecting.
                        int(not (is_wada and not sbb_frac)),
                        f"{r['entropy_boundary_fraction']:.6f}",
                        int(r["degenerate"]),
                        "" if r["structured"] is None else int(r["structured"]),
                        int(r["trusted"])])
    np.savez(out_dir / "wada_summary.npz",
             K=Ks, wada_fraction=fracs, strict_fraction=strict, D_f=Dfs,
             S_b=np.array([r["S_b"] for r in records], float),
             S_bb=np.array([r["S_bb"] for r in records], float),
             S_bb_gt_ln2=np.array([r["entropy_fractal"] for r in records], bool),
             n_basins=np.array([r["n_basins"] for r in records]),
             files=np.array([r["stem"] for r in records]))
    print(f"  wrote {csv_path.name}, wada_summary.npz")

    # ── cross-check the two criteria ────────────────────────────────────────
    ln2 = math.log(2.0)

    def _klist(rs):
        return ", ".join(format(r["K"], "g") for r in rs) or "none"

    # Only trusted slices may contribute a reported result. A degenerate
    # (salt-and-pepper) or structureless slice satisfies S_bb > ln 2 trivially,
    # so counting it would manufacture exactly the false positive this gate
    # exists to prevent.
    trusted = [r for r in records if r["trusted"]]
    quarantined = [r for r in records if not r["trusted"]]
    wada_hits = [r for r in trusted if r["strict_fraction"] >= args.wada_thresh]
    sbb_hits = [r for r in trusted if r["entropy_fractal"]]

    print(f"\n[criteria]  ln 2 = {ln2:.4f}   (entropy box eps = {args.entropy_eps} px)")
    print(f"[gate]      trusted {len(trusted)}/{len(records)} slices; "
          f"quarantined {len(quarantined)}   K = {_klist(quarantined)}")
    if quarantined:
        print("[gate]      quarantined slices are EXCLUDED below — their "
              "S_bb/D_f/Wada numbers measure a partition of noise, not geometry.")
    if not trusted:
        print("[criteria]  NO TRUSTED SLICES — nothing can be concluded about "
              "fractality or Wada from this directory.")
    print(f"[criteria]  S_bb > ln 2 (fractal boundary) : "
          f"{len(sbb_hits)}/{len(trusted)} trusted slices   K = {_klist(sbb_hits)}")
    print(f"[criteria]  strict Wada (>= {args.wada_thresh:.2f})       : "
          f"{len(wada_hits)}/{len(trusted)} trusted slices   K = {_klist(wada_hits)}")
    # Wada is strictly stronger than fractality: Wada without S_bb > ln 2 is a
    # contradiction and almost always means one of the tests is misconfigured.
    bad = [r for r in wada_hits if not r["entropy_fractal"]]
    if bad:
        print(f"[criteria]  INCONSISTENT: {len(bad)} slice(s) flagged Wada but "
              f"failing S_bb > ln 2 — inspect K = {_klist(bad)}")
    else:
        print("[criteria]  consistent: every Wada slice also satisfies S_bb > ln 2.")
    print(f"[analyze_wada] done → {out_dir}")


if __name__ == "__main__":
    main()
