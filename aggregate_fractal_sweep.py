#!/usr/bin/env python3
"""
aggregate_fractal_sweep.py — assemble the coupling-sweep results into a
presentation-ready fractal-dimension curve and an animation.

It does two independent things:

  1. Crawls  <root>/task_*  directories, reads the box-counting fractal
     dimension  D_f  out of each  basin_data.npz  (key `fractal_dim`), pairs
     it with the coupling strength K parsed from the folder name
     (task_0000_K0.4500 -> K = 0.4500), and plots  D_f  vs  K  with a cubic
     spline through the points.

  2. Compiles  <root>/frames/frame_*.png  into a 60 fps, high-bitrate mp4
     via ffmpeg (falling back to OpenCV's mp4v writer if ffmpeg is absent —
     e.g. a login node where the FFmpeg module has not been loaded).

Standalone: depends only on numpy, scipy, matplotlib (+ optional opencv for
the fallback). No pythongpu package import required.

Examples
--------
    # local default layout
    python aggregate_fractal_sweep.py

    # results written to scratch on ACRES
    python aggregate_fractal_sweep.py \
        --root /mnt/data/tmp/$USER/coupling_sweep

    # on ACRES first make ffmpeg available:
    #   module load FFmpeg/4.4.2-GCCcore-11.3.0
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# Matplotlib without a display (HPC login/compute nodes have no X server).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

# task_<idx>_K<coupling>, e.g. task_0007_K0.5237
_TASK_RE = re.compile(r"task_(\d+)_K([-+]?\d*\.?\d+)")


def _find_npz(task_dir: str):
    """Locate basin_data.npz within a task dir.

    lorenz_sweep currently writes it to the task root, but the plot PNGs go to
    <task>/derivatives/, so check derivatives/ first (in case the .npz is ever
    moved alongside them), then the root, then a recursive glob as a backstop.
    """
    for cand in (os.path.join(task_dir, "derivatives", "basin_data.npz"),
                 os.path.join(task_dir, "basin_data.npz")):
        if os.path.isfile(cand):
            return cand
    hits = glob.glob(os.path.join(task_dir, "**", "basin_data.npz"), recursive=True)
    return hits[0] if hits else None


# ─────────────────────────────────────────────────────────────────────────────
#  1. Harvest (K, D_f) from every task directory
# ─────────────────────────────────────────────────────────────────────────────
def harvest(root: str):
    """Return sorted-by-K arrays (K, D_f, R^2) and the list of skipped dirs."""
    task_dirs = sorted(glob.glob(os.path.join(root, "task_*")))
    if not task_dirs:
        sys.exit(f"[error] no task_* directories under {root!r}")

    rows, skipped = [], []
    for d in task_dirs:
        m = _TASK_RE.search(os.path.basename(d))
        npz = _find_npz(d)
        if m is None or npz is None:
            skipped.append(d)
            continue
        k = float(m.group(2))
        try:
            # np.load is lazy; touching only these keys avoids unpickling the
            # `config` object array, so allow_pickle stays off.
            with np.load(npz) as data:
                d_f = float(np.asarray(data["fractal_dim"]).ravel()[0])
                r2 = (float(np.asarray(data["r_squared"]).ravel()[0])
                      if "r_squared" in data.files else np.nan)
        except Exception as exc:                       # noqa: BLE001
            print(f"[warn] could not read {npz}: {exc}")
            skipped.append(d)
            continue
        rows.append((k, d_f, r2))

    if not rows:
        sys.exit(f"[error] found task dirs but no readable basin_data.npz under {root!r}")

    rows.sort(key=lambda r: r[0])
    K = np.array([r[0] for r in rows])
    Df = np.array([r[1] for r in rows])
    R2 = np.array([r[2] for r in rows])
    return K, Df, R2, skipped


# ─────────────────────────────────────────────────────────────────────────────
#  2. Plot D_f vs K with a cubic spline
# ─────────────────────────────────────────────────────────────────────────────
def plot_curve(K, Df, R2, out_png: str, csv_path: str | None = None):
    # Persist the raw numbers alongside the figure — handy as a backup slide
    # and for reproducing the plot without re-crawling.
    if csv_path:
        with open(csv_path, "w") as fh:
            fh.write("K,D_f,R_squared\n")
            for k, d, r in zip(K, Df, R2):
                fh.write(f"{k:.6f},{d:.6f},{r:.6f}\n")
        print(f"[saved]    {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Reference regimes for basin-boundary fractality.
    ax.axhspan(1.2, 1.8, color="tab:orange", alpha=0.10,
               label="expected chimera range (1.2–1.8)")
    ax.axhline(1.0, ls=":", lw=1, color="gray")
    ax.axhline(2.0, ls=":", lw=1, color="gray")
    ax.text(K.min(), 1.02, "smooth boundary  (D_f≈1)", fontsize=8, color="gray")
    ax.text(K.min(), 1.94, "space-filling  (D_f≈2)", fontsize=8, color="gray")

    # Cubic spline through the sweep points (needs strictly increasing K).
    if len(K) >= 4 and np.all(np.diff(K) > 0):
        xs = np.linspace(K.min(), K.max(), 400)
        ys = CubicSpline(K, Df)(xs)
        ax.plot(xs, ys, "-", color="tab:blue", lw=2, label="cubic spline")
    else:
        ax.plot(K, Df, "-", color="tab:blue", lw=2, label="D_f(K)")

    ax.plot(K, Df, "o", color="tab:red", ms=6, zorder=5, label="sweep points")

    ax.set_xlabel("coupling strength  K")
    ax.set_ylabel(r"basin-boundary fractal dimension  $D_f$")
    ax.set_title("Fractalization of DTI-coupled Lorenz basins vs. coupling")
    ax.set_ylim(0.95, 2.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[saved]    {out_png}   ({len(K)} points, "
          f"D_f in [{Df.min():.3f}, {Df.max():.3f}])")


# ─────────────────────────────────────────────────────────────────────────────
#  3. Compile the frame gallery into an mp4
# ─────────────────────────────────────────────────────────────────────────────
def compile_video(frames_dir: str, out_mp4: str, fps: int, bitrate: str,
                  hold: int = 1, duration: float | None = None):
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frames:
        print(f"[warn] no frames in {frames_dir}; skipping video")
        return

    # --duration overrides --hold: solve for the per-frame repeat count that
    # makes the whole clip last ~duration seconds at the chosen fps.
    if duration is not None:
        hold = max(1, round(duration * fps / len(frames)))
    hold = max(1, int(hold))

    # Materialise the playback order by repeating each K-frame `hold` times.
    seq = [f for f in frames for _ in range(hold)]
    dur = len(seq) / fps
    print(f"[video]    {len(frames)} frames x hold {hold} = {len(seq)} @ "
          f"{fps} fps  ->  {dur:.2f} s")
    if dur < 2:
        print("[hint]     still a short clip; raise --hold or set "
              "--duration 10 for a smooth ~10 s sweep.")

    if shutil.which("ffmpeg"):
        _ffmpeg(seq, out_mp4, fps, bitrate)
    else:
        print("[warn] ffmpeg not on PATH "
              "(on ACRES: module load FFmpeg/4.4.2-GCCcore-11.3.0); "
              "using OpenCV mp4v fallback.")
        _opencv(seq, out_mp4, fps)


def _ffmpeg(seq, out_mp4: str, fps: int, bitrate: str):
    # ffmpeg's %0Nd reader needs a contiguous numbered sequence, so stage the
    # (possibly repeated) playback order as symlinks in a temp dir. Symlinks
    # keep it cheap — no copying of the PNGs, even for long holds.
    tmp = tempfile.mkdtemp(prefix="framestage_")
    try:
        for j, src in enumerate(seq):
            os.symlink(os.path.abspath(src), os.path.join(tmp, f"f_{j:06d}.png"))
        # pad filter forces even dimensions (required by yuv420p / most players).
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", os.path.join(tmp, "f_%06d.png"),
            "-c:v", "libx264",
            "-preset", "slow",
            "-b:v", bitrate,
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            out_mp4,
        ]
        print("[ffmpeg]   " + " ".join(cmd[:1]) + " ... -> " + out_mp4)
        subprocess.run(cmd, check=True)
        print(f"[saved]    {out_mp4}")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"[error] ffmpeg failed (exit {exc.returncode})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _opencv(seq, out_mp4: str, fps: int):
    try:
        import cv2
    except ImportError:
        sys.exit("[error] neither ffmpeg nor OpenCV available; cannot make video")
    first = cv2.imread(seq[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    for f in seq:
        img = cv2.imread(f)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()
    print(f"[saved]    {out_mp4}  (OpenCV mp4v)")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/coupling_sweep",
                    help="sweep root containing task_*/ and frames/ "
                         "(default: data/coupling_sweep)")
    ap.add_argument("--curve-out", default=None,
                    help="output PNG for the D_f-vs-K plot "
                         "(default: <root>/fractal_dimension_vs_K.png)")
    ap.add_argument("--video-out", default=None,
                    help="output mp4 (default: <root>/coupling_sweep.mp4)")
    ap.add_argument("--fps", type=int, default=60, help="video frame rate")
    ap.add_argument("--bitrate", default="20M",
                    help="ffmpeg target video bitrate (default: 20M)")
    ap.add_argument("--hold", type=int, default=1,
                    help="repeat each K-frame this many times so it lingers "
                         "(default: 1)")
    ap.add_argument("--duration", type=float, default=None,
                    help="target clip length in seconds; overrides --hold by "
                         "auto-computing the per-frame repeat (e.g. --duration 10)")
    ap.add_argument("--no-video", action="store_true",
                    help="only build the D_f curve, skip the mp4")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    curve_out = args.curve_out or os.path.join(root, "fractal_dimension_vs_K.png")
    csv_out = os.path.splitext(curve_out)[0] + ".csv"
    video_out = args.video_out or os.path.join(root, "coupling_sweep.mp4")

    print(f"[root]     {root}")
    K, Df, R2, skipped = harvest(root)
    if skipped:
        print(f"[warn] skipped {len(skipped)} dir(s) without readable basin_data.npz")
    plot_curve(K, Df, R2, curve_out, csv_out)

    if not args.no_video:
        compile_video(os.path.join(root, "frames"), video_out, args.fps,
                      args.bitrate, hold=args.hold, duration=args.duration)


if __name__ == "__main__":
    main()
