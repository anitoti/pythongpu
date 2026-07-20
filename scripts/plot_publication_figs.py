#!/usr/bin/env python3
"""Aggregate CLV topology summaries and render publication figures.

The script scans a results tree containing per-coupling subdirectories with
`clv_topology_summary.json`, then produces:

1. A combined two-panel PNG with the full m=83 Lyapunov spectrum and the
   Kaplan–Yorke / burst-fraction trends.
2. Two single-panel companion PNGs for reuse in slides or manuscript
   assembly.

By default, unresolved Kaplan–Yorke points are shown as the lower bound 83,
with ceiling cases marked distinctly so the figure remains honest about the
missing portion of the spectrum.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PINK_SPECTRUM_COLORS = ["#f8bbd0", "#f48fb1", "#ec407a", "#ad1457"]
PINK_DKY_COLOR = "#b1126b"
PINK_BURST_COLOR = "#e91e63"
PINK_GRID = "#f3d7e3"
PINK_ZERO = "#d46a9a"


@dataclass(frozen=True)
class CLVSummary:
    coupling: float
    label: str
    spectrum: np.ndarray
    kaplan_yorke_dimension: float
    kaplan_yorke_is_ceiling: bool
    burst_fraction: float


def _parse_coupling_from_name(name: str) -> float | None:
    match = re.search(r"c(?P<whole>\d+)_(?P<frac>\d+)", name)
    if not match:
        return None
    return float(f"{match.group('whole')}.{match.group('frac')}")


def _load_summary(json_path: Path) -> CLVSummary:
    with json_path.open() as fh:
        payload = json.load(fh)

    coupling = payload.get("coupling")
    if coupling is None:
        coupling = _parse_coupling_from_name(json_path.parent.name)
    if coupling is None:
        raise ValueError(f"Unable to infer coupling from {json_path}")

    riddling = payload.get("riddling", {})
    return CLVSummary(
        coupling=float(coupling),
        label=payload.get("label", json_path.parent.name),
        spectrum=np.asarray(payload["lyapunov_exponents"], dtype=float),
        kaplan_yorke_dimension=float(payload["kaplan_yorke_dimension"]),
        kaplan_yorke_is_ceiling=bool(payload["kaplan_yorke_is_ceiling"]),
        burst_fraction=float(riddling["burst_fraction"]),
    )


def _collect_summaries(results_dir: Path) -> list[CLVSummary]:
    json_paths = sorted(results_dir.rglob("clv_topology_summary.json"))
    if not json_paths:
        raise FileNotFoundError(
            f"No clv_topology_summary.json files found under {results_dir}"
        )

    summaries = [_load_summary(path) for path in json_paths]
    summaries.sort(key=lambda item: (item.coupling, item.label))
    return summaries


def _configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 0.9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")


def _zero_crossings(x: np.ndarray, y: np.ndarray) -> list[tuple[float, float]]:
    crossings: list[tuple[float, float]] = []
    for idx in range(len(y) - 1):
        y0 = float(y[idx])
        y1 = float(y[idx + 1])
        if y0 == 0.0:
            crossings.append((float(x[idx]), float(idx + 1)))
            continue
        if y0 * y1 < 0.0:
            frac = y0 / (y0 - y1)
            x_cross = float(x[idx] + frac * (x[idx + 1] - x[idx]))
            index_cross = float((idx + 1) + frac)
            crossings.append((x_cross, index_cross))
    return crossings


def _plot_spectrum_panel(ax, summaries: Iterable[CLVSummary]) -> None:
    summaries = list(summaries)
    couplings = np.array([item.coupling for item in summaries], dtype=float)
    spectra = [item.spectrum for item in summaries]
    n_modes = max(len(spec) for spec in spectra)

    from matplotlib.colors import LinearSegmentedColormap

    discrete_cmap = LinearSegmentedColormap.from_list(
        "pink_bow",
        PINK_SPECTRUM_COLORS,
    )
    discrete_colors = [discrete_cmap(v) for v in np.linspace(0.1, 0.95, len(summaries))]

    for color, summary in zip(discrete_colors, summaries):
        x_values = np.arange(1, len(summary.spectrum) + 1)
        ax.plot(
            x_values,
            summary.spectrum,
            color=color,
            linewidth=1.6,
            alpha=0.95,
            label=fr"$c={summary.coupling:g}$",
        )
        for x_cross, idx_cross in _zero_crossings(x_values, summary.spectrum):
            ax.annotate(
                fr"$i={idx_cross:.2f}$",
                xy=(x_cross, 0.0),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                rotation=0,
                clip_on=True,
            )

    ax.axhline(0.0, color=PINK_ZERO, linewidth=0.9, linestyle="--", zorder=0)
    ax.set_xlim(1, n_modes)
    ax.set_xlabel("Lyapunov exponent index")
    ax.set_ylabel("Lyapunov exponent")
    ax.set_title("Full m=83 Lyapunov spectrum across coupling")
    ax.grid(True, axis="both", linewidth=0.4, alpha=0.35, color=PINK_GRID)
    ax.legend(loc="upper right", frameon=False, title="Coupling")


def _plot_trend_panel(ax, summaries: Iterable[CLVSummary]) -> None:
    summaries = list(summaries)
    couplings = np.array([item.coupling for item in summaries], dtype=float)
    dky_raw = np.array([item.kaplan_yorke_dimension for item in summaries], dtype=float)
    ceiling = np.array([item.kaplan_yorke_is_ceiling for item in summaries], dtype=bool)
    burst = np.array([item.burst_fraction for item in summaries], dtype=float)

    dky_plot = dky_raw.copy()
    dky_plot[ceiling] = 83.0

    ax.plot(
        couplings,
        dky_plot,
        color=PINK_DKY_COLOR,
        linewidth=1.8,
        marker="o",
        markersize=5.5,
        label="Kaplan–Yorke dimension",
        zorder=3,
    )

    if ceiling.any():
        ax.scatter(
            couplings[ceiling],
            dky_plot[ceiling],
            marker="^",
            s=70,
            facecolors="white",
            edgecolors=PINK_DKY_COLOR,
            linewidths=1.2,
            label="Ceiling cases shown at 83",
            zorder=4,
        )

    ax2 = ax.twinx()
    ax2.plot(
        couplings,
        burst,
        color=PINK_BURST_COLOR,
        linewidth=1.8,
        marker="s",
        markersize=5.0,
        label="Burst fraction",
        zorder=2,
    )

    ax.set_xlabel("Coupling strength")
    ax.set_ylabel("Kaplan–Yorke dimension $D_{KY}$")
    ax2.set_ylabel("Burst fraction")
    ax.set_title("Resolved dimension and riddling trend")
    ax.grid(True, axis="both", linewidth=0.4, alpha=0.35, color=PINK_GRID)

    upper = max(83.0, float(np.max(dky_plot)))
    ax.set_ylim(0.0, upper * 1.06)
    ax2.set_ylim(0.0, 1.0)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="lower left",
        frameon=True,
        framealpha=0.0,
        facecolor="none",
        edgecolor="none",
    )


def build_figures(results_dir: Path, outdir: Path) -> list[Path]:
    summaries = _collect_summaries(results_dir)
    _configure_matplotlib()

    import matplotlib.pyplot as plt

    coupling_values = ", ".join(f"{item.coupling:g}" for item in summaries)
    print(f"[data] loaded {len(summaries)} CLV summaries from {results_dir}")
    print(f"[data] coupling points: {coupling_values}")

    generated: list[Path] = []

    # Combined panel figure.
    fig, (ax_spec, ax_trend) = plt.subplots(
        1,
        2,
        figsize=(13.4, 5.0),
        constrained_layout=True,
    )
    _plot_spectrum_panel(ax_spec, summaries)
    _plot_trend_panel(ax_trend, summaries)

    combined_path = outdir / "plot_publication_figs__clv_riddling_summary.png"
    _save_figure(fig, combined_path)
    plt.close(fig)
    generated.append(combined_path)

    # Single-panel spectrum figure for direct reuse.
    fig_spec, ax_spec = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    _plot_spectrum_panel(ax_spec, summaries)
    spectrum_path = outdir / "plot_publication_figs__clv_spectrum.png"
    _save_figure(fig_spec, spectrum_path)
    plt.close(fig_spec)
    generated.append(spectrum_path)

    # Single-panel trend figure for direct reuse.
    fig_trend, ax_trend = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    _plot_trend_panel(ax_trend, summaries)
    trend_path = outdir / "plot_publication_figs__clv_trends.png"
    _save_figure(fig_trend, trend_path)
    plt.close(fig_trend)
    generated.append(trend_path)

    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate clv_topology_summary.json files and render publication-"
            "quality CLV figures."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "clv_results_final",
        help="Directory containing clv_c*/clv_topology_summary.json files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "data" / "derivatives",
        help="Directory where the PNG figures will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = build_figures(args.results_dir, args.outdir)
    for path in generated:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()