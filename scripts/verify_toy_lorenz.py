#!/usr/bin/env python3
"""Verify the shared CLV/Kaplan--Yorke pipeline on Lorenz-63.

This runs the standard three-dimensional Lorenz attractor
(`sigma=10`, `rho=28`, `beta=8/3`) through the same Ginelli/CLV machinery used
for the network experiments, but with a single node and zero coupling. The
script prints a comparison table against the common benchmark spectrum
``[0.90, 0.00, -14.57]`` and saves a high-resolution convergence plot of the
running Lyapunov exponent estimates over time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pythongpu.oscillators.lorenz import LorenzNetwork
from pythongpu.pipeline.clv_diagnostics import (
    CLVCalculator,
    kaplan_yorke_dimension,
    lyapunov_spectrum,
)
from pythongpu.pipeline.clv_topology import lorenz_clv_closures


BENCHMARK = np.array([0.90, 0.00, -14.57], dtype=np.float64)


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
    })


def _lorenz63_pipeline(
    *,
    steps: int,
    dt: float,
    qr_interval: int,
    discard_frac: float,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    L = torch.zeros((1, 1), dtype=torch.float32, device=device)
    lor = LorenzNetwork(L, sigma=10.0, rho=28.0, beta=8.0 / 3.0, coupling=0.0, dt=dt, device=device)
    rhs_flat, jac_flat = lorenz_clv_closures(lor, 1, device)
    calculator = CLVCalculator(
        rhs_fn=rhs_flat,
        jac_fn=jac_flat,
        n=3,
        dt=dt,
        device=device,
        qr_interval=qr_interval,
    )

    torch.manual_seed(seed)
    initial_state = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    _q_list, r_half_list, qr_steps = calculator.run_forward(
        initial_state=initial_state,
        total_steps=steps,
        m=3,
    )

    final_exponents = lyapunov_spectrum(
        r_half_list,
        qr_interval=qr_interval,
        dt=dt,
        discard_frac=discard_frac,
    )

    start = int(discard_frac * len(r_half_list))
    tau = qr_interval * dt
    log_diag = []
    for r_half in r_half_list[start:]:
        diag = torch.abs(torch.diag(r_half.to(dtype=torch.float32))).cpu().numpy()
        diag = np.where(diag < 1e-12, 1e-12, diag)
        log_diag.append(np.log(diag))
    log_diag = np.asarray(log_diag, dtype=np.float64)
    if log_diag.size == 0:
        raise ValueError("No QR samples remain after discard_frac; lower discard_frac or increase steps.")

    running = np.cumsum(log_diag, axis=0) / (np.arange(1, log_diag.shape[0] + 1)[:, None] * tau)
    times = np.asarray(qr_steps[start:], dtype=np.float64) * dt
    d_ky = kaplan_yorke_dimension(final_exponents)
    return final_exponents, running, d_ky, times


def _format_table(exponents: np.ndarray, benchmark: np.ndarray) -> str:
    lines = []
    header = f"{'mode':<6s}{'computed':>12s}{'benchmark':>12s}{'abs err':>12s}{'rel err':>12s}"
    lines.append(header)
    lines.append("-" * len(header))
    for idx, (comp, ref) in enumerate(zip(exponents, benchmark), start=1):
        abs_err = abs(comp - ref)
        rel_err = abs_err / abs(ref) if abs(ref) > 1e-12 else float("nan")
        rel_str = f"{'n/a':>12s}" if np.isnan(rel_err) else f"{rel_err:12.4e}"
        lines.append(
            f"{idx:<6d}{comp:12.5f}{ref:12.5f}{abs_err:12.5f}{rel_str}"
        )
    return "\n".join(lines)


def _save_outputs(
    outdir: Path,
    exponents: np.ndarray,
    benchmark: np.ndarray,
    d_ky: float,
    times: np.ndarray,
    running: np.ndarray,
) -> tuple[Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    table_text = _format_table(exponents, benchmark)
    summary_txt = outdir / "verify_toy_lorenz_summary.txt"
    summary_json = outdir / "verify_toy_lorenz_summary.json"
    figure_path = outdir / "verify_toy_lorenz_convergence.png"
    summary_txt.write_text(
        table_text
        + "\n\n"
        + f"D_KY = {d_ky:.5f}\n"
        + f"benchmark = {benchmark.tolist()}\n"
    )
    summary_json.write_text(
        json.dumps(
            {
                "benchmark": benchmark.tolist(),
                "kaplan_yorke_dimension": float(d_ky),
                "final_exponents": exponents.tolist(),
                "time_samples": times.tolist(),
            },
            indent=2,
        )
    )
    return summary_txt, summary_json, figure_path


def _plot_convergence(times: np.ndarray, running: np.ndarray, benchmark: np.ndarray, output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = ["#f48fb1", "#d81b60", "#8e24aa"]
    labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]

    fig, ax = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)
    for idx, (color, label) in enumerate(zip(colors, labels)):
        ax.plot(
            times,
            running[:, idx],
            color=color,
            linewidth=1.8,
            label=f"{label} computed",
        )
        ax.axhline(
            benchmark[idx],
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=f"{label} benchmark",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Lyapunov exponent")
    ax.set_title("Lorenz-63 exponent convergence")
    ax.grid(True, linewidth=0.4, alpha=0.35, color="#f3d7e3")
    ax.legend(loc="best", frameon=True, framealpha=0.9, facecolor="white")
    fig.savefig(output, dpi=300, format="png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the shared CLV/Kaplan-Yorke pipeline on Lorenz-63."
    )
    parser.add_argument("--steps", type=int, default=30000, help="RK4 steps to integrate.")
    parser.add_argument("--dt", type=float, default=0.01, help="Integration time step.")
    parser.add_argument("--qr-interval", type=int, default=10, help="QR orthonormalization interval.")
    parser.add_argument("--discard-frac", type=float, default=0.1, help="Fraction of QR samples to discard as transient.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (cpu or cuda).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for determinism.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "data" / "derivatives",
        help="Directory where outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_matplotlib()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    exponents, running, d_ky, times = _lorenz63_pipeline(
        steps=args.steps,
        dt=args.dt,
        qr_interval=args.qr_interval,
        discard_frac=args.discard_frac,
        device=device,
        seed=args.seed,
    )
    summary_txt, summary_json, figure_path = _save_outputs(
        args.outdir, exponents, BENCHMARK, d_ky, times, running
    )
    _plot_convergence(times, running, BENCHMARK, figure_path)

    print(_format_table(exponents, BENCHMARK))
    print(f"\nD_KY = {d_ky:.5f}")
    print(f"[saved] {summary_txt}")
    print(f"[saved] {summary_json}")
    print(f"[saved] {figure_path}")


if __name__ == "__main__":
    main()
