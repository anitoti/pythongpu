#!/usr/bin/env python3
"""
regime_scan.py — find the interesting regime cheaply, WITHOUT clustering.

Why this exists
---------------
Everything downstream of k-means (D_f, gamma, basin entropy, Wada) inherits the
same disease: k-means always returns k groups, so on featureless data it invents
basins and D_f -> 2 regardless. Scanning D_f or gamma to "find where structure
is" therefore cannot work — the observable is computed from the broken step.

This scans observables that need NO global clustering, so they can be trusted to
locate the regime worth spending GPU-hours on. Only then do you run the
expensive, gated basin mapping there.

Observables per (coupling K, node pair)
---------------------------------------
1. sync_err  — time-averaged cross-node std of x. ~0 means the network
   synchronised; large means it did not. A pure diagnostic of the dynamical
   regime.

2. f(eps)    — FINAL-STATE SENSITIVITY, the Grebogi-McDonald-Ott-Yorke
   construction. Take an initial condition, perturb it by eps, integrate both,
   and ask whether they end on the same attractor. f(eps) is the fraction that
   do NOT. This needs only a LOCAL, PAIRWISE decision ("same place or not?") —
   never a global "how many groups" — which is exactly why it survives where
   k-means fails. It discriminates every case on its own:

       f(eps) ~ 0 for all eps      ->  a single attractor: nothing to find
       f(eps) ∝ eps    (gamma=1)   ->  several basins, smooth boundary
       f(eps) ∝ eps^gamma, 0<g<1   ->  several basins, FRACTAL boundary  <-- target
       f(eps) ~ const  (gamma~0)   ->  riddled, or the labelling is noise

   and D = d - gamma (GMOY 1983) recovers the boundary dimension without ever
   box-counting a clustered map.

   "Same attractor" is decided by comparing long-time descriptors (per-node mean
   x, std x, mean |x| — averages, hence constant on one attractor) against a
   threshold CALIBRATED from the data itself: pairs perturbed by the smallest
   eps are on the same attractor by construction, so their descriptor distance
   sets the scale.

3. desc_spread — spread of the long-time descriptors across independent ICs,
   in units of that same-attractor scale. Large spread with small f(eps) would
   mean several attractors reached from far-apart ICs but not from nearby ones.

The point of the (K x node-pair) grid
-------------------------------------
Basin geometry has been studied here as a function of coupling ALONE. But which
nodes you perturb is an independent axis: perturbing a degree-2 leaf barely
propagates, while a degree-44 hub drives half the network. This scans both.

Usage
-----
    python3 -m pythongpu.pipeline.regime_scan --csv scan.csv
    python3 -m pythongpu.pipeline.regime_scan \
        --k-list 0.05 0.1 0.2 0.4 0.8 1.6 --pairs 74,73 74,82 16,43 28,79
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from pythongpu.networks.static_adjacency import load_dti_laplacian
from pythongpu.pipeline.lorenz_sweep import LorenzParams, rk4_step_batched


def _descriptors(x0: torch.Tensor, L, p, record_stride: int) -> torch.Tensor:
    """Integrate through the transient, then return long-time per-node stats."""
    x = x0
    for _ in range(p.steps_transient):
        x = rk4_step_batched(x, L, p)
    B, N, _ = x.shape
    s1 = torch.zeros((B, N), device=x.device)
    s2 = torch.zeros((B, N), device=x.device)
    sa = torch.zeros((B, N), device=x.device)
    sync = 0.0
    cnt = 0
    for step in range(p.steps_record):
        x = rk4_step_batched(x, L, p)
        if step % record_stride == 0:
            xc = x[..., 0]
            s1 += xc
            s2 += xc * xc
            sa += xc.abs()
            sync += float(xc.std(dim=1).mean())
            cnt += 1
    mean = s1 / cnt
    std = torch.sqrt(torch.clamp(s2 / cnt - mean * mean, min=0.0))
    desc = torch.cat([mean, std, sa / cnt], dim=1)
    return desc, sync / cnt


def _base_state(B: int, N: int, device, scatter: bool, rng_seed: int):
    g = torch.Generator(device="cpu").manual_seed(rng_seed)
    if scatter:
        # A genuine slice of state space: every node independent.
        x = torch.randn((B, N, 3), generator=g).to(device) * 8.0
    else:
        # The pipeline's historical slice: all nodes near (1,1,1), i.e. nearly
        # synchronous, with only the two probe nodes moved.
        x = torch.ones((B, N, 3)) + 0.05 * torch.randn((B, N, 3), generator=g)
        x = x.to(device)
    return x


def scan_one(L, N, K: float, node_a: int, node_b: int, args, device) -> dict:
    p = LorenzParams(coupling=K, rho=args.rho, dt=0.05,
                     t_transient=args.t_transient, tmax=args.tmax,
                     slice_node_x=node_a, slice_node_y=node_b, n_osc=N)
    B = args.n_ic
    eps_list = sorted(args.eps_list)

    # Reference ICs, spread over the probe plane.
    x0 = _base_state(B, N, device, args.scatter, args.seed)
    gx = torch.empty(B).uniform_(args.grid_lo, args.grid_hi, generator=torch.Generator().manual_seed(args.seed + 1))
    gy = torch.empty(B).uniform_(args.grid_lo, args.grid_hi, generator=torch.Generator().manual_seed(args.seed + 2))
    x0[:, node_a, 0] = gx.to(device)
    x0[:, node_b, 0] = gy.to(device)

    d_ref, sync_err = _descriptors(x0.clone(), L, p, args.record_stride)

    # Perturbed copies: same ICs nudged by eps along the probe axis.
    dists = {}
    gpert = torch.Generator(device="cpu").manual_seed(args.seed + 7)
    unit = torch.randn(x0.shape, generator=gpert)
    unit = (unit / torch.linalg.norm(unit.reshape(B, -1), dim=1)
            .reshape(B, 1, 1)).to(device)
    for eps in eps_list:
        xp = x0.clone()
        if args.perturb_all:
            # Proper GMOY: displace the WHOLE state by eps in a random direction.
            # Perturbing only the probe node's x measures the sensitivity of a
            # coordinate that may sit deep inside one basin, while the rest of the
            # state — held fixed — is the part actually near a boundary.
            xp = xp + eps * unit
        else:
            xp[:, node_a, 0] += eps
        d_eps, _ = _descriptors(xp, L, p, args.record_stride)
        dists[eps] = torch.linalg.norm(d_ref - d_eps, dim=1).cpu().numpy()

    # Calibrate "same attractor" from the SMALLEST eps: those pairs are on the
    # same attractor by construction, so their descriptor distance is the
    # finite-averaging noise floor. Anything well beyond it is a real change of
    # final state.
    floor = dists[eps_list[0]]
    theta = float(np.percentile(floor, 99)) * args.theta_mult
    theta = max(theta, 1e-9)

    f_of_eps = {eps: float(np.mean(dists[eps] > theta)) for eps in eps_list}

    # gamma from a log-log fit of f(eps) ~ eps^gamma, using only eps where f is
    # in (0,1) — a saturated or empty f carries no exponent.
    usable = [(e, f) for e, f in f_of_eps.items() if 0.0 < f < 1.0]
    if len(usable) >= 3:
        e_arr = np.log(np.array([u[0] for u in usable]))
        f_arr = np.log(np.array([u[1] for u in usable]))
        gamma = float(np.polyfit(e_arr, f_arr, 1)[0])
        resid = np.polyval(np.polyfit(e_arr, f_arr, 1), e_arr) - f_arr
        r2 = float(1.0 - resid.var() / (f_arr.var() + 1e-12))
    else:
        gamma, r2 = float("nan"), float("nan")

    # Spread of independent ICs' descriptors, in units of the same-attractor floor.
    spread = float(np.median(torch.cdist(d_ref, d_ref).cpu().numpy())) / theta

    f_max = max(f_of_eps.values())
    n_events = int(round(f_max * B))
    # POWER CHECK. f_max ~ 0 has two very different causes: genuinely one
    # attractor, or a probe line that simply crossed almost no boundaries. With
    # only a handful of flip events we cannot tell them apart, and calling that
    # "single attractor" is a confident claim built on nothing. (The rho=24
    # bistable control failed exactly here: 2 events out of 256 ICs were reported
    # as "SINGLE ATTRACTOR" for a system that is provably bistable.)
    if n_events < args.min_events:
        verdict = (f"INSUFFICIENT POWER — only {n_events} flip events "
                   f"(need >={args.min_events}); f(eps) unmeasurable here. "
                   f"NOT evidence of a single attractor.")
    elif f_max < args.f_floor:
        verdict = "SINGLE ATTRACTOR (no final-state sensitivity)"
    elif not np.isnan(gamma) and 0.05 < gamma < 0.95:
        verdict = f"FRACTAL BOUNDARY candidate (gamma={gamma:.2f}, D={2-gamma:.2f})"
    elif not np.isnan(gamma) and gamma >= 0.95:
        verdict = "smooth boundary (gamma~1)"
    else:
        verdict = "riddled / saturated — f(eps) flat, gamma~0"

    return dict(K=K, node_a=node_a, node_b=node_b, sync_err=sync_err,
                theta=theta, desc_spread=spread, gamma=gamma, gamma_r2=r2,
                f_max=f_max, n_events=n_events, n_ic=B, verdict=verdict,
                **{f"f_eps_{e:g}": v for e, v in f_of_eps.items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k-list", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
                    help="couplings to scan — deliberately spanning decades, not "
                         "a narrow window")
    ap.add_argument("--pairs", nargs="+",
                    default=["74,73", "74,82", "16,43", "28,79"],
                    help="probe node pairs 'a,b'. Defaults span topological "
                         "roles in DTI_A: hub+leaf, hub+hub, farthest apart, and "
                         "the historical (average,average) pair.")
    ap.add_argument("--eps-list", type=float, nargs="+",
                    default=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
                    help="perturbation sizes for f(eps)")
    ap.add_argument("--n-ic", type=int, default=256)
    ap.add_argument("--tmax", type=float, default=120.0)
    ap.add_argument("--t-transient", type=float, default=60.0)
    ap.add_argument("--record-stride", type=int, default=5)
    ap.add_argument("--grid-lo", type=float, default=-9.0)
    ap.add_argument("--grid-hi", type=float, default=9.0)
    ap.add_argument("--scatter", action="store_true",
                    help="scatter ALL nodes instead of the historical "
                         "near-synchronous base state (a genuine state-space slice)")
    ap.add_argument("--theta-mult", type=float, default=3.0,
                    help="multiple of the same-attractor noise floor above which "
                         "two ICs count as reaching different final states")
    ap.add_argument("--perturb-all", action="store_true",
                    help="displace the entire state by eps in a random direction "
                         "(the standard GMOY construction) instead of nudging only "
                         "the probe node's x-coordinate.")
    ap.add_argument("--min-events", type=int, default=20,
                    help="minimum number of final-state flips required before any "
                         "verdict is issued. Below this the probe has no power and "
                         "f(eps)~0 means 'we did not sample boundaries', NOT 'one "
                         "attractor'.")
    ap.add_argument("--f-floor", type=float, default=0.02,
                    help="max f(eps) below which we call it a single attractor")
    ap.add_argument("--rho", type=float, default=28.0,
                    help="Lorenz rho. 28 = chaotic. VALIDATION: rho=24 with K=0 "
                         "makes each oscillator BISTABLE (two stable fixed points "
                         "C+/C- with a known fractal basin boundary) — the tool "
                         "must detect f(eps)>0 there, or it cannot be trusted.")
    ap.add_argument("--dti-path", default="data/DTI-og.mat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, N = load_dti_laplacian(args.dti_path, device)
    pairs = [tuple(int(v) for v in s.split(",")) for s in args.pairs]

    print(f"[device]   {device}")
    print(f"[scan]     {len(args.k_list)} couplings x {len(pairs)} node pairs, "
          f"{args.n_ic} ICs, eps={args.eps_list}")
    print(f"[slice]    {'scattered (all nodes)' if args.scatter else 'near-synchronous base (historical)'}")

    rows = []
    for a, b in pairs:
        for K in args.k_list:
            r = scan_one(L, N, K, a, b, args, device)
            rows.append(r)
            g = f"{r['gamma']:.2f}" if not np.isnan(r["gamma"]) else " -- "
            print(f"  ({a:2d},{b:2d}) K={K:<5g} sync={r['sync_err']:5.2f} "
                  f"f_max={r['f_max']:5.3f} ({r['n_events']}/{r['n_ic']} events) "
                  f"gamma={g}\n           {r['verdict']}")
            if args.csv:
                Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
                with open(args.csv, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
    if args.csv:
        print(f"[csv]      {len(rows)} rows -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
