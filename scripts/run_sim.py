#!/usr/bin/env python3
import argparse

import torch
from pythongpu.gnm_random_graph import generate_gnm_laplacian
from src.pythongpu.simulate_network import simulate


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a Rössler simulation on an ER Laplacian.")
    ap.add_argument("--nodes", type=int, default=100, help="Number of nodes in the ER graph.")
    ap.add_argument("--prob", type=float, default=0.05, help="Edge probability for the ER graph.")
    ap.add_argument("--steps", type=int, default=5000, help="Number of simulation steps.")
    ap.add_argument("--dt", type=float, default=0.01, help="Integration time step.")
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device to run on.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    L = generate_gnm_laplacian(args.nodes, args.prob, device=device)
    final_state = simulate(L, steps=args.steps, dt=args.dt)

    print("simulation finished on:", final_state.device)
    print("output shape:", final_state.shape)


if __name__ == "__main__":
    main()
