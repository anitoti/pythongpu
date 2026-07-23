from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pythongpu.pipeline.clv_topology import run_hr_clv_topology
from pythongpu.pipeline.hr_clv_cli import load_seeded_graph_laplacian
import scripts.seeded_random_graph_sweep as sweep


def test_seeded_graph_laplacian_wires_into_hr_clv_pipeline(tmp_path):
    """End-to-end smoke test: build one seeded null-model graph, load its
    Laplacian the way hr_clv_cli.py does, and run a (tiny) real CLV
    analysis on it -- confirms the wiring works, not just that each piece
    works in isolation."""
    import numpy as np
    g = sweep.build_graph("ba", seed=0)
    import networkx as nx
    adj = nx.to_numpy_array(g)
    npz_path = tmp_path / "seed000.npz"
    np.savez_compressed(npz_path, adjacency=adj, seed=np.array([0]))

    device = torch.device("cpu")
    L, N = load_seeded_graph_laplacian(str(npz_path), device=device)
    assert N == sweep.N_NODES == 83
    assert L.shape == (83, 83)

    summary = run_hr_clv_topology(
        L, N, coupling=0.5, steps=100, m=5, K=5, qr_interval=5,
        device=device, out_prefix=str(tmp_path / "clv_angles_"), label="ba_seed000",
    )
    assert "riddling" in summary
    assert summary["riddling"]["verdict"] in ("RIDDLED", "SYNCHRONISED")
    assert len(summary["lyapunov_exponents"]) == 5
