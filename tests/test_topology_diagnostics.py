from __future__ import annotations

import numpy as np
import torch

from pythongpu.processing.entropic_regression import (
    entropic_regression,
    fuse_structural_functional,
)
from pythongpu.pipeline.clv_diagnostics import (
    lyapunov_spectrum,
    kaplan_yorke_dimension,
    detect_riddling_kmeans,
)
from pythongpu.networks.random_graphs import match_baselines_from_adjacency


def test_entropic_regression_kills_indirect_edge():
    """ER's backward pass must recover the chain 0->1->2 and drop the indirect
    edge 0->2 that a forward-only oCSE sweep would leave behind."""
    rng = np.random.default_rng(0)
    T = 4000
    x0 = rng.standard_normal(T)
    x1 = np.zeros(T)
    x2 = np.zeros(T)
    x3 = rng.standard_normal(T)  # independent distractor
    for t in range(1, T):
        x1[t] = 0.85 * x0[t - 1] + 0.3 * rng.standard_normal()
        x2[t] = 0.85 * x1[t - 1] + 0.3 * rng.standard_normal()
    ts = np.column_stack([x0, x1, x2, x3])
    ts = (ts - ts.mean(0)) / ts.std(0)

    adj = entropic_regression(ts, max_lag=1, alpha=0.01, n_jobs=1)
    edges = {(s, t) for s, t in zip(*np.where(adj > 0))}
    assert (0, 1) in edges
    assert (1, 2) in edges
    assert (0, 2) not in edges          # backward pass removed indirect edge
    assert not any(3 in e for e in edges)  # distractor stays disconnected


def test_fuse_structural_gates_functional():
    F = np.array([[0, 1.0, 2.0], [0, 0, 3.0], [0, 0, 0]])
    S = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # no 0-2 wire
    gated = fuse_structural_functional(F, S, mode="gate")
    assert gated[0, 1] == 1.0
    assert gated[0, 2] == 0.0           # functional edge with no structural substrate removed
    assert gated[1, 2] == 3.0


def test_kaplan_yorke_lorenz_value():
    """Canonical Lorenz spectrum gives the textbook D_KY ~ 2.06."""
    lam = np.array([0.906, 0.0, -14.572])
    d = kaplan_yorke_dimension(lam)
    assert 2.0 < d < 2.12


def test_kaplan_yorke_fixed_point_is_zero():
    assert kaplan_yorke_dimension(np.array([-0.1, -1.0, -2.0])) == 0.0


def test_kaplan_yorke_ceiling_when_sum_stays_positive():
    """If the running exponent sum never crosses zero within the computed
    spectrum, D_KY is pinned at the ceiling m (signals 'compute more CLVs')."""
    lam = np.array([0.4, 0.3, -0.1, -0.2])  # cumulative sum stays > 0 through all 4
    assert kaplan_yorke_dimension(lam) == 4.0
    # a spectrum that DOES cross resolves to a fractional value < m
    lam2 = np.array([0.4, 0.1, -0.9])       # crosses between index 2 and 3
    assert 2.0 < kaplan_yorke_dimension(lam2) < 3.0


def test_lyapunov_spectrum_from_R_diagonals():
    targets = np.array([0.5, 0.0, -1.0])
    tau = 10 * 0.01
    g = np.exp(targets * tau)
    R = torch.diag(torch.tensor(g, dtype=torch.float16))
    R_list = [R.clone() for _ in range(200)]
    lam = lyapunov_spectrum(R_list, qr_interval=10, dt=0.01, discard_frac=0.1)
    assert np.allclose(lam, targets, atol=0.05)


def test_riddling_detector_bimodal_vs_unimodal():
    rng = np.random.default_rng(1)
    bimodal = np.concatenate([rng.normal(0.02, 0.01, 900), rng.normal(0.9, 0.05, 100)])
    unimodal = np.abs(rng.normal(0.02, 0.01, 1000))
    assert detect_riddling_kmeans(bimodal)["verdict"] == "RIDDLED"
    assert detect_riddling_kmeans(unimodal)["verdict"] == "SYNCHRONISED"


def test_load_adjacency_handles_both_csv_formats(tmp_path):
    """The baseline loader must return a square matrix whether the CSV has the
    integer-name header row our writers emit (to_csv index=False) or is bare."""
    import pandas as pd
    from pythongpu.pipeline.baseline_models import _load_empirical_adjacency

    A = (np.random.default_rng(0).random((10, 10)) > 0.6).astype(float)
    np.fill_diagonal(A, 0)
    hdr = tmp_path / "hdr.csv"
    bare = tmp_path / "bare.csv"
    pd.DataFrame(A).to_csv(hdr, index=False)          # entropic_regression format
    np.savetxt(bare, A, delimiter=",")                # headerless
    for path in (hdr, bare):
        M = _load_empirical_adjacency(None, str(path))
        assert M.shape == (10, 10)
        assert np.allclose(M, A)


def test_run_clv_topology_smoke():
    """The shared CLV driver returns a well-formed summary on a small graph."""
    import networkx as nx
    import torch
    from pythongpu.utils import get_laplacian
    from pythongpu.pipeline.clv_topology import run_clv_topology

    g = nx.gnm_random_graph(8, 14, seed=0)
    L = get_laplacian(nx.to_numpy_array(g), device=torch.device("cpu"))
    s = run_clv_topology(
        L, 8, coupling=0.1, steps=60, m=4, K=4, qr_interval=5,
        device=torch.device("cpu"), out_prefix="/tmp/_clv_smoke_", label="smoke",
    )
    assert s["n_nodes"] == 8 and s["n_clvs_computed"] == 4
    assert len(s["lyapunov_exponents"]) == 4
    assert s["lyapunov_exponents"] == sorted(s["lyapunov_exponents"], reverse=True)
    assert s["riddling"]["verdict"] in ("RIDDLED", "SYNCHRONISED", "INSUFFICIENT_DATA")


def test_baselines_match_empirical_connectome():
    rng = np.random.default_rng(2)
    emp = np.zeros((30, 30))
    for j in range(1, 15):
        emp[0, j] = rng.random()          # hub
    for _ in range(40):
        i, j = rng.integers(0, 30, 2)
        if i != j:
            emp[i, j] = rng.random()
    res = match_baselines_from_adjacency(emp, seed=1)
    assert res["er"]["stats"]["n_edges"] == res["empirical"]["stats"]["n_edges"]
    assert res["er"]["stats"]["n_nodes"] == 30
    # preferential attachment => heavier-tailed degree distribution than ER
    assert (res["ba"]["stats"]["degree_heterogeneity"]
            > res["er"]["stats"]["degree_heterogeneity"])
    assert res["empirical"]["L"].shape == (30, 30)
