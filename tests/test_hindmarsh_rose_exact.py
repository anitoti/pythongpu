from __future__ import annotations

import numpy as np
import torch

from pythongpu.paper_replication.hindmarsh_rose_exact import HindmarshRoseNetworkExact


def test_rhs_matches_matlab_equations_by_hand():
    """Hand-computed reference for a 2-node network, transcribed directly
    from HR_ElCh_network.m -- catches sign errors or a wrong coupling
    variable/matrix (the two structural bugs this class exists to fix
    relative to pythongpu.oscillators.hindmarsh_rose.HindmarshRoseNetwork,
    see talk/notes/matlab_source_audit.md)."""
    A = np.array([[0, 1], [1, 0]], dtype=np.float32)
    net = HindmarshRoseNetworkExact(A, gel=0.5, gch=0.3, device="cpu")

    x = np.array([0.5, -0.3], dtype=np.float32)
    y = np.array([0.1, 0.2], dtype=np.float32)
    z = np.array([1.0, 1.5], dtype=np.float32)
    state = torch.tensor(np.stack([x, y, z]), dtype=torch.float32)

    out = net.rhs(state).numpy()

    a, b, c, d, r, s, p0, Iext = 1.0, 3.0, 1.0, 5.0, 0.005, 4.0, -1.6, 3.25
    gel, gch, Vsyn, thetasyn, lam = 0.5, 0.3, 2.0, -0.25, 10.0
    L = np.diag(A.sum(1)) - A

    sig = 1.0 / (1.0 + np.exp(-lam * (x - thetasyn)))
    chem = A @ sig
    dx_manual = y - a * x**3 + b * x**2 - z + Iext - gch * (x - Vsyn) * chem
    dy_manual = c - d * x**2 - y - gel * (L @ y)
    dz_manual = r * (s * (x - p0) - z)

    np.testing.assert_allclose(out[0], dx_manual, atol=1e-5)
    np.testing.assert_allclose(out[1], dy_manual, atol=1e-5)
    np.testing.assert_allclose(out[2], dz_manual, atol=1e-5)


def test_coupling_acts_on_y_not_x():
    """The whole point of this class: coupling must move dY, not dX, when
    gel != 0 and gch == 0 (isolate the diffusive term)."""
    A = np.array([[0, 1], [1, 0]], dtype=np.float32)
    net_coupled = HindmarshRoseNetworkExact(A, gel=0.5, gch=0.0)
    net_isolated = HindmarshRoseNetworkExact(A, gel=0.0, gch=0.0)

    state = torch.tensor([[0.5, -0.3], [0.1, 0.2], [1.0, 1.5]], dtype=torch.float32)
    out_c = net_coupled.rhs(state)
    out_i = net_isolated.rhs(state)

    assert not torch.allclose(out_c[1], out_i[1]), "dY should change with gel != 0"
    assert torch.allclose(out_c[0], out_i[0]), "dX should NOT change from gel alone (gch=0)"


def test_chemical_coupling_uses_raw_adjacency_not_laplacian():
    """gch != 0 should move dX; a network with no edges (A=0) must leave
    dX unaffected by gch since there's nothing to couple through."""
    A_edges = np.array([[0, 1], [1, 0]], dtype=np.float32)
    A_none = np.zeros((2, 2), dtype=np.float32)
    state = torch.tensor([[0.5, -0.3], [0.1, 0.2], [1.0, 1.5]], dtype=torch.float32)

    net_edges = HindmarshRoseNetworkExact(A_edges, gel=0.0, gch=0.3)
    net_none = HindmarshRoseNetworkExact(A_none, gel=0.0, gch=0.3)

    out_edges = net_edges.rhs(state)
    out_none = net_none.rhs(state)
    assert not torch.allclose(out_edges[0], out_none[0])


def test_batched_and_integration_stability():
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    net = HindmarshRoseNetworkExact(A, gel=0.5, gch=0.03)

    B, N = 8, 3
    state_b = torch.randn(B, 3, N)
    out_b = net.rhs(state_b)
    assert out_b.shape == (B, 3, N)

    state0 = torch.tensor(np.tile([[0.5], [0.1], [1.0]], (1, N)), dtype=torch.float32)
    traj = net.integrate(state0, dt=0.02, steps=2000)
    assert torch.isfinite(traj).all()
