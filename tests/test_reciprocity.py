"""Reciprocity of the complex-step-differentiated operator.

Dokumaci (*Duct Acoustics*, CUP 2021, Sec. 5.5) gives reciprocity of a two-port as
``det(T) = 1`` in pressure/volume-velocity variables.  This is a *physical* invariance --
unlike the edge-arrow reversal checked in ``test_invariances``, which is bookkeeping -- and it
is exactly the property a subtly wrong derivative breaks, so it is a searching test of an
operator assembled by complex-stepping the mean-flow residuals.

Nefes reports the transfer matrix in the ``primitive`` basis ``(p'/rho c, u')``.  Converting to
``(p, U = A u)`` scales the determinant by the ports' ``rho c A``, so the invariant checked here is

    | det(T_prim) * (z_b A_b) / (z_a A_a) | = 1,   z = rho c.

Reciprocity is *not* losslessness: a passive but dissipative element is still reciprocal, which
``test_reciprocity_survives_dissipation`` pins down.  The 2x2 acoustic reduction is only
meaningful where entropy does not feed the ``(p, u)`` rows, so each case asserts that leak is
negligible before reading the determinant.
"""

import numpy as np
import pytest

from nefes.assembly.recover import ES_AREA, ES_C, ES_RHO
from nefes.elements import catalog as cat
from nefes.perturbation import perturbation_response
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.thermo.configure import perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
CFG = perfect_gas(R_AIR, GAMMA)
FREQS = np.linspace(20.0, 400.0, 13)
FULL = ("acoustic", "entropy")
P0, T0 = 101325.0, 300.0
AREA = 0.05


def _solve(net, edges, mdot_ref):
    prob = build_problem(CFG, net, edges, mdot_ref, P0, CP * T0)
    res = solve(prob)
    assert res.converged
    return prob, res


def _det_pU(prob, res, a, b, edges):
    """(det(T) in (p, U) variables, entropy leak into the (p, u) rows)."""
    est = states_table(prob, res.x)
    resp = perturbation_response(prob, res.x, FREQS, excite=FULL)
    T3 = resp.transfer_matrix(a, b, basis="primitive")
    leak = max(np.abs(T3[:, 0, 2]).max(), np.abs(T3[:, 1, 2]).max())

    def zA(e):
        return float(est[ES_RHO, e]) * float(est[ES_C, e]) * float(est[ES_AREA, e])

    return np.linalg.det(T3[:, :2, :2]) * zA(b) / zA(a), leak


@pytest.mark.parametrize("pt_in", [101335.0, 108000.0])
def test_uniform_duct_is_reciprocal(pt_in):
    """|det(T)| = 1 for a lossless duct, at low and at appreciable mean Mach."""
    net = [cat.total_pressure_inlet(pt_in, T0), cat.duct(1.0), cat.pressure_outlet(P0, T0)]
    edges = [(0, 1, AREA), (1, 2, AREA)]
    prob, res = _solve(net, edges, 0.5 if pt_in < 105000.0 else 10.0)
    det, leak = _det_pU(prob, res, 0, 1, edges)
    assert leak < 1e-12  # entropy decouples exactly on a uniform duct
    assert np.allclose(np.abs(det), 1.0, atol=1e-12)


def test_uniform_duct_determinant_matches_the_closed_form():
    """det(T) is not merely unit-modulus: it is the convective phase exp(2 i Omega).

    For a uniform duct in (p/rho c, u), T = e^{i Omega} [[cos K, i sin K], [i sin K, cos K]]
    with Omega = k M L / (1 - M^2), so det(T) = e^{2 i Omega}.
    """
    from nefes.assembly.recover import ES_M

    length = 1.0
    net = [cat.total_pressure_inlet(101335.0, T0), cat.duct(length), cat.pressure_outlet(P0, T0)]
    edges = [(0, 1, AREA), (1, 2, AREA)]
    prob, res = _solve(net, edges, 0.5)
    est = states_table(prob, res.x)
    mach, c = float(est[ES_M, 0]), float(est[ES_C, 0])
    det, _ = _det_pU(prob, res, 0, 1, edges)
    k = 2.0 * np.pi * FREQS / c
    expected = np.exp(2j * k * mach * length / (1.0 - mach**2))
    assert np.allclose(det, expected, atol=1e-5)


def test_area_change_carries_the_volume_velocity_normalisation():
    """Across an area change the (p, u) determinant picks up the area ratio; (p, U) does not."""
    net = [
        cat.total_pressure_inlet(101335.0, T0),
        cat.duct(0.7),
        cat.isentropic_area_change(),
        cat.duct(1.1),
        cat.pressure_outlet(P0, T0),
    ]
    edges = [(0, 1, AREA), (1, 2, AREA), (2, 3, 0.03), (3, 4, 0.03)]
    prob, res = _solve(net, edges, 0.5)
    det, leak = _det_pU(prob, res, 0, 3, edges)
    assert leak < 1e-3  # near-quiescent: entropy barely touches the acoustic rows
    assert np.allclose(np.abs(det), 1.0, rtol=1e-3)


def test_reciprocity_survives_dissipation():
    """A lossy orifice is still reciprocal -- reciprocity is not an energy statement."""
    net = [
        cat.total_pressure_inlet(108000.0, T0),
        cat.duct(0.7),
        cat.orifice(0.6),
        cat.duct(1.1),
        cat.pressure_outlet(P0, T0),
    ]
    edges = [(0, 1, AREA), (1, 2, AREA), (2, 3, AREA), (3, 4, AREA)]
    prob, res = _solve(net, edges, 10.0)
    det, leak = _det_pU(prob, res, 0, 3, edges)
    assert leak < 1e-6
    assert np.allclose(np.abs(det), 1.0, atol=1e-7)
