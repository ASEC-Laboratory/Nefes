"""Transmission loss is a power ratio, not an amplitude ratio.

The scattering matrix gives the forward transmission coefficient ``tau`` with the anechoic
condition ``r_out = 0`` built in, so ``-20 log10|tau|`` is right whenever the two ports carry
the same ``rho c (1+M)^2 A``.  When they do not, the missing factor is Dokumaci's ``C_TL``
(*Duct Acoustics*, CUP 2021, Eq. 5.4) and the bare form is wrong -- badly enough to report a
*negative* transmission loss for a lossless contraction.

Every transmission-loss case shipped in this repository is an equal-port case, so the
correction is identically zero there and no published number moves; these tests pin the
general behaviour that the shipped cases cannot exercise.
"""

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.perturbation import perturbation_response, transmission_loss
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.thermo.configure import perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
CFG = perfect_gas(R_AIR, GAMMA)
FREQS = np.linspace(50.0, 500.0, 11)
P0, T0 = 101325.0, 300.0


def _area_change(area_in, area_out, pt_in=P0):
    """Quiescent duct - isentropic area change - duct; returns (response, e_in, e_out)."""
    net = [
        cat.total_pressure_inlet(pt_in, T0),
        cat.duct(0.6),
        cat.isentropic_area_change(),
        cat.duct(0.9),
        cat.pressure_outlet(P0, T0),
    ]
    edges = [(0, 1, area_in), (1, 2, area_in), (2, 3, area_out), (3, 4, area_out)]
    prob = build_problem(CFG, net, edges, 0.5, P0, CP * T0)
    res = solve(prob)
    assert res.converged
    return perturbation_response(prob, res.x, FREQS), 0, 3


def _bare(resp, e_in, e_out):
    """The amplitude-ratio form, i.e. transmission loss without the port correction."""
    tau = resp.acoustic_scattering_matrix(e_in, e_out)[:, 1, 0]
    return -20.0 * np.log10(np.abs(tau))


def test_equal_ports_correction_vanishes():
    """With matched ports the correction vanishes, so no shipped number moves."""
    resp, e_in, e_out = _area_change(0.05, 0.05)
    assert np.allclose(transmission_loss(resp, e_in, e_out), _bare(resp, e_in, e_out), atol=1e-9)


@pytest.mark.parametrize("area_in, area_out", [(0.05, 0.10), (0.05, 0.025), (0.02, 0.08)])
def test_correction_is_the_area_ratio_at_rest(area_in, area_out):
    """At rest and with one gas, C_TL reduces to 10 log10(A_in / A_out).

    The network is driven at zero pressure difference, so the residual Mach number is
    ``O(1e-5)`` rather than exactly zero; that leaves an ``O(1e-4)`` dB discrepancy against
    the pure area ratio, which is why the tolerance is not machine precision.
    """
    from nefes.assembly.recover import ES_M

    resp, e_in, e_out = _area_change(area_in, area_out)
    assert abs(float(resp.est[ES_M, e_in])) < 1e-3  # the case really is at rest
    delta = transmission_loss(resp, e_in, e_out) - _bare(resp, e_in, e_out)
    assert np.allclose(delta, 10.0 * np.log10(area_in / area_out), atol=1e-3)


def test_lossless_contraction_has_non_negative_transmission_loss():
    """The discriminating case: a lossless passive element cannot amplify.

    The bare amplitude ratio returns a negative transmission loss for a contraction, which
    would mean a passive area change generates acoustic power.  The power-based form does not.
    """
    resp, e_in, e_out = _area_change(0.05, 0.025)
    assert _bare(resp, e_in, e_out).min() < -1.0  # the bare form really does go negative
    assert transmission_loss(resp, e_in, e_out).min() >= -1e-9


def test_mean_flow_enters_through_the_convected_flux():
    """With flow the correction carries (1+M)^2 as well, not the area ratio alone."""
    resp, e_in, e_out = _area_change(0.05, 0.05, pt_in=125000.0)
    from nefes.assembly.recover import ES_AREA, ES_C, ES_M, ES_RHO

    est = resp.est

    def wplus(e):
        return float(est[ES_RHO, e]) * float(est[ES_C, e]) * (1 + float(est[ES_M, e])) ** 2 * float(est[ES_AREA, e])

    assert float(est[ES_M, e_in]) > 0.05  # the case really does carry mean flow
    delta = transmission_loss(resp, e_in, e_out) - _bare(resp, e_in, e_out)
    assert np.allclose(delta, 10.0 * np.log10(wplus(e_in) / wplus(e_out)), atol=1e-9)
