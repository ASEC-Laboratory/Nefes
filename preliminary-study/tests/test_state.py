"""State recovery: uniqueness, roundtrip, complex-step consistency."""

import numpy as np
import pytest

from fns import AIR, recover_state, StateError
from fns.state import solve_density


def make_xpht(rho, u, T, area, gas=AIR):
    """Forward map: physical state -> (mdot, p, ht)."""
    p = rho * gas.R * T
    ht = gas.cp * T + 0.5 * u * u
    mdot = rho * u * area
    return mdot, p, ht


@pytest.mark.parametrize(
    "rho,u,T,area",
    [
        (1.2, 30.0, 300.0, 0.5),
        (1.2, -30.0, 300.0, 0.5),  # reversed flow
        (0.5, 250.0, 600.0, 0.05),  # high subsonic
        (0.4, 600.0, 500.0, 0.01),  # supersonic (M ~ 1.34)
        (1.2, 1e-9, 300.0, 1.0),  # essentially quiescent
        (1.2, 0.0, 300.0, 1.0),  # exactly quiescent
    ],
)
def test_roundtrip(rho, u, T, area):
    mdot, p, ht = make_xpht(rho, u, T, area)
    st = recover_state(mdot, p, ht, area, AIR)
    assert st.rho == pytest.approx(rho, rel=1e-12)
    assert st.u == pytest.approx(u, rel=1e-12, abs=1e-12)
    assert st.T == pytest.approx(T, rel=1e-12)


def test_signed_quantities():
    mdot, p, ht = make_xpht(1.2, -50.0, 300.0, 0.5)
    st = recover_state(mdot, p, ht, 0.5, AIR)
    assert st.u < 0 and st.M < 0 and st.mdot < 0
    # direction independent quantities
    st_fwd = recover_state(-mdot, p, ht, 0.5, AIR)
    assert st.pt == pytest.approx(st_fwd.pt, rel=1e-14)
    assert st.T == pytest.approx(st_fwd.T, rel=1e-14)


def test_total_quantities_consistency():
    mdot, p, ht = make_xpht(1.0, 150.0, 400.0, 0.2)
    st = recover_state(mdot, p, ht, 0.2, AIR)
    g = AIR.gamma
    assert st.Tt == pytest.approx(st.T * (1 + 0.5 * (g - 1) * st.M**2), rel=1e-13)
    assert st.pt == pytest.approx(st.p * (st.Tt / st.T) ** (g / (g - 1)), rel=1e-13)


def test_nonphysical_raises():
    with pytest.raises(StateError):
        recover_state(1.0, -100.0, 3e5, 1.0, AIR)
    with pytest.raises(StateError):
        recover_state(1.0, 101325.0, -1.0, 1.0, AIR)


def test_density_complex_step_matches_fd():
    """IFT-propagated imaginary part == finite-difference derivative."""
    m, p, ht = 300.0, 90000.0, 4.0e5
    h = 1e-30
    for k, dx in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        args_c = [m + 1j * h * dx[0], p + 1j * h * dx[1], ht + 1j * h * dx[2]]
        d_cs = np.imag(solve_density(*args_c, AIR.K)) / h
        eps = 1e-6 * [m, p, ht][k]
        args_p = [m + eps * dx[0], p + eps * dx[1], ht + eps * dx[2]]
        args_m = [m - eps * dx[0], p - eps * dx[1], ht - eps * dx[2]]
        d_fd = (solve_density(*args_p, AIR.K) - solve_density(*args_m, AIR.K)) / (2 * eps)
        assert d_cs == pytest.approx(d_fd, rel=1e-6)
