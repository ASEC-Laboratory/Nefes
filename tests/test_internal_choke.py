"""Internal choking: the lumped normal shock in a diverging element.

A converging--diverging pair sharing the throat edge is the smallest truly *internal* choke:
the converging element stays isentropic, the throat edge pins at M = 1, and the diverging
element's complementarity row admits the total-pressure drop that stands in for a normal
shock somewhere inside it.  These tests pin that behavior quantitatively -- including that
the lumped loss corresponds to a *realizable* shock (Rankine--Hugoniot inversion) throughout
the shock-in-nozzle back-pressure window -- and that the post-solve check warns once the
back pressure leaves that window for the (out-of-scope) supersonic-exit regime.
"""

import warnings

import numpy as np
import pytest

import nefes
import nefes.shell.checks as checks
from nefes.elements import catalog as cat

R_AIR, GAMMA = 287.0, 1.4
PT, TT = 2.0e5, 300.0
A_IN, A_THROAT, A_EXIT = 0.020, 0.010, 0.016

FLUX_STAR = np.sqrt(GAMMA / R_AIR) * (2.0 / (GAMMA + 1.0)) ** ((GAMMA + 1.0) / (2.0 * (GAMMA - 1.0)))
MDOT_MAX = PT / np.sqrt(TT) * FLUX_STAR * A_THROAT


def _area_ratio(M):
    return (1.0 / M) * ((2.0 + (GAMMA - 1.0) * M * M) / (GAMMA + 1.0)) ** ((GAMMA + 1.0) / (2.0 * (GAMMA - 1.0)))


def _pt_ratio_shock(M1):
    a = ((GAMMA + 1.0) * M1 * M1 / (2.0 + (GAMMA - 1.0) * M1 * M1)) ** (GAMMA / (GAMMA - 1.0))
    b = ((GAMMA + 1.0) / (2.0 * GAMMA * M1 * M1 - (GAMMA - 1.0))) ** (1.0 / (GAMMA - 1.0))
    return a * b


def _bisect(f, lo, hi, n=80):
    flo = f(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0.0) == (flo > 0.0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# the divergent's exit-plane Mach numbers on the two isentropic branches
M_EXIT_SUB = _bisect(lambda M: _area_ratio(M) - A_EXIT / A_THROAT, 1e-6, 1.0 - 1e-9)
M_EXIT_SUP = _bisect(lambda M: _area_ratio(M) - A_EXIT / A_THROAT, 1.0 + 1e-9, 30.0)
# first critical: throat just sonic, divergent all subsonic
PB_FIRST_CRITICAL = PT * (1.0 + 0.5 * (GAMMA - 1.0) * M_EXIT_SUB**2) ** (-GAMMA / (GAMMA - 1.0))


def _cd_nozzle(pb):
    return nefes.Network(
        nefes.perfect_gas(R_AIR, GAMMA),
        nodes=[
            cat.total_pressure_inlet(PT, TT, name="reservoir"),
            cat.isentropic_area_change(name="converging"),
            cat.isentropic_area_change(name="diverging"),
            cat.pressure_outlet(pb, TT, name="back"),
        ],
        edges=[(0, 1, A_IN), (1, 2, A_THROAT), (2, 3, A_EXIT)],
    )


def test_internal_choke_pins_throat_and_caps_mass_flow():
    """Below first critical the throat edge is sonic, the mass flow caps, the exit is subsonic."""
    sol = _cd_nozzle(0.85 * PT).solve()
    assert sol.converged
    assert sol.edge(1)["M"] == pytest.approx(1.0, abs=5e-3)  # throat edge sonic
    assert sol.edge(1)["mdot"] == pytest.approx(MDOT_MAX, rel=5e-3)  # capped
    assert sol.edge(2)["M"] < 0.6  # divergent exit subsonic (shock-in-nozzle regime)
    assert sol.edge(2)["p"] == pytest.approx(0.85 * PT, rel=1e-3)  # back pressure met


def test_lumped_shock_is_realizable_through_the_shock_window():
    """The diverging element's converged loss inverts (Rankine--Hugoniot) to a shock the
    divergent can host, at every back pressure in the shock-in-nozzle window; the converging
    element stays lossless and the loss grows monotonically as the back pressure falls."""
    assert PB_FIRST_CRITICAL / PT == pytest.approx(0.897, abs=2e-3)  # window's upper edge
    prev_loss = 0.0
    for ratio in (0.88, 0.82, 0.75, 0.65):
        sol = _cd_nozzle(ratio * PT).solve()
        assert sol.converged
        pt_in, pt_thr, pt_ex = (sol.edge(e)["p_t"] for e in (0, 1, 2))
        assert pt_thr == pytest.approx(pt_in, rel=1e-4)  # converging side isentropic
        loss = 1.0 - pt_ex / pt_thr
        assert loss > prev_loss  # lumped-shock loss deepens as the back pressure falls
        prev_loss = loss
        m_shock = _bisect(lambda M: _pt_ratio_shock(M) - pt_ex / pt_thr, 1.0 + 1e-9, 30.0)
        assert 1.0 < m_shock < M_EXIT_SUP  # a normal shock inside the divergent can host it
        # the post-solve inspection reports the same numbers and flags it realizable
        (sh,) = [s for s in sol.unrealizable_lumped_shocks()]
        assert sh["realizable"]
        assert sh["implied_shock_mach"] == pytest.approx(m_shock, rel=1e-3)


def test_supersonic_exit_regime_warns():
    """Below the strongest realizable shock the converged subsonic root is not physical, and
    the solve says so: the implied shock exceeds the exit-plane Mach and a warning is raised."""
    with pytest.warns(UserWarning, match="supersonic-exit regime"):
        sol = _cd_nozzle(0.40 * PT).solve()
    assert sol.converged  # the root still converges; the *warning* carries the scope verdict
    (sh,) = sol.unrealizable_lumped_shocks()
    assert not sh["realizable"]
    assert sh["implied_shock_mach"] > M_EXIT_SUP
    assert any("supersonic-exit" in m for m in sol.verify())


def test_lumped_shock_check_can_be_disabled():
    """The gate follows the CHECK_LUMPED_SHOCK toggle, like every discretionary check."""
    checks.CHECK_LUMPED_SHOCK = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sol = _cd_nozzle(0.40 * PT).solve()
        assert sol.converged
        assert not [w for w in caught if "supersonic-exit" in str(w.message)]
    finally:
        checks.CHECK_LUMPED_SHOCK = True


def test_subsonic_venturi_reports_no_lumped_shock():
    """Above first critical the pair is a subsonic venturi: lossless, and the inspection is empty."""
    sol = _cd_nozzle(0.95 * PT).solve()
    assert sol.converged
    assert sol.edge(1)["M"] < 1.0 - 1e-3
    assert sol.unrealizable_lumped_shocks() == []
    assert sol.verify() == []
