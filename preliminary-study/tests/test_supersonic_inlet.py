"""Started supersonic intake: SupersonicInlet boundary + terminal shock.

Validates the count-redistribution mechanics (the inlet supplies two
equations; the sonic-fed diverging element's complementarity self-vacates)
and the full quasi-1D intake solution: supersonic compression, sonic throat,
terminal normal shock at the analytically constructed station, subsonic
pressure recovery.
"""

import numpy as np
import pytest

from fns import (
    AIR,
    Network,
    SupersonicInlet,
    IsentropicAreaChange,
    PressureOutlet,
    solve,
    shock_report,
)
from fns.shock import area_ratio, normal_shock_post_mach, normal_shock_pt_ratio

G, R = AIR.gamma, AIR.R

M0, P0, T0 = 1.8, 30000.0, 220.0
BETA0 = 1 + 0.2 * M0 * M0
PT0, TT0 = P0 * BETA0**3.5, T0 * BETA0
A0 = 0.20
AT = A0 / area_ratio(M0)  # exactly-critical throat
AREAS = [A0, 0.165, AT, 0.18, 0.24]


def mach_from_ar(ar, supersonic):
    lo, hi = (1.0 + 1e-12, 50.0) if supersonic else (1e-9, 1.0 - 1e-12)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (area_ratio(mid) > ar) == supersonic:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def capture_mdot():
    rho = P0 / (R * T0)
    return rho * M0 * np.sqrt(G * R * T0) * A0


def back_pressure(M_shock):
    A_s = AT * area_ratio(M_shock)
    pt2 = normal_shock_pt_ratio(M_shock) * PT0
    M2 = normal_shock_post_mach(M_shock)
    A2s = A_s / area_ratio(M2)
    M_exit = mach_from_ar(AREAS[4] / A2s, False)
    return pt2 * (1 + 0.2 * M_exit**2) ** -3.5, pt2, A_s


def build(p_b):
    net = Network(AIR, p_ref=PT0, T_ref=TT0, mdot_ref=60.0)
    el = [net.add(SupersonicInlet(M=M0, pt=PT0, Tt=TT0, name="freestream"))]
    for nm in ("c1", "c2", "d1", "d2"):
        el.append(net.add(IsentropicAreaChange(name=nm)))
    el.append(net.add(PressureOutlet(p=p_b, name="engine-face")))
    for k in range(5):
        net.connect(el[k], el[k + 1], area=AREAS[k])
    return net


def analytic_x0(M_shock, perturb=0.0, seed=0):
    _, pt2, A_s = back_pressure(M_shock)
    Ms = [M0, mach_from_ar(AREAS[1] / AT, True), 1.0]
    pts = [PT0, PT0, PT0]
    M2 = normal_shock_post_mach(M_shock)
    A2s = A_s / area_ratio(M2)
    for A in AREAS[3:]:
        Ms.append(mach_from_ar(A / A2s, False))
        pts.append(pt2)
    x0 = np.empty(15)
    for k in range(5):
        b = 1 + 0.2 * Ms[k] ** 2
        p, T = pts[k] * b**-3.5, TT0 / b
        rho = p / (R * T)
        x0[3 * k : 3 * k + 3] = [rho * Ms[k] * np.sqrt(G * R * T) * AREAS[k], p, AIR.cp * TT0]
    if perturb:
        rng = np.random.default_rng(seed)
        x0 *= 1 + perturb * (rng.random(15) - 0.5)
    return x0


@pytest.mark.parametrize("M_shock", [1.1, 1.5])
def test_started_intake_vs_analytic(M_shock):
    p_b, pt2, A_s = back_pressure(M_shock)
    net = build(p_b)
    res = solve(net, x0=analytic_x0(M_shock, perturb=0.2, seed=3),
                tol=2e-9, stab_stages=(0.0,))
    assert res.converged, res
    st = net.states(res.x)

    assert st[0].M == pytest.approx(M0, rel=1e-6)  # flight condition held
    assert st[0].mdot == pytest.approx(capture_mdot(), rel=1e-5)  # full capture
    assert st[1].M > 1.0  # supersonic compression
    assert st[2].M == pytest.approx(1.0, abs=2e-3)  # sonic throat edge
    assert st[4].M < 1.0  # subsonic delivery
    assert st[4].pt == pytest.approx(pt2, rel=1e-3)  # pressure recovery

    sh = shock_report(net, res.x)
    assert len(sh) == 1 and sh[0]["valid"]
    assert sh[0]["M_shock"] == pytest.approx(M_shock, rel=5e-3)
    assert sh[0]["A_shock"] == pytest.approx(A_s, rel=1e-2)


def test_supercritical_shock_moves_with_back_pressure():
    """Lower back pressure swallows the shock deeper (higher M_s, more loss)."""
    sh_machs = []
    for M_shock in (1.15, 1.35, 1.55):
        p_b, _, _ = back_pressure(M_shock)
        net = build(p_b)
        res = solve(net, x0=analytic_x0(M_shock, perturb=0.1, seed=1),
                    tol=2e-9, stab_stages=(0.0,))
        assert res.converged
        sh = shock_report(net, res.x)
        sh_machs.append(sh[0]["M_shock"])
    assert sh_machs[0] < sh_machs[1] < sh_machs[2]


def test_supersonic_inlet_requires_supersonic_mach():
    with pytest.raises(ValueError):
        SupersonicInlet(M=0.8, pt=1e5, Tt=300.0)
