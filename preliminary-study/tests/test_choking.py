"""Validation of emergent choking against quasi-1D compressible-flow analytics.

The configurations exercise the Fischer-Burmeister complementarity rows of
IsentropicAreaChange (throat-as-edge choking with lumped internal shock) and
PressureOutlet (choked-orifice discharge), including the regime transitions.
"""

import numpy as np
import pytest

from fns import (
    AIR,
    Network,
    MassFlowInlet,
    TotalPressureInlet,
    PressureOutlet,
    IsentropicAreaChange,
    LosslessSplitter,
    JunctionStaticP,
    solve,
    complex_step_jacobian,
    shock_report,
    normal_shock_pt_ratio,
)
from fns.shock import area_ratio, normal_shock_post_mach

GAS = AIR
G = GAS.gamma
R = GAS.R

PT, TT = 2.0e5, 400.0
A_FEED, A_THROAT, A_EXIT = 0.10, 0.03, 0.06

FLUX_STAR = np.sqrt(G / R) * (2 / (G + 1)) ** ((G + 1) / (2 * (G - 1)))
MDOT_STAR = PT / np.sqrt(TT) * FLUX_STAR * A_THROAT


def phi(M):
    """Nondimensional mass-flux function, maximal at M = 1."""
    return M * (1 + 0.5 * (G - 1) * M * M) ** (-(G + 1) / (2 * (G - 1)))


def mach_from_area_ratio(ar, supersonic):
    lo, hi = (1.0 + 1e-12, 50.0) if supersonic else (1e-9, 1.0 - 1e-12)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (area_ratio(mid) > ar) == supersonic:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def isentropic_p(M, pt):
    return pt * (1 + 0.5 * (G - 1) * M * M) ** (-G / (G - 1))


def cd_nozzle(p_b):
    """Reservoir -> converging IAC -> throat edge -> diverging IAC -> outlet."""
    net = Network(GAS, p_ref=PT, T_ref=TT, mdot_ref=15.0)
    inl = net.add(TotalPressureInlet(pt=PT, Tt=TT, name="res"))
    conv = net.add(IsentropicAreaChange(name="conv"))
    div = net.add(IsentropicAreaChange(name="div"))
    out = net.add(PressureOutlet(p=p_b, name="exit"))
    net.connect(inl, conv, area=A_FEED)
    net.connect(conv, div, area=A_THROAT)
    net.connect(div, out, area=A_EXIT)
    return net


def shock_case_back_pressure(M_shock):
    """Analytic back pressure that places a normal shock at pre-shock Mach
    M_shock inside the diverging section (full quasi-1D chain)."""
    A_s = A_THROAT * area_ratio(M_shock)  # shock area (A* = throat)
    sigma = normal_shock_pt_ratio(M_shock)
    pt2 = sigma * PT
    M2 = normal_shock_post_mach(M_shock)
    A2_star = A_s / area_ratio(M2)  # post-shock sonic reference area
    M_exit = mach_from_area_ratio(A_EXIT / A2_star, supersonic=False)
    p_exit = isentropic_p(M_exit, pt2)
    return p_exit, pt2, M_exit, A_s


# ---------------------------------------------------------------------------


def test_venturi_regime_matches_isentropic():
    """Back pressure above the first critical: subsonic everywhere, lossless,
    mass flow strictly below the choking value."""
    M_exit_ven = mach_from_area_ratio(A_EXIT / A_THROAT, supersonic=False)
    p_first_crit = isentropic_p(M_exit_ven, PT)

    p_b = 0.5 * (PT + p_first_crit)  # comfortably in the venturi regime
    net = cd_nozzle(p_b)
    res = solve(net)
    assert res.converged
    st = net.states(res.x)

    M_exit = mach_from_area_ratio(A_EXIT * phi(1.0) / (st[2].mdot * np.sqrt(TT) / PT / np.sqrt(G / R)) / 1.0, True) if False else None
    # exact venturi solution: exit subsonic with p = p_b at pt = PT
    M_ex = np.sqrt(2 / (G - 1) * ((p_b / PT) ** (-(G - 1) / G) - 1))
    rho = p_b / (R * (TT / (1 + 0.5 * (G - 1) * M_ex**2)))
    u = M_ex * np.sqrt(G * R * TT / (1 + 0.5 * (G - 1) * M_ex**2))
    assert st[2].mdot == pytest.approx(rho * u * A_EXIT, rel=1e-7)
    assert st[2].mdot < MDOT_STAR
    for s in st:
        assert s.pt == pytest.approx(PT, rel=1e-6)  # lossless
        assert abs(s.M) < 1.0
    assert shock_report(net, res.x) == []


@pytest.mark.parametrize("M_shock", [1.3, 1.8, 2.15])
def test_shock_in_nozzle_vs_analytic(M_shock):
    """Choked operation with an internal normal shock: mass flow capped at
    the choking value, exit state and total-pressure loss exactly matching
    the quasi-1D Rankine-Hugoniot construction, and the shock Mach/area
    recovered by the post-processing diagnostics."""
    p_b, pt2_exact, M_exit_exact, A_s_exact = shock_case_back_pressure(M_shock)
    net = cd_nozzle(p_b)
    res = solve(net)
    assert res.converged, f"M_shock={M_shock}: {res}"
    st = net.states(res.x)

    assert st[0].mdot == pytest.approx(MDOT_STAR, rel=1e-5)
    assert st[1].M == pytest.approx(1.0, abs=2e-3)  # throat edge sonic
    assert st[2].pt == pytest.approx(pt2_exact, rel=1e-4)
    assert abs(st[2].M) == pytest.approx(M_exit_exact, rel=1e-3)
    assert st[2].p == pytest.approx(p_b, rel=1e-6)

    shocks = shock_report(net, res.x)
    assert len(shocks) == 1
    sh = shocks[0]
    assert sh["element"] == "div"
    assert sh["M_shock"] == pytest.approx(M_shock, rel=2e-3)
    assert sh["A_shock"] == pytest.approx(A_s_exact, rel=5e-3)
    assert sh["valid"]


def test_mass_flow_saturation_continuity():
    """Sweep the back pressure across the first critical point: the mass flow
    rises, saturates at the choking value, and never exceeds it."""
    M_exit_ven = mach_from_area_ratio(A_EXIT / A_THROAT, supersonic=False)
    p_first_crit = isentropic_p(M_exit_ven, PT)

    p_bs = np.linspace(0.999 * PT, 0.7 * p_first_crit, 12)
    mdots = []
    x0 = None
    for p_b in p_bs:
        net = cd_nozzle(float(p_b))
        res = solve(net, x0=x0)
        assert res.converged, f"p_b={p_b}"
        x0 = res.x
        mdots.append(net.states(res.x)[1].mdot)
    mdots = np.asarray(mdots)

    assert np.all(np.diff(mdots) > -1e-6 * MDOT_STAR)  # monotone rise
    assert np.all(mdots <= MDOT_STAR * (1 + 1e-6))  # never exceeds choking
    assert mdots[-1] == pytest.approx(MDOT_STAR, rel=1e-5)  # saturated
    assert mdots[0] < 0.5 * MDOT_STAR  # clearly unchoked at high p_b


def test_choked_solution_is_well_conditioned():
    """The complementarity formulation removes the fold: the scaled Jacobian
    at a choked solution is far from singular (the fold gave sigma_min ~ 5e-5)."""
    p_b, _, _, _ = shock_case_back_pressure(1.8)
    net = cd_nozzle(p_b)
    res = solve(net)
    assert res.converged

    xs, rs = net.variable_scales(), net.residual_scales()
    J = complex_step_jacobian(lambda y: net.residual(y * xs) / rs, res.x / xs)
    sv = np.linalg.svd(J, compute_uv=False)
    assert sv[-1] > 1e-3


def test_overdriven_transfer_loop_capped():
    """The test_1.yaml failure class: two outlets at very different pressures
    joined by a lossless path.  With emergent choking the transfer loop caps
    at the choking flow of the smallest edge and the case converges."""
    net = Network(GAS, p_ref=101325.0, T_ref=300.0, mdot_ref=10.0)
    inl = net.add(MassFlowInlet(mdot=1.0, Tt=300.0, name="inlet"))
    iac1 = net.add(IsentropicAreaChange(name="iac1"))
    spl = net.add(LosslessSplitter(3, name="split"))
    sx = net.add(IsentropicAreaChange(name="brA"))
    iac2 = net.add(IsentropicAreaChange(name="brB"))
    outA = net.add(PressureOutlet(p=130000.0, name="outA"))
    outB = net.add(PressureOutlet(p=101325.0, name="outB"))
    net.connect(inl, iac1, area=1.0)
    net.connect(iac1, spl, area=2.0)
    net.connect(spl, sx, area=1.5)
    eB = net.connect(spl, iac2, area=1.0)  # the limiting passage
    net.connect(sx, outA, area=5.0)
    net.connect(iac2, outB, area=4.0)

    res = solve(net, x0=net.initial_guess(mdot0=0.0))
    assert res.converged, res
    st = net.states(res.x)

    # loop flow capped at the choke of the area-1 edge at its local (pt, Tt)
    s = st[eB]
    mdot_cap = s.pt / np.sqrt(s.Tt) * FLUX_STAR * s.area
    assert s.mdot == pytest.approx(mdot_cap, rel=1e-5)
    assert abs(s.M) == pytest.approx(1.0, abs=2e-3)
    # net mass balance: inlet feed leaves through the outlets
    m_A = st[4].mdot  # sx -> outA edge (index 4)
    m_B = st[5].mdot
    assert m_A + m_B == pytest.approx(1.0, rel=1e-6)


def test_subsonic_cases_unaffected_by_fb_rows():
    """The complementarity reduces to total-pressure equality in subsonic
    operation: a moderate case matches the lossless analytic solution."""
    net = Network(GAS, p_ref=101325.0, T_ref=300.0)
    inl = net.add(MassFlowInlet(mdot=25.0, Tt=500.0, name="inlet"))
    iac = net.add(IsentropicAreaChange(name="iac"))
    out = net.add(PressureOutlet(p=101325.0, name="outlet"))
    net.connect(inl, iac, area=0.3)
    net.connect(iac, out, area=0.12)
    res = solve(net)
    assert res.converged
    s0, s1 = net.states(res.x)
    assert s0.pt == pytest.approx(s1.pt, rel=1e-8)
    assert s1.p == pytest.approx(101325.0, rel=1e-6)
    assert shock_report(net, res.x) == []
