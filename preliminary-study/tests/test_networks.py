"""Network solutions against analytic 1-D compressible flow relations."""

import numpy as np
import pytest
from scipy.optimize import brentq

from fns import (
    AIR,
    Network,
    MassFlowInlet,
    TotalPressureInlet,
    PressureOutlet,
    IsentropicAreaChange,
    SuddenAreaChange,
    LossElement,
    JunctionStaticP,
    LosslessSplitter,
    solve,
)

GAS = AIR
G = GAS.gamma


def mach_from_mass_flux(mdot, area, pt, Tt, gas=GAS, supersonic=False):
    """Solve mdot = A * pt * sqrt(g/(R*Tt)) * M * (1+...)^-(g+1)/(2(g-1)) for M."""

    def f(M):
        return (
            area
            * pt
            * np.sqrt(gas.gamma / (gas.R * Tt))
            * M
            * (1 + 0.5 * (gas.gamma - 1) * M * M) ** (-(gas.gamma + 1) / (2 * (gas.gamma - 1)))
            - mdot
        )

    lo, hi = (1.0 + 1e-12, 50.0) if supersonic else (1e-12, 1.0)
    return brentq(f, lo, hi, xtol=1e-15)


def isentropic_state(M, pt, Tt, gas=GAS):
    beta = 1 + 0.5 * (gas.gamma - 1) * M * M
    T = Tt / beta
    p = pt * beta ** (-gas.gamma / (gas.gamma - 1))
    rho = p / (gas.R * T)
    u = M * np.sqrt(gas.gamma * gas.R * T)
    return p, T, rho, u


def conservation_ok(net, x, mtol=1e-8, etol=1e-3):
    """All interior elements conserve mass and energy (boundary elements carry the throughflow)."""
    n_ports = {el.name: len(p) for el, p in zip(net.elements, net._ports)}
    for el, dm, de in net.conservation_report(x):
        if n_ports[el] < 2:  # boundary element: through-flow by design
            continue
        assert abs(dm) < mtol, f"{el}: mass imbalance {dm}"
        assert abs(de) < etol * 3e5, f"{el}: energy imbalance {de}"


def test_single_iac_vs_exact():
    """Mass-flow inlet -> isentropic area change -> pressure outlet."""
    mdot, Tt, p_out = 25.0, 500.0, 101325.0
    a0, a1 = 0.3, 0.12

    net = Network(GAS, p_ref=101325.0, T_ref=300.0)
    inl = net.add(MassFlowInlet(mdot=mdot, Tt=Tt, name="inlet"))
    iac = net.add(IsentropicAreaChange(name="iac"))
    out = net.add(PressureOutlet(p=p_out, name="outlet"))
    net.connect(inl, iac, area=a0)
    net.connect(iac, out, area=a1)

    res = solve(net)
    assert res.converged
    s0, s1 = net.states(res.x)

    # exact: outlet edge fixes p_t from (mdot, Tt, p_out, a1)
    def f(pt):
        M = mach_from_mass_flux(mdot, a1, pt, Tt)
        return isentropic_state(M, pt, Tt)[0] - p_out

    # lower bracket: smallest pt that passes mdot through a1 (choking limit)
    flux_star = np.sqrt(G / GAS.R) * (2 / (G + 1)) ** ((G + 1) / (2 * (G - 1)))
    pt_choke = mdot / a1 * np.sqrt(Tt) / flux_star
    pt_exact = brentq(f, pt_choke * (1 + 1e-9), p_out * 5, xtol=1e-10)
    M1 = mach_from_mass_flux(mdot, a1, pt_exact, Tt)
    M0 = mach_from_mass_flux(mdot, a0, pt_exact, Tt)
    p0_exact = isentropic_state(M0, pt_exact, Tt)[0]

    assert s1.pt == pytest.approx(pt_exact, rel=1e-9)
    assert s0.pt == pytest.approx(pt_exact, rel=1e-9)
    assert abs(s1.M) == pytest.approx(M1, rel=1e-9)
    assert s0.p == pytest.approx(p0_exact, rel=1e-9)
    assert s0.Tt == pytest.approx(Tt, rel=1e-8)
    assert s1.Tt == pytest.approx(Tt, rel=1e-8)


def test_multi_iac_chain_from_quiescent():
    """The historically failing configuration: several isentropic elements in
    series, solved from an exactly quiescent, uniform, cold initial state."""
    areas = [1.0, 0.5, 0.15, 0.5, 2.0, 0.12]
    mdot, Tt, p_out = 25.0, 500.0, 101325.0

    net = Network(GAS, p_ref=101325.0, T_ref=300.0)
    el = [net.add(MassFlowInlet(mdot=mdot, Tt=Tt, name="inlet"))]
    for k in range(len(areas) - 1):
        el.append(net.add(IsentropicAreaChange(name=f"iac{k}")))
    el.append(net.add(PressureOutlet(p=p_out, name="outlet")))
    for k, a in enumerate(areas):
        net.connect(el[k], el[k + 1], area=a)

    x0 = net.initial_guess(mdot0=0.0, p0=101325.0, Tt0=300.0)
    res = solve(net, x0=x0)
    assert res.converged
    states = net.states(res.x)

    # uniform mdot, pt, Tt along the chain
    for st in states:
        assert st.mdot == pytest.approx(mdot, rel=1e-10)
        assert st.pt == pytest.approx(states[-1].pt, rel=1e-9)
        assert st.Tt == pytest.approx(Tt, rel=1e-8)
    # the smooth boundary-regime blend leaves an O((eps/mdot)^2) imprint
    assert states[-1].p == pytest.approx(p_out, rel=1e-8)
    conservation_ok(net, res.x)


def test_edge_direction_flip_invariance():
    """Flipping edge directions must not change the physics; mdot and u flip sign."""
    areas = [0.4, 0.15, 0.3]

    def build(flips):
        net = Network(GAS, p_ref=101325.0, T_ref=300.0)
        el = [net.add(MassFlowInlet(mdot=20.0, Tt=450.0, name="inlet"))]
        el.append(net.add(IsentropicAreaChange(name="iac0")))
        el.append(net.add(SuddenAreaChange(name="sud0")))
        el.append(net.add(PressureOutlet(p=101325.0, name="outlet")))
        for k, a in enumerate(areas):
            if flips[k]:
                net.connect(el[k + 1], el[k], area=a)  # reversed edge
            else:
                net.connect(el[k], el[k + 1], area=a)
        return net

    base = build([False, False, False])
    res_base = solve(base)
    assert res_base.converged
    st_base = base.states(res_base.x)

    for flips in ([True, False, False], [False, True, False], [True, True, True]):
        net = build(flips)
        res = solve(net)
        assert res.converged, f"flips={flips}"
        st = net.states(res.x)
        for k in range(len(areas)):
            sgn = -1.0 if flips[k] else 1.0
            assert st[k].mdot == pytest.approx(sgn * st_base[k].mdot, rel=1e-8)
            assert st[k].u == pytest.approx(sgn * st_base[k].u, rel=1e-8)
            assert st[k].p == pytest.approx(st_base[k].p, rel=1e-8)
            assert st[k].ht == pytest.approx(st_base[k].ht, rel=1e-8)


def test_sudden_expansion_borda_carnot():
    """Momentum balance with back-wall pressure + entropy must increase."""
    net = Network(GAS, p_ref=101325.0, T_ref=300.0)
    inl = net.add(MassFlowInlet(mdot=30.0, Tt=350.0, name="inlet"))
    sx = net.add(SuddenAreaChange(name="sx"))
    out = net.add(PressureOutlet(p=101325.0, name="outlet"))
    net.connect(inl, sx, area=0.1)
    net.connect(sx, out, area=0.4)
    res = solve(net)
    assert res.converged
    s0, s1 = net.states(res.x)

    lhs = s0.mdot * s0.u + s0.p * s0.area + s0.p * (s1.area - s0.area)
    rhs = s1.mdot * s1.u + s1.p * s1.area
    assert lhs == pytest.approx(rhs, rel=1e-6)
    assert s1.pt < s0.pt  # total pressure loss
    assert s1.s > s0.s  # entropy rises (2nd law)
    assert s0.ht == pytest.approx(s1.ht, rel=1e-8)
    conservation_ok(net, res.x)


def test_sudden_contraction_is_lossless():
    """Reverse regime of SuddenAreaChange: large -> small modeled isentropic."""
    net = Network(GAS, p_ref=101325.0, T_ref=300.0)
    inl = net.add(MassFlowInlet(mdot=30.0, Tt=350.0, name="inlet"))
    sx = net.add(SuddenAreaChange(name="sx"))
    out = net.add(PressureOutlet(p=101325.0, name="outlet"))
    net.connect(inl, sx, area=0.4)  # large side first: flow large -> small
    net.connect(sx, out, area=0.1)
    res = solve(net)
    assert res.converged
    s0, s1 = net.states(res.x)
    assert s1.pt == pytest.approx(s0.pt, rel=1e-7)
    assert s1.s == pytest.approx(s0.s, rel=1e-7)


def test_pressure_driven_forward_vs_exact():
    """pt inlet / p outlet, no mass flow specified anywhere."""
    pt_in, Tt, p_out = 110000.0, 300.0, 101325.0
    a_exit = 0.5

    net = Network(GAS, p_ref=101325.0, T_ref=300.0, mdot_ref=10.0)
    inl = net.add(TotalPressureInlet(pt=pt_in, Tt=Tt, name="pt-in"))
    iac = net.add(IsentropicAreaChange(name="iac"))
    out = net.add(PressureOutlet(p=p_out, name="p-out"))
    net.connect(inl, iac, area=0.3)
    net.connect(iac, out, area=a_exit)
    res = solve(net)
    assert res.converged
    s0, s1 = net.states(res.x)

    M_exit = np.sqrt(2 / (G - 1) * ((pt_in / p_out) ** ((G - 1) / G) - 1))
    p, T, rho, u = isentropic_state(M_exit, pt_in, Tt)
    assert s1.M == pytest.approx(M_exit, rel=1e-7)
    assert s1.mdot == pytest.approx(rho * u * a_exit, rel=1e-7)
    assert s0.pt == pytest.approx(pt_in, rel=1e-9)


def test_reversed_flow_through_boundaries():
    """Receiver pt below discharge p: the flow must run against both edges."""
    pt_in, p_out, Tt_back = 90000.0, 101325.0, 400.0

    net = Network(GAS, p_ref=101325.0, T_ref=300.0, mdot_ref=10.0)
    inl = net.add(TotalPressureInlet(pt=pt_in, Tt=300.0, name="pt-in"))
    iac = net.add(IsentropicAreaChange(name="iac"))
    out = net.add(PressureOutlet(p=p_out, Tt_backflow=Tt_back, name="p-out"))
    net.connect(inl, iac, area=0.3)
    net.connect(iac, out, area=0.5)
    res = solve(net)
    assert res.converged
    s0, s1 = net.states(res.x)

    # exact: backflow stream from the outlet reservoir (pt = p_out, Tt = Tt_back)
    # discharging into the receiver at static pt_in
    M0 = np.sqrt(2 / (G - 1) * ((p_out / pt_in) ** ((G - 1) / G) - 1))
    p, T, rho, u = isentropic_state(M0, p_out, Tt_back)
    assert s0.mdot == pytest.approx(-rho * u * 0.3, rel=1e-7)
    assert s0.mdot < 0 and s1.mdot < 0
    assert s0.Tt == pytest.approx(Tt_back, rel=1e-8)  # advected from the outlet
    assert s1.Tt == pytest.approx(Tt_back, rel=1e-8)


def test_lossless_diamond_split_fractions():
    """Splitter -> two lossless branches -> static-p junction.

    With everything lossless and a common junction static pressure, the two
    branch edge states at the junction are identical, so the split fractions
    are proportional to the branch areas at the junction.
    """
    aA, aB = 0.3, 0.18
    net = Network(GAS, p_ref=101325.0, T_ref=300.0)
    inl = net.add(MassFlowInlet(mdot=20.0, Tt=400.0, name="inlet"))
    spl = net.add(LosslessSplitter(3, name="split"))
    jun = net.add(JunctionStaticP(3, name="junction"))
    out = net.add(PressureOutlet(p=101325.0, name="outlet"))
    net.connect(inl, spl, area=0.5)
    eA = net.connect(spl, jun, area=aA)
    eB = net.connect(spl, jun, area=aB)
    net.connect(jun, out, area=0.5)

    res = solve(net, x0=net.initial_guess(mdot0=0.0))
    assert res.converged
    st = net.states(res.x)
    split_A = st[eA].mdot / 20.0
    assert split_A == pytest.approx(aA / (aA + aB), rel=1e-8)
    conservation_ok(net, res.x)


def test_diamond_with_loss_branch():
    """A loss in one branch shifts the split toward the clean branch and the
    junction mixes the enthalpies consistently."""
    net = Network(GAS, p_ref=101325.0, T_ref=300.0)
    inl = net.add(MassFlowInlet(mdot=20.0, Tt=400.0, name="inlet"))
    spl = net.add(LosslessSplitter(3, name="split"))
    ia = net.add(IsentropicAreaChange(name="branchA"))
    lb = net.add(LossElement(K=5.0, name="branchB"))
    jun = net.add(JunctionStaticP(3, name="junction"))
    out = net.add(PressureOutlet(p=101325.0, name="outlet"))
    net.connect(inl, spl, area=0.5)
    eA1 = net.connect(spl, ia, area=0.25)
    eB1 = net.connect(spl, lb, area=0.25)
    net.connect(ia, jun, area=0.3)
    net.connect(lb, jun, area=0.25)
    net.connect(jun, out, area=0.5)

    res = solve(net, x0=net.initial_guess(mdot0=0.0))
    assert res.converged
    st = net.states(res.x)
    assert st[eA1].mdot > st[eB1].mdot > 0  # loss branch starves
    assert st[eA1].mdot + st[eB1].mdot == pytest.approx(20.0, rel=1e-9)
    conservation_ok(net, res.x)


def test_energy_conservation_with_mixed_temperatures():
    """Two inlets at different Tt merging: outflow enthalpy = mass-weighted mix."""
    net = Network(GAS, p_ref=101325.0, T_ref=300.0)
    in1 = net.add(MassFlowInlet(mdot=10.0, Tt=300.0, name="in-cold"))
    in2 = net.add(MassFlowInlet(mdot=5.0, Tt=600.0, name="in-hot"))
    jun = net.add(JunctionStaticP(3, name="junction"))
    out = net.add(PressureOutlet(p=101325.0, name="outlet"))
    net.connect(in1, jun, area=0.2)
    net.connect(in2, jun, area=0.1)
    e_out = net.connect(jun, out, area=0.3)

    res = solve(net)
    assert res.converged
    st = net.states(res.x)
    Tt_mix = (10.0 * 300.0 + 5.0 * 600.0) / 15.0
    assert st[e_out].Tt == pytest.approx(Tt_mix, rel=1e-6)
    assert st[e_out].mdot == pytest.approx(15.0, rel=1e-10)


def test_high_subsonic_vs_exact():
    """Near-choking accuracy: exit Mach 0.99, compare with exact relations."""
    pt_in, Tt_in, a_exit = 2.0e5, 400.0, 0.03
    PR = 0.535
    net = Network(GAS, p_ref=pt_in, T_ref=Tt_in, mdot_ref=15.0)
    inl = net.add(TotalPressureInlet(pt=pt_in, Tt=Tt_in, name="reservoir"))
    iac = net.add(IsentropicAreaChange(name="nozzle"))
    out = net.add(PressureOutlet(p=PR * pt_in, name="exit"))
    net.connect(inl, iac, area=0.10)
    net.connect(iac, out, area=a_exit)
    res = solve(net)
    assert res.converged
    st = net.states(res.x)

    M_ex = np.sqrt(2 / (G - 1) * (PR ** (-(G - 1) / G) - 1))
    p, T, rho, u = isentropic_state(M_ex, pt_in, Tt_in)
    assert st[1].M == pytest.approx(M_ex, rel=1e-7)
    assert st[1].mdot == pytest.approx(rho * u * a_exit, rel=1e-7)
    assert st[1].M > 0.98  # genuinely near choking


def test_choked_orifice_below_critical_pr():
    """Below the critical pressure ratio a converging exit chokes: the mass
    flow caps at the choking value, the exit edge sits at exactly M = 1, and
    the exit pressure detaches upward from the specification (underexpanded
    discharge) -- all emergent from the complementarity rows, no switches."""
    pt_in, Tt_in = 2.0e5, 400.0
    a_exit = 0.03
    net = Network(GAS, p_ref=pt_in, T_ref=Tt_in, mdot_ref=15.0)
    inl = net.add(TotalPressureInlet(pt=pt_in, Tt=Tt_in, name="reservoir"))
    iac = net.add(IsentropicAreaChange(name="nozzle"))
    out = net.add(PressureOutlet(p=0.45 * pt_in, name="exit"))
    net.connect(inl, iac, area=0.10)
    net.connect(iac, out, area=a_exit)
    res = solve(net)
    assert res.converged
    st = net.states(res.x)

    flux_star = np.sqrt(G / GAS.R) * (2 / (G + 1)) ** ((G + 1) / (2 * (G - 1)))
    mdot_star = pt_in / np.sqrt(Tt_in) * flux_star * a_exit
    p_sonic = pt_in * (2 / (G + 1)) ** (G / (G - 1))
    assert st[1].mdot == pytest.approx(mdot_star, rel=1e-6)
    assert st[1].M == pytest.approx(1.0, abs=1e-4)
    assert st[1].p == pytest.approx(p_sonic, rel=1e-4)
    assert st[1].p > 0.45 * pt_in  # detached above the specification
    assert st[1].pt == pytest.approx(pt_in, rel=1e-6)  # lossless to the throat

    # the choking diagnostic flags the sonic edge
    flagged = net.choking_report(res.x)
    assert any(e.index == 1 for e, M, ratio in flagged)
    # subsonic networks report nothing
    net2 = Network(GAS, p_ref=101325.0, T_ref=300.0)
    inl = net2.add(MassFlowInlet(mdot=5.0, Tt=300.0, name="inlet"))
    out = net2.add(PressureOutlet(p=101325.0, name="outlet"))
    net2.connect(inl, out, area=0.5)
    res2 = solve(net2)
    assert res2.converged
    assert net2.choking_report(res2.x) == []


def _bridge_network(K1, K2, K3, K4):
    net = Network(GAS, p_ref=2.5e5, T_ref=500.0, mdot_ref=5.0)
    inl = net.add(TotalPressureInlet(pt=3.0e5, Tt=500.0, name="src"))
    spl = net.add(LosslessSplitter(3, name="split"))
    kA1 = net.add(LossElement(K=K1, name="K1"))
    kA2 = net.add(LossElement(K=K2, name="K2"))
    kB1 = net.add(LossElement(K=K3, name="K3"))
    kB2 = net.add(LossElement(K=K4, name="K4"))
    jt = net.add(JunctionStaticP(3, name="mid-top"))
    jb = net.add(JunctionStaticP(3, name="mid-bot"))
    jun = net.add(JunctionStaticP(3, name="merge"))
    out = net.add(PressureOutlet(p=2.0e5, name="sink"))
    a = 0.02
    net.connect(inl, spl, area=2 * a)
    net.connect(spl, kA1, area=a)
    net.connect(kA1, jt, area=a)
    net.connect(spl, kB1, area=a)
    net.connect(kB1, jb, area=a)
    e_bridge = net.connect(jt, jb, area=0.5 * a)
    net.connect(jt, kA2, area=a)
    net.connect(kA2, jun, area=a)
    net.connect(jb, kB2, area=a)
    net.connect(kB2, jun, area=a)
    net.connect(jun, out, area=2 * a)
    return net, e_bridge


def test_wheatstone_bridge_reversal_and_balance():
    """Interior edge whose flow direction is decided by the physics: the
    bridge flow is antisymmetric under mirroring the resistances and exactly
    zero for a balanced bridge."""
    net1, eb1 = _bridge_network(2.0, 8.0, 8.0, 2.0)
    r1 = solve(net1, x0=net1.initial_guess(mdot0=0.0))
    assert r1.converged
    m1 = net1.states(r1.x)[eb1].mdot
    assert m1 > 0.1  # top -> bottom

    net2, eb2 = _bridge_network(8.0, 2.0, 2.0, 8.0)
    r2 = solve(net2, x0=net2.initial_guess(mdot0=0.0))
    assert r2.converged
    m2 = net2.states(r2.x)[eb2].mdot
    assert m2 == pytest.approx(-m1, rel=1e-6)  # exact antisymmetry

    net3, eb3 = _bridge_network(4.0, 4.0, 4.0, 4.0)
    r3 = solve(net3, x0=net3.initial_guess(mdot0=0.0))
    assert r3.converged
    assert abs(net3.states(r3.x)[eb3].mdot) < 1e-6  # balanced bridge


def test_manifold_network_with_mixing():
    """Larger heterogeneous network (examples/ex4): two-temperature sources,
    mixing chamber, lossless distribution manifold, three dissimilar
    branches, high-subsonic Mach -- from a quiescent start."""
    net = Network(GAS, p_ref=2.5e5, T_ref=500.0, mdot_ref=10.0)
    cold = net.add(MassFlowInlet(mdot=8.0, Tt=450.0, name="cold-src"))
    hot = net.add(TotalPressureInlet(pt=4.5e5, Tt=800.0, name="hot-src"))
    mix = net.add(JunctionStaticP(3, name="mix-chamber"))
    feed = net.add(IsentropicAreaChange(name="feed-pipe"))
    man = net.add(LosslessSplitter(4, name="manifold"))
    k1 = net.add(LossElement(K=3.0, name="brA-loss"))
    outA = net.add(PressureOutlet(p=2.2e5, name="outA"))
    sx = net.add(SuddenAreaChange(name="brB-dump"))
    outB = net.add(PressureOutlet(p=3.4e5, name="outB"))
    orf = net.add(IsentropicAreaChange(name="brC-nozzle"))
    k2 = net.add(LossElement(K=1.5, name="brC-loss"))
    outC = net.add(PressureOutlet(p=2.7e5, name="outC"))
    e_cold = net.connect(cold, mix, area=0.06)
    e_hot = net.connect(hot, mix, area=0.03)
    e_m1 = net.connect(mix, feed, area=0.08)
    e_m2 = net.connect(feed, man, area=0.035)
    eA1 = net.connect(man, k1, area=0.02)
    net.connect(k1, outA, area=0.02)
    eB1 = net.connect(man, sx, area=0.012)
    net.connect(sx, outB, area=0.03)
    eC1 = net.connect(man, orf, area=0.02)
    net.connect(orf, k2, area=0.008)
    net.connect(k2, outC, area=0.008)

    res = solve(net, x0=net.initial_guess(mdot0=0.0))
    assert res.converged
    st = net.states(res.x)

    # mixing: edge after the chamber carries the mass-weighted enthalpy
    Tt_mix = (st[e_cold].mdot * 450.0 + st[e_hot].mdot * 800.0) / (
        st[e_cold].mdot + st[e_hot].mdot
    )
    assert st[e_m1].Tt == pytest.approx(Tt_mix, rel=1e-6)
    # splits add up to the feed flow
    assert st[eA1].mdot + st[eB1].mdot + st[eC1].mdot == pytest.approx(
        st[e_m2].mdot, rel=1e-9
    )
    # high-subsonic but not supersonic anywhere
    maxM = max(abs(s.M) for s in st)
    assert 0.6 < maxM < 1.0
    conservation_ok(net, res.x)
