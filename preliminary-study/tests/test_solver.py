"""Solver machinery: complex-step Jacobian correctness, robustness to bad starts."""

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
)


def _net():
    net = Network(AIR, p_ref=101325.0, T_ref=300.0)
    inl = net.add(MassFlowInlet(mdot=15.0, Tt=400.0, name="inlet"))
    iac = net.add(IsentropicAreaChange(name="iac"))
    out = net.add(PressureOutlet(p=101325.0, name="outlet"))
    net.connect(inl, iac, area=0.3)
    net.connect(iac, out, area=0.2)
    return net


def test_complex_step_jacobian_matches_fd():
    """Compare on the nondimensionalized residual so entries are O(1)."""
    net = _net()
    xs = net.variable_scales()
    rs = net.residual_scales()

    def f(y):
        return net.residual(y * xs) / rs

    y = net.initial_guess(mdot0=10.0) / xs
    J_cs = complex_step_jacobian(f, y)

    J_fd = np.empty_like(J_cs)
    for j in range(y.size):
        e = np.zeros_like(y)
        e[j] = 1e-7 * max(abs(y[j]), 1.0)
        J_fd[:, j] = (f(y + e) - f(y - e)) / (2 * e[j])

    assert np.allclose(J_cs, J_fd, rtol=1e-5, atol=1e-6 * np.max(np.abs(J_cs)))


@pytest.mark.parametrize("mdot0", [0.0, 0.5, -5.0, 40.0])
def test_converges_from_poor_starts(mdot0):
    net = _net()
    res = solve(net, x0=net.initial_guess(mdot0=mdot0))
    assert res.converged
    st = net.states(res.x)
    assert st[0].mdot == pytest.approx(15.0, rel=1e-9)


def test_quiescent_symmetric_branching_start():
    """Indeterminate split at zero flow: PTC must regularize the singular
    Jacobian of the perfectly symmetric quiescent state."""
    net = Network(AIR, p_ref=101325.0, T_ref=300.0)
    inl = net.add(MassFlowInlet(mdot=10.0, Tt=300.0, name="inlet"))
    spl = net.add(LosslessSplitter(3, name="split"))
    jun = net.add(JunctionStaticP(3, name="junction"))
    out = net.add(PressureOutlet(p=101325.0, name="outlet"))
    net.connect(inl, spl, area=0.2)
    eA = net.connect(spl, jun, area=0.1)
    eB = net.connect(spl, jun, area=0.1)
    net.connect(jun, out, area=0.2)

    res = solve(net, x0=net.initial_guess(mdot0=0.0))
    assert res.converged
    st = net.states(res.x)
    assert st[eA].mdot == pytest.approx(5.0, rel=1e-6)
    assert st[eB].mdot == pytest.approx(5.0, rel=1e-6)


def test_final_stage_solves_exact_equations():
    """The homotopy must hand back residuals of the UNMODIFIED equations."""
    net = _net()
    res = solve(net)
    assert res.converged
    R = net.residual(res.x, stab=0.0)
    rs = net.residual_scales()
    assert np.max(np.abs(R / rs)) < 1e-10


def test_pure_newton_option():
    net = _net()
    res = solve(net, x0=net.initial_guess(mdot0=10.0), stab_stages=(0.0,))
    assert res.converged
    assert res.iterations <= 12
