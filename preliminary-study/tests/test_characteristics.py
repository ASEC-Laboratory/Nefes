"""Characteristic-variable maps and the Newton-equivalence demonstration."""

import numpy as np
import pytest

from fns import (
    AIR,
    Network,
    MassFlowInlet,
    IsentropicAreaChange,
    SuddenAreaChange,
    PressureOutlet,
    recover_state,
    solve,
    complex_step_jacobian,
)
from fns.characteristics import (
    char_to_dq,
    dq_to_char,
    char_to_dx,
    dx_to_char,
    transformation_blocks,
)


def _example_state():
    rho, u, T, area = 0.9, 120.0, 400.0, 0.25
    p = rho * AIR.R * T
    ht = AIR.cp * T + 0.5 * u * u
    return recover_state(rho * u * area, p, ht, area, AIR)


def test_char_maps_are_inverses():
    st = _example_state()
    R = char_to_dq(st, AIR)
    L = dq_to_char(st, AIR)
    assert np.allclose(L @ R, np.eye(3), atol=1e-13)
    T = char_to_dx(st, AIR)
    Ti = dx_to_char(st, AIR)
    assert np.allclose(Ti @ T, np.eye(3), atol=1e-12)


def test_char_to_dx_against_complex_step():
    """The analytic map (f,g,h) -> (dmdot, dp, dht) must match differentiating
    the nonlinear forward map numerically."""
    st = _example_state()
    T = char_to_dx(st, AIR)

    def xvec(drho, du, dp):
        rho, u, p = st.rho + drho, st.u + du, st.p + dp
        Tloc = p / (rho * AIR.R)
        return np.array([rho * u * st.area, p, AIR.cp * Tloc + 0.5 * u * u])

    x0 = xvec(0.0, 0.0, 0.0)
    R = char_to_dq(st, AIR)
    eps = 1e-6
    for k, w in enumerate(np.eye(3)):
        dq = R @ w  # (drho, du, dp) of a unit characteristic
        dx_fd = (xvec(*(eps * dq)) - xvec(*(-eps * dq))) / (2 * eps)
        assert np.allclose(dx_fd, T @ w, rtol=1e-6, atol=1e-8), f"char {k}"


def _small_network():
    net = Network(AIR, p_ref=101325.0, T_ref=300.0)
    inl = net.add(MassFlowInlet(mdot=20.0, Tt=450.0, name="inlet"))
    iac = net.add(IsentropicAreaChange(name="iac"))
    sx = net.add(SuddenAreaChange(name="sx"))
    out = net.add(PressureOutlet(p=101325.0, name="outlet"))
    net.connect(inl, iac, area=0.4)
    net.connect(iac, sx, area=0.15)
    net.connect(sx, out, area=0.3)
    return net


def test_newton_in_characteristics_equals_newton_in_primitives():
    """The central theoretical point (REVIEW.md): solving the Newton system in
    characteristic amplitudes w and mapping back, dx = T w, yields the
    IDENTICAL update as solving directly for dx.  A nonsingular change of the
    solution basis cannot alter Newton's iterates, so by itself it cannot
    improve mean-flow convergence."""
    net = _small_network()
    x = net.initial_guess(mdot0=12.0)

    def res(z):
        return net.residual(z)

    J = complex_step_jacobian(res, x)
    R = net.residual(x)

    dx_direct = np.linalg.solve(J, -R)

    T = transformation_blocks(net, x)
    w = np.linalg.solve(J @ T, -R)  # Newton system in characteristic unknowns
    dx_via_char = T @ w

    assert np.allclose(dx_direct, dx_via_char, rtol=1e-9, atol=1e-12)


def test_characteristic_amplitudes_of_converged_solution():
    """At the converged mean state the Newton update, expressed in
    characteristic amplitudes, vanishes (residual-driven waves are zero)."""
    net = _small_network()
    sol = solve(net)
    assert sol.converged

    J = complex_step_jacobian(net.residual, sol.x)
    R = net.residual(sol.x)
    T = transformation_blocks(net, sol.x)
    w = np.linalg.solve(J @ T, -R)
    assert np.max(np.abs(w)) < 1e-7
