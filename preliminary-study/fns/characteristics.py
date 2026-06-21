"""Characteristic variables (f, g, h) of the 1-D Euler equations and their
relation to the network solution variables (mdot, p, h_t).

Definitions (the user's convention):

    u' = f - g
    p' = rho * c * (f + g)
    rho' = h + p' / c^2

f: downstream-running acoustic wave, g: upstream-running acoustic wave,
h: entropy wave (advected with the flow).

This module provides the exact linear maps between the characteristic
perturbations w = (f, g, h) and the perturbations of the network unknowns
dx = (d mdot, d p, d h_t) at a given edge state.  Two uses:

1. Diagnostics: express a Newton update or a residual in wave amplitudes.
2. The acoustic (perturbation) network: the element Jacobian blocks evaluated
   at the converged mean state, transformed with these maps, ARE the
   zero-frequency acoustic jump conditions -- the "100% consistency" goal of
   the framework.  (Finite-frequency terms add edge propagation phases and
   element storage terms on top of the same matrices.)

Note on mean-flow convergence (proved numerically in
tests/test_characteristics.py): solving the Newton system in w instead of dx
is a *similarity transformation of the linear system* and returns the
identical update dx = T w.  A change of solution basis cannot change Newton's
convergence; the robustness problems of the earlier prototypes were caused by
the equation set (degenerate advection rows, sign() upwinding, scaling), not
by the choice of primitive vs characteristic unknowns.
"""

import numpy as np


def char_to_dq(state, gas):
    """3x3 matrix R: (d rho, d u, d p) = R @ (f, g, h)."""
    rho, c = state.rho, state.c
    return np.array(
        [
            [rho / c, rho / c, 1.0],
            [1.0, -1.0, 0.0],
            [rho * c, rho * c, 0.0],
        ]
    )


def dq_to_char(state, gas):
    """Inverse map: (f, g, h) = L @ (d rho, d u, d p)."""
    rho, c = state.rho, state.c
    return np.array(
        [
            [0.0, 0.5, 0.5 / (rho * c)],
            [0.0, -0.5, 0.5 / (rho * c)],
            [1.0, 0.0, -1.0 / (c * c)],
        ]
    )


def dq_to_dx(state, gas):
    """3x3 matrix: (d mdot, d p, d h_t) from (d rho, d u, d p).

    mdot = rho * u * A
    h_t  = cp/R * p/rho + u^2/2   (calorically perfect gas)
    """
    rho, u, p, A = state.rho, state.u, state.p, state.area
    K = gas.K
    return np.array(
        [
            [u * A, rho * A, 0.0],
            [0.0, 0.0, 1.0],
            [-K * p / rho**2, u, K / rho],
        ]
    )


def char_to_dx(state, gas):
    """T: (d mdot, d p, d h_t) = T @ (f, g, h)."""
    return dq_to_dx(state, gas) @ char_to_dq(state, gas)


def dx_to_char(state, gas):
    return np.linalg.inv(char_to_dx(state, gas))


def transformation_blocks(network, x):
    """Block-diagonal T for the whole network: dx = T @ w (w stacked per edge)."""
    states = network.states(x)
    n = network.n_unknowns
    T = np.zeros((n, n))
    for e in network.edges:
        i = 3 * e.index
        T[i : i + 3, i : i + 3] = char_to_dx(states[e.index], network.gas)
    return T
