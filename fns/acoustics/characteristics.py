"""Characteristic-variable maps at a mean edge state (theory.md s12.2).

Characteristic amplitudes ``w = (f, g, h)`` of the 1-D Euler equations:

    u'   = f - g
    p'   = rho*c*(f + g)
    rho' = h + p'/c^2

f: downstream acoustic wave (speed u+c); g: upstream (u-c); h: entropy (speed u).
``char_to_dx`` is the per-edge block ``T_e`` mapping ``w`` to the perturbations of
the network unknowns ``(d_mdot, d_p, d_h_t)``; ``dx_to_char`` is ``L_e = T_e^-1``.
At the converged mean state these blocks turn the algebraic Jacobian into the
zero-frequency acoustic jump conditions.
"""

import numpy as np

from ..derive import ES_RHO, ES_U, ES_P, ES_C, ES_AREA


def char_to_dq(rho, c):
    """R: (d_rho, d_u, d_p) = R @ (f, g, h)."""
    return np.array(
        [
            [rho / c, rho / c, 1.0],
            [1.0, -1.0, 0.0],
            [rho * c, rho * c, 0.0],
        ]
    )


def dq_to_dx(rho, u, p, area, K):
    """(d_mdot, d_p, d_h_t) from (d_rho, d_u, d_p) for a calorically perfect gas.

    mdot = rho*u*A ;  h_t = (cp/R) p/rho + u^2/2.
    """
    return np.array(
        [
            [u * area, rho * area, 0.0],
            [0.0, 0.0, 1.0],
            [-K * p / rho**2, u, K / rho],
        ]
    )


def char_to_dx(rho, c, u, p, area, K):
    """T_e: (d_mdot, d_p, d_h_t) = T_e @ (f, g, h)."""
    return dq_to_dx(rho, u, p, area, K) @ char_to_dq(rho, c)


def dx_to_char(rho, c, u, p, area, K):
    """L_e = T_e^-1: (f, g, h) = L_e @ (d_mdot, d_p, d_h_t)."""
    return np.linalg.inv(char_to_dx(rho, c, u, p, area, K))


def edge_transforms(est, K):
    """Per-edge (T_e, L_e) lists from the mean edge-state table ``est``."""
    E = est.shape[1]
    Ts, Ls = [], []
    for e in range(E):
        rho = est[ES_RHO, e]
        c = est[ES_C, e]
        u = est[ES_U, e]
        p = est[ES_P, e]
        area = est[ES_AREA, e]
        T = char_to_dx(rho, c, u, p, area, K)
        Ts.append(T)
        Ls.append(np.linalg.inv(T))
    return Ts, Ls
