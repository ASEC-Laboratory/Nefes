"""Smooth (C-infinity), complex-step-safe replacements for non-smooth functions.

Every function here is analytic in a neighbourhood of the real axis, so the
complex-step derivative trick (x + i*h) propagates exact derivatives through
them.  Never use ``abs``, ``sign``, ``max`` or branching on the solution
variables inside residual definitions -- use these instead.

The regularization scale ``delta`` should be chosen relative to a problem
reference value (e.g. ``1e-3 * mdot_ref``).  The error introduced at a
converged state with |x| >> delta is O(delta^2 / x^2), i.e. quadratically
small.
"""

import numpy as np


def smooth_abs(x, delta):
    """|x| regularized as sqrt(x^2 + delta^2) - delta is NOT subtracted.

    smooth_abs(0) = delta, smooth_abs(x) -> |x| + delta^2/(2|x|) for |x| >> delta.
    """
    return np.sqrt(x * x + delta * delta)


def smooth_pos(x, delta):
    """max(x, 0) regularized: 0.5 * (x + sqrt(x^2 + delta^2)).

    smooth_pos(0) = delta/2; for x >> delta -> x; for x << -delta -> delta^2/(4|x|).
    """
    return 0.5 * (x + np.sqrt(x * x + delta * delta))


def smooth_step(x, delta):
    """Heaviside step regularized: 0.5 * (1 + x / sqrt(x^2 + delta^2)).

    smooth_step(0) = 1/2; -> 1 for x >> delta, -> 0 for x << -delta.
    """
    return 0.5 * (1.0 + x / np.sqrt(x * x + delta * delta))


def smooth_sign_sq(x, delta):
    """x * |x| regularized as x * sqrt(x^2 + delta^2) (smooth at x = 0)."""
    return x * np.sqrt(x * x + delta * delta)


def fischer_burmeister(a, b, eps):
    """Smoothed Fischer-Burmeister complementarity residual.

    The exact function phi(a, b) = a + b - sqrt(a^2 + b^2) vanishes iff
    a >= 0, b >= 0 and a*b = 0 -- i.e. it encodes an either/or regime switch
    ("either a is slack and b is zero, or a is zero and b is slack") as a
    SINGLE residual with no branching.  The eps-smoothing rounds the corner:
    on the smoothed root manifold 2*a*b = eps^2, so within a regime the
    "off" variable is pinned to eps^2/(2*active) -- a quadratically small
    bias, same philosophy as the rest of fns's regularizations.

    Used for emergent regime switches: choking (subsonic-and-lossless vs
    sonic-and-lossy) in area changes and outlets.
    """
    return a + b - np.sqrt(a * a + b * b + eps * eps)
