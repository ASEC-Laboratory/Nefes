"""Smooth (C-infinity), complex-step-safe replacements for non-smooth functions.

Every function here is analytic in a neighbourhood of the real axis, so the
complex-step derivative trick (``x + 1j*h``) propagates exact derivatives
through them.  Never use ``abs``, ``sign``, ``min``, ``max`` or branch on a
solution variable inside a residual definition -- use these instead.

The regularization scale ``delta`` is chosen relative to a problem reference
value (e.g. ``1e-3 * mdot_ref``).  The error introduced at a converged state
with ``|x| >> delta`` is ``O(delta^2 / x^2)`` -- quadratically small.

All functions are ``@njit`` and dtype-generic: numba compiles a ``float64``
specialization (used to evaluate the real residual) and a ``complex128``
specialization (used to seed the complex-step Jacobian) from the same source.
``np.sqrt`` is analytic in both because every radicand here is strictly
positive on the real axis (the ``+ delta^2`` / ``+ eps^2`` floor guarantees it),
so the complex branch never approaches the cut on ``(-inf, 0]``.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def smooth_abs(x, delta):
    """``|x|`` regularized as ``sqrt(x^2 + delta^2)`` (delta is NOT subtracted).

    ``smooth_abs(0) = delta``; for ``|x| >> delta`` it tends to
    ``|x| + delta^2/(2|x|)``.
    """
    return np.sqrt(x * x + delta * delta)


@njit(cache=True)
def smooth_pos(x, delta):
    """``max(x, 0)`` regularized: ``0.5 * (x + sqrt(x^2 + delta^2))``.

    ``smooth_pos(0) = delta/2``; ``-> x`` for ``x >> delta``;
    ``-> delta^2/(4|x|)`` for ``x << -delta``.
    """
    return 0.5 * (x + np.sqrt(x * x + delta * delta))


@njit(cache=True)
def smooth_step(x, delta):
    """Heaviside step regularized: ``0.5 * (1 + x / sqrt(x^2 + delta^2))``.

    ``smooth_step(0) = 1/2``; ``-> 1`` for ``x >> delta``; ``-> 0`` for
    ``x << -delta``.  This is the smooth upwind weight used in the edge
    enthalpy-transport equation.
    """
    return 0.5 * (1.0 + x / np.sqrt(x * x + delta * delta))


@njit(cache=True)
def smooth_sign_sq(x, delta):
    """``x * |x|`` regularized as ``x * sqrt(x^2 + delta^2)`` (smooth at 0).

    The direction-aware dynamic-pressure term used by loss elements: a loss
    proportional to ``x * |x|`` opposes the flow in both directions (second
    law) and stays differentiable through ``x = 0``.
    """
    return x * np.sqrt(x * x + delta * delta)


@njit(cache=True)
def fischer_burmeister(a, b, eps):
    """Smoothed Fischer-Burmeister complementarity residual.

    The exact function ``phi(a, b) = a + b - sqrt(a^2 + b^2)`` vanishes iff
    ``a >= 0``, ``b >= 0`` and ``a*b = 0`` -- it encodes an either/or regime
    switch as a SINGLE residual with no branching.  The ``eps``-smoothing
    rounds the corner: on the smoothed root manifold ``2*a*b = eps^2``, so
    within a regime the "off" variable is pinned to ``eps^2/(2*active)`` -- a
    quadratically small bias, same philosophy as the other regularizations.

    Used for emergent regime switches: choking (subsonic-and-lossless vs
    sonic-and-lossy) in area changes and pressure outlets.
    """
    return a + b - np.sqrt(a * a + b * b + eps * eps)
