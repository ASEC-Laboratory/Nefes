"""Edge state recovery from the solution variables (mdot, p, h_t).

Solution variables per edge:
    mdot : mass flow rate, *signed along the edge direction*  [kg/s]
    p    : static pressure                                     [Pa]
    h_t  : specific total enthalpy                             [J/kg]

The full thermodynamic/kinematic state follows from the implicit density
equation (calorically perfect gas):

    F(rho) = rho - p*K / (h_t - m^2 / (2 rho^2)) = 0,   m = mdot/A,  K = cp/R

Key property (the reason this variable set was kept from the user's `01`
iteration):  F is *strictly monotone increasing* on (rho_min, inf) with
F -> -inf at rho_min = |m|/sqrt(2 h_t) and F -> +inf, so the root exists and
is **unique for any mdot** as long as p > 0 and h_t > 0.  There is no
subsonic/supersonic ambiguity (unlike recovery from (mdot, p_t, T_t)), no
choking singularity in the recovery, and reversed flow (mdot < 0) is handled
by exactly the same expressions.

Complex-step support: the real root is found with a safeguarded Newton
iteration; if any input carries an imaginary part, the imaginary part of rho
is attached via the implicit function theorem, which is exact to first order
(all that complex-step differentiation requires).
"""

from dataclasses import dataclass

import numpy as np


class StateError(ValueError):
    """Raised when (mdot, p, h_t) does not describe a physical state."""


def _solve_density_real(m: float, p: float, ht: float, K: float) -> float:
    """Unique real root of F(rho) = rho - p*K/(ht - m^2/(2 rho^2))."""
    if not (p > 0.0) or not (ht > 0.0):
        raise StateError(f"non-physical state: p={p}, h_t={ht}")

    m2 = m * m
    # F(p*K/ht) <= 0 and F is increasing, so the root lies in [p*K/ht, inf).
    lo = p * K / ht
    rho = lo

    # Expanding upper bracket.
    hi = max(2.0 * lo, np.sqrt(m2 / (2.0 * ht)) * 4.0 + lo)
    for _ in range(200):
        H = ht - m2 / (2.0 * hi * hi)
        if H > 0.0 and hi - p * K / H > 0.0:
            break
        hi *= 2.0
    else:  # pragma: no cover
        raise StateError("density bracket expansion failed")

    # Safeguarded Newton (bisection fallback keeps it inside [lo, hi]).
    for _ in range(100):
        H = ht - m2 / (2.0 * rho * rho)
        if H <= 0.0:
            rho = 0.5 * (rho + hi)
            continue
        F = rho - p * K / H
        if F > 0.0:
            hi = rho
        else:
            lo = rho
        dF = 1.0 + p * K * m2 / (rho**3 * H * H)
        step = F / dF
        rho_new = rho - step
        if not (lo < rho_new < hi):
            rho_new = 0.5 * (lo + hi)
        if abs(rho_new - rho) <= 1e-14 * rho:
            return rho_new
        rho = rho_new
    return rho


def solve_density(m, p, ht, K):
    """Density from mass flux density m = rho*u, static p and total enthalpy.

    Accepts real or complex scalars; complex parts are propagated through the
    implicit relation exactly (implicit function theorem).
    """
    mr, pr, hr = float(np.real(m)), float(np.real(p)), float(np.real(ht))
    rho = _solve_density_real(mr, pr, hr, K)

    if isinstance(m, complex) or isinstance(p, complex) or isinstance(ht, complex) or (
        np.iscomplexobj(m) or np.iscomplexobj(p) or np.iscomplexobj(ht)
    ):
        m2 = mr * mr
        H = hr - m2 / (2.0 * rho * rho)
        F_rho = 1.0 + pr * K * m2 / (rho**3 * H * H)
        F_m = -pr * K * mr / (rho * rho * H * H)
        F_p = -K / H
        F_h = pr * K / (H * H)
        drho = -(F_m * np.imag(m) + F_p * np.imag(p) + F_h * np.imag(ht)) / F_rho
        return rho + 1j * drho
    return rho


@dataclass
class EdgeState:
    """Fully recovered state at an edge (velocity signed along edge direction)."""

    mdot: float  # signed mass flow rate along edge direction
    p: float  # static pressure
    ht: float  # total enthalpy
    area: float
    rho: float
    u: float  # signed velocity along edge direction
    T: float  # static temperature
    c: float  # speed of sound
    M: float  # signed Mach number u/c
    pt: float  # total pressure (direction independent)
    Tt: float  # total temperature
    s: float  # entropy invariant p / rho^gamma


def recover_state(mdot, p, ht, area: float, gas) -> EdgeState:
    """Recover the full edge state from (mdot, p, h_t).  Complex-step safe."""
    m = mdot / area
    rho = solve_density(m, p, ht, gas.K)
    u = m / rho
    T = (ht - 0.5 * u * u) / gas.cp
    c = np.sqrt(gas.gamma * gas.R * T)
    M = u / c
    pt = p * (1.0 + 0.5 * (gas.gamma - 1.0) * M * M) ** (gas.gamma / (gas.gamma - 1.0))
    Tt = ht / gas.cp
    s = p / rho**gas.gamma
    return EdgeState(mdot=mdot, p=p, ht=ht, area=area, rho=rho, u=u, T=T, c=c, M=M, pt=pt, Tt=Tt, s=s)
