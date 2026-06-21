"""Closure adapter: the AD-3 thermo boundary used by the solver.

Maps the band-1 unknowns ``(mdot, p, h_t, Z_el)`` plus the edge ``area`` to the
density ``rho`` and static enthalpy ``h``, resolving the kinetic-energy coupling
``h = h_t - u^2/2`` with ``u = mdot/(rho*A)``.

For a perfect gas this collapses to the single density root (the kinetic-energy
term is already inside ``F(rho)``), so there is one root-find, not a nested fixed
point.  A future opaque-thermo backend (equilibrium/table) implements the same
interface with an outer fixed point; both return ``(rho, h)`` with the
IFT-spliced complex-step seed, keeping the boundary backend-agnostic.
"""

from numba import njit

from .thermo.api import PERFECT_GAS
from .thermo.perfect_gas import pg_solve_density


@njit(cache=True)
def closure_solve(model_id, tf, ti, mdot, p, h_t, Z_el, area):
    """Return ``(rho, h)`` for the edge state.  Dtype-generic, complex-safe."""
    if model_id == PERFECT_GAS:
        K = tf[0] / tf[1]  # cp / R
        m = mdot / area
        rho = pg_solve_density(m, p, h_t, K)
        u = m / rho
        h = h_t - 0.5 * u * u
        return rho, h
    raise ValueError("unknown thermo model_id")
