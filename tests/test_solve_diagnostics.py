"""Failure diagnostics of ``Network.solve``.

A solve that does not converge returns its (partial) ``Solution`` and warns at the solve
boundary, rather than failing silently and only surfacing later when a field is read.
"""

import warnings

import numpy as np

import nefes
from nefes.elements import catalog as cat
from nefes.thermo.configure import perfect_gas

CFG = perfect_gas(R=287.0, gamma=1.4)


def _duct_network():
    nodes = [
        cat.mass_flow_inlet(2.0, 300.0),
        cat.duct(0.5, name="d"),
        cat.pressure_outlet(101325.0),
    ]
    return nefes.Network(gas=CFG, nodes=nodes, edges=[(0, 1, 0.02), (1, 2, 0.02)])


def test_solve_warns_on_non_convergence():
    """An unreachable tolerance forces non-convergence, which is surfaced as a warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = _duct_network().solve(tol=1e-30, max_iter=3)  # below the round-off floor
    assert not sol.converged
    assert any("did not converge" in str(rec.message) for rec in w)


def test_converged_solve_does_not_warn_about_convergence():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = _duct_network().solve()
    assert sol.converged
    assert not any("did not converge" in str(rec.message) for rec in w)
    assert np.all(np.isfinite(sol.field("T")))  # a converged solve recovers its fields
