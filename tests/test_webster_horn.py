"""Validation: a resolved ``tapered_duct`` converges to the true Webster horn.

The taper composite's convergence is otherwise only ever checked against *itself* (Cauchy
refinement, `test_composite_elements.py`), which answers "does it settle?" but not "does it
settle on the right answer?".  This module supplies the independent reference.

For zero mean flow the one-dimensional equations of a passage of area ``A(x)`` are

    dp/dx = -i omega rho u
    du/dx = -i omega p / (rho c^2)  -  u A'(x)/A(x)

(continuity ``i omega p/c^2 + (rho/A) d(A u)/dx = 0`` and momentum ``i omega rho u = -dp/dx``,
in the ``e^{+i omega t}`` convention).  Integrating ``Y' = M(x) Y`` from ``Y(0) = I`` gives the
exact ``(p, u)`` transfer matrix.  This is an *independent* solution -- different equations,
different discretisation, a different code path, and no closed form taken on trust; the
reference is itself checked against the reciprocity identity ``det T = A_in/A_out``.

The taper discretises the horn into ``N`` uniform ducts joined by ``N`` compact area changes.
That staircase is endpoint-biased rather than centre-frozen, so convergence is first order in
``1/N`` -- which is what these tests pin, both the rate and the limit.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import nefes
from nefes.assembly.recover import ES_C, ES_RHO
from nefes.elements import catalog as cat
from nefes.solver.report import states_table

CFG = nefes.perfect_gas(287.0, 1.4)
P0, T0 = 101325.0, 300.0

A_IN, A_OUT, LENGTH = 0.01, 0.04, 0.5
FLARE = np.log(A_OUT / A_IN) / LENGTH  # exponential flare rate m, A(x) = A_in exp(m x)
FREQS = np.array([120.0, 250.0, 400.0, 650.0])  # all above the m/2 cut-on (~77 Hz)


def _area(x):
    return A_IN * np.exp(FLARE * x)


def _webster_transfer_matrix(freq, rho, c):
    """Exact ``(p, u)`` transfer matrix over the horn by numerical integration."""
    omega = 2.0 * np.pi * freq

    def rhs(_x, y):
        block = np.array([[0.0, -1j * omega * rho], [-1j * omega / (rho * c * c), -FLARE]], dtype=complex)
        return (block @ y.reshape(2, 2)).ravel()

    sol = solve_ivp(rhs, (0.0, LENGTH), np.eye(2, dtype=complex).ravel(), rtol=1e-12, atol=1e-14)
    assert sol.success
    return sol.y[:, -1].reshape(2, 2)


def _nefes_transfer_matrix(n_segments, freqs):
    """Nefes' ``(p, u)`` transfer matrix across the taper, plus the mean ``(rho, c)``."""
    net = nefes.Network(
        CFG,
        nodes=[
            cat.total_pressure_inlet(P0, T0),  # equal in/out pressure -> quiescent, as Webster assumes
            cat.tapered_duct(_area, length=LENGTH, n_segments=n_segments, name="horn"),
            cat.pressure_outlet(P0, T0),
        ],
        edges=[(0, 1, A_IN), (1, 2, A_OUT)],
        # a quiescent network has no realized flow to derive a mass scale from, so the
        # residual scaling needs one supplied explicitly
        mdot_ref=0.5,
    )
    sol = net.solve()
    assert sol.converged, (sol.residual_norm, sol.print_residuals())
    est = states_table(sol.problem, sol.x)
    rho, c = float(est[ES_RHO, 0]), float(est[ES_C, 0])
    resp = sol.perturbation_response(freqs, excite=("acoustic",))
    t_char = np.asarray(resp.transfer_matrix(0, 1, basis="char")).reshape(-1, 2, 2)
    # (p, u) = B (f, g) with p = rho c (f+g), u = f-g; at rest rho c matches at both ends
    basis = np.array([[rho * c, rho * c], [1.0, -1.0]], dtype=complex)
    return np.einsum("ij,kjl,lm->kim", basis, t_char, np.linalg.inv(basis)), rho, c


def _max_rel_error(n_segments):
    t_nefes, rho, c = _nefes_transfer_matrix(n_segments, FREQS)
    errs = []
    for j, freq in enumerate(FREQS):
        t_ref = _webster_transfer_matrix(freq, rho, c)
        errs.append(np.abs(t_nefes[j] - t_ref).max() / np.abs(t_ref).max())
    return max(errs)


def test_webster_reference_is_reciprocal():
    """Sanity on the reference itself: det(T) = A_in/A_out for a lossless passage at rest."""
    _, rho, c = _nefes_transfer_matrix(2, FREQS[:1])
    for freq in FREQS:
        det = np.linalg.det(_webster_transfer_matrix(freq, rho, c))
        assert det == pytest.approx(A_IN / A_OUT, abs=1e-10)


def test_tapered_duct_converges_to_the_webster_horn():
    """The taper converges on the true horn, not merely on itself."""
    assert _max_rel_error(64) < 2.0e-2


def test_taper_convergence_is_first_order():
    """Doubling N halves the error -- the documented O(1/N) rate of the endpoint-biased staircase."""
    errors = {n: _max_rel_error(n) for n in (8, 16, 32, 64)}
    assert all(errors[2 * n] < errors[n] for n in (8, 16, 32))  # monotone
    orders = [np.log2(errors[n] / errors[2 * n]) for n in (8, 16, 32)]
    assert all(0.8 < o < 1.3 for o in orders), orders
