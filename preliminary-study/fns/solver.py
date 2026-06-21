"""Scaled, damped Newton solver with pseudo-transient continuation (PTC).

Three robustness ingredients, each addressing a specific failure mode of the
earlier prototypes (see REVIEW.md):

1. **Scaling.**  The system is solved in nondimensional variables
   (mdot/mdot_ref, p/p_ref, ht/h_ref) with residual rows scaled by their
   physical reference (mass rows by mdot_ref, pressure rows by p_ref, ...).
   Absolute tolerances and the PTC parameter then act uniformly; without this
   the energy rows (~3e5 J/kg) numerically drown the mass rows (~1 kg/s).

2. **Levenberg-Marquardt damping.**  Solve (J^T J + lam*I) dy = -J^T R: for
   lam -> 0 this is the exact Newton step (square nonsingular J), for large
   lam it turns toward steepest descent of the merit ||R||^2 -- every step is
   a guaranteed descent direction, unlike a naively damped (J + lam*I) step,
   which can deadlock in curved valleys.  The damping also regularizes
   states where the steady Jacobian is exactly singular, e.g. the
   indeterminate flow split of a symmetric branching network at the
   quiescent initial state.  lam shrinks on accepted steps (recovering
   quadratic Newton convergence near the solution) and grows on rejected
   ones.

3. **Physicality guard.**  A step is rejected (and lam increased) if any
   edge state becomes unrecoverable (p > 0, h_t > 0 are required; given
   those, the density recovery cannot fail) or the residual norm does not
   decrease.

The Jacobian is computed by complex-step differentiation: exact to machine
precision, no analytic-derivative bookkeeping to get wrong.  All residual
code is written to be complex-analytic (fns.smooth, fns.state).
"""

from dataclasses import dataclass, field

import numpy as np

from .state import StateError


def complex_step_jacobian(fun, y, h=1e-30):
    n = y.size
    r0 = fun(y)
    J = np.empty((r0.size, n))
    for j in range(n):
        yc = y.astype(complex)
        yc[j] += 1j * h
        J[:, j] = np.imag(fun(yc)) / h
    return J


@dataclass
class SolveResult:
    x: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    history: list = field(default_factory=list)

    def __repr__(self):
        s = "converged" if self.converged else "NOT converged"
        return f"<SolveResult {s} in {self.iterations} its, ||R|| = {self.residual_norm:.3e}>"


def solve(
    network,
    x0=None,
    tol: float = 1e-10,
    max_iter: int = 100,
    lam0: float = 1e-2,
    lam_min: float = 1e-14,
    lam_max: float = 1e6,
    stab_stages=(0.1, 0.01, 0.0),
    verbose: bool = False,
) -> SolveResult:
    """Solve the steady network equations. Returns SolveResult (x in physical units).

    ``stab_stages`` is the vanishing-friction homotopy schedule: the system is
    solved once per stage (warm-started), with interior pressure rows carrying
    a linear resistance ``stage * p_ref/mdot_ref * mdot``.  The last stage
    must be 0.0 so that the converged equations are the exact ones.  Pass
    ``stab_stages=(0.0,)`` for a single pure Newton solve.
    """
    if stab_stages[-1] != 0.0:
        raise ValueError("the last homotopy stage must be 0.0 (exact equations)")
    network.check_square()

    x = network.initial_guess() if x0 is None else np.asarray(x0, dtype=float)
    total_its = 0
    history = []
    for k, stage in enumerate(stab_stages):
        last = k == len(stab_stages) - 1
        stage_tol = tol if last else max(tol, 1e-8)
        if verbose and len(stab_stages) > 1:
            print(f" homotopy stage {k + 1}/{len(stab_stages)}: stab = {stage}")
        result = _solve_stage(
            network, x, stage, stage_tol, max_iter, lam0, lam_min, lam_max, verbose
        )
        x = result.x
        total_its += result.iterations
        history.extend(result.history)
        if not result.converged and last:
            return SolveResult(x, False, total_its, result.residual_norm, history)
    return SolveResult(x, True, total_its, result.residual_norm, history)


def _solve_stage(
    network, x0, stab, tol, max_iter, lam0, lam_min, lam_max, verbose
) -> SolveResult:
    xs = network.variable_scales()
    rs = network.residual_scales()

    def scaled_residual(y):
        return network.residual(y * xs, stab=stab) / rs

    def norm_or_inf(y):
        try:
            return float(np.linalg.norm(scaled_residual(np.real(y)))), True
        except StateError:
            return np.inf, False

    y = np.asarray(x0, dtype=float) / xs

    R = scaled_residual(y)
    norm = float(np.linalg.norm(R))
    lam = lam0
    history = [norm]
    n = y.size

    for it in range(1, max_iter + 1):
        if not np.all(np.isfinite(R)):
            raise FloatingPointError("non-finite residual encountered")
        if norm < tol:
            return SolveResult(y * xs, True, it - 1, norm, history)

        J = complex_step_jacobian(scaled_residual, y)
        JtJ = J.T @ J
        JtR = J.T @ R

        # Levenberg-Marquardt inner loop: increase lam until a step is
        # accepted (residual decreases and all states stay physical).
        accepted = False
        for _ in range(40):
            try:
                dy = np.linalg.solve(JtJ + lam * np.eye(n), -JtR)
            except np.linalg.LinAlgError:
                lam = max(3.0 * lam, 1e-12)
                continue
            norm_new, ok = norm_or_inf(y + dy)
            if ok and norm_new < norm:
                accepted = True
                break
            lam = max(3.0 * lam, 1e-12)
        if not accepted:
            return SolveResult(np.real(y) * xs, False, it, norm, history)

        y = y + dy
        R = scaled_residual(y)
        norm = norm_new
        history.append(norm)
        lam = max(lam * 0.3, lam_min)

        if verbose:
            print(f"  it {it:3d}  ||R|| = {norm:12.5e}   lam = {lam:9.2e}")

    return SolveResult(y * xs, norm < tol, max_iter, norm, history)
