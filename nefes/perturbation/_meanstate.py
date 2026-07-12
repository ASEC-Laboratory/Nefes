"""Let the perturbation entry points take a solved ``Solution`` in place of the
low-level ``(problem, mean state)`` pair.

The perturbation routines operate on a compiled problem and a converged mean-flow
state vector.  A user working through the :class:`nefes.Network` / :class:`nefes.Solution`
front door holds a ``Solution``, not that pair, so the :func:`accepts_solution` decorator
lets every routine accept the ``Solution`` directly and unpack it internally.

This module exports :func:`accepts_solution`.
"""

from __future__ import annotations

import functools
from typing import Callable


def _is_solution(obj) -> bool:
    """True when ``obj`` stands in for a ``(problem, mean state)`` pair.

    A :class:`nefes.Solution` exposes both the compiled problem (``.problem``) and the
    converged mean state (``.x``); a ``CompiledProblem`` exposes neither.  Testing for the
    two attributes distinguishes them without importing the shell layer, which would be a
    circular import.
    """
    return hasattr(obj, "problem") and hasattr(obj, "x")


def accepts_solution(fn: Callable) -> Callable:
    """Wrap ``fn(prob, x_bar, ...)`` so its first argument may instead be a ``Solution``.

    A solved :class:`nefes.Solution` passed as the first argument is expanded to
    ``fn(sol.problem, sol.x, ...)``; a ``CompiledProblem`` with an explicit mean state is
    forwarded unchanged.  Pass either the solution or the ``(problem, mean state)`` pair,
    never both.

    Parameters
    ----------
    fn : callable
        A perturbation routine whose first two positional arguments are the compiled
        problem and the mean-flow state.

    Returns
    -------
    callable
        The wrapped routine, additionally accepting a ``Solution`` as its first argument.
    """

    @functools.wraps(fn)
    def wrapper(state, *args, **kwargs):
        if _is_solution(state):
            return fn(state.problem, state.x, *args, **kwargs)
        return fn(state, *args, **kwargs)

    return wrapper
