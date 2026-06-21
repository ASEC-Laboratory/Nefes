"""Mean-flow Newton solver (control layer, above the @njit line)."""

from .control import solve, SolveResult, initial_guess

__all__ = ["solve", "SolveResult", "initial_guess"]
