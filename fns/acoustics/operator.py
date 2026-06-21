"""The acoustic operator A(omega) = J_alg + i*omega*M + P(omega) + S(omega).

``J_alg`` is the converged mean-flow Jacobian -- the zero-frequency acoustic
operator (theory.md s12.1) -- reused verbatim from the @njit complex-step
machinery (no new kernel).  ``M`` is the storage block (compliance/inertance);
it is non-zero only for finite-volume elements, which v1's catalog does not yet
carry, so it defaults to the zero block of the same pattern.  The propagation
``P`` and source ``S`` stamps live with their element models (see ``duct.py``).
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..assemble import jacobian


@dataclass
class AcousticBlocks:
    """Frequency-independent blocks, built once and cached for a sweep."""

    J_alg: sp.csc_matrix  # complex, the converged Jacobian
    M: sp.csc_matrix  # complex, storage (zero unless volumes are present)
    n: int


def build_acoustic_blocks(prob, x_bar, eps=None, eps_fb=1e-6):
    """Build ``(J_alg, M)`` at the frozen mean state ``x_bar`` (shape (3, E)).

    ``J_alg`` is assembled with the regularizations turned down (the
    un-regularized variant of theory.md s12.6) at ``stab = 0``.
    """
    if eps is None:
        eps = 1e-4 * prob.var_scale[0]
    J = jacobian(prob, np.ascontiguousarray(x_bar), eps, eps_fb, 0.0).astype(np.complex128)
    n = J.shape[0]
    M = sp.csc_matrix((n, n), dtype=np.complex128)
    return AcousticBlocks(J_alg=J.tocsc(), M=M, n=n)


def assemble_acoustic(omega, blocks: AcousticBlocks):
    """Stamp ``A(omega) = J_alg + i*omega*M`` (plus element P/S stamps elsewhere).

    At ``omega = 0`` this returns exactly ``J_alg`` -- the founding consistency
    between the steady Jacobian and the acoustic operator.
    """
    return (blocks.J_alg + 1j * omega * blocks.M).tocsc()
