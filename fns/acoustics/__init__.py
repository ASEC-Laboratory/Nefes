"""Acoustic / perturbation network built on a converged mean flow (theory.md s12).

A second analysis over the same compiled network and converged state -- not a
second solver.  It reuses the connectivity, the complex-step Jacobian (which is
the zero-frequency acoustic operator), and the frozen mean thermo state.  All
Python/SciPy, no new @njit kernel.
"""

from .characteristics import char_to_dx, dx_to_char, edge_transforms
from .operator import build_acoustic_blocks, assemble_acoustic, AcousticBlocks
from .duct import duct_modes, DuctAcoustics
from .drivers import modes_from_det, scattering_2port

__all__ = [
    "char_to_dx",
    "dx_to_char",
    "edge_transforms",
    "build_acoustic_blocks",
    "assemble_acoustic",
    "AcousticBlocks",
    "duct_modes",
    "DuctAcoustics",
    "modes_from_det",
    "scattering_2port",
]
