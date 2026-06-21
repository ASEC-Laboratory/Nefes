"""Perturbation (acoustic) boundary conditions for single-port terminals.

A single-port element fixes the *mean* boundary condition (a mass flow, a total
pressure, a static pressure, or a wall).  The *perturbation* boundary condition is
extra information the mean BC cannot supply: how the terminal closes the linear
fluctuation problem.  theory.md s12.4 settles the form -- every terminal BC is a
**reflection relation**, diagonal in the wave amplitudes ``w = (f, g, h)``::

    w_incoming - R(omega) * w_outgoing = b(omega)

with ``incoming``/``outgoing`` the characteristic indices fixed by which end of its
edge the terminal sits on (tail -> incoming ``f``, outgoing ``g``; head -> incoming
``g``, outgoing ``f``), ``R`` a reflection coefficient (possibly frequency- and
mean-state-dependent), and ``b`` an optional excitation forcing.  Every flavor a
user reaches for is a choice of ``R`` and ``b``:

===================  =========================================================
kind                 reflection coefficient ``R``
===================  =========================================================
``inherit``          *no stamp* -- keep the linearized mean BC already in J_alg
``hard_wall``        ``+1``                       (``u' = 0``)
``open_end``         ``-1``                       (``p' = 0``, pressure release)
``mean_flow_open_end`` ``-(1 - M)/(1 + M)``       convective open end (``M`` = the
                                                  outward-normal mean Mach)
``anechoic``         ``0``                        (reflection-free termination)
``reflection``       user ``R(omega)``            constant / table / callable
``impedance``        ``(Z - rho c)/(Z + rho c)``  from a (specific or absolute) ``Z``
``excitation``       ``base_R`` (default ``0``), with forcing ``b``
===================  =========================================================

The impedance map uses the **outward-normal** velocity convention, so it is uniform
at an inlet and an outlet: a rigid wall ``Z -> inf`` gives ``R = +1``, a
pressure-release end ``Z -> 0`` gives ``R = -1``, and the matched impedance
``Z = rho c`` gives ``R = 0``.

Each numeric carrier (``R``, ``Z``, ``amplitude``, ``entropy_in``) may be a complex
constant, a frequency table ``(omegas, values)`` interpolated in ``omega``, or a
callable ``omega -> complex`` (Python API only; YAML/UI use constants or a table).
This object lives entirely *above* the @njit line -- it is evaluated on the frozen
mean state at assembly time, so no complex-step differentiation flows through it.
"""

import cmath
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

KINDS = (
    "inherit",
    "hard_wall",
    "open_end",
    "mean_flow_open_end",
    "anechoic",
    "reflection",
    "impedance",
    "excitation",
)


def _eval(value, omega):
    """Evaluate a coefficient at angular frequency ``omega`` (rad/s).

    ``value`` is a complex constant, a callable ``omega -> complex``, or a frequency
    table ``(omegas, values)`` linearly interpolated (real and imaginary parts) in
    ``omega`` and held flat outside its range.
    """
    if value is None:
        return 0.0 + 0.0j
    if callable(value):
        return complex(value(omega))
    if isinstance(value, tuple) and len(value) == 2:
        xs, ys = np.asarray(value[0], dtype=float), np.asarray(value[1], dtype=complex)
        re = np.interp(omega, xs, ys.real)
        im = np.interp(omega, xs, ys.imag)
        return complex(re, im)
    return complex(value)


@dataclass
class PerturbationBC:
    """Acoustic/perturbation closure for a single-port terminal (theory.md s12.4).

    Build one with the named constructors (:meth:`hard_wall`, :meth:`open_end`,
    :meth:`anechoic`, :meth:`reflection`, :meth:`impedance`, :meth:`excitation`,
    :meth:`mean_flow_open_end`); :meth:`inherit` (the default) leaves the linearized
    mean boundary row untouched.

    Attributes
    ----------
    kind : str
        One of :data:`KINDS`.
    R : complex or callable or tuple, optional
        Reflection coefficient for ``kind == "reflection"``.
    Z : complex or callable or tuple, optional
        Acoustic impedance for ``kind == "impedance"`` (specific if ``specific``).
    specific : bool
        If True, ``Z`` is normalized by the characteristic impedance ``rho c``.
    amplitude : complex or callable or tuple, optional
        Acoustic excitation forcing ``b`` for ``kind == "excitation"``.
    base_R : complex or callable or tuple, optional
        Reflection coefficient of an excitation terminal (default ``0`` -- a clean,
        reflection-free source).
    entropy_in : complex or callable or tuple, optional
        Incoming entropy-wave amplitude seated at an inflow terminal (default ``0``).
    family : str
        ``"acoustic"`` (default) or ``"entropy"`` -- which incoming wave an
        :meth:`excitation` drives.
    """

    kind: str = "inherit"
    R: object = None
    Z: object = None
    specific: bool = False
    amplitude: object = None
    base_R: object = None
    entropy_in: object = None
    family: str = "acoustic"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown perturbation BC kind {self.kind!r}; choose from {KINDS}")

    # -- evaluation on the frozen mean state --------------------------------

    def reflection_coefficient(self, omega, rho, c, M) -> Optional[complex]:
        """Reflection coefficient ``R`` at ``omega`` and the terminal mean state.

        Returns ``None`` for ``inherit`` (signalling "do not stamp this terminal").
        ``M`` is the **outward-normal** mean Mach number at the terminal edge.
        """
        k = self.kind
        if k == "inherit":
            return None
        if k == "hard_wall":
            return 1.0 + 0.0j
        if k == "open_end":
            return -1.0 + 0.0j
        if k == "mean_flow_open_end":
            return complex(-(1.0 - M) / (1.0 + M))
        if k == "anechoic":
            return 0.0 + 0.0j
        if k == "reflection":
            return _eval(self.R, omega)
        if k == "impedance":
            z = _eval(self.Z, omega)
            if self.specific:
                z = z * (rho * c)
            zc = rho * c
            return (z - zc) / (z + zc)
        if k == "excitation":
            return _eval(self.base_R, omega)
        raise ValueError(f"unhandled perturbation BC kind {k!r}")  # pragma: no cover

    def forcing(self, omega) -> complex:
        """Acoustic-row excitation forcing ``b`` at ``omega`` (0 unless excitation)."""
        if self.kind == "excitation" and self.family == "acoustic":
            return _eval(self.amplitude, omega)
        return 0.0 + 0.0j

    def entropy_forcing(self, omega) -> complex:
        """Incoming entropy amplitude seated at an inflow terminal, at ``omega``."""
        b = _eval(self.entropy_in, omega) if self.entropy_in is not None else 0.0 + 0.0j
        if self.kind == "excitation" and self.family == "entropy":
            b = b + _eval(self.amplitude, omega)
        return b

    @property
    def stamps_terminal(self) -> bool:
        """True if this BC overwrites the terminal row (everything but ``inherit``)."""
        return self.kind != "inherit"

    # -- named constructors -------------------------------------------------

    @classmethod
    def inherit(cls) -> "PerturbationBC":
        """Keep the linearized mean boundary row (the default)."""
        return cls("inherit")

    @classmethod
    def hard_wall(cls, entropy_in=None) -> "PerturbationBC":
        """Rigid wall, ``u' = 0`` (``R = +1``)."""
        return cls("hard_wall", entropy_in=entropy_in)

    @classmethod
    def open_end(cls, entropy_in=None) -> "PerturbationBC":
        """Ideal pressure-release open end, ``p' = 0`` (``R = -1``)."""
        return cls("open_end", entropy_in=entropy_in)

    @classmethod
    def mean_flow_open_end(cls, entropy_in=None) -> "PerturbationBC":
        """Convective open end, ``R = -(1 - M)/(1 + M)`` (``-1`` at ``M=0``)."""
        return cls("mean_flow_open_end", entropy_in=entropy_in)

    @classmethod
    def anechoic(cls, entropy_in=None) -> "PerturbationBC":
        """Reflection-free termination (``R = 0``)."""
        return cls("anechoic", entropy_in=entropy_in)

    @classmethod
    def reflection(cls, R, entropy_in=None) -> "PerturbationBC":
        """Prescribed reflection coefficient ``R`` (constant, table, or callable)."""
        return cls("reflection", R=R, entropy_in=entropy_in)

    @classmethod
    def impedance(cls, Z, specific=False, entropy_in=None) -> "PerturbationBC":
        """Acoustic impedance ``Z`` (absolute Pa.s/m, or specific if ``specific``)."""
        return cls("impedance", Z=Z, specific=specific, entropy_in=entropy_in)

    @classmethod
    def impedance_polar(cls, magnitude, phase_deg=0.0, specific=True, entropy_in=None) -> "PerturbationBC":
        """Impedance from a magnitude and phase (degrees); specific (``Z/rho c``) by default.

        This is the closure the UI exposes: ``magnitude = 1, phase = 0`` is the matched
        (anechoic) termination, and the rigid-wall limit is ``magnitude -> inf``.
        """
        Z = float(magnitude) * cmath.exp(1j * math.radians(float(phase_deg)))
        return cls("impedance", Z=Z, specific=specific, entropy_in=entropy_in)

    @classmethod
    def excitation(cls, amplitude, family="acoustic", base_R=0.0, entropy_in=None) -> "PerturbationBC":
        """Drive an incoming wave with forcing ``amplitude`` on top of ``base_R``.

        ``family`` selects which incoming wave is driven (``"acoustic"`` or
        ``"entropy"``); ``base_R`` is the terminal's own reflection (default ``0`` --
        a clean source).
        """
        if family not in ("acoustic", "entropy"):
            raise ValueError(f"excitation family must be 'acoustic' or 'entropy'; got {family!r}")
        return cls("excitation", amplitude=amplitude, base_R=base_R, family=family, entropy_in=entropy_in)
