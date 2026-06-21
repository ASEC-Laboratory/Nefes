"""Parse-time field registry: the single authority on field name <-> index.

Fields are numbered once, contiguously, in band order across all three bands
(implementation-plan.md section 4):

  band 1  primary unknowns   mdot, p, h_t [, Z_el...]      -> solution vector x
  band 2  thermo-derived     T, rho, c, W [, species...]   -> derived cache
  band 3  flow-derived       u, M, q                       -> derived cache

``finalize()`` freezes the layout: band widths, the band-boundary indices, the
name<->index maps (Python-only -- no string ever crosses into @njit), the
``field_layout`` array of flexible offsets, and the per-band-1-variable
``scale`` reference magnitudes used to nondimensionalize the Newton system.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# Truly-fixed band-1 core indices (model-independent -> module constants).
MDOT = 0
P = 1
HT = 2


@dataclass(frozen=True)
class Layout:
    """Frozen field layout returned by ``FieldRegistry.finalize()``."""

    names: List[str]
    index: Dict[str, int]
    n_solve: int  # band-1 width (3 + n_elem)
    unknown_end: int  # == n_solve
    thermo_end: int
    edge_n_vars: int
    n_elem: int
    field_layout: np.ndarray  # int64[::1] flexible offsets
    scale: np.ndarray  # float64[::1] band-1 reference magnitudes

    def thermo_slice(self):
        return slice(self.unknown_end, self.thermo_end)

    def flow_slice(self):
        return slice(self.thermo_end, self.edge_n_vars)


# field_layout slot meanings (the handful of model-flexible offsets).
FL_N_ELEM = 0
FL_ZEL_START = 1
FL_N_SPECIES = 2
FL_SPECIES_START = 3
FL_UNKNOWN_END = 4
FL_THERMO_END = 5
FL_EDGE_N_VARS = 6
FL_LEN = 7


class FieldRegistry:
    """Append fields in band order, then finalize to an immutable Layout."""

    def __init__(self):
        self._unknowns: List[str] = []
        self._thermo: List[str] = []
        self._flow: List[str] = []
        self._n_species = 0

    def add_unknowns(self, *names: str) -> "FieldRegistry":
        self._unknowns.extend(names)
        return self

    def add_thermo(self, *names: str) -> "FieldRegistry":
        self._thermo.extend(names)
        return self

    def add_flow(self, *names: str) -> "FieldRegistry":
        self._flow.extend(names)
        return self

    def set_n_species(self, n: int) -> "FieldRegistry":
        self._n_species = n
        return self

    def finalize(self, scale_map: Dict[str, float]) -> Layout:
        names = self._unknowns + self._thermo + self._flow
        if names[:3] != ["mdot", "p", "h_t"]:
            raise ValueError("band-1 must start with mdot, p, h_t")
        index = {nm: i for i, nm in enumerate(names)}
        if len(index) != len(names):
            raise ValueError("duplicate field name in registry")

        n_solve = len(self._unknowns)
        n_elem = n_solve - 3
        unknown_end = n_solve
        thermo_end = n_solve + len(self._thermo)
        edge_n_vars = thermo_end + len(self._flow)

        field_layout = np.zeros(FL_LEN, dtype=np.int64)
        field_layout[FL_N_ELEM] = n_elem
        field_layout[FL_ZEL_START] = 3
        field_layout[FL_N_SPECIES] = self._n_species
        field_layout[FL_SPECIES_START] = thermo_end - self._n_species
        field_layout[FL_UNKNOWN_END] = unknown_end
        field_layout[FL_THERMO_END] = thermo_end
        field_layout[FL_EDGE_N_VARS] = edge_n_vars

        scale = np.ones(n_solve, dtype=np.float64)
        for nm in self._unknowns:
            if nm not in scale_map:
                raise ValueError(f"no scale provided for band-1 unknown '{nm}'")
            scale[index[nm]] = scale_map[nm]

        return Layout(
            names=names,
            index=index,
            n_solve=n_solve,
            unknown_end=unknown_end,
            thermo_end=thermo_end,
            edge_n_vars=edge_n_vars,
            n_elem=n_elem,
            field_layout=field_layout,
            scale=scale,
        )


def perfect_gas_layout(mdot_ref: float, p_ref: float, h_ref: float) -> Layout:
    """The standard v1 layout for a perfect gas (n_elem = 0, no species)."""
    reg = FieldRegistry()
    reg.add_unknowns("mdot", "p", "h_t")
    reg.add_thermo("T", "rho", "c", "W")
    reg.add_flow("u", "M", "q")
    return reg.finalize({"mdot": mdot_ref, "p": p_ref, "h_t": h_ref})
