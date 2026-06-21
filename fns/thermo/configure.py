"""Parse-time thermo configuration (pure Python).

Builds the immutable ``(model_id, tf, ti)`` bundle and a manifest describing the
transported composition (empty for a perfect gas).  The bundle has a fixed dtype
/ contiguity signature across all models so a single compiled ``thermo_update``
serves every backend.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .api import PERFECT_GAS

# Universal gas constant [J/(kmol*K)]; molar mass W = RU / R_specific [kg/kmol].
RU = 8314.462618


@dataclass(frozen=True)
class ThermoConfig:
    """Immutable thermo bundle passed read-only through the kernels."""

    model_id: int
    tf: np.ndarray  # float64[::1]
    ti: np.ndarray  # int64[::1]
    element_names: List[str] = field(default_factory=list)
    species_names: List[str] = field(default_factory=list)

    @property
    def n_elem(self) -> int:
        return len(self.element_names)

    @property
    def n_species(self) -> int:
        return len(self.species_names)


def perfect_gas(R: float = 287.0, gamma: float = 1.4) -> ThermoConfig:
    """Calorically-perfect-gas configuration (default: dry air)."""
    cp = gamma * R / (gamma - 1.0)
    W = RU / R
    tf = np.ascontiguousarray([cp, R, W], dtype=np.float64)
    ti = np.empty(0, dtype=np.int64)
    return ThermoConfig(model_id=PERFECT_GAS, tf=tf, ti=ti)
