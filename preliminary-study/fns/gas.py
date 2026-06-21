"""Calorically perfect gas model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PerfectGas:
    """Calorically perfect (ideal) gas.

    Parameters
    ----------
    R : float
        Specific gas constant [J/(kg K)].
    gamma : float
        Heat capacity ratio c_p/c_v [-].
    """

    R: float = 287.0
    gamma: float = 1.4

    @property
    def cp(self) -> float:
        return self.gamma * self.R / (self.gamma - 1.0)

    @property
    def cv(self) -> float:
        return self.R / (self.gamma - 1.0)

    @property
    def K(self) -> float:
        """c_p / R = gamma / (gamma - 1). Appears in the density recovery."""
        return self.cp / self.R


AIR = PerfectGas(R=287.0, gamma=1.4)
