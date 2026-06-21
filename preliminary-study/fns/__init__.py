"""fns: compressible flow network prototype."""

from .gas import PerfectGas, AIR
from .network import Network
from .solver import solve, SolveResult, complex_step_jacobian
from .state import recover_state, solve_density, EdgeState, StateError
from .shock import shock_report, normal_shock_pt_ratio, shock_mach_from_pt_ratio
from .elements import (
    MassFlowInlet,
    TotalPressureInlet,
    SupersonicInlet,
    SupersonicOutlet,
    PressureOutlet,
    IsentropicAreaChange,
    SuddenAreaChange,
    LossElement,
    JunctionStaticP,
    LosslessSplitter,
)

__all__ = [
    "PerfectGas",
    "AIR",
    "Network",
    "solve",
    "SolveResult",
    "complex_step_jacobian",
    "recover_state",
    "solve_density",
    "EdgeState",
    "StateError",
    "MassFlowInlet",
    "TotalPressureInlet",
    "SupersonicInlet",
    "SupersonicOutlet",
    "PressureOutlet",
    "IsentropicAreaChange",
    "SuddenAreaChange",
    "LossElement",
    "JunctionStaticP",
    "LosslessSplitter",
    "shock_report",
    "normal_shock_pt_ratio",
    "shock_mach_from_pt_ratio",
]
