"""Element residual-type ids and per-type metadata (Python-side constants).

The integer ``residual_id`` is the @njit dispatch key (a big switch that
constant-folds).  Supersonic boundaries are reserved but not implemented in v1.
"""

MASS_FLOW_INLET = 0
PT_INLET = 1
P_OUTLET = 2
ISEN_AREA_CHANGE = 3
SUDDEN_AREA_CHANGE = 4
LOSS = 5
JUNCTION = 6
SPLITTER = 7
DUCT = 8
SUPERSONIC_INLET = 9  # reserved (deferred)
SUPERSONIC_OUTLET = 10  # reserved (deferred)

# Acoustic-face ids (implementation-plan.md s8.3): which acoustic stamp an
# element overrides its default CSD face with.  Only DUCT is active in v1;
# VOLUME (storage M) and FLAME (sources S) are reserved provisions.
ACOUSTIC_DEFAULT = 0  # contributes only through J_alg (the CSD linearization)
ACOUSTIC_DUCT = 1  # phase-propagation stamp P(omega)
ACOUSTIC_VOLUME = 2  # finite-volume storage stamp M (reserved)
ACOUSTIC_FLAME = 3  # heat-release source stamp S(omega) (reserved)

# Equation-kind tags (for residual-row scaling); mirror prototype KIND_*.
KIND_MASS = 0
KIND_PRESSURE = 1
KIND_ENTHALPY = 2

# Fixed n_ports for fixed-arity elements (None -> variable: junction/splitter).
FIXED_NPORTS = {
    MASS_FLOW_INLET: 1,
    PT_INLET: 1,
    P_OUTLET: 1,
    ISEN_AREA_CHANGE: 2,
    SUDDEN_AREA_CHANGE: 2,
    LOSS: 2,
    DUCT: 2,
}
