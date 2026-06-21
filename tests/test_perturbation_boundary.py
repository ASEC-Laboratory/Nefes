"""Verification of the perturbation boundary conditions (theory.md s12.4).

Every terminal BC is a reflection relation ``w_incoming - R(omega) w_outgoing = b``.
The physical, non-circular check is a **duct terminated by the BC**: drive an acoustic
wave at one end and read the input reflection at that end, which transmission-line
theory fixes as

    Gamma_in(omega) = R_term * exp(-i omega (tau_+ + tau_-)),

with ``tau_+ = L/(u + c)`` and ``tau_- = L/(c - u)`` the duct round-trip delays.  This
exercises the BC stamp, the duct propagation, and the forced-response driver together,
and reproduces the analytic reflection coefficient of each closure.  The wall element
is checked both for its mean flow (``mdot = 0``) and for terminating a duct as an
acoustic hard wall.  A separate block unit-tests :class:`PerturbationBC` itself.
"""

import os

import numpy as np
import pytest
import yaml

from fns.shell import Network
from fns.elements import catalog as cat
from fns.elements.ids import WALL
from fns.io import load_case
from fns.thermo.configure import perfect_gas
from fns.derive import ES_U, ES_C, ES_RHO
from fns.perturbation import PerturbationBC, boundary_response

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")

R_AIR, GAMMA = 287.0, 1.4
CFG = perfect_gas(R_AIR, GAMMA)
OMEGAS = np.linspace(80.0, 3200.0, 9)
LDUCT = 0.5


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _duct_case(inlet_bc, outlet_bc, *, pt_in=104000.0, p_out=101325.0, L=LDUCT, area=0.05):
    """[total-pressure inlet] -- duct(L) -- [pressure outlet], with the given BCs."""
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(pt_in, 300.0, perturbation_bc=inlet_bc))
    net.add(cat.duct(L))
    net.add(cat.pressure_outlet(p_out, 300.0, perturbation_bc=outlet_bc))
    net.connect(0, 1, area)
    net.connect(1, 2, area)
    sol = net.solve()
    assert sol.converged
    return net, sol


def _uc(sol, e=0):
    est = sol.table()
    return float(est[ES_U, e]), float(est[ES_C, e])


def _roundtrip(L, u, c, omegas):
    return np.exp(-1j * omegas * (L / (u + c) + L / (c - u)))


# --------------------------------------------------------------------------
# 1. Terminated-duct input reflection: one analytic R per closure.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outlet_bc, Rval",
    [
        (PerturbationBC.hard_wall(), 1.0),
        (PerturbationBC.open_end(), -1.0),
        (PerturbationBC.anechoic(), 0.0),
        (PerturbationBC.reflection(0.5 - 0.3j), 0.5 - 0.3j),
    ],
)
def test_terminated_duct_reflection(outlet_bc, Rval):
    _, sol = _duct_case(PerturbationBC.excitation(1.0), outlet_bc)
    u, c = _uc(sol)
    fr = boundary_response(sol.problem, sol.x, OMEGAS)
    expected = Rval * _roundtrip(LDUCT, u, c, OMEGAS)
    assert np.allclose(fr.reflection_at(0), expected, atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize("zeta", [2.0, 0.5, 1.5 - 0.4j])
def test_impedance_specific_and_absolute(zeta):
    R = (zeta - 1.0) / (zeta + 1.0)
    # specific impedance: R independent of rho*c
    _, sol = _duct_case(PerturbationBC.excitation(1.0), PerturbationBC.impedance(zeta, specific=True))
    u, c = _uc(sol)
    fr = boundary_response(sol.problem, sol.x, OMEGAS)
    assert np.allclose(fr.reflection_at(0), R * _roundtrip(LDUCT, u, c, OMEGAS), atol=1e-9)

    # absolute impedance Z = zeta * (rho c) read at the outlet edge -> same R
    est = sol.table()
    rho_c = float(est[ES_RHO, 1]) * float(est[ES_C, 1])
    _, sol2 = _duct_case(PerturbationBC.excitation(1.0), PerturbationBC.impedance(zeta * rho_c, specific=False))
    u2, c2 = _uc(sol2)
    fr2 = boundary_response(sol2.problem, sol2.x, OMEGAS)
    assert np.allclose(fr2.reflection_at(0), R * _roundtrip(LDUCT, u2, c2, OMEGAS), atol=1e-9)


def test_mean_flow_open_end():
    _, sol = _duct_case(PerturbationBC.excitation(1.0), PerturbationBC.mean_flow_open_end(), pt_in=115000.0)
    u, c = _uc(sol)
    M = u / c
    R = -(1.0 - M) / (1.0 + M)
    assert M > 0.2 and abs(R + 1.0) > 0.05  # genuinely mean-flow-corrected, not the ideal open end
    fr = boundary_response(sol.problem, sol.x, OMEGAS)
    assert np.allclose(fr.reflection_at(0), R * _roundtrip(LDUCT, u, c, OMEGAS), atol=1e-9)


def test_frequency_dependent_reflection_table():
    om_tab = np.linspace(0.0, 6000.0, 13)
    R_tab = 0.2 + 0.1j + (om_tab / 6000.0) * (0.5 - 0.2j)  # ramps with frequency
    _, sol = _duct_case(PerturbationBC.excitation(1.0), PerturbationBC.reflection((om_tab, R_tab)))
    u, c = _uc(sol)
    fr = boundary_response(sol.problem, sol.x, OMEGAS)
    R_at = np.interp(OMEGAS, om_tab, R_tab.real) + 1j * np.interp(OMEGAS, om_tab, R_tab.imag)
    assert np.allclose(fr.reflection_at(0), R_at * _roundtrip(LDUCT, u, c, OMEGAS), atol=1e-9)


def test_excitation_pins_incoming_wave_and_propagates():
    amp = 0.7 - 0.2j
    _, sol = _duct_case(PerturbationBC.excitation(amp), PerturbationBC.anechoic())
    u, c = _uc(sol)
    fr = boundary_response(sol.problem, sol.x, OMEGAS)
    # the excitation pins the incoming downstream wave f at the inlet edge
    assert np.allclose(fr.waves(0)[:, 0], amp, atol=1e-9)
    # anechoic outlet: no reflected (upstream) wave returns
    assert np.allclose(fr.waves(1)[:, 1], 0.0, atol=1e-9)
    # f propagates down the duct with the downstream phase exp(-i w tau_+)
    assert np.allclose(fr.waves(1)[:, 0], amp * np.exp(-1j * OMEGAS * (LDUCT / (u + c))), atol=1e-9)


def test_inherited_pressure_outlet_is_pressure_release():
    # An outlet left at 'inherit' keeps its linearized mean BC; for a subsonic
    # pressure outlet that is p' = 0 -- the ideal open end R = -1 (theory.md s12.4,
    # "continuity with the steady solution").
    _, sol = _duct_case(PerturbationBC.excitation(1.0), PerturbationBC.inherit())
    u, c = _uc(sol)
    fr = boundary_response(sol.problem, sol.x, OMEGAS)
    assert np.allclose(fr.reflection_at(0), -1.0 * _roundtrip(LDUCT, u, c, OMEGAS), atol=1e-8)


# --------------------------------------------------------------------------
# 2. The wall element: mean flow blocked; acoustically a hard wall.
# --------------------------------------------------------------------------


def test_wall_blocks_mean_flow_dead_end():
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(105000.0, 300.0))
    net.add(cat.junction())
    net.add(cat.pressure_outlet(101325.0, 300.0))
    net.add(cat.wall())
    net.connect(0, 1, 0.05)  # feed
    net.connect(1, 2, 0.05)  # main
    net.connect(1, 3, 0.05)  # dead-end into the wall
    sol = net.solve()
    assert sol.converged
    main, dead = sol.edge(1), sol.edge(2)
    assert abs(dead["mdot"]) < 1e-9  # no mass crosses the wall
    assert dead["p"] == pytest.approx(main["p"], rel=1e-6)  # junction common static pressure
    assert dead["h_t"] == pytest.approx(main["h_t"], rel=1e-6)  # enthalpy-transparent donor


def test_wall_terminated_duct_is_hard_wall():
    # A wall closes the duct -> mean flow is blocked (M = 0, a quiescent duct); the
    # wall's default closure must reflect as a hard wall, R = +1.
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(101325.0, 300.0, perturbation_bc=PerturbationBC.excitation(1.0)))
    net.add(cat.duct(LDUCT))
    net.add(cat.wall())
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    sol = net.solve()
    assert sol.converged
    u, c = _uc(sol)
    assert abs(u) < 1e-9  # quiescent
    fr = boundary_response(sol.problem, sol.x, OMEGAS)
    assert np.allclose(fr.reflection_at(0), np.exp(-2j * OMEGAS * LDUCT / c), atol=1e-9)


# --------------------------------------------------------------------------
# 3. PerturbationBC unit tests (reflection map, impedance, tables, forcing).
# --------------------------------------------------------------------------


def test_bc_reflection_presets():
    rho, c = 1.2, 340.0
    assert PerturbationBC.inherit().reflection_coefficient(0.0, rho, c, 0.0) is None
    assert PerturbationBC.hard_wall().reflection_coefficient(0.0, rho, c, 0.0) == 1.0
    assert PerturbationBC.open_end().reflection_coefficient(0.0, rho, c, 0.0) == -1.0
    assert PerturbationBC.anechoic().reflection_coefficient(0.0, rho, c, 0.0) == 0.0
    for M in (0.0, 0.3, 0.7):
        R = PerturbationBC.mean_flow_open_end().reflection_coefficient(0.0, rho, c, M)
        assert R == pytest.approx(-(1.0 - M) / (1.0 + M))


def test_bc_impedance_to_reflection():
    rho, c = 1.2, 340.0
    zc = rho * c
    for Z in (2.0 * zc, zc, (1.5 + 0.3j) * zc):
        R = PerturbationBC.impedance(Z).reflection_coefficient(0.0, rho, c, 0.0)
        assert R == pytest.approx((Z - zc) / (Z + zc))
    # specific impedance, and the rigid / pressure-release limits
    assert PerturbationBC.impedance(2.0, specific=True).reflection_coefficient(0.0, rho, c, 0.0) == pytest.approx(1 / 3)
    assert PerturbationBC.impedance(1e12 * zc).reflection_coefficient(0.0, rho, c, 0.0) == pytest.approx(1.0, abs=1e-6)
    assert PerturbationBC.impedance(0.0).reflection_coefficient(0.0, rho, c, 0.0) == pytest.approx(-1.0)


def test_bc_impedance_polar():
    # the UI closure: specific magnitude + phase (deg)
    rho, c = 1.2, 340.0
    assert PerturbationBC.impedance_polar(2.0, 0.0).reflection_coefficient(0.0, rho, c, 0.0) == pytest.approx(1 / 3)
    # magnitude 1, phase 0 -> matched (anechoic)
    assert PerturbationBC.impedance_polar(1.0, 0.0).reflection_coefficient(0.0, rho, c, 0.0) == pytest.approx(0.0)
    # large magnitude -> rigid wall; phase 90 deg (zeta = i) -> |R| = 1
    assert PerturbationBC.impedance_polar(1e9, 0.0).reflection_coefficient(0.0, rho, c, 0.0) == pytest.approx(
        1, abs=1e-6
    )
    assert abs(PerturbationBC.impedance_polar(1.0, 90.0).reflection_coefficient(0.0, rho, c, 0.0)) == pytest.approx(1.0)


def test_bc_table_interpolation_in_omega():
    om = np.array([0.0, 100.0, 200.0])
    val = np.array([0.0 + 0j, 1.0 + 0j, 1.0 + 2j])
    bc = PerturbationBC.reflection((om, val))
    assert bc.reflection_coefficient(50.0, 1.0, 1.0, 0.0) == pytest.approx(0.5 + 0j)
    assert bc.reflection_coefficient(150.0, 1.0, 1.0, 0.0) == pytest.approx(1.0 + 1j)


def test_bc_forcing_and_entropy():
    bc = PerturbationBC.excitation(0.7 - 0.2j)
    assert bc.forcing(0.0) == pytest.approx(0.7 - 0.2j)
    assert bc.entropy_forcing(0.0) == 0.0
    bce = PerturbationBC.excitation(1.0, family="entropy")
    assert bce.forcing(0.0) == 0.0
    assert bce.entropy_forcing(0.0) == pytest.approx(1.0)
    assert PerturbationBC.anechoic(entropy_in=0.3 + 0.1j).entropy_forcing(0.0) == pytest.approx(0.3 + 0.1j)


def test_bc_rejects_unknown_kind_and_family():
    with pytest.raises(ValueError):
        PerturbationBC(kind="nonsense")
    with pytest.raises(ValueError):
        PerturbationBC.excitation(1.0, family="vortical")


# --------------------------------------------------------------------------
# 4. UI/YAML loader: BC attributes parse into PerturbationBCs on the elements.
# --------------------------------------------------------------------------


def test_loader_parses_boundary_conditions_and_runs():
    net = load_case(os.path.join(_EXAMPLES, "acoustic_terminations.yaml"))
    kinds = {el.name: (None if el.perturbation_bc is None else el.perturbation_bc.kind) for el in net._elements}
    assert kinds["liner"] == "impedance"  # specific-impedance liner
    assert kinds["resonator-end"] == "hard_wall"  # rigid wall
    assert kinds["reservoir"] == "impedance"  # matched default (magnitude 1)
    assert kinds["tee"] is None  # interior element -> no BC
    sol = net.solve()
    assert sol.converged
    assert abs(sol.edge(3)["mdot"]) < 1e-9  # branch behind the wall: no mean flow
    fr = boundary_response(sol.problem, sol.x, np.linspace(100.0, 2000.0, 4))
    assert fr.X.shape == (4, 3 * net.compile().n_edges)


def test_loader_default_inherit_keeps_old_cases():
    # boundary nodes with no acoustic fields default to inherit (None)
    net = load_case(os.path.join(_EXAMPLES, "converging_nozzle.yaml"))
    assert all(el.perturbation_bc is None for el in net._elements)


def _ui_case(outlet_attrs):
    return {
        "version": "2.0.0",
        "model": {
            "id": "fns-flow-network",
            "globalAttributes": {
                "gasConstant": 287.0,
                "heatCapacityRatio": 1.4,
                "referencePressure": 101325.0,
                "referenceTemperature": 300.0,
                "referenceMassFlow": 5.0,
            },
            "nodes": [
                {
                    "id": "in",
                    "type": "TotalPressureInlet",
                    "attributes": {"index": 0, "label": "in", "totalPressure": 104000.0, "totalTemperature": 300.0},
                },
                {"id": "d", "type": "Duct", "attributes": {"index": 1, "label": "d", "length": 0.5}},
                {
                    "id": "out",
                    "type": "PressureOutlet",
                    "attributes": dict({"index": 2, "label": "out", "pressure": 101325.0}, **outlet_attrs),
                },
            ],
            "edges": [
                {
                    "id": "e0",
                    "source": "in",
                    "target": "d",
                    "sourceHandle": "in-port-0",
                    "targetHandle": "d-port-0",
                    "attributes": {"index": 0, "area": 0.05},
                },
                {
                    "id": "e1",
                    "source": "d",
                    "target": "out",
                    "sourceHandle": "d-port-1",
                    "targetHandle": "out-port-0",
                    "attributes": {"index": 1, "area": 0.05},
                },
            ],
        },
    }


def _load_case_dict(case, tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(case))
    return load_case(str(path))


def test_loader_rigid_maps_to_hard_wall(tmp_path):
    net = _load_case_dict(_ui_case({"rigid": True}), tmp_path)
    assert net._elements[2].perturbation_bc.kind == "hard_wall"


def test_loader_open_maps_to_open_end(tmp_path):
    # the 'open' checkbox -> ideal pressure-release end (p'=0, R=-1)
    net = _load_case_dict(_ui_case({"rigid": False, "open": True}), tmp_path)
    bc = net._elements[2].perturbation_bc
    assert bc.kind == "open_end"
    assert bc.reflection_coefficient(0.0, 1.2, 340.0, 0.0) == pytest.approx(-1.0)


def test_loader_rigid_takes_precedence_over_open(tmp_path):
    net = _load_case_dict(_ui_case({"rigid": True, "open": True}), tmp_path)
    assert net._elements[2].perturbation_bc.kind == "hard_wall"


def test_loader_open_takes_precedence_over_impedance(tmp_path):
    net = _load_case_dict(_ui_case({"rigid": False, "open": True, "impedanceMagnitude": 2.0}), tmp_path)
    assert net._elements[2].perturbation_bc.kind == "open_end"


def test_loader_impedance_polar(tmp_path):
    # specific impedance magnitude 2, phase 0 -> zeta = 2 -> R = (2-1)/(2+1) = 1/3
    net = _load_case_dict(_ui_case({"rigid": False, "impedanceMagnitude": 2.0, "impedancePhase": 0.0}), tmp_path)
    bc = net._elements[2].perturbation_bc
    assert bc.kind == "impedance" and bc.specific
    assert bc.reflection_coefficient(0.0, 1.2, 340.0, 0.0) == pytest.approx(1.0 / 3.0)


def test_loader_no_acoustic_fields_is_inherit(tmp_path):
    net = _load_case_dict(_ui_case({}), tmp_path)
    assert net._elements[2].perturbation_bc is None


def test_loader_builds_wall_element(tmp_path):
    case = _ui_case({})
    case["model"]["nodes"][2] = {
        "id": "out",
        "type": "Wall",
        "attributes": {"index": 2, "label": "wall", "rigid": True},
    }
    net = _load_case_dict(case, tmp_path)
    wall = net._elements[2]
    assert wall.residual_id == WALL
    assert wall.perturbation_bc.kind == "hard_wall"
