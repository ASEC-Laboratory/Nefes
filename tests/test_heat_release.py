"""Flame heating power as a post-processing value: ``Solution.heat_release()``.

The perfect-gas heat-addition flame must report back its own ``Qdot`` parameter; the
reacting equilibrium flame -- whose power is an outcome of the equilibrium, not an
input -- must report the formation-enthalpy drop from frozen reactants to equilibrium
products, cross-checked against an independent species-level evaluation and against
the fuel's heating value.  The same number must feed the flame-transfer-function
de-normalization when no explicit ``q_mean`` is given.
"""

import numpy as np
import pytest

import nefes
from nefes.chem import equivalence_ratio_mixture
from nefes.chem.chemistry import T_REF
from nefes.elements import catalog as cat
from nefes.elements.dynamic_source import n_tau_flame

CH4_LHV = 50.0e6  # methane lower heating value [J/kg], gaseous water products

A = 0.05


@pytest.fixture(scope="module")
def reacting():
    """Converged lean CH4/air network with an equilibrium flame."""
    mix = equivalence_ratio_mixture({"CH4": 1.0}, {"O2": 0.21, "N2": 0.79}, phi=0.8)
    net = nefes.Network(
        nefes.equilibrium(),
        nodes=[
            cat.mass_flow_inlet(
                1.0, 300.0, composition=mix, name="feed", perturbation_bc=nefes.PerturbationBC.hard_wall()
            ),
            cat.duct(0.3),
            cat.equilibrium_flame(name="flame", dynamic_source=n_tau_flame(1.0, 2e-3, ref_edge=0)),
            cat.duct(0.7),
            cat.pressure_outlet(101325.0, 300.0, name="out", perturbation_bc=nefes.PerturbationBC.open_end()),
        ],
        edges=[(0, 1, A), (1, 2, A), (2, 3, A), (3, 4, A)],
    )
    sol = net.solve()
    assert sol.converged
    return sol


def test_perfect_gas_flame_reports_its_qdot():
    """The heat-addition flame's reported power is its ``Qdot`` parameter."""
    Qdot = 1.0e6
    sol = nefes.Network(
        nodes=[
            cat.mass_flow_inlet(10.0, 300.0),
            cat.heat_release_flame(Qdot, name="flame"),
            cat.pressure_outlet(1.0e5, name="out"),
        ],
        edges=[(0, 1, A), (1, 2, A)],
    ).solve()
    assert sol.converged
    hr = sol.heat_release()
    assert set(hr) == {"flame"}
    assert hr["flame"] == pytest.approx(Qdot, rel=1e-6)


def test_no_flame_is_empty():
    """A flame-less network reports no heat release."""
    sol = nefes.Network(
        nodes=[cat.mass_flow_inlet(2.0, 300.0), cat.duct(0.5), cat.pressure_outlet(1.0e5)],
        edges=[(0, 1, A), (1, 2, A)],
    ).solve()
    assert sol.converged
    assert sol.heat_release() == {}


def test_equilibrium_flame_formation_drop(reacting):
    """The reported power equals the formation-enthalpy drop, evaluated independently.

    The flame conserves the absolute total enthalpy, so its sensible rise is the drop in
    the 298.15 K formation enthalpy between the frozen reactant blend and the converged
    equilibrium products -- recomputed here from the species dictionaries and the NASA
    polynomials, a fully separate route from the packed-bundle evaluation under test.
    """
    sol = reacting
    from nefes.thermo import Thermo

    gas = Thermo(sol.network.gas.species_set)
    lib = gas.species_set

    def h_form(edge):
        Y = np.zeros(lib.n_species)
        for name, frac in sol.species(edge, basis="mass").items():
            Y[lib.species_index[name]] = frac
        return gas.enthalpy_mass(Y, T_REF)

    mdot = sol.edge(1)["mdot"]
    Q_ref = mdot * (h_form(1) - h_form(2))  # flame sits between edges 1 and 2
    hr = sol.heat_release()
    assert set(hr) == {"flame"}
    # the two routes weigh the mixture differently (per-species molar masses vs the
    # element matrix times atomic weights), which the NASA data only keeps consistent
    # to ~1e-5; agreement to that level is the exactness this check can certify
    assert hr["flame"] == pytest.approx(Q_ref, rel=1e-4)
    # physical anchor: heat per kg of fuel just under the LHV (mild dissociation at phi = 0.8)
    q_per_fuel = hr["flame"] / (mdot * sol.species(0, basis="mass")["CH4"])
    assert 0.9 * CH4_LHV < q_per_fuel < CH4_LHV


def test_ftf_denormalization_uses_the_exact_power(reacting):
    """With no explicit ``q_mean``, the flame source stamp carries ``heat_release()``.

    The heat-release source writes ``-Q_bar / mdot`` on the downstream energy row, so the
    stamped factor times the through-flow recovers the de-normalization power exactly.
    """
    sol = reacting
    from nefes.perturbation.operator.stamps import build_source_stamps

    stamps, flame_edges = build_source_stamps(sol.problem, sol.x)
    assert len(stamps) == 1 and flame_edges
    mdot = sol.edge(1)["mdot"]
    q_stamped = -float(stamps[0].factors[0]) * mdot
    assert q_stamped == pytest.approx(sol.heat_release()["flame"], rel=1e-10)
