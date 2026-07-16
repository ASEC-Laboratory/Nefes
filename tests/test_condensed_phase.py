"""Condensed-phase (e.g. graphite) chemical equilibrium.

The element-potential kernel admits high-temperature condensed species (CEA phase != 0 with a
wide NASA range, such as graphite ``C(gr)``) as pure phases at unit activity, so rich
combustion (C/O > 1) converges to the correct sooting equilibrium instead of failing.  A
liquid fuel (a low-temperature condensed species) stays feed-only.

Reference values are from NASA-CEA-equivalent Cantera (gri30 gas + graphite phase), hardcoded
so the tests are self-contained.  Graphite onsets only for C/O > 1, its amount matches the
reference, elements are conserved, the complex-step derivative through the phase is exact, and
a rich reacting network that fails gas-only now converges.
"""

import os

import numpy as np
import pytest

from nefes.chem.composition import enthalpy_mass, species_mass_fractions
from nefes.thermo import SpeciesSet, Thermo

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nefes", "thermo", "data")
THERMO_INP = os.path.join(DATA, "thermo.inp")
GAS = ["CO", "CO2", "O2", "H2", "H2O", "OH", "O", "H", "CH4", "N2", "NO"]


def _lib(extra=()):
    if not os.path.isfile(THERMO_INP):
        pytest.skip("thermo.inp not present")
    return SpeciesSet.from_cea(species=GAS + list(extra))


def _Zh(lib, gas, feed_mole, T):
    """Elemental mass fractions and specific enthalpy of a mole-fraction feed at T."""
    Y = species_mass_fractions(lib, feed_mole, "mole")
    return gas.elemental_mass_fractions(Y), enthalpy_mass(lib, Y, T)


def _graphite_index(lib):
    return [s.name for s in lib.species].index("C(gr)")


# -- data model ------------------------------------------------------------------------------


def test_graphite_admitted_as_product_liquid_fuel_not():
    lib = SpeciesSet.from_cea(species=["C(gr)", "CO2", "O2", "Jet-A(L)"])
    idx = {s.name: j for j, s in enumerate(lib.species)}
    assert lib.product_mask[idx["C(gr)"]]  # high-T condensed -> product
    assert not lib.product_mask[idx["Jet-A(L)"]]  # low-T condensed -> feed-only
    assert lib.product_mask[idx["CO2"]] and lib.product_mask[idx["O2"]]  # gas -> products


# -- TP equilibrium: onset and amount vs Cantera ---------------------------------------------


@pytest.mark.parametrize(
    "feed, T, expect_graphite, ref_frac",
    [
        ({"CH4": 1.0, "O2": 2.0}, 2000.0, False, 0.0),  # C/O = 0.25, lean
        ({"CH4": 1.0, "O2": 0.5}, 1500.0, False, 0.0),  # C/O = 1, the soot boundary
        ({"CH4": 1.0, "O2": 0.3}, 1500.0, True, 0.1327),  # C/O = 1.67, sooting (Cantera)
        ({"CH4": 1.0, "O2": 0.15}, 1200.0, True, 0.2311),  # C/O = 3.3 (Cantera)
    ],
)
def test_tp_graphite_onset_and_amount(feed, T, expect_graphite, ref_frac):
    lib = _lib(extra=["C(gr)"])
    gas = Thermo(lib)
    Z, _ = _Zh(lib, gas, feed, 300.0)
    r = gas.equilibrate_TP(Z, T, 1.0e5)
    assert r.converged
    frac = r.n[_graphite_index(lib)] / r.n.sum()
    if expect_graphite:
        assert frac > 1e-4
        assert frac == pytest.approx(ref_frac, abs=0.01)  # ~1% (data source differs from CEA)
    else:
        assert frac < 1e-6
    # element conservation
    b_in = Z / lib.element_weights
    assert np.max(np.abs(lib.element_matrix @ r.n - b_in)) < 1e-9


# -- HP (adiabatic) flame with graphite ------------------------------------------------------


def test_hp_adiabatic_flame_with_graphite_vs_cantera():
    """CH4 + 0.4 O2 from 300 K -> Cantera HP-with-graphite gives T~983.5 K, graphite frac ~0.116."""
    lib = _lib(extra=["C(gr)"])
    gas = Thermo(lib)
    Z, h = _Zh(lib, gas, {"CH4": 1.0, "O2": 0.4}, 300.0)
    r = gas.equilibrate_HP(Z, h, 1.0e5, T_guess=2000.0)
    assert r.converged
    assert r.T == pytest.approx(983.5, abs=5.0)
    frac = r.n[_graphite_index(lib)] / r.n.sum()
    assert frac == pytest.approx(0.1161, abs=0.01)


# -- complex-step derivative through the condensed phase -------------------------------------


def test_complex_step_matches_fd_through_graphite():
    """dT/dh and dnC/dh through a graphite-forming HP equilibrium: complex-step == finite diff."""
    lib = _lib(extra=["C(gr)"])
    gas = Thermo(lib)
    Z, h = _Zh(lib, gas, {"CH4": 1.0, "O2": 0.3}, 400.0)
    iC = _graphite_index(lib)

    d, fd = 1e-30, 1.0e-2
    rc = gas.equilibrate_HP(np.asarray(Z, complex), complex(h, d), complex(1.0e5, 0.0), T_guess=1800.0)
    rp = gas.equilibrate_HP(Z, h + fd, 1.0e5, T_guess=1800.0)
    rm = gas.equilibrate_HP(Z, h - fd, 1.0e5, T_guess=1800.0)
    assert rc.n[iC].real > 1e-4  # graphite is actually present at this point

    dT_cs, dT_fd = rc.T.imag / d, (rp.T - rm.T) / (2 * fd)
    assert dT_cs == pytest.approx(dT_fd, rel=1e-4)
    dnC_cs, dnC_fd = rc.n[iC].imag / d, (rp.n[iC] - rm.n[iC]) / (2 * fd)
    assert dnC_cs == pytest.approx(dnC_fd, rel=1e-3)


# -- live Cantera cross-check (skipped when Cantera is absent) --------------------------------


def test_graphite_matches_live_cantera():
    """Sweep C/O across the sooting onset; graphite fraction tracks a gri30 + graphite Mixture."""
    ct = pytest.importorskip("cantera")
    lib = _lib(extra=["C(gr)"])
    gas = Thermo(lib)
    ct_gas = ct.Solution("gri30.yaml")
    ct_graph = ct.Solution("graphite.yaml")
    iC = _graphite_index(lib)
    for feed, T in [
        ({"CH4": 1.0, "O2": 0.6}, 1600.0),
        ({"CH4": 1.0, "O2": 0.35}, 1500.0),
        ({"CH4": 1.0, "O2": 0.2}, 1300.0),
    ]:
        Z, _ = _Zh(lib, gas, feed, 300.0)
        r = gas.equilibrate_TP(Z, T, 1.0e5)
        assert r.converged
        frac = r.n[iC] / r.n.sum()

        ct_gas.TPX = T, 1.0e5, feed
        mix = ct.Mixture([(ct_gas, 1.0), (ct_graph, 0.0)])
        mix.T, mix.P = T, 1.0e5
        mix.equilibrate("TP", max_steps=5000, log_level=0)
        frac_ct = mix.phase_moles(1) / (mix.phase_moles(0) + mix.phase_moles(1))
        assert frac == pytest.approx(frac_ct, abs=0.01)


# -- the full mixture envelope: 100% air -> 100% Jet-A ---------------------------------------


@pytest.mark.parametrize("burn", [False, True])
def test_equilibrium_across_air_to_jetA(burn):
    """The equilibrium solver converges across the full air/Jet-A(L) blend, into deep sooting.

    Sweeps the fuel mass fraction of an air/Jet-A blend from pure air through stoichiometric and
    into the rich sooting regime up to near-pure fuel pyrolysis (~50% of the carbon condensing
    as graphite), solving TP (``burn`` False) and adiabatic HP (``burn`` True) equilibrium at
    each point.  Every solve converges with elements conserved and graphite growing monotonically
    once the mixture is rich enough.  (The exact zero-oxidiser limit, w = 1, is a measure-zero
    single-species-gas singularity and is excluded.)
    """
    air = {"O2": 0.2095, "N2": 0.7808, "Ar": 0.0093, "CO2": 0.0004}
    species = ["Jet-A(L)", "O2", "N2", "Ar", "CO2", "H2O", "CO", "H2", "OH", "O", "H", "NO", "C(gr)"]
    lib = SpeciesSet.from_cea(species=species)
    gas = Thermo(lib)
    iC = _graphite_index(lib)
    Yair = species_mass_fractions(lib, air, "mole")
    Yfuel = species_mass_fractions(lib, {"Jet-A(L)": 1.0}, "mole")
    h_air = enthalpy_mass(lib, Yair, 700.0)
    h_fuel = enthalpy_mass(lib, Yfuel, 700.0)

    saw_graphite = False
    prev_frac = -1.0
    for w in np.linspace(0.0, 0.999, 21):  # fuel mass fraction of the blend (up to near-pure fuel)
        Ymix = (1.0 - w) * Yair + w * Yfuel
        Z = gas.elemental_mass_fractions(Ymix)
        b_in = Z / lib.element_weights
        if burn:
            h = (1.0 - w) * h_air + w * h_fuel
            r = gas.equilibrate_HP(Z, h, 10.0e5, T_guess=1500.0)
        else:
            r = gas.equilibrate_TP(Z, 1600.0, 10.0e5)
        assert r.converged, f"failed at fuel fraction w={w:.3f}"
        assert np.max(np.abs(lib.element_matrix @ r.n - b_in)) < 1e-8
        assert np.all(np.isfinite(r.n)) and r.T > 0
        frac = r.n[iC] / r.n.sum()
        if frac > 1e-4:
            saw_graphite = True
            assert frac + 1e-9 >= prev_frac  # soot grows monotonically as the blend richens
        prev_frac = max(prev_frac, frac)
    assert saw_graphite  # the rich end of the sweep forms soot


# -- rich reacting network converges (previously failed gas-only) ----------------------------


def test_rich_flame_network_converges_with_graphite():
    from nefes.elements import catalog as cat
    from nefes.shell import Network
    from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL
    from nefes.thermo.configure import equilibrium

    air = {"O2": 0.2095, "N2": 0.7808, "Ar": 0.0093, "CO2": 0.0004}

    def flame(species, mdot_fuel):
        lib = SpeciesSet.from_cea(species=species)
        h_air = enthalpy_mass(lib, species_mass_fractions(lib, air, "mole"), 300.0)
        nodes = [
            cat.mass_flow_inlet(0.9, 720.0, composition=air, name="air"),
            cat.mass_source(mdot_fuel, 720.0, composition={"CH4": 1.0}, name="fuel"),
            cat.equilibrium_flame(name="flame"),
            cat.pressure_outlet(11e5, Tt_backflow=720.0, composition=air, name="out"),
        ]
        net = Network(
            gas=equilibrium(lib),
            p_ref=12e5,
            T_ref=720.0,
            mdot_ref=0.9,
            h_ref=abs(h_air),
            nodes=nodes,
            edges=[(0, 1, 0.01), (1, 2, 0.01), (2, 3, 0.01)],
            edge_models=[EQ_FROZEN, EQ_FROZEN, EQ_KERNEL],
        )
        return net.solve()

    base = ["CH4", "O2", "N2", "Ar", "CO2", "H2O", "CO", "H2", "OH", "O", "H", "NO"]
    mdot = 0.9 * 0.35  # very rich (C/O > 1): the sooting regime a gas-only slate cannot represent
    assert flame(base + ["C(gr)"], mdot).converged  # graphite closes the equilibrium
