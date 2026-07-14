"""Fuel/oxidizer + equivalence-ratio mixture helper (`equivalence_ratio_mixture`)
and the declared-stream blend resolver (`resolve_stream_blend`)."""

import numpy as np
import pytest

from nefes.chem.composition import (
    _o2_demand,
    equivalence_ratio_mixture,
    resolve_stream_blend,
    species_mass_fractions,
    species_mole_fractions,
    stream_mean_molar_masses,
)
from nefes.thermo import SpeciesLibrary


@pytest.fixture(scope="module")
def lib():
    return SpeciesLibrary.from_cea(species=["CH4", "O2", "N2", "CO2", "H2O", "H2"])


AIR = {"O2": 0.21, "N2": 0.79}


def _phi_of(lib, mix, basis="mole"):
    """Recover the equivalence ratio of a blend from its net O2 balance.

    For a blend with net demand ``d`` and pure-oxidizer/pure-fuel demands, phi is the
    actual fuel/oxidizer ratio over the stoichiometric one; equivalently the blend is
    stoichiometric when its net O2 demand is zero.  Here we just check the sign/zero.
    """
    X = species_mole_fractions(lib, mix, basis)
    return _o2_demand(lib, X)


def test_stoichiometric_methane_air_is_9_5_percent(lib):
    mix = equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, 1.0)
    # Textbook stoichiometric methane-in-air mole fraction ~ 9.5 %.
    assert mix["CH4"] == pytest.approx(0.0950, abs=2e-3)
    # The unburnt blend carries only the reactants, normalized.
    assert set(mix) == {"CH4", "O2", "N2"}
    assert sum(mix.values()) == pytest.approx(1.0)


def test_stoichiometric_blend_has_zero_oxygen_balance(lib):
    mix = equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, 1.0)
    assert _phi_of(lib, mix) == pytest.approx(0.0, abs=1e-12)


def test_rich_is_positive_lean_is_negative_balance(lib):
    rich = equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, 2.0)
    lean = equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, 0.5)
    assert _phi_of(lib, rich) > 0.0  # excess fuel -> still wants O2
    assert _phi_of(lib, lean) < 0.0  # excess O2 -> supplies O2


def test_phi_scales_fuel_oxidizer_mole_ratio(lib):
    """Doubling phi doubles the fuel/oxidizer mole ratio."""
    one = equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, 1.0)
    two = equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, 2.0)
    r1 = one["CH4"] / one["N2"]
    r2 = two["CH4"] / two["N2"]
    assert r2 / r1 == pytest.approx(2.0, rel=1e-10)


def test_hydrogen_oxygen_stoichiometric_is_two_to_one(lib):
    mix = equivalence_ratio_mixture(lib, {"H2": 1.0}, {"O2": 1.0}, 1.0)
    assert mix["H2"] / mix["O2"] == pytest.approx(2.0, rel=1e-10)
    assert mix["H2"] == pytest.approx(2.0 / 3.0, rel=1e-10)


def test_phi_zero_returns_pure_oxidizer(lib):
    mix = equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, 0.0)
    assert "CH4" not in mix
    assert mix["O2"] == pytest.approx(0.21, rel=1e-12)
    assert mix["N2"] == pytest.approx(0.79, rel=1e-12)


def test_mass_basis_sums_to_one_and_differs_from_mole(lib):
    mole = equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, 1.0, basis="mole")
    mass = equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, 1.0, basis="mass")
    assert sum(mass.values()) == pytest.approx(1.0)
    # CH4 (light) has a smaller mass fraction than mole fraction in air.
    assert mass["CH4"] < mole["CH4"]


def test_mass_basis_fuel_input(lib):
    """A fuel given by mass still mixes correctly (same blend as the mole spec for a pure fuel)."""
    by_mole = equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, 1.0)
    by_mass = equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, 1.0, fuel_basis="mass")
    assert by_mass["CH4"] == pytest.approx(by_mole["CH4"], rel=1e-12)


def test_rejects_bad_arguments(lib):
    with pytest.raises(ValueError, match="non-negative"):
        equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, -1.0)
    with pytest.raises(ValueError, match="no net oxygen demand"):
        equivalence_ratio_mixture(lib, {"N2": 1.0}, AIR, 1.0)  # N2 is not a fuel
    with pytest.raises(ValueError, match="supplies no oxygen"):
        equivalence_ratio_mixture(lib, {"CH4": 1.0}, {"N2": 1.0}, 1.0)  # N2 is not an oxidizer


# --- declared-stream blend resolver -----------------------------------------------------

STREAMS = {"air": AIR, "H2": {"H2": 1.0}}


def _basis(lib, streams=STREAMS, basis="mole"):
    """The (labels, stream_Y) declared basis for a set of named species mixtures."""
    labels = list(streams)
    stream_Y = np.array([species_mass_fractions(lib, streams[k], basis) for k in labels])
    return labels, stream_Y


def test_mass_blend_is_the_normalized_ratio(lib):
    labels, sY = _basis(lib)
    xi = resolve_stream_blend(lib, labels, sY, {"air": 20.0, "H2": 1.0}, basis="mass")
    assert xi == pytest.approx([20.0 / 21.0, 1.0 / 21.0])
    assert xi.sum() == pytest.approx(1.0)


def test_mole_blend_reconstructs_the_premix(lib):
    """The physics anchor: a mole blend's ``xi @ stream_Y`` equals the premix species mass
    fractions, so keeping the streams separate does not change the mean composition."""
    labels, sY = _basis(lib)
    xi = resolve_stream_blend(lib, labels, sY, {"air": 20.0, "H2": 1.0}, basis="mole")
    premix = {"O2": 20.0 * 0.21, "N2": 20.0 * 0.79, "H2": 1.0}  # 20 mol air + 1 mol H2
    Y_premix = species_mass_fractions(lib, premix, "mole")
    assert xi @ sY == pytest.approx(Y_premix, rel=1e-12)
    assert xi.sum() == pytest.approx(1.0)


def test_blend_default_basis_is_mole(lib):
    labels, sY = _basis(lib)
    default = resolve_stream_blend(lib, labels, sY, {"air": 20.0, "H2": 1.0})
    mole = resolve_stream_blend(lib, labels, sY, {"air": 20.0, "H2": 1.0}, basis="mole")
    assert default == pytest.approx(mole)


def test_pure_stream_blend_is_one_hot(lib):
    labels, sY = _basis(lib)
    xi = resolve_stream_blend(lib, labels, sY, {"air": 5.0}, basis="mass")
    assert xi == pytest.approx([1.0, 0.0])


def test_stream_mean_molar_masses(lib):
    _labels, sY = _basis(lib)
    W = stream_mean_molar_masses(lib, sY)
    assert W[0] == pytest.approx(0.02885, abs=1e-3)  # air ~ 28.85 g/mol
    assert W[1] == pytest.approx(0.002016, abs=5e-5)  # H2 ~ 2.016 g/mol


def test_blend_rejects_bad_input(lib):
    labels, sY = _basis(lib)
    with pytest.raises(KeyError, match="not a declared stream"):
        resolve_stream_blend(lib, labels, sY, {"argon": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        resolve_stream_blend(lib, labels, sY, {"air": -1.0})
    with pytest.raises(ValueError, match="positive"):
        resolve_stream_blend(lib, labels, sY, {})
    with pytest.raises(ValueError, match="basis"):
        resolve_stream_blend(lib, labels, sY, {"air": 1.0}, basis="volume")
