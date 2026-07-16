import os

import pytest

from nefes.thermo import Mechanism, SpeciesDatabase, SpeciesSet

DATA = os.path.join(os.path.dirname(__file__), os.pardir, "nefes", "thermo", "data")
H2O2 = os.path.join(DATA, "h2o2.yaml")
THERMO_INP = os.path.join(DATA, "thermo.inp")


@pytest.fixture(scope="session")
def cantera_mech():
    """The H2/O2/N2/Ar mechanism (species_set+reactions) parsed from the packaged Cantera YAML."""
    return Mechanism.from_cantera(H2O2)


@pytest.fixture(scope="session")
def cantera_lib():
    """The same data as a bare SpeciesSet (no reactions)."""
    return SpeciesSet.from_cantera(H2O2)


@pytest.fixture(scope="session")
def thermo_inp():
    """Parsed NASA Glenn / CEA ``thermo.inp`` database, or skip if absent."""
    if not os.path.isfile(THERMO_INP):
        pytest.skip("data/thermo.inp not present")
    return SpeciesDatabase(THERMO_INP)


@pytest.fixture(scope="session")
def cantera():
    """Cantera module, or skip the test if it is not installed."""
    ct = pytest.importorskip("cantera")
    return ct
