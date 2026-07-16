"""The master species database and the NASA Glenn / CEA ``thermo.inp`` reader.

A :class:`SpeciesDatabase` is the master source of species thermodynamic data: the source
you draw a working :class:`~nefes.thermo.species.SpeciesSet` from. By default it is the
packaged NASA Glenn / CEA ``thermo.inp`` (the canonical NASA-9 database of McBride &
Gordon, ~2000 species in a fixed-column FORTRAN format); a user may point it at another
``thermo.inp``-format file or at a Cantera-YAML file (only the species block is read)
through :meth:`SpeciesDatabase.from_file`. Parse once, :meth:`~SpeciesDatabase.search` by
name, then :meth:`~SpeciesDatabase.select` the handful of species the equilibrium/property
code consumes directly.

    from nefes.thermo import SpeciesDatabase, Thermo
    db          = SpeciesDatabase()            # the packaged thermo.inp
    db.search("H2O")                           # -> ['H2O', 'H2O2', 'H2O(cr)', ...]
    species_set = db.select(["H2", "O2", "H2O", "OH", "H", "O", "N2"])
    gas         = Thermo(species_set)          # equilibrium, properties, ...

Record layout (per species), mirroring CEA's ``thermo.inp`` reader:

* line 1: ``name`` then a free-text reference/comment;
* line 2: interval count, code, up to five ``element count`` pairs, a phase flag, the
  molar mass [g/mol] and the formation enthalpy;
* then, per interval, three lines: an interval header (``T_lo T_hi n_coef`` and the term
  exponents) followed by two coefficient lines in FORTRAN ``D`` exponent notation.

The exponents are the standard NASA-9 set ``[-2,-1,0,1,2,3,4]``; coefficients are stored
as the canonical 9-term row ``[a1..a7, b1, b2]``.

Public: :class:`SpeciesDatabase`, :func:`read_thermo_inp`, :func:`default_thermo_inp`.
"""

from __future__ import annotations

import os

import numpy as np
import yaml

from .constants import P_REF, P_REF_BAR
from .elements import normalize_element
from .species import NASA9, Species, SpeciesSet, _parse_cantera_doc

__all__ = ["SpeciesDatabase", "read_thermo_inp", "default_thermo_inp"]

# The NASA Glenn / CEA database is vendored next to this module (``nefes/thermo/data``)
# and shipped as package data, so it is available without the user naming a path.
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_DEFAULT_THERMO_INP = os.path.join(_DATA_DIR, "thermo.inp")


def default_thermo_inp() -> str:
    """Filesystem path to the packaged NASA Glenn / CEA ``thermo.inp`` database.

    This is the default species database used whenever a ``thermo.inp`` path is not
    given explicitly (``SpeciesDatabase()``, ``read_thermo_inp()``, ``SpeciesSet.from_cea()``).

    Returns
    -------
    str
        Absolute path to the vendored ``thermo.inp``.

    Raises
    ------
    FileNotFoundError
        If the packaged database is missing (a broken install).
    """
    if not os.path.isfile(_DEFAULT_THERMO_INP):
        raise FileNotFoundError(
            f"the packaged thermo.inp is missing at {_DEFAULT_THERMO_INP!r}; "
            "reinstall nefes or pass an explicit path"
        )
    return _DEFAULT_THERMO_INP


# Column slices for line 2 element pairs (symbol at n:n+2, count at n+2:n+8).
_ELEMENT_COLS = (10, 18, 26, 34, 42)


def _f(text):
    """Parse a FORTRAN float, accepting the ``D`` exponent marker."""
    return float(text.strip().replace("D", "E").replace("d", "e"))


def _parse_record_2(line):
    n_intervals = int(line[0:2])
    composition = {}
    for n in _ELEMENT_COLS:
        sym = line[n : n + 2].strip()
        if sym and sym[0].isalpha():
            count = _f(line[n + 2 : n + 8])
            if count:
                composition[normalize_element(sym)] = int(count) if float(count).is_integer() else count
    # Phase flag sits just before the molar-mass field: 0 = gas, non-zero = condensed.
    try:
        phase = int(line[50:52].strip() or "0")
    except ValueError:
        phase = 0
    molar_mass_g = _f(line[52:65])  # g/mol
    return n_intervals, composition, molar_mass_g, phase


def _parse_interval(lines):
    """Return ``(T_lo, T_hi, coeffs9)`` for one 3-line interval block."""
    T_lo = _f(lines[0][0:11])
    T_hi = _f(lines[0][11:22])
    n_coef = int(lines[0][22])
    if n_coef != 7:  # pragma: no cover - all standard records use 7 terms
        raise ValueError(f"thermo.inp: unsupported n_coef={n_coef} (expected 7)")
    vals = [
        _f(lines[1][0:16]),
        _f(lines[1][16:32]),
        _f(lines[1][32:48]),
        _f(lines[1][48:64]),
        _f(lines[1][64:80]),
        _f(lines[2][0:16]),
        _f(lines[2][16:32]),  # a6, a7
        _f(lines[2][48:64]),
        _f(lines[2][64:80]),  # b1, b2
    ]
    return T_lo, T_hi, np.array(vals, float)


def read_thermo_inp(path=None):
    """Parse ``thermo.inp`` into an ordered ``{name: Species}`` dict.

    Single-point records (interval count 0) and records outside the standard
    7-term layout are skipped; everything evaluable over a range is kept,
    gaseous and condensed alike (the phase flag is preserved in the note).

    ``path`` defaults to the packaged database (:func:`default_thermo_inp`).
    """
    if path is None:
        path = default_thermo_inp()
    with open(path, "r") as fh:
        lines = fh.readlines()

    # Find the data start: the line that is exactly "thermo", then skip the
    # global temperature-range header line that follows it.
    start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "thermo":
            start = i + 2
            break

    out = {}
    i = start
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped or line[0] in "!#" or not line[0].isalpha():
            i += 1
            continue
        if stripped.upper().startswith("END"):
            i += 1
            continue

        name = line.split()[0]
        comment = line[len(name) :].strip()
        n_intervals, composition, molar_mass_g, phase = _parse_record_2(lines[i + 1])

        if n_intervals < 1:
            # Reference-only / single-point record: not evaluable over a range.
            i += 3
            continue

        Tranges = []
        coeffs = []
        for k in range(n_intervals):
            blk = lines[i + 2 + 3 * k : i + 5 + 3 * k]
            T_lo, T_hi, c9 = _parse_interval(blk)
            if not Tranges:
                Tranges.append(T_lo)
            Tranges.append(T_hi)
            coeffs.append(c9)

        out[name] = Species(
            name=name,
            composition=composition,
            thermo=NASA9(Tranges, np.array(coeffs)),
            molar_mass=molar_mass_g * 1e-3,  # g/mol -> kg/mol
            note=comment,
            phase=phase,
        )
        i += 2 + 3 * n_intervals

    return out


class SpeciesDatabase:
    """The master species database: a searchable set of species with thermodynamic data.

    A database is the source a working :class:`~nefes.thermo.species.SpeciesSet` is drawn
    from. By default it is the packaged NASA Glenn / CEA ``thermo.inp`` (~2000 species); a
    user may point it at another ``thermo.inp``-format file or at a Cantera-YAML file (only
    the species block is read) through :meth:`from_file`. Parse once, :meth:`search` by
    name, then :meth:`select` the species the equilibrium/property code consumes.

    The default constructor (or :meth:`from_thermo_inp`) opens a ``thermo.inp`` database,
    whose standard state is one bar; :meth:`from_cantera` opens a Cantera-YAML species set,
    whose standard state is one atm. :meth:`select` inherits that reference pressure unless
    an explicit ``P_ref`` is passed.

    Parameters
    ----------
    path : str, optional
        A ``thermo.inp``-format file. Defaults to the packaged database
        (:func:`default_thermo_inp`).
    species : dict, optional
        Pre-parsed ``{name: Species}`` records (used by :meth:`from_cantera`); mutually
        exclusive with ``path``.
    source : str, optional
        A label for the ``species`` records, used in messages when no ``path`` is set.

    See Also
    --------
    SpeciesDatabase.from_file : Load a database, dispatching on file extension.
    """

    def __init__(self, path=None, *, species=None, source=None):
        if species is not None:
            # In-memory records (e.g. parsed from a Cantera-YAML document); their standard
            # state is one atm, matching the Cantera/YAML convention.
            self.species = dict(species)
            self.path = None
            self.source = source or "<in-memory>"
            self.P_ref = P_REF
            return
        if path is None:
            path = default_thermo_inp()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"species database file not found: {path!r}")
        self.path = path
        self.source = path
        self.species = read_thermo_inp(path)
        # The NASA Glenn / CEA standard state is one bar.
        self.P_ref = P_REF_BAR

    # -- constructors ----------------------------------------------------
    @classmethod
    def default(cls) -> "SpeciesDatabase":
        """Return the packaged NASA Glenn / CEA ``thermo.inp`` database."""
        return cls()

    @classmethod
    def from_thermo_inp(cls, path=None) -> "SpeciesDatabase":
        """Load a ``thermo.inp``-format database (the packaged default if ``path`` is ``None``)."""
        return cls(path)

    @classmethod
    def from_cantera(cls, source) -> "SpeciesDatabase":
        """Load a database from a Cantera-YAML file or parsed document (species block only).

        Only the species records (thermo and composition) are read; any reactions and
        transport data are ignored. ``source`` may be a file path or an already-parsed
        ``dict`` of such a document.
        """
        if isinstance(source, (str, os.PathLike)):
            with open(source, "r") as fh:
                doc = yaml.safe_load(fh)
            src = os.fspath(source)
        else:
            doc, src = source, "<dict>"
        _, species, _ = _parse_cantera_doc(doc)
        return cls(species={s.name: s for s in species}, source=src)

    @classmethod
    def from_file(cls, path) -> "SpeciesDatabase":
        """Load a database, dispatching on the file extension.

        A ``.yaml``/``.yml`` file is read as Cantera-YAML (species block only); any other
        extension is read as a ``thermo.inp``-format database.
        """
        ext = os.path.splitext(os.fspath(path))[1].lower()
        if ext in (".yaml", ".yml"):
            return cls.from_cantera(path)
        return cls.from_thermo_inp(path)

    # -- mapping protocol ------------------------------------------------
    def __contains__(self, name):
        return name in self.species

    def __getitem__(self, name):
        return self.species[name]

    def __len__(self):
        return len(self.species)

    @property
    def names(self):
        return list(self.species)

    def search(self, substring, case_sensitive=False):
        """Return species names containing ``substring``."""
        if case_sensitive:
            return [n for n in self.species if substring in n]
        s = substring.lower()
        return [n for n in self.species if s in n.lower()]

    def select(self, names=None, P_ref=None) -> SpeciesSet:
        """Build a :class:`~nefes.thermo.species.SpeciesSet` from ``names`` (all if ``None``).

        ``P_ref`` defaults to the database's own standard-state pressure (one bar for a
        ``thermo.inp`` database, one atm for a Cantera-YAML one).
        """
        if names is None:
            chosen = list(self.species.values())
        else:
            missing = [n for n in names if n not in self.species]
            if missing:
                label = os.path.basename(self.path) if self.path else self.source
                raise KeyError(f"species not in {label}: {missing}")
            chosen = [self.species[n] for n in names]

        elements = []
        for sp in chosen:
            for el in sp.composition:
                if el not in elements:
                    elements.append(el)
        return SpeciesSet(
            elements=elements,
            species=chosen,
            P_ref=self.P_ref if P_ref is None else P_ref,
        )

    def candidate_species(self, elements, *, gas_only=True, exclude_ions=True):
        """Database species reachable from a pool of ``elements`` (CEA-style product slate).

        Returns every species whose elemental composition is a subset of ``elements``, the
        candidate equilibrium products that can form from the fed-in atoms. This is the
        un-reduced slate; a :class:`~nefes.thermo.reduction.SpeciesReducer` trims it down.

        Parameters
        ----------
        elements : iterable of str
            Element symbols present in the feed (the reachable element pool).
        gas_only : bool, optional
            Drop condensed-phase species (``phase != 0``).  Defaults to ``True``;
            equilibrium products are gaseous (condensed species are feed-only).
        exclude_ions : bool, optional
            Drop ionic species (a ``+``/``-`` in the name or an electron ``E`` in the
            composition). Defaults to ``True``, since ionization is negligible for subsonic
            combustion and the charge balance adds cost for no benefit.

        Returns
        -------
        list of str
            Candidate species names, in database order.
        """
        pool = set(elements)
        out = []
        for name, sp in self.species.items():
            els = set(sp.composition) - {"E"}
            if not els.issubset(pool):
                continue
            if exclude_ions and ("+" in name or "-" in name or "E" in sp.composition):
                continue
            if gas_only and getattr(sp, "phase", 0) != 0:
                continue
            out.append(name)
        return out
