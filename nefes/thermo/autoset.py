"""Automatic (CEA-style) product-species set from the network feeds.

A reacting network need not carry a hand-curated species list.  Given the feed
compositions and a species database, the reachable product slate is *every* gas-phase
species buildable from the fed-in elements, reduced (when that slate is large) to the
species that are non-trace at equilibrium across the feed-mixing range.  The final
species set also carries the declared feed species so the frozen closure and the enthalpy
datum can be evaluated.

This is the single policy shared by both entry points that need it: the YAML / case
loader (:mod:`nefes.io.yaml_in`), which resolves the species set while parsing a case, and
the Python network build (:func:`nefes.shell.build.finalize_thermo`), which resolves it
when a deferred :func:`nefes.thermo.equilibrium` config meets its network.

Exports :func:`auto_product_set`.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from ..chem.composition import elemental_Z, species_mass_fractions
from .edge_state import AUTO_REDUCE_THRESHOLD
from .reduction import SampleState, get_reducer


def _dedup(seq: Iterable[str]) -> List[str]:
    """De-duplicate a sequence, preserving first-seen order."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _feed_species(feed_specs) -> List[str]:
    """Feed/source species named anywhere in the network (the reactants), first-seen order.

    Each ``spec`` is any object carrying a ``composition_spec`` mapping (a node spec or an
    :class:`~nefes.shell.build.ElementSpec`); specs without a composition are skipped.
    """
    out: List[str] = []
    for sp in feed_specs:
        comp = getattr(sp, "composition_spec", None)
        if not comp:
            continue
        for name in comp:
            if name not in out:
                out.append(name)
    return out


def _feed_sample_states(feed_lib, feed_specs, *, p_ref: float, T_init: float):
    """Representative equilibrium probe states along the feed-mixing line.

    Each feed stream contributes its elemental composition; convex (mass) combinations of
    the distinct streams span the lean-to-rich range the network can realize, probed at a
    couple of temperatures bracketing the burnt-gas guess.  Used to drive slate reduction.
    """
    T_samples = sorted({T_init, max(1500.0, 0.7 * T_init)})

    feeds = []
    for sp in feed_specs:
        comp = getattr(sp, "composition_spec", None)
        if not comp:
            continue
        Y = species_mass_fractions(feed_lib, comp, getattr(sp, "basis", "mole"))
        feeds.append(elemental_Z(feed_lib, Y))

    uniq = []
    for Z in feeds:
        if not any(np.allclose(Z, U, atol=1e-9) for U in uniq):
            uniq.append(Z)

    elems = list(feed_lib.elements)

    def zdict(Z):
        return {elems[i]: float(Z[i]) for i in range(len(elems))}

    states = []
    for Z in uniq:
        states += [SampleState(zdict(Z), T, p_ref) for T in T_samples]
    for ia in range(len(uniq)):
        for ib in range(ia + 1, len(uniq)):
            for w in (0.1, 0.3, 0.5, 0.7, 0.9):
                Zm = w * uniq[ia] + (1.0 - w) * uniq[ib]
                states += [SampleState(zdict(Zm), T, p_ref) for T in T_samples]
    return states


def auto_product_set(
    db,
    feed_specs,
    *,
    p_ref: float,
    T_init: float,
    reducer_name: str = "equilibrium_sampling",
    threshold: float = None,
    reduce_above: int = None,
):
    """CEA-style automatic product slate over a ``SpeciesDatabase`` database ``db``.

    Declared feed species fix the reachable element pool; the candidate gas-phase slate is
    every species buildable from those elements, reduced (when large) to the species that
    are non-trace at equilibrium across the feed-mixing range.  The final species_set also
    carries the declared feed species (including condensed fuels) so the frozen closure and
    the enthalpy datum can be evaluated; the equilibrium kernel masks condensed species out
    of the products.

    The slate size has three dials: which reducer runs (``reducer_name``), how deep it trims
    (``threshold``), and the candidate count above which it runs at all (``reduce_above``).

    Parameters
    ----------
    db : nefes.thermo.SpeciesDatabase
        The species database (the packaged NASA Glenn / CEA data, or a ``thermo.inp`` path).
    feed_specs : iterable
        The network's stream-introducing specs (inlets, sources, backflow-bearing outlets),
        each carrying a ``composition_spec`` and a ``basis`` naming its feed species.
    p_ref : float
        Reference pressure [Pa] the equilibrium probe states are evaluated at.
    T_init : float
        Burnt-gas temperature guess [K]; sets the probe-state temperatures for reduction.
    reducer_name : str, optional
        Registry key of the slate reducer (default ``"equilibrium_sampling"``); ``"none"``
        keeps every candidate.  Runs only when the candidate count exceeds ``reduce_above``.
    threshold : float, optional
        Trace mole-fraction threshold forwarded to the reducer: a species is kept when its
        peak equilibrium mole fraction across the feed-mixing samples clears it (subject to
        the reducer's safety margin).  Larger keeps fewer species, smaller keeps more.
        ``None`` uses the reducer's own default.
    reduce_above : int, optional
        Reduction runs only when the candidate count exceeds this; a smaller value forces
        reduction on a lean slate, a larger one keeps a broad slate whole.  ``None`` uses
        :data:`~nefes.thermo.edge_state.AUTO_REDUCE_THRESHOLD`.

    Returns
    -------
    nefes.thermo.SpeciesSet
        The resolved species_set, carrying a ``reduction_report`` attribute recording which
        products were selected and why.

    Raises
    ------
    ValueError
        If no feed or source declares a composition.
    KeyError
        If a feed species is absent from the database.
    """
    declared = _feed_species(feed_specs)
    if not declared:
        raise ValueError(
            "the reacting (equilibrium) model with automatic species needs at least one feed "
            "or source composition (pass an explicit species set to override)"
        )
    missing = [n for n in declared if n not in db]
    if missing:
        raise KeyError(f"feed species not in thermo.inp: {missing}")

    pool = set()
    for name in declared:
        pool.update(el for el in db[name].composition if el != "E")
    candidates = db.candidate_species(pool, gas_only=True, exclude_ions=True)
    declared_gas = [n for n in declared if db[n].phase == 0]

    gate = AUTO_REDUCE_THRESHOLD if reduce_above is None else int(reduce_above)
    if len(candidates) <= gate:
        report = {"reducer": "none", "n_candidates": len(candidates), "n_kept": len(candidates)}
        final = _dedup(candidates + declared)
    else:
        feed_lib = db.select(_dedup(declared))
        samples = _feed_sample_states(feed_lib, feed_specs, p_ref=p_ref, T_init=T_init)
        reducer_kwargs = {} if threshold is None else {"threshold": float(threshold)}
        reducer = get_reducer(str(reducer_name or "equilibrium_sampling"), **reducer_kwargs)
        result = reducer.reduce(db.select(candidates), samples, always_keep=declared_gas)
        report = result.report
        final = _dedup(result.species + declared)

    lib = db.select(final)
    lib.reduction_report = report  # auditable: which products were selected and why
    return lib
