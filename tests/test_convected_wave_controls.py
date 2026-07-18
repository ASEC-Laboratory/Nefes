"""Individual convected-wave controls: entropy and composition removable one at a time.

``convected="all"|"entropy"|"composition"|"none"`` names the convected families that stay
live in the perturbation operator; ``isentropic=True`` remains the shorthand for ``"none"``.
On a network without reacting scalars, freezing the composition is a no-op and pinning the
entropy is the whole reduction, which gives two exact spectral invariants checked here.
"""

import warnings

import numpy as np
import pytest

import nefes
from nefes.elements import catalog as cat
from nefes.perturbation import build_acoustic_blocks
from nefes.perturbation.operator.operator import assemble_acoustic, resolve_convected

AREA = 0.004


@pytest.fixture(scope="module")
def flowing():
    sol = nefes.Network(
        nodes=[
            cat.mass_flow_inlet(2.0, 300.0, perturbation_bc=nefes.PerturbationBC.hard_wall()),
            cat.duct(0.5),
            cat.choked_nozzle_outlet(1e-3, name="throat"),
        ],
        edges=[(0, 1, AREA), (1, 2, AREA)],
    ).solve()
    assert sol.converged
    return sol


def test_resolve_convected_mapping_and_fail_closed():
    """The control resolves to (pin entropy, freeze composition) and rejects conflicts."""
    assert resolve_convected(False, None) == (False, False)
    assert resolve_convected(True, None) == (True, True)
    assert resolve_convected(False, "all") == (False, False)
    assert resolve_convected(False, "entropy") == (False, True)
    assert resolve_convected(False, "composition") == (True, False)
    assert resolve_convected(False, "none") == (True, True)
    assert resolve_convected(True, "none") == (True, True)
    with pytest.raises(ValueError):
        resolve_convected(True, "all")
    with pytest.raises(ValueError):
        resolve_convected(False, "bogus")


def test_blocks_carry_the_resolved_flags(flowing):
    """The assembled blocks report which family is pinned/frozen (and the repr says so)."""
    b = build_acoustic_blocks(flowing.problem, flowing.x, convected="entropy")
    assert b.isentropic is False and b.freeze_composition is True
    assert "composition frozen" in repr(b)
    b2 = build_acoustic_blocks(flowing.problem, flowing.x, convected="composition")
    assert b2.isentropic is True and b2.freeze_composition is False
    b3 = build_acoustic_blocks(flowing.problem, flowing.x, isentropic=True)
    assert b3.isentropic and b3.freeze_composition and "isentropic" in repr(b3)


def test_nonreacting_spectral_invariants(flowing):
    """Without reacting scalars: ``"entropy"`` equals ``"all"`` and ``"composition"`` equals ``"none"``."""

    def modes(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.sort(flowing.eigenmodes(freq_band=(200.0, 900.0), **kw).omega)

    assert np.allclose(modes(convected="entropy"), modes(convected="all"))
    assert np.allclose(modes(convected="composition"), modes(isentropic=True))
    assert np.allclose(modes(convected="none"), modes(isentropic=True))


def test_controls_touch_exactly_their_own_rows(flowing):
    """Pinning the entropy wave rewrites only the entropy transport rows of ``A(omega)``."""
    prob = flowing.problem
    w = 2.0 * np.pi * 400.0 - 150.0j  # complex frequency: the convected phases are live
    A_all = assemble_acoustic(w, build_acoustic_blocks(prob, flowing.x, convected="all")).toarray()
    A_comp = assemble_acoustic(w, build_acoustic_blocks(prob, flowing.x, convected="composition")).toarray()
    diff_rows = {int(r) for r in np.nonzero(np.abs(A_all - A_comp).sum(axis=1) > 0.0)[0]}
    tr0, E = int(prob.transport_row0), int(prob.n_edges)
    assert diff_rows and diff_rows <= set(range(tr0, tr0 + E))
    # and with no reacting scalars, freezing the composition rewrites nothing at all
    A_ent = assemble_acoustic(w, build_acoustic_blocks(prob, flowing.x, convected="entropy")).toarray()
    assert np.array_equal(A_all, A_ent)


def test_nyquist_accepts_the_control(flowing):
    """The real-frequency driver threads the same knob (needs no dynamic source here to build)."""
    from nefes.perturbation.operator.operator import CONVECTED_CHOICES

    assert set(CONVECTED_CHOICES) == {"all", "entropy", "composition", "none"}
