"""Declared feed streams with premixed feeds.

In ``"declared"`` mode the reacting model's transported streams are named up front
(``equilibrium(streams=...)``) and every feed states its composition **in terms of those
streams**.  A premixed inlet declared as a blend keeps its constituent streams separate:
it carries one live mixture fraction per stream (so their ratio is a real composition degree
of freedom), while its mean species composition is identical to the same premix given as a
single mixture in ``"auto"`` mode.  A stream may be declared without any dedicated inlet.
"""

import warnings

import numpy as np
import pytest

import nefes
from nefes.chem.composition import resolve_stream_blend, species_mass_fractions
from nefes.elements import catalog as cat
from nefes.perturbation import CompositionalNoiseWarning, forced_response
from nefes.perturbation.operator.boundary_bc import PerturbationBC
from nefes.thermo.api import EQ_FROZEN
from nefes.thermo.configure import equilibrium

AIR = {"O2": 0.21, "N2": 0.79}
STREAMS = {"air": AIR, "H2": {"H2": 1.0}}


def _frozen_premix_network(lib, blend, basis="mole"):
    """A premixed inlet -> duct -> outlet, all frozen, over the declared air/H2 streams."""
    cfg = equilibrium(lib, streams=STREAMS, mode="declared")
    nodes = [
        cat.mass_flow_inlet(0.01, 300.0, composition=blend, basis=basis, name="premix"),
        cat.duct(0.1, name="duct"),
        cat.pressure_outlet(1.0e5, name="outlet"),
    ]
    edges = [(0, 1, 0.01), (1, 2, 0.01)]
    net = nefes.Network(cfg, nodes=nodes, edges=edges, edge_models=[EQ_FROZEN, EQ_FROZEN])
    sol = net.solve()
    assert sol.converged
    return sol


# --- the reacting-model stream-mode switch ----------------------------------------------


def test_mode_defaults_and_property(cantera_lib):
    assert equilibrium(cantera_lib).stream_mode == "auto"
    assert equilibrium(cantera_lib, streams=STREAMS).stream_mode == "declared"


def test_declared_mode_requires_streams(cantera_lib):
    with pytest.raises(ValueError, match="requires streams"):
        equilibrium(cantera_lib, mode="declared")


def test_auto_mode_rejects_streams(cantera_lib):
    with pytest.raises(ValueError, match="drop streams"):
        equilibrium(cantera_lib, streams=STREAMS, mode="auto")


# --- premixed (blended) feeds -----------------------------------------------------------


def test_premixed_inlet_carries_two_live_streams(cantera_lib):
    """The non-degeneracy that makes an equivalence-ratio wave expressible: both streams
    are present on the feed edge, so their ratio is a live degree of freedom."""
    sol = _frozen_premix_network(cantera_lib, {"air": 20.0, "H2": 1.0})
    xi = sol.mixture_fractions(1)  # the duct edge, downstream of the premixed inlet
    assert set(xi) == {"air", "H2"}
    assert xi["air"] > 0.0 and xi["H2"] > 0.0
    assert sum(xi.values()) == pytest.approx(1.0)


def test_transported_fractions_match_the_blend(cantera_lib):
    sol = _frozen_premix_network(cantera_lib, {"air": 20.0, "H2": 1.0})
    labels = ["air", "H2"]
    stream_Y = np.array([species_mass_fractions(cantera_lib, STREAMS[k], "mole") for k in labels])
    expect = resolve_stream_blend(cantera_lib, labels, stream_Y, {"air": 20.0, "H2": 1.0}, "mole")
    xi = sol.mixture_fractions(1)
    assert xi["air"] == pytest.approx(expect[0], rel=1e-6)
    assert xi["H2"] == pytest.approx(expect[1], rel=1e-6)


def test_sum_of_fractions_is_one_on_every_edge(cantera_lib):
    sol = _frozen_premix_network(cantera_lib, {"air": 20.0, "H2": 1.0})
    for e in range(sol.problem.n_edges):
        assert sum(sol.mixture_fractions(e).values()) == pytest.approx(1.0, abs=1e-7)


def test_species_equal_the_single_composition_premix(cantera_lib):
    """The mean species composition of a blended feed equals the same premix given as one
    mixture (auto-discovered single stream): keeping the streams separate is bookkeeping only."""
    blended = _frozen_premix_network(cantera_lib, {"air": 20.0, "H2": 1.0}).species(1)

    # the identical premix as a single composition, auto-discovered (one stream)
    premix = {"O2": 20.0 * 0.21, "N2": 20.0 * 0.79, "H2": 1.0}
    cfg = equilibrium(cantera_lib)
    nodes = [
        cat.mass_flow_inlet(0.01, 300.0, composition=premix, name="premix"),
        cat.duct(0.1, name="duct"),
        cat.pressure_outlet(1.0e5, name="outlet"),
    ]
    edges = [(0, 1, 0.01), (1, 2, 0.01)]
    single = nefes.Network(cfg, nodes=nodes, edges=edges, edge_models=[EQ_FROZEN, EQ_FROZEN]).solve()
    assert single.converged
    ref = single.species(1)

    assert set(blended) == set(ref)
    for name in ref:
        assert blended[name] == pytest.approx(ref[name], rel=1e-9)
    assert ref["H2"] == pytest.approx(1.0 / 21.0, rel=1e-6)  # 1 mol H2 in 21 mol of premix


def test_mass_and_mole_blends_differ(cantera_lib):
    mole = _frozen_premix_network(cantera_lib, {"air": 20.0, "H2": 1.0}, basis="mole").mixture_fractions(1)
    mass = _frozen_premix_network(cantera_lib, {"air": 20.0, "H2": 1.0}, basis="mass").mixture_fractions(1)
    assert mass["H2"] == pytest.approx(1.0 / 21.0, rel=1e-6)  # mass ratio 20:1
    # H2 is light: for the same 20:1 amounts, one mole of H2 carries far less mass than one
    # mole of air, so the mole-basis H2 mass fraction is well below the mass-basis one.
    assert mole["H2"] < mass["H2"]


def test_pure_declared_stream_is_one_hot(cantera_lib):
    """A pure feed names a single declared stream."""
    sol = _frozen_premix_network(cantera_lib, {"air": 1.0})
    xi = sol.mixture_fractions(1)
    assert xi["air"] == pytest.approx(1.0)
    assert xi["H2"] == pytest.approx(0.0, abs=1e-12)


def test_undeclared_stream_in_declared_mode_is_rejected(cantera_lib):
    """In declared mode a feed must name declared streams, not raw species."""
    cfg = equilibrium(cantera_lib, streams=STREAMS, mode="declared")
    nodes = [
        cat.mass_flow_inlet(0.01, 300.0, composition={"O2": 1.0}, name="pure-O2"),  # a species, not a stream
        cat.duct(0.1, name="duct"),
        cat.pressure_outlet(1.0e5, name="outlet"),
    ]
    edges = [(0, 1, 0.01), (1, 2, 0.01)]
    with pytest.raises(KeyError, match="not a declared stream"):
        nefes.Network(cfg, nodes=nodes, edges=edges, edge_models=[EQ_FROZEN, EQ_FROZEN]).solve()


# --- driving the equivalence-ratio (composition) wave at a premixed inlet ----------------

FREQS = np.linspace(50.0, 500.0, 8)


def _premix_forced(lib, outlet_bc, drive="H2", amp=0.3, mdot=0.02):
    """Premixed (air+H2) inlet driven in composition -> duct -> outlet; frozen forced response."""
    cfg = equilibrium(lib, streams=STREAMS, mode="declared")
    inlet_bc = PerturbationBC.anechoic(driven=(drive,), amplitudes={drive: amp})
    nodes = [
        cat.mass_flow_inlet(mdot, 300.0, composition={"air": 20.0, "H2": 1.0}, perturbation_bc=inlet_bc, name="premix"),
        cat.duct(0.3, name="duct"),
        outlet_bc,
    ]
    edges = [(0, 1, 0.01), (1, 2, 0.01)]
    sol = nefes.Network(cfg, nodes=nodes, edges=edges, edge_models=[EQ_FROZEN, EQ_FROZEN]).solve()
    assert sol.converged
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompositionalNoiseWarning)
        fr = forced_response(sol.problem, sol.x, FREQS, isentropic=False)
    return sol, fr


def _acoustic_indices(fr, streams):
    """Indices of the non-composition (acoustic f/g + entropy h) waves."""
    return [i for i, lab in enumerate(fr.wave_labels) if lab not in streams]


def test_equivalence_ratio_wave_is_silent(cantera_lib):
    """Driving the H2 stream at a premixed inlet raises H2 and lowers air by the same amount
    (sum-zero, constant mass), and generates no sound in the plain duct: every acoustic and
    entropy wave is zero on the feed edge."""
    sol, fr = _premix_forced(cantera_lib, cat.pressure_outlet(1.0e5, perturbation_bc=PerturbationBC.anechoic()))
    labels = list(fr.wave_labels)
    w0 = fr.waves(0)  # the driven inlet edge
    # the constant-mass trade: H2 up by amp, air down by amp -> sum of composition perturbations = 0
    assert np.allclose(w0[:, labels.index("H2")], 0.3, atol=1e-6)
    assert np.allclose(w0[:, labels.index("air")], -0.3, atol=1e-6)
    assert np.allclose(w0[:, labels.index("H2")] + w0[:, labels.index("air")], 0.0, atol=1e-9)
    # silent: no acoustic or entropy content (no p', no mdot')
    for i in _acoustic_indices(fr, {"air", "H2"}):
        assert np.allclose(w0[:, i], 0.0, atol=1e-6), f"wave {labels[i]!r} is not silent"


def test_composition_wave_convects_down_the_duct(cantera_lib):
    """The silent composition wave rides the mean speed to the duct exit with phase exp(-i w L/u)."""
    sol, fr = _premix_forced(cantera_lib, cat.pressure_outlet(1.0e5, perturbation_bc=PerturbationBC.anechoic()))
    from nefes.assembly.recover import ES_U

    u = float(sol.table()[ES_U, 0])
    labels = list(fr.wave_labels)
    iH2 = labels.index("H2")
    phase = np.exp(-1j * (2.0 * np.pi * FREQS) * (0.3 / u))
    assert np.allclose(fr.waves(1)[:, iH2], fr.waves(0)[:, iH2] * phase, atol=1e-6)


def test_lone_stream_drive_is_rejected_as_loading_mode(cantera_lib):
    """A pure single-stream feed has nothing to trade against, so driving its only stream is the
    non-physical mass-loading mode and is rejected."""
    cfg = equilibrium(cantera_lib, streams=STREAMS, mode="declared")
    inlet_bc = PerturbationBC.anechoic(driven=("air",))
    nodes = [
        cat.mass_flow_inlet(0.02, 300.0, composition={"air": 1.0}, perturbation_bc=inlet_bc, name="pure-air"),
        cat.duct(0.3, name="duct"),
        cat.pressure_outlet(1.0e5, perturbation_bc=PerturbationBC.anechoic()),
    ]
    edges = [(0, 1, 0.01), (1, 2, 0.01)]
    sol = nefes.Network(cfg, nodes=nodes, edges=edges, edge_models=[EQ_FROZEN, EQ_FROZEN]).solve()
    with pytest.raises(ValueError, match="mass-loading mode"):
        forced_response(sol.problem, sol.x, FREQS, isentropic=False)


def test_declared_mode_yaml_round_trip(tmp_path):
    """A declared-stream network saves to and reloads from the UI YAML: the declared basis, the
    stream mode, and the feed blends survive, and the reloaded network reproduces the mixture
    fractions exactly.  Uses a thermo.inp-based species_set so the species slate reloads without a
    recorded mechanism file (a from_cantera species_set round-trips only with its mechanism)."""
    from nefes.thermo import SpeciesSet

    lib = SpeciesSet.from_cea(species=["O2", "N2", "H2", "H2O", "OH", "H", "O"])
    cfg = equilibrium(lib, streams=STREAMS, mode="declared")
    nodes = [
        cat.mass_flow_inlet(0.02, 300.0, composition={"air": 20.0, "H2": 1.0}, basis="mass", name="premix"),
        cat.duct(0.3, name="duct"),
        cat.pressure_outlet(1.0e5, name="outlet"),
    ]
    edges = [(0, 1, 0.01), (1, 2, 0.01)]
    net = nefes.Network(cfg, nodes=nodes, edges=edges, edge_models=[EQ_FROZEN, EQ_FROZEN])
    path = str(tmp_path / "declared.yaml")
    net.to_yaml(path)

    net2 = nefes.Network.from_yaml(path)
    assert net2.gas.stream_mode == "declared"
    assert list(net2.gas.element_names) == ["air", "H2"]

    sol1, sol2 = net.solve(), net2.solve()
    assert sol1.converged and sol2.converged
    for e in range(sol1.problem.n_edges):
        xi1, xi2 = sol1.mixture_fractions(e), sol2.mixture_fractions(e)
        assert set(xi1) == set(xi2)
        for k in xi1:
            assert xi2[k] == pytest.approx(xi1[k], abs=1e-6)


def test_composition_wave_radiates_sound_at_a_nozzle(cantera_lib):
    """The wave that is silent in the duct converts to sound at a compact nozzle: the inherited
    (complex-step) nozzle keeps the composition -> acoustic coupling, so an acoustic wave appears
    that is absent with the non-reflecting outlet."""
    _sol_silent, fr_silent = _premix_forced(
        cantera_lib, cat.pressure_outlet(1.0e5, perturbation_bc=PerturbationBC.anechoic())
    )
    _sol_nozzle, fr_nozzle = _premix_forced(cantera_lib, cat.choked_nozzle_outlet(0.008))  # inherited closure

    ac = _acoustic_indices(fr_nozzle, {"air", "H2"})
    reflected_silent = max(np.max(np.abs(fr_silent.waves(0)[:, i])) for i in ac)
    reflected_nozzle = max(np.max(np.abs(fr_nozzle.waves(0)[:, i])) for i in ac)
    assert reflected_silent < 1e-6  # non-reflecting duct: no sound
    assert reflected_nozzle > 1e-3  # the nozzle turns the composition wave into sound
