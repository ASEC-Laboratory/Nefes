"""The mixing junction: a variable-port merge that obeys the second law.

The static-pressure ``junction`` ties every port to a common static pressure, which at a fast
port hands the branch its velocity head as extra total pressure (more than the feed carries) --
free energy the second law forbids.  The ``mixing_junction`` ties every port to a common
*effective* total pressure instead: each inflow gives up the unrecovered fraction of its
dynamic head on entering, so the node total pressure never rises above the feeds and the
mass-averaged outflow entropy never falls below the feed mean.

These tests pin the guarantees (non-negative entropy production, no manufactured total
pressure), the limits (low-Mach merge -> junction, ``recovery = 1`` -> splitter), the parameter
addressing and YAML round-trip, and that the acoustic operator accepts the new element.
"""

import warnings

import numpy as np
import pytest

import nefes
from nefes.assembly.assemble import residual
from nefes.assembly.recover import ES_M, ES_P, ES_PT
from nefes.elements import catalog as cat
from nefes.perturbation import PerturbationBC, perturbation_response
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.solver.report import states_table

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)


def _gas():
    return nefes.perfect_gas(R_AIR, GAMMA)


def _entropy(T, p):
    """Specific entropy of the perfect gas (the additive constant cancels in a difference)."""
    return CP * np.log(T) - R_AIR * np.log(p)


def _merge_network(manifold, pt_hi=2.2e5, pt_lo=2.0e5, tt_hi=400.0, tt_lo=300.0, a_in=0.02, a_out=0.05, p_out=1.8e5):
    """Two total-pressure feeds merging through ``manifold`` into one pressure outlet.

    Node order: 0, 1 feeds; 2 manifold; 3 outlet.  Edges: e0 (0->2), e1 (1->2), e2 (2->3), so
    the two feeds flow into the manifold and edge 2 carries the merged stream out.
    """
    nodes = [
        cat.total_pressure_inlet(pt_hi, tt_hi),
        cat.total_pressure_inlet(pt_lo, tt_lo),
        manifold,
        cat.pressure_outlet(p_out),
    ]
    edges = [(0, 2, a_in), (1, 2, a_in), (2, 3, a_out)]
    return nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=350.0)


def _node_entropy_production(sol, in_edges, out_edges):
    """Entropy generated at a node: sum of (mass flux * s) leaving minus entering."""
    mdot, T, p = sol.field("mdot"), sol.field("T"), sol.field("p")
    s = _entropy(T, p)
    out = sum(mdot[e] * s[e] for e in out_edges)
    inn = sum(mdot[e] * s[e] for e in in_edges)
    return out - inn


def test_merge_converges_and_respects_second_law():
    """A merge of two unequal streams converges, stays subsonic, and generates entropy."""
    sol = _merge_network(cat.mixing_junction(0.0)).solve()
    assert sol.converged, (sol.residual_norm, sol.print_residuals())
    assert sol.verify() == []
    assert np.abs(sol.field("M")).max() < 1.0

    sgen = _node_entropy_production(sol, in_edges=(0, 1), out_edges=(2,))
    assert sgen > 0.0  # adiabatic mixing generates entropy

    # No manufactured total pressure: the merged stream leaves at or below every feed.
    pt = sol.field("p_t")
    assert pt[2] <= min(pt[0], pt[1]) * (1.0 + 1e-6)


def test_junction_manufactures_total_pressure_where_mixing_junction_does_not():
    """The documented failure: a slow plenum feeding a fast branch.

    The static-pressure junction hands the fast branch more total pressure than the feed
    carries (entropy production goes negative); the mixing junction keeps the branch at or
    below the feed and generates entropy.
    """

    def build(manifold):
        # slow feed -> manifold -> [slow branch (large area), fast branch (small area, low back p)]
        nodes = [
            cat.total_pressure_inlet(2.0e5, 300.0),
            manifold,
            cat.pressure_outlet(1.95e5),
            cat.pressure_outlet(1.1e5),
        ]
        edges = [(0, 1, 0.10), (1, 2, 0.10), (1, 3, 0.010)]
        return nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=300.0).solve()

    jun = build(cat.junction())
    mix = build(cat.mixing_junction(0.0))
    assert jun.converged and mix.converged

    feed_pt = jun.field("p_t")[0]
    # The junction manufactures total pressure on the fast branch (edge 2 = node1->node3).
    assert jun.field("p_t")[2] > feed_pt * 1.05
    assert _node_entropy_production(jun, in_edges=(0,), out_edges=(1, 2)) < 0.0  # second-law violation

    # The mixing junction does not: the fast branch stays at or below the feed, entropy grows.
    assert mix.field("p_t")[2] <= feed_pt * (1.0 + 1e-6)
    assert _node_entropy_production(mix, in_edges=(0,), out_edges=(1, 2)) > 0.0


def _restricted_merge_problem(manifold, a=0.05, K=50.0):
    """Two feeds merging through ``manifold``, then a downstream loss to a pressure outlet.

    The loss carries the pressure drop, so the manifold ports run at low (but not vanishing)
    Mach.  Built through the low-level ``build_problem`` so its :class:`CompiledProblem` (and
    hence its residual) is directly evaluable.  Node order: 0, 1 feeds; 2 manifold; 3 loss; 4
    outlet.  Edge 2 (2->3) is the manifold's single outflow.
    """
    els = [
        cat.total_pressure_inlet(2.02e5, 320.0),
        cat.total_pressure_inlet(2.00e5, 300.0),
        manifold,
        cat.loss(K),
        cat.pressure_outlet(1.0e5),
    ]
    edges = [(0, 2, a), (1, 2, a), (2, 3, a), (3, 4, a)]
    return build_problem(_gas(), els, edges, mdot_ref=5.0, p_ref=1.0e5, h_ref=CP * 310.0)


def test_reduces_to_junction_residual_at_low_mach():
    """As the port Mach falls the mixing-junction and junction kernels coincide.

    The two residuals differ only in the manifold rows, and there by exactly the outflow
    dynamic head ``p_t - p`` (the term the mixing junction dumps and the junction keeps).  So
    the mixing-junction solution nearly solves the *junction* problem, with a residual equal to
    that dynamic head, which is ``O(M^2)`` and vanishes as ``M -> 0``.
    """
    prob_mix = _restricted_merge_problem(cat.mixing_junction(0.0))
    prob_jun = _restricted_merge_problem(cat.junction())  # identical topology and state layout
    res = solve(prob_mix)
    assert res.converged
    x = res.x

    eps = 1.0e-4 * 5.0  # the converged smoothing scale (max(0.3*kappa, 1e-4) * mdot_ref, kappa -> 0)
    r_mix = residual(prob_mix, x, eps, 1.0e-5, 0.0)
    r_jun = residual(prob_jun, x, eps, 1.0e-5, 0.0)
    est = states_table(prob_mix, x)

    assert np.abs(r_mix).max() < 1.0e-4  # the mixing junction solved its own problem
    manifold_mach = np.abs(est[ES_M, :3]).max()
    dyn_head = float(est[ES_PT, 2] - est[ES_P, 2])  # dynamic head on the manifold outflow edge
    # The junction residual at the mixing solution equals that (small) outflow dynamic head.
    assert np.abs(r_jun).max() == pytest.approx(dyn_head, rel=1e-3)
    assert dyn_head / 1.0e5 < 2.0e-2  # the O(M^2) smallness at this low Mach
    assert manifold_mach < 0.15


def test_recovery_one_matches_splitter():
    """recovery = 1 recovers the full dynamic head, i.e. the lossless splitter exactly."""

    # A gentle distribution case (one inflow, two outflows) where the splitter is well posed:
    # a large inflow area keeps the inflow well below choke.
    def build(manifold):
        nodes = [
            cat.total_pressure_inlet(2.0e5, 300.0),
            manifold,
            cat.pressure_outlet(1.98e5),
            cat.pressure_outlet(1.97e5),
        ]
        edges = [(0, 1, 0.10), (1, 2, 0.03), (1, 3, 0.03)]
        return nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=300.0).solve()

    mix = build(cat.mixing_junction(1.0))
    spl = build(cat.splitter())
    assert mix.converged and spl.converged
    for name in ("mdot", "T", "p_t", "p"):
        assert np.allclose(mix.field(name), spl.field(name), rtol=1e-6, atol=1e-6), name


def test_general_merge_where_splitter_fails():
    """The mixing junction is the general merge element: it converges where the splitter cannot.

    Merging two streams of unequal total pressure is infeasible for the lossless splitter
    (which forces a single common total pressure); the mixing junction reconciles them through
    the dump loss and converges.
    """
    mix = _merge_network(cat.mixing_junction(0.0), pt_hi=2.4e5, pt_lo=2.0e5).solve()
    with warnings.catch_warnings():  # the splitter deliberately fails to converge here
        warnings.simplefilter("ignore")
        spl = _merge_network(cat.splitter(), pt_hi=2.4e5, pt_lo=2.0e5).solve()
    assert mix.converged and mix.verify() == []
    assert not spl.converged


def test_recovery_parameter_addressing():
    """``recovery`` is a named parameter: readable, writable through with_params, validated."""
    net = _merge_network(cat.mixing_junction(0.0, name="mix"))
    assert net.get("mix.recovery") == 0.0
    tuned = net.with_params({"mix.recovery": 0.5})
    assert tuned.get("mix.recovery") == 0.5
    assert net.get("mix.recovery") == 0.0  # base network is untouched

    with pytest.raises(ValueError):
        cat.mixing_junction(1.5)
    with pytest.raises(ValueError):
        cat.mixing_junction(-0.1)


def test_yaml_roundtrip_preserves_recovery(tmp_path):
    """A saved-and-reloaded case keeps the mixing junction and its recovery."""
    net = _merge_network(cat.mixing_junction(0.4, name="mix"))
    path = str(tmp_path / "merge.yaml")
    net.save(path)
    back = nefes.load_case(path)
    assert back.get("mix.recovery") == 0.4
    sol = back.solve()
    assert sol.converged


def test_perturbation_operator_builds():
    """The acoustic layer accepts the mixing junction (auto-linearized, no storage stamp)."""
    net = nefes.Network(
        _gas(),
        nodes=[
            cat.total_pressure_inlet(2.0e5, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
            cat.total_pressure_inlet(2.0e5, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
            cat.mixing_junction(0.0),
            cat.duct(0.5),
            cat.pressure_outlet(1.8e5, perturbation_bc=PerturbationBC.open_end()),
        ],
        edges=[(0, 2, 0.02), (1, 2, 0.02), (2, 3, 0.05), (3, 4, 0.05)],
    )
    sol = net.solve()
    assert sol.converged
    resp = perturbation_response(sol, np.array([200.0, 600.0]))
    tm = resp.transfer_matrix(2, 3)
    assert np.all(np.isfinite(tm))
