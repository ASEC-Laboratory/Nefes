"""Generate the UI showcase: UI-openable cases + matching results.

Writes, into examples/ui_showcase/, pairs of files per case:
    <case>.yaml          -- UI save file (open via the UI's load button;
                            the UI switches to the FNS Flow Network model)
    <case>-results.json  -- solver results in the UI data-pane format (load it
                            to color nodes/edges with the solution)

Cases:
  Converging-diverging nozzle (segment-resolved profile), three regimes:
    cd_nozzle_venturi          subsonic everywhere, lossless
    cd_nozzle_shock_weak       choked, internal normal shock at M_s ~ 1.30
    cd_nozzle_shock_strong     choked, internal normal shock at M_s ~ 1.70
    cd_nozzle_supersonic       full supersonic operation at the design pressure
                               (SupersonicOutlet, exit M = 2.20)
  Supersonic intake (SupersonicInlet boundary, M0 = 1.8), two operating points:
    intake_near_critical       terminal shock just past the throat (M_s ~ 1.10)
    intake_supercritical       terminal shock deeper in the diffuser (M_s ~ 1.50)
  Gas-turbine secondary-air system (all subsonic):
    gas_turbine_splits         plenum -> manifold -> three dissimilar branches
    gas_turbine_large          two sources, main manifold, three sub-manifolds
                               with a cross-bridge loop, 15 metered branches;
                               > 50 elements

Every case is solved through the same bridge the UI workflow uses
(`run_ui_case.py` would reproduce the results files), and validated against
the analytic expectations before being written.
"""

import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fns import solve
from fns.shock import area_ratio, normal_shock_post_mach, normal_shock_pt_ratio
from fns.ui_bridge import load_case, results_payload, supersonic_chain_guess

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_showcase")

G, R = 1.4, 287.0
FLUX_STAR = np.sqrt(G / R) * (2 / (G + 1)) ** ((G + 1) / (2 * (G - 1)))

# ---------------------------------------------------------------------------
# analytic helpers
# ---------------------------------------------------------------------------


def mach_from_area_ratio(ar, supersonic):
    if ar <= 1.0:
        return 1.0
    lo, hi = (1.0 + 1e-12, 50.0) if supersonic else (1e-9, 1.0 - 1e-12)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (area_ratio(mid) > ar) == supersonic:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def isentropic_p(M, pt):
    return pt * (1 + 0.2 * M * M) ** -3.5


def back_pressure_for_shock(M_shock, pt0, Tt, A_throat, A_exit):
    """Quasi-1D: back pressure placing a normal shock at pre-shock Mach M_shock."""
    A_s = A_throat * area_ratio(M_shock)
    pt2 = normal_shock_pt_ratio(M_shock) * pt0
    M2 = normal_shock_post_mach(M_shock)
    A2_star = A_s / area_ratio(M2)
    M_exit = mach_from_area_ratio(A_exit / A2_star, supersonic=False)
    return isentropic_p(M_exit, pt2), pt2, A_s


# ---------------------------------------------------------------------------
# save-file composer (mirrors the UI's generateSaveData / handle conventions)
# ---------------------------------------------------------------------------

PORT_LAYOUT = {
    # type -> (n_target_ports_fn, n_source_ports_fn) given the node attrs
    "MassFlowInlet": (lambda a: 0, lambda a: 1),
    "TotalPressureInlet": (lambda a: 0, lambda a: 1),
    "SupersonicInlet": (lambda a: 0, lambda a: 1),
    "PressureOutlet": (lambda a: 1, lambda a: 0),
    "IsentropicAreaChange": (lambda a: 1, lambda a: 1),
    "SuddenAreaChange": (lambda a: 1, lambda a: 1),
    "LossElement": (lambda a: 1, lambda a: 1),
    "JunctionStaticP": (lambda a: a.get("leftPorts", 2), lambda a: a.get("rightPorts", 1)),
    "LosslessSplitter": (lambda a: 1, lambda a: a.get("rightPorts", 2)),
}


class CaseBuilder:
    """Composes a UI save-file payload."""

    def __init__(self, name):
        self.name = name
        self.nodes = []  # dicts
        self.edges = []
        self._counters = {}

    def node(self, ntype, label, x, y, **attrs):
        n = self._counters.get(ntype, 0) + 1
        self._counters[ntype] = n
        node_id = f"{ntype}_{n}"
        self.nodes.append(
            {
                "id": node_id,
                "type": ntype,
                "label": label,
                "position": {"x": float(x), "y": float(y)},
                "attrs": attrs,
            }
        )
        return node_id

    def edge(self, source, target, area, src_port=None, tgt_port=None):
        """Connect source -> target.  Ports are *per-side* ordinals (0-based);
        handle ids use the node-wide ordinal (targets first, then sources)."""
        def node_of(nid):
            return next(n for n in self.nodes if n["id"] == nid)

        src = node_of(source)
        tgt = node_of(target)
        nt_src = PORT_LAYOUT[src["type"]][0](src["attrs"])
        src_ordinal = nt_src + (src_port or 0)
        tgt_ordinal = tgt_port or 0
        self.edges.append(
            {
                "id": f"edge_{len(self.edges) + 1}",
                "source": source,
                "target": target,
                "sourceHandle": f"{source}-port-{src_ordinal}",
                "targetHandle": f"{target}-port-{tgt_ordinal}",
                "area": float(area),
            }
        )

    def payload(self, global_attrs):
        model_nodes = []
        for idx, n in enumerate(self.nodes):
            attrs = {"label": n["label"], "index": idx}
            attrs.update(n["attrs"])
            model_nodes.append({"id": n["id"], "type": n["type"], "attributes": attrs})
        model_edges = []
        for idx, e in enumerate(self.edges):
            model_edges.append(
                {
                    "id": e["id"],
                    "source": e["source"],
                    "target": e["target"],
                    "sourceHandle": e["sourceHandle"],
                    "targetHandle": e["targetHandle"],
                    "type": "flow",
                    "attributes": {"index": idx, "area": e["area"]},
                }
            )
        return {
            "version": "2.0.0",
            "model": {
                "id": "fns-flow-network",
                "globalAttributes": global_attrs,
                "nodes": model_nodes,
                "edges": model_edges,
            },
            "uiAttributes": {
                "nodes": [{"id": n["id"], "position": n["position"]} for n in self.nodes]
            },
            "uiState": {
                "counters": {
                    "nodeCounters": dict(self._counters),
                    "totalNodeCounters": dict(self._counters),
                }
            },
        }


def emit(builder, global_attrs, x0_fn=None, tol=1e-10, stab_stages=None, checks=None,
         max_iter=100):
    """Write the case YAML, solve it through the bridge, validate, write results."""
    os.makedirs(OUT_DIR, exist_ok=True)
    case_path = os.path.join(OUT_DIR, f"{builder.name}.yaml")
    with open(case_path, "w") as fh:
        yaml.safe_dump(builder.payload(global_attrs), fh, sort_keys=False)

    net, meta = load_case(case_path)
    x0 = x0_fn(net) if x0_fn else net.initial_guess(mdot0=0.0)
    kwargs = {"tol": tol, "max_iter": max_iter}
    if stab_stages is not None:
        kwargs["stab_stages"] = stab_stages
    result = solve(net, x0=x0, **kwargs)
    assert result.converged, f"{builder.name}: {result}"
    states = net.states(result.x)
    if checks:
        checks(net, result, states)

    with open(os.path.join(OUT_DIR, f"{builder.name}-results.json"), "w") as fh:
        json.dump(results_payload(net, meta, result.x), fh, indent=2)

    max_m = max(abs(s.M) for s in states)
    print(f"  {builder.name:<26s} converged ({result.iterations:3d} its), "
          f"mdot = {states[0].mdot:8.4f} kg/s, max |M| = {max_m:.3f}")
    return net, result, states


# ---------------------------------------------------------------------------
# Case set 1: converging-diverging nozzle (segment-resolved)
# ---------------------------------------------------------------------------

NOZZLE_PT, NOZZLE_TT = 2.0e5, 400.0
NOZZLE_AREAS = [0.10, 0.05, 0.03, 0.045, 0.06]  # feed, conv-mid, THROAT, div-mid, exit
NOZZLE_MDOT_STAR = NOZZLE_PT / np.sqrt(NOZZLE_TT) * FLUX_STAR * NOZZLE_AREAS[2]


def build_cd_nozzle(name, p_b):
    b = CaseBuilder(name)
    res = b.node("TotalPressureInlet", "reservoir", 0, 0,
                 totalPressure=NOZZLE_PT, totalTemperature=NOZZLE_TT)
    segs = []
    for k, nm in enumerate(("conv-1", "conv-2", "div-1", "div-2")):
        segs.append(b.node("IsentropicAreaChange", nm, 230 * (k + 1), 0))
    out = b.node("PressureOutlet", "exit", 230 * 5, 0,
                 pressure=float(p_b), backflowTotalTemperature=NOZZLE_TT)
    chain = [res] + segs + [out]
    for k in range(5):
        b.edge(chain[k], chain[k + 1], NOZZLE_AREAS[k])
    return b


def nozzle_global_attrs():
    return {
        "gasConstant": R, "heatCapacityRatio": G,
        "referencePressure": NOZZLE_PT, "referenceTemperature": NOZZLE_TT,
        "referenceMassFlow": 15.0,
    }


def make_nozzle_cases():
    print("converging-diverging nozzle:")
    # venturi: back pressure above the first critical point
    M_ven = mach_from_area_ratio(NOZZLE_AREAS[4] / NOZZLE_AREAS[2], supersonic=False)
    p_first_crit = isentropic_p(M_ven, NOZZLE_PT)

    def check_venturi(net, result, st):
        assert st[2].mdot < NOZZLE_MDOT_STAR * 0.999
        for s in st:
            assert abs(s.M) < 1.0 and abs(s.pt / NOZZLE_PT - 1) < 1e-5

    emit(build_cd_nozzle("cd_nozzle_venturi", 0.5 * (NOZZLE_PT + p_first_crit)),
         nozzle_global_attrs(), checks=check_venturi)

    # choked with internal shock: the shock must land inside div-1
    # (area range throat..0.045 -> representable shock Machs ~ (1, 1.85))
    for name, M_s in (("cd_nozzle_shock_weak", 1.30), ("cd_nozzle_shock_strong", 1.70)):
        p_b, pt2, A_s = back_pressure_for_shock(
            M_s, NOZZLE_PT, NOZZLE_TT, NOZZLE_AREAS[2], NOZZLE_AREAS[4])
        assert NOZZLE_AREAS[2] < A_s < NOZZLE_AREAS[3], "shock must sit inside div-1"

        def check_shock(net, result, st, M_s=M_s, pt2=pt2, A_s=A_s):
            from fns import shock_report
            assert st[2].mdot == pytest_approx(NOZZLE_MDOT_STAR, 1e-5)
            assert abs(st[2].M - 1.0) < 2e-3  # throat edge sonic
            assert pytest_close(st[4].pt, pt2, 1e-3)
            sh = shock_report(net, result.x)
            assert len(sh) == 1 and sh[0]["valid"]
            assert pytest_close(sh[0]["M_shock"], M_s, 5e-3)
            assert pytest_close(sh[0]["A_shock"], A_s, 1e-2)

        emit(build_cd_nozzle(name, p_b), nozzle_global_attrs(), checks=check_shock)


def pytest_approx(val, rel):
    class _A:
        def __eq__(self, other):  # noqa
            return abs(other - val) <= rel * abs(val)
    return _A()


def pytest_close(a, b, rel):
    return abs(a - b) <= rel * abs(b)


# ---------------------------------------------------------------------------
# Case set 2: supersonic intake (started, critical/supercritical)
# ---------------------------------------------------------------------------

INTAKE_M0, INTAKE_P0, INTAKE_T0 = 1.8, 30000.0, 220.0
_beta0 = 1 + 0.2 * INTAKE_M0**2
INTAKE_PT, INTAKE_TT = INTAKE_P0 * _beta0**3.5, INTAKE_T0 * _beta0
INTAKE_A0 = 0.20
INTAKE_AT = INTAKE_A0 / area_ratio(INTAKE_M0)  # exactly-critical throat
INTAKE_AREAS = [INTAKE_A0, 0.165, INTAKE_AT, 0.18, 0.24]
INTAKE_MDOT = None  # filled below


def intake_capture_mdot():
    T = INTAKE_T0
    rho = INTAKE_P0 / (R * T)
    return rho * INTAKE_M0 * np.sqrt(G * R * T) * INTAKE_A0


def build_intake(name, p_b):
    b = CaseBuilder(name)
    inl = b.node("SupersonicInlet", "freestream", 0, 0,
                 machNumber=INTAKE_M0, totalPressure=INTAKE_PT, totalTemperature=INTAKE_TT)
    segs = []
    for k, nm in enumerate(("ramp-1", "ramp-2", "diffuser-1", "diffuser-2")):
        segs.append(b.node("IsentropicAreaChange", nm, 230 * (k + 1), 0))
    out = b.node("PressureOutlet", "engine-face", 230 * 5, 0,
                 pressure=float(p_b), backflowTotalTemperature=INTAKE_TT)
    chain = [inl] + segs + [out]
    for k in range(5):
        b.edge(chain[k], chain[k + 1], INTAKE_AREAS[k])
    return b


def make_intake_cases():
    print("supersonic intake (M0 = 1.8, exactly-critical throat):")
    mdot_capture = intake_capture_mdot()
    for name, M_s in (("intake_near_critical", 1.10), ("intake_supercritical", 1.50)):
        p_b, pt2, A_s = back_pressure_for_shock(
            M_s, INTAKE_PT, INTAKE_TT, INTAKE_AT, INTAKE_AREAS[4])
        assert INTAKE_AT < A_s < INTAKE_AREAS[3], "terminal shock must sit inside diffuser-1"

        def check_intake(net, result, st, M_s=M_s, pt2=pt2):
            from fns import shock_report
            assert pytest_close(st[0].mdot, mdot_capture, 1e-5)  # full capture
            assert pytest_close(st[0].M, INTAKE_M0, 1e-6)
            assert st[1].M > 1.0 and abs(st[2].M - 1.0) < 2e-3
            assert pytest_close(st[4].pt, pt2, 1e-3)  # pressure recovery
            sh = shock_report(net, result.x)
            assert len(sh) == 1 and pytest_close(sh[0]["M_shock"], M_s, 5e-3)

        emit(
            build_intake(name, p_b),
            {
                "gasConstant": R, "heatCapacityRatio": G,
                "referencePressure": INTAKE_PT, "referenceTemperature": INTAKE_TT,
                "referenceMassFlow": 60.0,
            },
            x0_fn=supersonic_chain_guess,
            tol=2e-9,
            stab_stages=(0.0,),
            checks=check_intake,
        )
        print(f"      pressure recovery pt2/pt0 = {normal_shock_pt_ratio(M_s):.4f} "
              f"(terminal shock at M = {M_s})")


# ---------------------------------------------------------------------------
# Case set 3: gas-turbine secondary-air splits (all subsonic)
# ---------------------------------------------------------------------------


def make_gas_turbine_case():
    print("gas-turbine secondary-air system:")
    pt_bleed, Tt_bleed = 8.0e5, 650.0
    b = CaseBuilder("gas_turbine_splits")

    src = b.node("TotalPressureInlet", "compressor-bleed", 0, 150,
                 totalPressure=pt_bleed, totalTemperature=Tt_bleed)
    feed = b.node("IsentropicAreaChange", "feed-pipe", 230, 150)
    man = b.node("LosslessSplitter", "manifold", 460, 150, rightPorts=3)

    k1 = b.node("LossElement", "rim-seal", 700, 0, lossCoefficient=1.5)
    o1 = b.node("PressureOutlet", "front-cavity", 940, 0,
                pressure=7.2e5, backflowTotalTemperature=Tt_bleed)

    i2 = b.node("IsentropicAreaChange", "blade-feed", 700, 150)
    s2 = b.node("SuddenAreaChange", "blade-plenum", 940, 150)
    o2 = b.node("PressureOutlet", "blade-root", 1180, 150,
                pressure=6.8e5, backflowTotalTemperature=Tt_bleed)

    i3 = b.node("IsentropicAreaChange", "bearing-orifice", 700, 300)
    k3 = b.node("LossElement", "labyrinth", 940, 300, lossCoefficient=3.0)
    o3 = b.node("PressureOutlet", "bearing-chamber", 1180, 300,
                pressure=6.0e5, backflowTotalTemperature=Tt_bleed)

    b.edge(src, feed, 0.015)
    b.edge(feed, man, 0.020)
    b.edge(man, k1, 0.006, src_port=0)
    b.edge(k1, o1, 0.006)
    b.edge(man, i2, 0.006, src_port=1)
    b.edge(i2, s2, 0.0035)
    b.edge(s2, o2, 0.009)
    b.edge(man, i3, 0.006, src_port=2)
    b.edge(i3, k3, 0.003)
    b.edge(k3, o3, 0.003)

    def check_gt(net, result, st):
        max_m = max(abs(s.M) for s in st)
        assert max_m < 0.8, f"meant to be fully subsonic, max M = {max_m}"
        assert all(s.mdot > 0 for s in st)
        total = st[1].mdot
        assert abs(st[2].mdot + st[4].mdot + st[7].mdot - total) < 1e-9 * total

    net, result, st = emit(
        b,
        {
            "gasConstant": R, "heatCapacityRatio": G,
            "referencePressure": pt_bleed, "referenceTemperature": Tt_bleed,
            "referenceMassFlow": 5.0,
        },
        checks=check_gt,
    )
    total = st[1].mdot
    for nm, k in (("rim-seal", 2), ("blade-feed", 4), ("bearing", 7)):
        print(f"      branch {nm:<10s}: {st[k].mdot:7.4f} kg/s ({st[k].mdot / total * 100:5.1f} %)")




# ---------------------------------------------------------------------------
# Case 1b: full supersonic nozzle operation (design point, SupersonicOutlet)
# ---------------------------------------------------------------------------


def make_supersonic_nozzle_case():
    print("supersonic nozzle (design point, declared supersonic exit):")
    areas = [0.10, 0.05, 0.03, 0.06]  # feed, conv-mid, THROAT, exit (single div element)
    M_exit = mach_from_area_ratio(areas[3] / areas[2], supersonic=True)
    p_design = isentropic_p(M_exit, NOZZLE_PT)
    mdot_star = NOZZLE_PT / np.sqrt(NOZZLE_TT) * FLUX_STAR * areas[2]

    b = CaseBuilder("cd_nozzle_supersonic")
    res = b.node("TotalPressureInlet", "reservoir", 0, 0,
                 totalPressure=NOZZLE_PT, totalTemperature=NOZZLE_TT)
    c1 = b.node("IsentropicAreaChange", "conv-1", 230, 0)
    c2 = b.node("IsentropicAreaChange", "conv-2", 460, 0)
    d = b.node("IsentropicAreaChange", "diverging", 690, 0)
    out = b.node("SupersonicOutlet", "design-exit", 920, 0, pressure=float(p_design))
    chain = [res, c1, c2, d, out]
    for k in range(4):
        b.edge(chain[k], chain[k + 1], areas[k])

    def check_supersonic(net, result, st):
        assert pytest_close(st[0].mdot, mdot_star, 1e-5)  # choked flow
        assert abs(st[2].M - 1.0) < 2e-3  # sonic throat edge
        assert pytest_close(st[3].M, M_exit, 1e-4)  # design exit Mach
        for s in st:
            assert pytest_close(s.pt, NOZZLE_PT, 1e-4)  # shock-free (lossless)

    emit(b, nozzle_global_attrs(), x0_fn=supersonic_chain_guess,
         tol=2e-9, stab_stages=(0.0,), checks=check_supersonic)
    print(f"      exit M = {M_exit:.4f} at design pressure {p_design:.0f} Pa, "
          f"shock-free, mass flow = choked value")


# ---------------------------------------------------------------------------
# Case set 3b: large gas-turbine secondary-air system (> 50 elements)
# ---------------------------------------------------------------------------


def make_gas_turbine_large_case():
    print("large gas-turbine secondary-air system:")
    pt_hp, Tt_hp = 9.0e5, 700.0
    Tt_lp = 480.0
    b = CaseBuilder("gas_turbine_large")

    hp = b.node("TotalPressureInlet", "hp-bleed", 0, 60,
                totalPressure=pt_hp, totalTemperature=Tt_hp)
    lp = b.node("MassFlowInlet", "lp-bleed", 0, 260, massFlowRate=3.0,
                totalTemperature=Tt_lp)
    mix = b.node("JunctionStaticP", "mix-plenum", 240, 160, leftPorts=2, rightPorts=1)
    feed = b.node("IsentropicAreaChange", "main-feed", 480, 160)
    main = b.node("LosslessSplitter", "main-manifold", 720, 160, rightPorts=3)

    b.edge(hp, mix, 0.020, tgt_port=0)
    b.edge(lp, mix, 0.015, tgt_port=1)
    b.edge(mix, feed, 0.030)
    b.edge(feed, main, 0.025)

    # branch archetypes: (elements..., outlet pressure factor)
    # each returns the number of elements it created
    def branch(sub_id, port, x, y, kind, p_out):
        prev, area = sub_id, 0.004
        if kind == 0:  # orifice + loss
            o = b.node("IsentropicAreaChange", f"orf-{x}-{y}", x, y)
            b.edge(prev, o, area, src_port=port)
            l = b.node("LossElement", f"loss-{x}-{y}", x + 200, y, lossCoefficient=2.0)
            b.edge(o, l, 0.002)
            e = b.node("PressureOutlet", f"sink-{x}-{y}", x + 400, y,
                       pressure=p_out, backflowTotalTemperature=Tt_hp)
            b.edge(l, e, 0.002)
            return 3
        if kind == 1:  # plain loss
            l = b.node("LossElement", f"loss-{x}-{y}", x, y, lossCoefficient=4.0)
            b.edge(prev, l, area, src_port=port)
            e = b.node("PressureOutlet", f"sink-{x}-{y}", x + 200, y,
                       pressure=p_out, backflowTotalTemperature=Tt_hp)
            b.edge(l, e, area)
            return 2
        if kind == 2:  # nozzle + dump + loss
            o = b.node("IsentropicAreaChange", f"noz-{x}-{y}", x, y)
            b.edge(prev, o, area, src_port=port)
            s = b.node("SuddenAreaChange", f"dump-{x}-{y}", x + 200, y)
            b.edge(o, s, 0.0025)
            l = b.node("LossElement", f"loss-{x}-{y}", x + 400, y, lossCoefficient=1.0)
            b.edge(s, l, 0.006)
            e = b.node("PressureOutlet", f"sink-{x}-{y}", x + 600, y,
                       pressure=p_out, backflowTotalTemperature=Tt_hp)
            b.edge(l, e, 0.006)
            return 4
        # kind == 3: double loss (labyrinth stack)
        l1 = b.node("LossElement", f"lab1-{x}-{y}", x, y, lossCoefficient=2.5)
        b.edge(prev, l1, area, src_port=port)
        l2 = b.node("LossElement", f"lab2-{x}-{y}", x + 200, y, lossCoefficient=2.5)
        b.edge(l1, l2, area)
        e = b.node("PressureOutlet", f"sink-{x}-{y}", x + 400, y,
                   pressure=p_out, backflowTotalTemperature=Tt_hp)
        b.edge(l2, e, area)
        return 3

    n_elements = 5
    subs = []
    p_outs = [8.3e5, 8.1e5, 7.9e5, 7.7e5, 7.5e5]
    for si, (sy, n_left) in enumerate(((-560, 1), (160, 2), (880, 1))):
        fl = b.node("LossElement", f"sub-feed-{si}", 960, sy + 200,
                    lossCoefficient=0.8 + 0.4 * si)
        b.edge(main, fl, 0.010, src_port=si)
        n_right = 6 if si == 0 else 5  # sub 0 carries the cross-bridge
        sub = b.node("JunctionStaticP", f"sub-manifold-{si}", 1200, sy + 200,
                     leftPorts=n_left, rightPorts=n_right)
        b.edge(fl, sub, 0.012, tgt_port=0)
        subs.append(sub)
        n_elements += 2
        for bi in range(5):
            kind = (si + bi) % 4
            n_elements += branch(sub, bi, 1480, sy + 110 * bi,
                                 kind, p_outs[bi] - 0.2e5 * si)

    # cross-bridge: sub-manifold-0 (6th right port) -> loss -> sub-manifold-1
    # (2nd left port).  Creates a loop whose flow direction is emergent.
    bridge = b.node("LossElement", "cross-bridge", 1200, -120, lossCoefficient=5.0)
    b.edge(subs[0], bridge, 0.003, src_port=5)
    b.edge(bridge, subs[1], 0.003, tgt_port=1)
    n_elements += 1

    def check_large(net, result, st):
        assert len(net.elements) >= 50, f"only {len(net.elements)} elements"
        max_m = max(abs(s.M) for s in st)
        assert max_m < 0.85, f"meant to be fully subsonic, max M = {max_m}"
        for el, dm, de in net.conservation_report(result.x):
            if len(net._ports[[e.name for e in net.elements].index(el)]) >= 2:
                assert abs(dm) < 1e-7, f"{el}: {dm}"

    net, result, st = emit(
        b,
        {
            "gasConstant": R, "heatCapacityRatio": G,
            "referencePressure": pt_hp, "referenceTemperature": Tt_hp,
            "referenceMassFlow": 20.0,
        },
        checks=check_large,
        max_iter=400,
    )
    bridge_edge = next(k for k, e in enumerate(net.edges) if "cross-bridge" in
                       net.elements[e.head].name and "sub-manifold-0" in net.elements[e.tail].name)
    print(f"      {len(net.elements)} elements, {net.n_edges} edges, "
          f"{net.n_unknowns} unknowns")
    print(f"      cross-bridge flow: {st[bridge_edge].mdot:+.4f} kg/s "
          f"(direction found by the solver)")


if __name__ == "__main__":
    make_nozzle_cases()
    make_supersonic_nozzle_case()
    make_intake_cases()
    make_gas_turbine_case()
    make_gas_turbine_large_case()
    print(f"\nall cases written to {OUT_DIR}")
    print("open the .yaml files with the UI's load button; load the matching")
    print("-results.json in the data pane to color the canvas.")
