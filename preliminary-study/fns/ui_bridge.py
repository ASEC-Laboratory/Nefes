"""Bridge between the node-graph UI editor and the fns solver.

Workflow:
  1. Build the network in the UI editor using the "FNS Flow Network" model
     (public/models/fns-flow-network.yaml) and save it (YAML save file).
  2. ``load_case(path)`` parses the save file and assembles a
     :class:`fns.Network`.
  3. ``solve`` it as usual.
  4. ``write_results(...)`` produces a JSON data file in the UI's dataset
     format (docs/example-data.json): per-edge solution fields and per-node
     conservation residuals, ordered by the ``index`` attribute the UI
     assigns on save -- load it in the UI's data pane to color the canvas.

Save-file facts this module relies on (see the UI's src/types/flow.ts):
  * the file is YAML (js-yaml dump) with sections model/uiAttributes/uiState;
  * model.nodes[*]  = {id, type, attributes{label, index, <params>}};
  * model.edges[*]  = {id, source, target, sourceHandle, targetHandle,
                       attributes{label?, index, area}};
  * handle ids end in ``-port-<ordinal>`` where the ordinal counts a node's
    target ports first, then its source ports;
  * indices are renumbered on save (RENUMBER_ON_SAVE), so ``attributes.index``
    is a dense 0-based ordering for nodes and edges separately.
"""

import json
import re

import numpy as np

from .gas import PerfectGas
from .network import Network
from .elements import (
    MassFlowInlet,
    TotalPressureInlet,
    SupersonicInlet,
    SupersonicOutlet,
    PressureOutlet,
    IsentropicAreaChange,
    SuddenAreaChange,
    LossElement,
    JunctionStaticP,
    LosslessSplitter,
)

MODEL_ID = "fns-flow-network"

_HANDLE_RE = re.compile(r"-port-(\d+)$")

# Global (model-level) attribute defaults; kept in sync with the YAML model.
_GLOBAL_DEFAULTS = {
    "gasConstant": 287.0,
    "heatCapacityRatio": 1.4,
    "referencePressure": 101325.0,
    "referenceTemperature": 300.0,
    "referenceMassFlow": 0.0,
}


class UICaseError(ValueError):
    """Raised when a save file cannot be translated into a solvable network."""


def _num(attrs, key, default=None, where=""):
    value = attrs.get(key, default)
    if value is None:
        raise UICaseError(f"missing required parameter '{key}' {where}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise UICaseError(f"parameter '{key}' {where} is not numeric: {value!r}")


def _handle_ordinal(handle):
    if not handle:
        return None
    match = _HANDLE_RE.search(str(handle))
    return int(match.group(1)) if match else None


# Builders: UI node type -> (element factory, fixed port count or None)
def _build_element(node):
    ntype = node.get("type")
    attrs = node.get("attributes") or {}
    name = str(attrs.get("label") or node.get("id") or ntype)
    where = f"on node '{name}'"

    if ntype == "MassFlowInlet":
        return MassFlowInlet(
            mdot=_num(attrs, "massFlowRate", where=where),
            Tt=_num(attrs, "totalTemperature", where=where),
            name=name,
        ), 1
    if ntype == "TotalPressureInlet":
        return TotalPressureInlet(
            pt=_num(attrs, "totalPressure", where=where),
            Tt=_num(attrs, "totalTemperature", where=where),
            name=name,
        ), 1
    if ntype == "SupersonicInlet":
        return SupersonicInlet(
            M=_num(attrs, "machNumber", where=where),
            pt=_num(attrs, "totalPressure", where=where),
            Tt=_num(attrs, "totalTemperature", where=where),
            name=name,
        ), 1
    if ntype == "SupersonicOutlet":
        return SupersonicOutlet(p=_num(attrs, "pressure", where=where), name=name), 1
    if ntype == "PressureOutlet":
        return PressureOutlet(
            p=_num(attrs, "pressure", where=where),
            Tt_backflow=_num(attrs, "backflowTotalTemperature", 300.0, where=where),
            name=name,
        ), 1
    if ntype == "IsentropicAreaChange":
        return IsentropicAreaChange(name=name), 2
    if ntype == "SuddenAreaChange":
        return SuddenAreaChange(name=name), 2
    if ntype == "LossElement":
        return LossElement(K=_num(attrs, "lossCoefficient", where=where), name=name), 2
    if ntype == "JunctionStaticP":
        # sized later from the actual number of connected edges
        return JunctionStaticP(0, name=name), None
    if ntype == "LosslessSplitter":
        return LosslessSplitter(0, name=name), None

    raise UICaseError(f"unknown node type '{ntype}' (node '{name}')")


def load_case(path):
    """Parse a UI save file and assemble a fns Network.

    Returns ``(net, meta)`` where ``meta`` holds the index orderings needed to
    write result datasets: ``meta['edge_order']`` / ``meta['node_order']`` are
    lists of fns edge / element indices sorted by the UI's index
    attribute, plus the corresponding label lists.
    """
    import yaml

    with open(path) as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, dict) or "model" not in payload:
        raise UICaseError(f"{path}: not a UI save file (no 'model' section)")
    model = payload["model"]
    if model.get("id") not in (None, MODEL_ID):
        raise UICaseError(
            f"{path}: save file targets model '{model.get('id')}', expected '{MODEL_ID}'"
        )

    g = dict(_GLOBAL_DEFAULTS)
    g.update({k: v for k, v in (model.get("globalAttributes") or {}).items() if v is not None})
    gas = PerfectGas(R=float(g["gasConstant"]), gamma=float(g["heatCapacityRatio"]))
    mdot_ref = float(g["referenceMassFlow"]) or None
    net = Network(
        gas,
        p_ref=float(g["referencePressure"]),
        T_ref=float(g["referenceTemperature"]),
        mdot_ref=mdot_ref,
    )

    ui_nodes = model.get("nodes") or []
    ui_edges = model.get("edges") or []
    if not ui_nodes or not ui_edges:
        raise UICaseError(f"{path}: the network has no nodes or no edges")

    # elements
    elem_of = {}  # UI node id -> fns element index
    expected_ports = {}
    labels = {}
    for node in ui_nodes:
        element, n_fixed = _build_element(node)
        idx = net.add(element)
        elem_of[node["id"]] = idx
        expected_ports[idx] = n_fixed
        labels[idx] = element.name

    # edges (direction: UI source -> target == fns tail -> head)
    handle_ordinals = []  # per fns edge: (tail-side ordinal, head-side ordinal)
    for edge in ui_edges:
        attrs = edge.get("attributes") or {}
        ename = str(attrs.get("label") or edge.get("id"))
        for end in ("source", "target"):
            if edge.get(end) not in elem_of:
                raise UICaseError(f"edge '{ename}' references unknown node '{edge.get(end)}'")
        area = _num(attrs, "area", where=f"on edge '{ename}'")
        if area <= 0:
            raise UICaseError(f"edge '{ename}' has non-positive area {area}")
        net.connect(elem_of[edge["source"]], elem_of[edge["target"]], area=area, name=ename)
        handle_ordinals.append(
            (_handle_ordinal(edge.get("sourceHandle")), _handle_ordinal(edge.get("targetHandle")))
        )

    # Order each element's ports by the UI handle ordinal so that port-0
    # conventions (e.g. the LossElement reference area) match the canvas.
    for ei, plist in enumerate(net._ports):
        def port_key(entry):
            edge_idx, sigma = entry
            ordinal = handle_ordinals[edge_idx][0 if sigma > 0 else 1]
            return ordinal if ordinal is not None else edge_idx
        plist.sort(key=port_key)

    # Size dynamic elements from their actual connections; check fixed ones.
    for ei, element in enumerate(net.elements):
        n_connected = len(net._ports[ei])
        expected = expected_ports[ei]
        if expected is None:
            if n_connected < 2:
                raise UICaseError(
                    f"element '{labels[ei]}' has {n_connected} connected edge(s); "
                    "junctions/splitters need at least 2"
                )
            element.n_ports = n_connected
        elif n_connected != expected:
            raise UICaseError(
                f"element '{labels[ei]}' has {n_connected} connected edge(s), expects {expected}"
            )

    net.check_square()

    def order_by_index(items, kind):
        """items: list of (fns_index, ui_index_or_None, label)."""
        if all(ui is not None for _, ui, _ in items):
            ranked = sorted(items, key=lambda t: t[1])
            indices = [ui for _, ui, _ in ranked]
            if indices != list(range(len(items))):
                raise UICaseError(f"{kind} 'index' attributes are not a dense 0-based range")
        else:
            ranked = sorted(items, key=lambda t: t[0])
        return [i for i, _, _ in ranked], [lbl for _, _, lbl in ranked]

    node_items = [
        (elem_of[n["id"]], (n.get("attributes") or {}).get("index"), labels[elem_of[n["id"]]])
        for n in ui_nodes
    ]
    edge_items = [
        (k, (e.get("attributes") or {}).get("index"), net.edges[k].name)
        for k, e in enumerate(ui_edges)
    ]
    node_order, node_labels = order_by_index(node_items, "node")
    edge_order, edge_labels = order_by_index(edge_items, "edge")

    meta = {
        "node_order": node_order,
        "node_labels": node_labels,
        "edge_order": edge_order,
        "edge_labels": edge_labels,
    }
    return net, meta


def results_payload(net, meta, x):
    """Build the UI data-file payload (datasets keyed to the UI element indices)."""
    states = net.states(x)

    def edge_values(fn):
        return [float(fn(states[k])) for k in meta["edge_order"]]

    datasets = [
        {"name": "Mass flow", "target": "edge", "unit": "kg/s", "values": edge_values(lambda s: s.mdot)},
        {"name": "Velocity", "target": "edge", "unit": "m/s", "values": edge_values(lambda s: s.u)},
        {"name": "Mach number", "target": "edge", "unit": "-", "values": edge_values(lambda s: s.M)},
        {"name": "Static pressure", "target": "edge", "unit": "Pa", "values": edge_values(lambda s: s.p)},
        {"name": "Total pressure", "target": "edge", "unit": "Pa", "values": edge_values(lambda s: s.pt)},
        {"name": "Static temperature", "target": "edge", "unit": "K", "values": edge_values(lambda s: s.T)},
        {"name": "Total temperature", "target": "edge", "unit": "K", "values": edge_values(lambda s: s.Tt)},
        {"name": "Density", "target": "edge", "unit": "kg/m^3", "values": edge_values(lambda s: s.rho)},
    ]

    # Node diagnostics: conservation residuals (zero for boundary elements,
    # whose single port legitimately carries the network through-flow).
    report = net.conservation_report(x)
    dm = {}
    de = {}
    for ei, (_, m_imb, e_imb) in enumerate(report):
        boundary = len(net._ports[ei]) < 2
        dm[ei] = 0.0 if boundary else m_imb
        de[ei] = 0.0 if boundary else e_imb
    datasets.append(
        {"name": "Mass imbalance", "target": "node", "unit": "kg/s",
         "values": [float(dm[k]) for k in meta["node_order"]]}
    )
    datasets.append(
        {"name": "Energy imbalance", "target": "node", "unit": "W",
         "values": [float(de[k]) for k in meta["node_order"]]}
    )
    return {"datasets": datasets}


def write_results(net, meta, x, path):
    with open(path, "w") as fh:
        json.dump(results_payload(net, meta, x), fh, indent=2)


def supersonic_chain_guess(net):
    """Initial guess selecting the supersonic branch on transonic chains.

    Handles two declared-supersonic configurations:
      * SupersonicInlet (started intake): supersonic isentropic branch from
        the inlet up to the minimum-area edge of the chain, sonic there, a
        mildly lossy subsonic branch beyond (generic terminal-shock profile);
      * SupersonicOutlet (full-supersonic nozzle): subsonic branch from the
        feeding boundary down to the minimum-area edge, sonic there,
        lossless supersonic branch from the throat to the outlet.

    Chains are walks through 2-port elements; edges are assumed drawn along
    the flow direction.  Edges not on such chains keep the default
    initialization.  Returns None when neither boundary type is present.
    """
    from .shock import area_ratio

    gas = net.gas
    G = gas.gamma

    def mach_from_ar(ar, supersonic):
        if ar <= 1.0:
            return 1.0
        lo, hi = (1.0 + 1e-12, 50.0) if supersonic else (1e-9, 1.0 - 1e-12)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if (area_ratio(mid, G) > ar) == supersonic:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def chain_from(elem_idx):
        """Edges along the 2-port-element walk starting at a boundary element."""
        chain = []
        edge, _ = net._ports[elem_idx][0]
        prev_elem = elem_idx
        while True:
            chain.append(edge)
            e = net.edges[edge]
            nxt = e.head if e.tail == prev_elem else e.tail
            plist = net._ports[nxt]
            if len(plist) != 2:
                return chain, nxt
            other = [k for k, _ in plist if k != edge]
            if not other:
                return chain, nxt
            prev_elem = nxt
            edge = other[0]

    def set_edge(x0, k, M, pt, Tt):
        A = net.edges[k].area
        beta = 1 + 0.5 * (G - 1) * M * M
        p = pt * beta ** (-G / (G - 1))
        T = Tt / beta
        rho = p / (gas.R * T)
        x0[3 * k] = rho * M * np.sqrt(G * gas.R * T) * A
        x0[3 * k + 1] = p
        x0[3 * k + 2] = gas.cp * Tt

    ss_in = [i for i, el in enumerate(net.elements) if isinstance(el, SupersonicInlet)]
    ss_out = [i for i, el in enumerate(net.elements) if isinstance(el, SupersonicOutlet)]
    if not ss_in and not ss_out:
        return None

    x0 = net.initial_guess()

    # started-intake chains: supersonic before the throat, subsonic after
    for ei0 in ss_in:
        inlet = net.elements[ei0]
        chain, _ = chain_from(ei0)
        a_min = min(net.edges[k].area for k in chain)
        supersonic_side = True
        for k in chain:
            ar = net.edges[k].area / a_min
            if ar <= 1.0 + 1e-12:
                set_edge(x0, k, 1.0, inlet.pt, inlet.Tt)
                supersonic_side = False
            elif supersonic_side:
                set_edge(x0, k, mach_from_ar(ar, True), inlet.pt, inlet.Tt)
            else:
                set_edge(x0, k, mach_from_ar(ar, False), 0.92 * inlet.pt, inlet.Tt)

    # full-supersonic nozzle chains: subsonic before the throat, supersonic after
    for eo in ss_out:
        chain, far_elem = chain_from(eo)
        source = net.elements[far_elem]
        pt0 = getattr(source, "pt", net.p_ref)
        Tt0 = getattr(source, "Tt", net.T_ref)
        chain = list(reversed(chain))  # now source -> outlet
        a_min = min(net.edges[k].area for k in chain)
        subsonic_side = True
        for k in chain:
            ar = net.edges[k].area / a_min
            if ar <= 1.0 + 1e-12:
                set_edge(x0, k, 1.0, pt0, Tt0)
                subsonic_side = False
            else:
                set_edge(x0, k, mach_from_ar(ar, not subsonic_side), pt0, Tt0)
    return x0
