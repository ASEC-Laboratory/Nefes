"""End-to-end test of the UI save-file bridge.

The fixture below mirrors exactly what the UI's generateSaveData() emits for
the diamond network of test_networks.py::test_diamond_with_loss_branch:
inlet -> splitter -> (isentropic branch | loss branch) -> junction -> outlet.
"""

import json

import numpy as np
import pytest

from fns import (
    AIR,
    Network,
    MassFlowInlet,
    LosslessSplitter,
    IsentropicAreaChange,
    LossElement,
    JunctionStaticP,
    PressureOutlet,
    solve,
)
from fns.ui_bridge import UICaseError, load_case, results_payload

SAVE_FILE = """
version: '1.0'
timestamp: '2026-06-11T12:00:00.000Z'
model:
  id: fns-flow-network
  globalAttributes:
    gasConstant: 287
    heatCapacityRatio: 1.4
    referencePressure: 101325
    referenceTemperature: 300
    referenceMassFlow: 0
  nodes:
    - id: MassFlowInlet_1
      type: MassFlowInlet
      attributes:
        label: inlet
        index: 0
        massFlowRate: 20
        totalTemperature: 400
    - id: LosslessSplitter_1
      type: LosslessSplitter
      attributes:
        label: split
        index: 1
        rightPorts: 2
    - id: IsentropicAreaChange_1
      type: IsentropicAreaChange
      attributes:
        label: branchA
        index: 2
    - id: LossElement_1
      type: LossElement
      attributes:
        label: branchB
        index: 3
        lossCoefficient: 5
    - id: JunctionStaticP_1
      type: JunctionStaticP
      attributes:
        label: junction
        index: 4
        leftPorts: 2
        rightPorts: 1
    - id: PressureOutlet_1
      type: PressureOutlet
      attributes:
        label: outlet
        index: 5
        pressure: 101325
        backflowTotalTemperature: 300
  edges:
    - id: edge_1
      source: MassFlowInlet_1
      target: LosslessSplitter_1
      sourceHandle: MassFlowInlet_1-port-0
      targetHandle: LosslessSplitter_1-port-0
      type: flow
      attributes:
        index: 0
        area: 0.5
    - id: edge_2
      source: LosslessSplitter_1
      target: IsentropicAreaChange_1
      sourceHandle: LosslessSplitter_1-port-1
      targetHandle: IsentropicAreaChange_1-port-0
      type: flow
      attributes:
        index: 1
        area: 0.25
    - id: edge_3
      source: LosslessSplitter_1
      target: LossElement_1
      sourceHandle: LosslessSplitter_1-port-2
      targetHandle: LossElement_1-port-0
      type: flow
      attributes:
        index: 2
        area: 0.25
    - id: edge_4
      source: IsentropicAreaChange_1
      target: JunctionStaticP_1
      sourceHandle: IsentropicAreaChange_1-port-1
      targetHandle: JunctionStaticP_1-port-0
      type: flow
      attributes:
        index: 3
        area: 0.3
    - id: edge_5
      source: LossElement_1
      target: JunctionStaticP_1
      sourceHandle: LossElement_1-port-1
      targetHandle: JunctionStaticP_1-port-1
      type: flow
      attributes:
        index: 4
        area: 0.25
    - id: edge_6
      source: JunctionStaticP_1
      target: PressureOutlet_1
      sourceHandle: JunctionStaticP_1-port-2
      targetHandle: PressureOutlet_1-port-0
      type: flow
      attributes:
        index: 5
        area: 0.5
uiAttributes:
  nodes: []
uiState:
  counters:
    nodeCounters: {}
    totalNodeCounters: {}
"""


@pytest.fixture
def case_file(tmp_path):
    path = tmp_path / "diamond.yaml"
    path.write_text(SAVE_FILE)
    return str(path)


def reference_solution():
    net = Network(AIR, p_ref=101325.0, T_ref=300.0)
    inl = net.add(MassFlowInlet(mdot=20.0, Tt=400.0, name="inlet"))
    spl = net.add(LosslessSplitter(3, name="split"))
    ia = net.add(IsentropicAreaChange(name="branchA"))
    lb = net.add(LossElement(K=5.0, name="branchB"))
    jun = net.add(JunctionStaticP(3, name="junction"))
    out = net.add(PressureOutlet(p=101325.0, name="outlet"))
    net.connect(inl, spl, area=0.5)
    net.connect(spl, ia, area=0.25)
    net.connect(spl, lb, area=0.25)
    net.connect(ia, jun, area=0.3)
    net.connect(lb, jun, area=0.25)
    net.connect(jun, out, area=0.5)
    res = solve(net, x0=net.initial_guess(mdot0=0.0))
    assert res.converged
    return net, res


def test_bridge_matches_directly_built_network(case_file):
    net, meta = load_case(case_file)
    res = solve(net, x0=net.initial_guess(mdot0=0.0))
    assert res.converged

    ref_net, ref_res = reference_solution()
    st = net.states(res.x)
    ref = ref_net.states(ref_res.x)
    for k in range(net.n_edges):
        assert st[k].mdot == pytest.approx(ref[k].mdot, rel=1e-8)
        assert st[k].p == pytest.approx(ref[k].p, rel=1e-8)
        assert st[k].Tt == pytest.approx(ref[k].Tt, rel=1e-8)


def test_results_payload_format(case_file):
    net, meta = load_case(case_file)
    res = solve(net, x0=net.initial_guess(mdot0=0.0))
    payload = results_payload(net, meta, res.x)

    json.dumps(payload)  # must be JSON-serializable
    names = {d["name"]: d for d in payload["datasets"]}
    assert "Mass flow" in names and "Mach number" in names and "Mass imbalance" in names

    for d in payload["datasets"]:
        assert d["target"] in ("node", "edge")
        expected = net.n_edges if d["target"] == "edge" else len(net.elements)
        assert len(d["values"]) == expected

    # values are ordered by the UI edge index: edge 0 is the inlet feed
    mdot = names["Mass flow"]["values"]
    assert mdot[0] == pytest.approx(20.0, rel=1e-9)
    assert mdot[1] + mdot[2] == pytest.approx(20.0, rel=1e-9)  # branch split
    # boundary nodes report zero imbalance; interior ones conserve
    assert max(abs(v) for v in names["Mass imbalance"]["values"]) < 1e-8


def test_load_rejects_wrong_model(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  id: acoustic-network\n  nodes: []\n  edges: []\n")
    with pytest.raises(UICaseError):
        load_case(str(path))


def test_load_rejects_dangling_two_port(tmp_path, case_file):
    import yaml

    data = yaml.safe_load(SAVE_FILE)
    data["model"]["edges"] = data["model"]["edges"][:-1]  # junction -> outlet removed
    path = tmp_path / "dangling.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(UICaseError):
        load_case(str(path))
