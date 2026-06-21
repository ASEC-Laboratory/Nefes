# FNS — Flow Network Solver

This repository is the **preliminary study** for FNS, a compressible-flow
network analysis tool. It holds two things: a working prototype (the `fns`
package) and the **design specification** it was built to validate (`docs/`).
The prototype carries branching networks with emergent flow directions and
choking end to end, which is what confirmed the design holds up; the documents
are the blueprint for the tool itself.

## The design spec (`docs/`)

*What* FNS is and *why* lives here — the theory, the equation structure, the
implementation plan, and the extension paths. This README is deliberately just
usage and orientation; it does not restate any of it.

* `docs/theory.md` — the complete theory: framework, governing equations,
  element formulations, characteristic variables, the solver, emergent
  choking/shocks, and the acoustic / perturbation network (§12).
* `docs/implementation-plan.md` — the plan for the first version: connectivity,
  Jacobian, storage, thermo, solver, the acoustic layer, and the OO user shell.
  (`docs/examples/ConnectivityDemonstrator.yaml` is an example UI-export input,
  the worked example of the connectivity format in §2.)
* `docs/modeling-guide.md` — mapping standard catalogue restrictions (orifices,
  valves, nozzles) onto the existing element library.
* `docs/reactive-flow-requirements.md` — requirements for reactive-flow support
  and the standalone thermochemistry library.

## Quick start

```python
from fns import (AIR, Network, MassFlowInlet, IsentropicAreaChange,
                     PressureOutlet, solve)

net = Network(AIR, p_ref=101325.0, T_ref=300.0)
inl = net.add(MassFlowInlet(mdot=25.0, Tt=500.0, name="inlet"))
iac = net.add(IsentropicAreaChange(name="iac"))
out = net.add(PressureOutlet(p=101325.0, name="outlet"))
net.connect(inl, iac, area=0.3)     # edge 0, directed inlet -> iac
net.connect(iac, out, area=0.12)    # edge 1

result = solve(net)                 # converged in a handful of iterations
print(net.report(result.x))
```

## Running the prototype

Needs numpy + scipy; the tests need pytest.

```bash
python examples/ex1_nozzle_chain.py      # multi-element chain, quiescent start
python examples/ex2_split_fractions.py   # branching network, split fractions
python examples/ex3_reverse_flow.py      # flow against the edge directions
python examples/ex4_manifold.py          # two-temperature sources, mixing, 3 branches
python examples/ex5_compressibility.py   # pressure-ratio sweep, emergent choking
python examples/ex6_bridge.py            # compressible Wheatstone bridge, interior reversal
python -m pytest tests/                  # validation vs analytic relations
```

## Building cases with the node-graph UI editor

The element library is available as a model in the (separate) node-graph network
editor, as the **FNS Flow Network** model (`fns-flow-network`):

1. Start the UI, load the **FNS Flow Network** model, and draw the network.
   Edge arrows are only sign conventions — the solver finds the actual flow
   directions. Set edge areas and element parameters in the info pane; gas
   properties and reference scales live in the Model pane.
2. Save the canvas (YAML save file) and solve it:

   ```bash
   python run_ui_case.py my-case.yaml          # writes my-case-results.json
   ```

   A ready-made example: `python run_ui_case.py examples/ui_case_diamond.yaml`.
3. Load the produced JSON in the UI's data pane to color nodes/edges with the
   solution (mass flow, Mach number, pressures, temperatures per edge;
   conservation residuals per node).

`fns/ui_bridge.py` implements the translation (`load_case`,
`write_results`) if you need it programmatically.

**Ready-made showcase cases** (`examples/ui_showcase/`, regenerate with
`python examples/make_ui_showcase.py`) — each `<case>.yaml` opens directly in
the UI and ships with its solved `<case>-results.json` for the data pane:

| case | demonstrates |
|---|---|
| `cd_nozzle_venturi` | C-D nozzle, subsonic lossless operation |
| `cd_nozzle_shock_weak` / `_strong` | C-D nozzle choked with internal normal shock (M_s = 1.3 / 1.7), shock position deduced |
| `cd_nozzle_supersonic` | full supersonic nozzle operation at the design pressure (declared `SupersonicOutlet`): sonic throat, shock-free exit at M = 2.20 |
| `intake_near_critical` / `intake_supercritical` | started supersonic intake (M0 = 1.8): supersonic compression, sonic throat, terminal shock, pressure recovery |
| `gas_turbine_splits` | all-subsonic secondary-air system: plenum, manifold, three dissimilar branches, split fractions |
| `gas_turbine_large` | 57-element secondary-air system: two sources, mixing plenum, main manifold, three sub-manifolds, 15 metered branches, cross-bridge loop with emergent flow direction (solve with `--max-iter 400`) |
