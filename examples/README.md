# Examples

- **`converging_nozzle.yaml`** — a network **saved from the FNetLibUI tool** (the
  `fns-flow-network` model): reservoir → isentropic contraction → back-pressure
  outlet.
- **`solve_from_yaml.ipynb`** — loads the UI case, solves the steady mean flow,
  prints the converged edge states, sweeps the back pressure to show emergent
  choking (mass-flow saturation at `M = 1`), and computes closed–closed duct
  acoustic modes on the side.

## Running the notebook

The notebook adds the repo root to `sys.path`, so no install of `fns` is needed —
just run it with a Python that has the project dependencies (`numpy`, `scipy`,
`numba`, `pyyaml`, `matplotlib`) plus `ipykernel`:

```bash
conda activate fns
jupyter lab examples/solve_from_yaml.ipynb
```

Or solve a UI case in two lines:

```python
from fns.io import load_case
sol = load_case("examples/converging_nozzle.yaml").solve()
print(sol.edge(1))   # throat state: mdot, M, p, p_t, T, ...
```

## The UI case format

`load_case` reads the native YAML the **FNetLibUI** tool writes out for the
`fns-flow-network` model (defined in that repo under `public/models/`). The
relevant sections:

```yaml
model:
  id: fns-flow-network
  globalAttributes: {gasConstant: 287.0, heatCapacityRatio: 1.4,
                     referencePressure: 101325.0, referenceTemperature: 300.0,
                     referenceMassFlow: 5.0}      # 0 -> auto
  nodes:
    - {id: TotalPressureInlet_1, type: TotalPressureInlet,
       attributes: {label: reservoir, index: 0, totalPressure: 2.0e5, totalTemperature: 300.0}}
    - {id: IsentropicAreaChange_1, type: IsentropicAreaChange, attributes: {label: nozzle, index: 1}}
    - {id: PressureOutlet_1, type: PressureOutlet,
       attributes: {label: back-pressure, index: 2, pressure: 1.5e5, backflowTotalTemperature: 300.0}}
  edges:
    - {id: edge_1, source: TotalPressureInlet_1, target: IsentropicAreaChange_1,
       sourceHandle: TotalPressureInlet_1-port-0, targetHandle: IsentropicAreaChange_1-port-0,
       type: flow, attributes: {label: feed, index: 0, area: 0.020}}
    - {id: edge_2, source: IsentropicAreaChange_1, target: PressureOutlet_1,
       sourceHandle: IsentropicAreaChange_1-port-1, targetHandle: PressureOutlet_1-port-0,
       type: flow, attributes: {label: throat, index: 1, area: 0.010}}
```

**Ports matter and are preserved.** Each edge's `sourceHandle`/`targetHandle`
ends in `-port-<ordinal>`; the loader keeps those ordinals and densifies each
element's incident ports to `0..d-1`, so port-0 conventions (the LossElement
reference area, the junction/splitter reference port) match the canvas. Element
`type` names map to the FNS catalog: `MassFlowInlet`, `TotalPressureInlet`,
`PressureOutlet`, `IsentropicAreaChange`, `SuddenAreaChange`, `LossElement`,
`Duct`, `JunctionStaticP`, `LosslessSplitter`. Supersonic boundaries are deferred
in v1 and raise a clear error.
