---
name: nefes-user
description: >-
  Use when building, solving, or analyzing a Nefes compressible-flow network as a
  user of the tool -- writing a notebook or script that imports `nefes`, constructs a
  `Network`, calls `.solve()`, or runs the acoustic/perturbation layer. Covers:
  assembling elements (inlets, outlets, orifices, valves, nozzles, ducts, junctions,
  plena, flames) and solving the steady mean flow; choosing which element represents a
  real component; diagnosing non-convergence or "no steady solution"; acoustic
  transfer/scattering matrices, entropy noise, and thermoacoustic stability
  (flame transfer functions, eigenmodes, growth rates) on a converged mean flow;
  reacting/combustion networks; parameter sweeps; and saving or loading cases. Trigger
  it even when the user does not say "Nefes" but is clearly modeling a gas-flow network
  of pipes, orifices, plena, or burners with this package. Do NOT use it for developing
  Nefes itself -- editing solver/kernel/element source, writing package tests, or
  changing the docs; that is not this skill's job.
---

# Nefes (user): build, solve, and analyze cases with fidelity

You are using Nefes to model a fluid system, not developing the package.
Nefes models the system as a directed graph and solves for the steady mean flow, and the linear acoustic/entropy behavior around it, without resolving the full 3-D field.

## The authority: `docs/best-practices.md`

`docs/best-practices.md` is the single source of truth for how Nefes is meant to be used.
It is line-cited to the source and its code blocks were executed, so it is the current, correct API; this skill deliberately does **not** restate signatures, because a second copy drifts and then produces confident, wrong code.

Read the relevant section of `docs/best-practices.md` before writing Nefes code, and prefer its patterns over anything you find in an older notebook.
When you need a signature or a parameter meaning, get it from that doc or from the installed package (below), never from memory.

## Hold the mental model

Every part of the API follows from these; getting one wrong produces a wrong or unsolvable model that does not announce itself.

- The system is a **directed graph**: elements are nodes, the flow state lives on edges. Each edge carries `(mdot, p, h_t)` plus any transported scalars.
- **Area is a property of the edge, not the element.** An area-change element reads the differing areas of its two incident edges; there is no area argument on the factory.
- The solver **discovers** flow directions, choke points, and the operating state. Edge directions are bookkeeping; the solution is invariant to them.
- The **acoustic layer is a linearization around a converged mean flow**, not a second solver. Always solve the mean flow first.
- Present scope is **subsonic** (flowing or quiescent). A model that wants a supersonic mean will not converge to a physical state.

## The non-negotiable loop

An unexecuted notebook that "looks right" is the primary way AI-built Nefes cases are wrong.
Do not report a result you have not run and checked. For every case:

1. **Read** the matching section of `docs/best-practices.md`.
2. **Verify** each signature you use against the installed package (do not recall it).
3. **Build** the `Network`; capture node indices from `add` and edge ids from `connect` (or recover them with `net.element_index`, `net.edge_between`, `net.edges_of`) instead of hand-counting.
4. **Solve**, then gate on the result before interpreting any number:

   ```python
   sol = net.solve()
   assert sol.converged, (sol.residual_norm, sol.print_residuals())
   sol.verify()          # physical-consistency checks (mass balance, choking, ...)
   ```

5. **Sanity-check the physics**: subsonic Mach on every edge, mass balance closes, choke points are where you expect, temperatures/compositions are plausible.
6. Only a case that ran, converged, and passed `verify()` is trustworthy.
   Nefes fails loudly: a bad setup raises with a message, a hard problem returns an inspectable `converged=False`, a stretched assumption warns. A number returned with no warning is trustworthy; do not manufacture confidence the tool did not give you.

If `converged=False`, read `sol.result.history` before changing anything: still-decreasing means warm-start or raise `max_iter`; flat/stalled means the demand is physically infeasible and the setup must change.
The full decision tree is `docs/best-practices.md` §14 (Troubleshooting).

## Verify, don't recall

Run Nefes in the project conda env from `environment.yml` (env name `nefes`), or against an editable install (`pip install -e .[dev]`).

Before using an unfamiliar call, confirm it exists and check its signature against the live package rather than trusting memory:

```python
import inspect, nefes
from nefes.elements import catalog as cat
inspect.signature(cat.orifice)          # real parameters, not remembered ones
[n for n in dir(nefes) if not n.startswith("_")]   # what is actually exported
```

Hallucinated calls are the dominant failure mode: `PerturbationBC.excitation`, `excitation(...)`, and `boundary_response(...)` have all been invented by agents and do **not** exist.
`docs/best-practices.md` §"Anti-patterns" lists the invented and superseded calls with their real replacements.

## Guardrails (the traps that produce silently wrong models)

Short list; the symptom -> cause -> fix detail is in `docs/best-practices.md` and `docs/reference/modeling-guide.md`.

- **`junction` feeding a fast branch has no steady solution.** Use `splitter` wherever a plenum distributes to fast branches; reserve `junction` for low-Mach merges of comparable streams.
- **Total vs. static pressure.** Inlets pin *total* pressure; outlets pin *static* pressure; loss elements are total-pressure relations. Misreading this shifts the whole operating point.
- **Area lives on edges.** Model an area change by giving the two incident edges different areas.
- **Subsonic only.** Keep throats at or below sonic; model a choked exit with `choked_nozzle_outlet`, not a resolved M=1 throat.
- **Acoustics need a converged mean flow, aligned flow, and length.** Solve first, then check preconditions (`verify_perturbation`), and give every length-bearing passage its length or its acoustic phase is lost.
- **Reacting runs** need a composition on the inlets and an absolute-enthalpy datum; follow the worked recipe rather than improvising.

## Existing notebooks are suspect

Many notebooks in this repo predate the current public API and are internally inconsistent; `docs/best-practices.md` says so and lists their anti-patterns.
When extending or fixing an existing notebook, reconcile it against `docs/best-practices.md` first; do not copy its style forward.
Tells of a stale notebook: `sys.path.insert(...)` bootstraps, `from nefes.shell import build_problem`, `from nefes.solver import solve`, `ES_M`/`ES_P` index constants, deep `from nefes.perturbation.operator...` imports.

## Notebook conventions

- Do **not** save notebook outputs (filesize); leave cells unexecuted-on-disk or clear outputs before saving.
- Plot with **plotly** and the bundled theme: `from nefes.plotting import use_nefes_theme; use_nefes_theme()`.

## Where to go for what

Route to the source of truth; sanity-check that it is current rather than assuming.

| Need | Go to |
| --- | --- |
| How Nefes is meant to be used (the workflow, every step) | `docs/best-practices.md` |
| Why a model is wrong or will not solve | `docs/best-practices.md` §14, `docs/reference/modeling-guide.md` |
| Full element parameters and coefficients | `docs/reference/atomic-elements.md`, `docs/reference/composite-elements.md`, `docs/reference/modeling-guide.md` |
| Runnable worked cases (build, sweep, reacting, acoustics) | `examples/` (start: `examples/getting-started/converging_nozzle.ipynb`) |
| Theory: governing equations, elements, choking, acoustics | `docs/theory/` |
| A signature or export you are unsure of | the installed package (`inspect.signature`, `dir(nefes)`) |
