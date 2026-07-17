---
title: 'Nefes: a network solver for reacting compressible flows and thermoacoustics'
tags:
- Python
- thermoacoustics
- combustion instability
- duct acoustics
- compressible flow
- network model
- gas turbines
authors:
- name: Çetin Ozan Alanyalıoğlu
  orcid: 0000-0002-8498-3088
  corresponding: true
  affiliation: 1
- name: Hendrik Nicolai
  orcid: 0000-0002-0355-2252
  affiliation: 2
affiliations:
- name: TBD
  index: 1
- name: TBD
  index: 2
date: TBD
bibliography: paper.bib
---

# Summary

Nefes is an open-source Python package that models a compressible
internal-flow system, such as a gas-turbine combustor or a duct network,
as a directed graph of connected elements instead of resolving the full
three-dimensional field. On this graph it first solves the steady mean
flow: pressures, velocities, temperatures, and gas composition, with
chemical equilibrium wherever streams burn or mix, and smooth choking at
a sonic throat. The solver finds flow directions itself from a quiescent
start: the user supplies geometry and boundary conditions, not a guess
of the answer. On that solved mean flow, Nefes then computes the linear
perturbations superimposed on it — sound waves, convected hot spots
(entropy waves), and composition fluctuations — as the linearization of
the very same network model, so the acoustic description cannot drift
from the flow it is based on. Such waves couple with unsteady combustion
into thermoacoustic instability, a first-order design concern for
low-emission gas turbines and rocket engines; network models are the
standard fast screening tool at the design stage. The same machinery
serves duct acoustics more broadly, including resonances, damping, and
scattering at area changes and junctions, with perturbations restricted
to flow-aligned (longitudinal) waves.

# Statement of need

Many practical questions about a combustor or a ducted gas system are
questions about a network: how the mass flow divides among parallel
passages, what pressures and temperatures establish where streams merge
and burn, whether a nozzle operates in the choked regime or not.
Open-source steady network solvers serve pipeline hydraulics and
thermodynamic cycle analysis well, but, as detailed below, none covers
the regime such systems occupy. To the best of our knowledge, no open
tool combines momentum-resolved compressible duct flow, including Mach
effects and a smooth, emergent treatment of choking, with
chemical-equilibrium thermochemistry evaluated inside the solve, on an
arbitrary graph whose flow directions the solver finds itself from a
quiescent start. Nefes provides that combination; its mean-flow solver
is useful entirely on its own.

The same solver is also the foundation for the layer that motivated it
in the first place. Predicting thermoacoustic instability
[@juniper2018sensitivity] requires a mean flow together with the
linear waves that are superimposed on it, and stability verdicts can
hinge on mean-flow details that are easy to get slightly wrong in
practice. In existing network tools the two layers are separate models,
and nothing enforces agreement between them. Because Nefes solves the
mean flow, it constructs the perturbation problem, including acoustic,
entropy, and compositional waves [@magri2016compositional], as the
linearization of the same network model about the converged state, so
the two layers agree by construction. We are not aware of another
released tool that couples a solved reacting compressible mean flow to a
multi-wave perturbation network. A companion methods paper
[@alanyalioglu2026operator] presents the formulation and its
verification, and demonstrates that seemingly small inconsistencies
between the mean flow and the acoustic model can flip a stability
verdict.

There is a practical gap as well: the established open thermoacoustic
network tools are MATLAB-based — OSCILOS [@li2015oscilos] and taX
[@emmert2014tax], the latter also needing Simulink and commercial
toolboxes — so running them requires a commercial platform. Nefes is
pure Python on the open numerical stack [@harris2020numpy;
@virtanen2020scipy; @lam2015numba], installable with pip, permissively
licensed (BSD-3-Clause), and scriptable end to end.

Nefes is intended for researchers modeling compressible reacting flow on
networks, with or without the acoustic layer; for combustion and
thermoacoustics groups screening designs for instability; for
duct-acoustics work on transmission, reflection, and scattering of
longitudinal waves in ducts [@munjal2014ducts]; for teaching, where a
complete instability analysis fits in a short notebook; and as an
extensible base for design-stage analysis in industrial gas-turbine
practice.

# State of the field

Steady flow-network solvers exist in several adjacent regimes. Pipeline
tools such as pandapipes [@lohmeier2020pandapipes] handle arbitrary
topology and genuinely compressible gases, including hydrogen, but in
the low-Mach, friction-dominated pipeline regime, without Mach effects,
choking, or reaction. Cycle-analysis tools such as pyCycle
[@hendricks2019pycycle] do branch and mix streams and do carry
chemical-equilibrium thermochemistry, but they abstract the system as
zero-dimensional stations in a fixed cycle layout, with no wave layer.
TESPy [@witte2020tespy] models plant flowsheets with lumped components
and simplified combustion. GFSSP [@majumdar2013gfssp], the nearest
regime match with momentum-resolved branches and choking, is closed
source and treats reaction outside the solve. None of these tools
resolves the duct momentum balance through smooth choking, computes its
mean state with equilibrium chemistry, and discovers the flow on an
arbitrary graph rather than a prescribed pipeline or cycle; that
combination is the spot Nefes fills.

Low-order thermoacoustic network models themselves are long established
[@dowling1995calculation; @dowling2003acoustic]. Among released
implementations, OSCILOS [@li2015oscilos] marches one-dimensional jump
relations along an essentially serial chain of modules for its mean
state, and taX [@emmert2014tax] propagates user-prescribed mean values
through local per-element relations, with branch flow splits supplied by
the user; LOTAN [@stow2001annular] is not public. These tools carry
entropy as well as acoustic waves, but none solves a coupled mean-flow
problem on the network or derives its acoustic elements from the same
equations as its mean state; among the released tools, none runs without
a commercial platform.

The closest methodological relative is the framework of
@merk2025jacobian, who derive acoustic, entropic, and compositional jump
conditions of compact elements as Jacobians of steady conservation
relations. Nefes realizes the corresponding construction for the
assembled system — the network Jacobian evaluated at the converged mean
state — in released software, with the mean state solved rather than
supplied. We built a new tool rather than extending an existing one
because this consistency requires both layers to share one set of
element equations from the outset: it is an architectural property, and
in the established tools the two layers are separate models by
construction.

# Software design

Nefes is based on a fundamental modeling approach: every element, such
as a duct, an orifice, a flame, or a junction, contributes algebraic
residual equations in the flow states of its incident edges, and a
network is the assembly of these residuals on a directed graph. The
steady mean flow is the solution of the assembled system; the linear
perturbation problem is the exact linearization of the same residuals
about that solution, extended, in ducts, with a dedicated
wave-propagation model that is transparent to the mean-flow solve.
Consistency between the mean flow and the acoustics is therefore an
architectural property rather than a discipline the user must maintain.

The exactness requirement of the linearization process imposes a
constraint on the element equations: every residual must be smooth in
the flow state, with no branches or switches, so that derivatives exact
to machine precision can be taken by complex-step differentiation
[@martins2013derivatives]. Some elements become less convenient to
write: choking, for example, enters as a smooth reformulation of the
sonic condition rather than a change of regime, and every function that
is not smooth in the flow state must be wrapped in a smooth
approximation.

![The Nefes architecture: the network description is frozen into an
immutable compiled problem; one compiled kernel returns residuals and
exact (complex-step) Jacobians to both the mean-flow solve and the
perturbation analysis.](figures/architecture.png)

In return, the Jacobian used by the solver, and with it the acoustic
operator built from it, cannot drift from the residuals, since both are
generated from the same source (Figure 1).

Assembly and user-facing layers are pure Python behind a small set of
objects and interfaces; the element kernels are compiled just in time
with Numba [@lam2015numba] and hidden from the user. A typical
mean-flow solution with its full stability analysis runs in seconds to
minutes on a laptop, making design-stage parameter sweeps practical.

# Features

The mean-flow layer:

- Solves the steady compressible mean flow on an arbitrary network as
  one coupled system, on arbitrary edge directions from a quiescent
  start; smooth residuals allow the solver to reach hard operating
  points without hand-tuned initial guesses.
- Evaluates chemical-equilibrium thermochemistry (NASA-Glenn data)
  inside the solve, so temperature, density, and composition follow from
  equilibrium wherever streams burn or mix; verified against Cantera
  [@cantera].
- Treats choking as an emergent, smooth part of the solution rather than
  as a boundary condition selected by the user.
- Provides an element catalog of inlets and outlets, ducts and pipes
  with friction, orifices, area changes and nozzles, junctions and
  splitters, cavities, heat sources and equilibrium flames, plus
  composite elements (tapered ducts, Fanno pipes, Helmholtz resonators)
  that expand into subgraphs before the solve.

The perturbation layer, on the converged mean flow:

- Computes eigenmodes (with a count of modes in the searched band), a
  real-frequency Nyquist stability criterion, forced responses, and
  acoustic scattering and transfer matrices of any element or
  subnetwork.
- Transports acoustic, entropy, and compositional waves, including the
  entropy-to-sound conversion at accelerating and choked nozzles.
- Allows attaching prescribed dynamic responses (such as flame transfer
  functions) to supported elements, and supports the reverse operation:
  extracting an unknown element’s response from a measured response of
  the surrounding network.

Across both layers: deterministic numerics, cases that serialize to and
from YAML, documentation built from executable notebooks, and a test
suite (88 files) with a claim-to-test validation map. Networks can also
be built interactively in Nemo, a companion browser-based editor.

# Examples

The repository ships a gallery of runnable example notebooks, each
regenerating the figures it shows: steady reacting networks from a
single flame to a full gas-turbine combustor and can-annular
architectures; duct-acoustics cases including expansion chambers,
Helmholtz resonators, and frequency-dependent boundaries; and
thermoacoustic analyses spanning the Rijke tube, a gas-turbine
combustor, flame identification, and entropy- and composition-driven
instability, together with the validation cases behind the benchmarks
cited below.

# Research impact statement

Nefes is newly released software; its case for significance rests on
verification and validation rather than a prior publication record. The
shipped benchmarks reproduce published results across all layers: branch
flows and nodal pressures of a compressed air pipe network
[@greyvenstein1994segregated], the transmission loss of an expansion
chamber [@dokumaci2021duct], the compact-nozzle acoustic and entropy
response of @marble1977nozzle, and, on a published laboratory-combustor
case [@li2015oscilosreport], a thermoacoustic mode within one percent
of the OSCILOS result in frequency and growth rate. The examples
reproduce a published swirl-burner stability analysis [@emmert2017brs]
at figure level, down to its intrinsic-mode branch and
reflection-coefficient sweep. The methods foundation is set out in the
companion paper [@alanyalioglu2026operator], and the package is in
active use in the authors’ research activities. Every figure in the
documentation is generated by a runnable notebook shipped with the
repository.

# AI usage disclosure

Generative AI (Anthropic’s Claude) assisted with implementation and
documentation drafting. Development followed a specification-driven
workflow: the authors designed the architecture—data structures,
interfaces, and algorithms—and authored the corresponding specification
sheets, implementation examples where necessary, and acceptance tests
that define intended behavior; AI then produced the bulk of the source
code to satisfy those specifications. First drafts of the package
documentation were likewise AI-generated and subsequently reviewed and
edited by the authors. All AI-assisted output was reviewed, tested, and
edited by the authors; correctness is gated by the package’s test suite
and the validation benchmarks cited above.

# Acknowledgements

TBD.

# References
