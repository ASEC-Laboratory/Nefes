# The network abstraction

The network abstraction is the structure on which everything else rests: a fluid system is represented as a directed graph, and the entire physical model is a set of relations attached to that graph.
This document describes what the graph is, what the quantities carried on it represent, where the flow state lives, and the sign convention that lets the solver work without knowing in advance which way the gas flows.
It also shows that the abstraction requires a thermodynamic closure, without committing to a particular one.

## Standing assumptions {#sec-abstraction-assumptions}

This document inherits the standing assumptions of the [overview](overview.qmd); the two that are specific to the network abstraction, and are examined in the sections that follow, are stated here:

1. **Port quantities are adequately described by their cross-sectional averages.** Each edge carries a single representative state standing for the average over its port; the profile-shape factor and the treatment of turbulence are developed with the governing equations ([edge-state closure](governing-equations.md#sec-govern-edge-closure); [turbulence](governing-equations.md#sec-govern-turbulence)).
2. **A thermodynamic closure supplies the state relations.** The abstraction requires a closure that returns the thermodynamic state, and its derivatives, from the carried variables; it does not depend on whether that closure is a calorically perfect gas or a chemical-equilibrium mixture.

## The network as a directed graph {#sec-abstraction-graph}

A *flow network* is a directed graph whose nodes are *elements* and whose edges represent the shared cross-sections between them.
The elements can be thought as analogous to volumes in the classical finite-volume (FV) method, and edges to the faces.
From this perspective the network discretization appears similar to a face-based finite volume discretization, however, this similarity is only conceptual: unlike FV, the network discretization does not have the notion of spatial coordinates.

An *element* is a component — a nozzle, orifice, junction, plenum, boundary, or flame — modeled as a *control volume on which the governing conservation laws are applied*.
The FV method typically uses spatial gradients to establish a relation between the face values and the cell-center values.
In absence of a spatial coordinate for the network discretization, establishing a similar relation between element states and edge states is not a well-defined problem.
For the edge-based network discretization used in $\textsf{Nefes}$, elements only enforce relations across the state vectors of their incident edges - they do not carry an internal state.
The mean flow layer deals with the steady-state form of the governing equations, which boils down to the flux balance that can be naturally expressed in terms of edge states.
Modeling compliance and inertia effects in the perturbation layer, that arise from the transient term in the governing equations, *do* require an element state — which is overcome by the assumption of compactness ([governing equations](governing-equations.md#sec-govern-zero-volume); [perturbation network](perturbation-network.md#sec-perturb-storage-block)).

An *edge* represents a surface shared between two neighbouring elements, or between an element and the exterior, and it carries the *state vector* $\mathbf{x}_e$ and has an area $A_e$ associated with it.
The state vector is the minimal set of variables from which every other flow quantity on the edge — density, velocity, temperature, sound speed, Mach number, stagnation state — can be recovered; the particular choice, and the recovery, are the subject of [state and recovery](state-and-recovery.qmd).

A network definition therefore consists of three ingredients: the element types and their parameters, the edge (port) areas $A_e$, and the connectivity — which edges attach to which elements, and at which local ports.
Each element of degree $d$ numbers its incident ports $0, 1, \dots, d-1$, and the ordering is significant wherever an element treats its ports asymmetrically (for instance the reference port of a loss element or a junction), so it is preserved from the network definition through to the residual.
The connectivity is exactly a signed node–edge incidence: each edge is incident to two elements, entering one as an outgoing port and the other as an incoming port, and the sign of that incidence is the orientation factor introduced below.
The counting and assignment of equations on this incidence, and its storage as the sparsity pattern of the Jacobian, are treated in [equation structure](equation-structure.md).

## Edge quantities as section averages {#sec-abstraction-averages}

The state on an edge is a single set of scalars, yet the port it represents is a two-dimensional surface over which the flow is in general non-uniform.
The reconciliation is that each edge quantity represents the *cross-sectional average* over the surface it represents.
We write the section (area) average of a field $\phi$ over a port of area $A$ as:

$$
\langle \phi \rangle \equiv \frac{1}{A}\int_A \phi\,\mathrm{d}A,
$$

where the integral runs over the port cross-section.
A network edge therefore carries one representative state for that average — the same single-face idea as one-dimensional gas dynamics — rather than a resolved profile.
How that one state is used to form convected fluxes when the profile is non-uniform (the profile-shape factor $\beta_\psi$), and how turbulence is treated relative to Reynolds stresses and to the acoustic fluctuation $X'$, are part of the reduction from the integral conservation laws ([governing equations](governing-equations.md#sec-govern-edge-closure); [turbulence](governing-equations.md#sec-govern-turbulence)).

## Orientation and the sign convention {#sec-abstraction-orientation}

Each edge $e$ is given a reference direction at build time, drawn from its *tail* element to its *head* element.
An essential point is that this arrow makes no physical claim: it does **not** assert that the gas flows from tail to head, but only fixes what "positive" means for the signed edge quantities — the mass flow rate $\dot m_e$, the velocity, and the Mach number.
It is the direct analogue of the outward surface-normal chosen for a face in a classical finite-volume method: an orientation fixed once, against which signed fluxes are measured, and whose particular choice does not affect the result.
If the converged solution returns a negative $\dot m_e$, the gas flows against the arrow, and the sign resolves itself as part of the solution rather than being prescribed.

To write an element's balance without reference to the global arrow directions, we introduce the orientation factor, defined as:

$$
\sigma_{P,e} =
\begin{cases}
+1 & \text{if } e \text{ points away from } P \text{ ($P$ is the tail)},\\
-1 & \text{if } e \text{ points towards } P \text{ ($P$ is the head)},
\end{cases}
$$

where $P$ is an element and $e$ an incident edge, so that $\sigma_{P,e}$ is precisely the sign of the signed node–edge incidence.
Then $\sigma_{P,e}\,\dot m_e$ is the mass flow *leaving* $P$ through $e$, whatever the global arrow conventions are, and it is convenient to name the two signed fluxes explicitly:

$$
\dot m^{\text{out}}_{P,e} = \sigma_{P,e}\,\dot m_e,
\qquad
\dot m^{\text{in}}_{P,e} = -\sigma_{P,e}\,\dot m_e,
$$

where $\dot m^{\text{out}}_{P,e}$ and $\dot m^{\text{in}}_{P,e}$ are the mass flows out of and into element $P$ through edge $e$.
The role of $\sigma_{P,e}$ is exactly that of the outward-normal sign in a finite-volume scheme: it fixes the sign with which each face contributes to the balance of its cell, so an edge shared by two elements enters their two balances with opposite signs, and mass is conserved across the shared face by construction.

A core requirement follows, and it is a requirement the abstraction must *satisfy*, not merely assume: the choice of edge arrows has no influence on the physical solution.
Reversing an edge's direction negates its $\dot m_e$ and flips every $\sigma_{P,e}$ that references it, and the two negations must cancel throughout, so that the recovered pressures, temperatures, and flow magnitudes are unchanged.
This *direction-flip invariance* is verified numerically rather than taken on faith (test: `test_edge_direction_invariance`).

## The thermodynamic closure {#sec-abstraction-closure}

The network abstraction is independent of the gas thermodynamics: it requires only that a *thermodynamic closure* be supplied, one that returns the thermodynamic state — and, for the exact Jacobian, its derivatives — from the carried variables and the composition.
Two closures are provided, and the element residuals see only the recovered state either produces, never the closure that produced it (see [state and recovery](state-and-recovery.qmd#sec-state-recovery-closure)).

The first, and the simplest, is the *calorically perfect gas*, for which the state relations are given as:

$$
p = \varrho R T,
\qquad
h = c_p T,
\qquad
c^2 = \gamma R T,
\qquad
\gamma = \frac{c_p}{c_v},
$$

where $p$ is the static pressure, $\varrho$ the density, $T$ the static temperature, $h$ the static specific enthalpy, $c$ the speed of sound, $R$ the specific gas constant, $c_p$ and $c_v$ the specific heats at constant pressure and volume, and $\gamma$ their ratio.
It is a good approximation for air and for combustion products over moderate temperature ranges, and it makes the abstraction concrete.
Throughout the documentation we use the caloric constant:

$$
\Gamma \equiv \frac{c_p}{R} = \frac{\gamma}{\gamma - 1}
\qquad (\approx 3.5 \text{ for air}),
$$

where $\Gamma$ groups the combination of specific heats that recurs in the state recovery and the isentropic relations, and it is the natural constant in which the total-enthalpy and stagnation relations are most compactly written.

The second is a *chemical-equilibrium mixture*, in which the specific heats and molar mass become temperature- and composition-dependent and are obtained from an equilibrium solve, and each reacting edge additionally carries a transported composition.
It enters the abstraction through the same interface — a recovered state and its derivatives — and its construction, together with the transported mixture fractions and the frozen/burnt closures, is deferred to [thermochemistry](thermochemistry.md).
The symbols used here and elsewhere are collected in the [nomenclature](../nomenclature.md).
