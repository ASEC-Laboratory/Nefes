# Theory of the Compressible Flow Network

This document explains the complete theory behind the `fns` prototype:
what the method computes, why the equations look the way they do, how every
element is modeled, and how the solver finds the solution reliably.

**How to read this document.**  It is written in two voices.  Each section
opens with a plain-language explanation — what the idea is and why it
matters — marked like this:

> **In plain terms:** these blocks carry the intuition.  If you read only
> them, you will understand what the method does and why.

The mathematics that follows each block makes the same statement precise,
with full derivations.  Technical terms (residual, Jacobian, characteristic,
choking, …) are explained where they first appear.  Cross-references of the
form *(test: ...)* point to the automated checks in `tests/` that verify the
claim; *(example: ...)* points to a runnable demonstration in `examples/`.

Companion documents: `../README.md` (usage), `implementation-plan.md` (the implementation plan
for the first version), `modeling-guide.md` (modeling standard catalogue
restrictions — orifices, valves, nozzles), and `reactive-flow-requirements.md`
(reactive-flow / thermochemistry requirements).

---

## Contents

1. [The big picture](#1-the-big-picture)
2. [Framework and notation](#2-framework-and-notation)
3. [Governing equations and network discretization](#3-governing-equations-and-network-discretization)
4. [Solution variables and state recovery](#4-solution-variables-and-state-recovery)
5. [Equation structure: counting and assignment](#5-equation-structure-counting-and-assignment)
6. [Total-enthalpy transport on edges](#6-total-enthalpy-transport-on-edges)
7. [Element formulations](#7-element-formulations)
8. [Why naive formulations fail](#8-why-naive-formulations-fail)
9. [Characteristic variables](#9-characteristic-variables)
10. [Numerical solution](#10-numerical-solution)
11. [Compressible-flow behavior: emergent choking and shocks](#11-compressible-flow-behavior-emergent-choking-and-shocks)
12. [The acoustic / perturbation network](#12-the-acoustic--perturbation-network)
13. [Worked examples](#13-worked-examples)
14. [Validation map](#14-validation-map)
15. [Limitations and the development path](#15-limitations-and-the-development-path)
- [Appendix A: smooth regularized functions](#appendix-a-smooth-regularized-functions)
- [Appendix B: symbols and terms](#appendix-b-symbols-and-terms)

---

## 1. The big picture

> **In plain terms:**  A flow network is the fluid-dynamic analogue of an
> electrical circuit.  Components (nozzles, orifices, junctions, plenums)
> are connected by ports; gas flows through them driven by pressure
> differences, the way current is driven by voltage differences.  A network
> solver answers questions like: *how does the flow split between these
> branches?  What pressure and temperature does each component see?  How
> much flow does this supply deliver?* — without resolving the full 3-D flow
> field, which would take a CFD computation hours instead of milliseconds.
>
> Unlike an electrical circuit (or an incompressible water network), a gas
> network carries *three* coupled quantities along every connection: mass
> flow, pressure, and temperature (energy).  And because the gas is
> compressible, the relations between them are nonlinear and have genuinely
> new physics: flow rate saturates ("choking"), density varies along the
> path, and hot and cold streams mix.  This document describes a framework
> that handles all of that, *and* — the long-term goal of the project — is
> built so that the very same equations can later be reused to predict the
> acoustic (pulsation/thermoacoustic) behavior of the network around the
> computed operating point.

The framework rests on four design decisions, each derived in the sections
below:

1. **All state lives on the connections ("edges"), none in the components
   ("elements").**  Components only impose physical *relations* between the
   states of the connections that meet there (§3).
2. **Each edge carries the unknowns (mass flow, static pressure, total
   enthalpy)** — a choice with the unusual property that every other flow
   quantity can be computed from them *uniquely and smoothly*, no matter how
   fast or which way the gas flows (§4).
3. **A fixed bookkeeping rule** — every element contributes exactly as many
   equations as it has ports; every edge contributes one transport equation —
   makes the equation system square regardless of which way the gas decides
   to flow (§5–6).  This is what lets the solver *discover* flow directions
   instead of being told.
4. **A solution procedure built around smoothness** (§10): the equations are
   written without any if/else switching on the flow state, so the standard
   workhorse for nonlinear systems (Newton's method, with safeguards) works
   even from a "cold" start where nothing is flowing yet.

---

## 2. Framework and notation

> **In plain terms:**  We describe the system as a graph.  *Elements* are
> the components; *edges* are the flow cross-sections where two components
> meet (or where a component meets the outside world).  Every edge is given
> an arrow — an arbitrary reference direction chosen when the network is
> built.  The arrow does **not** claim the gas flows that way; it only fixes
> what "positive" means.  If the solution comes out with a negative mass
> flow on an edge, the gas flows against the arrow.  This is exactly like
> assuming current directions in circuit analysis and letting the signs
> sort themselves out.

A network consists of **elements** (graph nodes) and **edges**.  Elements
are control volumes in the zero-volume limit; they own *equations* but no
state.  Edges represent the planar port surfaces shared between neighbouring
elements; they own the *state*.  A network definition consists of element
types and parameters, edge areas $A_e$, and the connectivity.

Each edge $e$ is directed from its *tail* element to its *head* element.
All signed edge quantities (mass flow $\dot m_e$, velocity, Mach number) are
measured along this direction.  A core requirement is that the choice of
edge directions has **no influence on the physical solution** *(test:
`test_edge_direction_flip_invariance`)*.

For element $P$ and an adjacent edge $e$ define the orientation factor

$$
\sigma_{P,e} \;=\;
\begin{cases}
+1 & \text{if } e \text{ points away from } P \text{ (}P\text{ is the tail)},\\[2pt]
-1 & \text{if } e \text{ points towards } P \text{ (}P\text{ is the head)}.
\end{cases}
$$

Then $\sigma_{P,e}\,\dot m_e$ is the mass flow *leaving* $P$ through $e$,
whatever the arrow conventions are.  We abbreviate

$$
\dot m^{\text{out}}_{P,e} = \sigma_{P,e}\,\dot m_e,
\qquad
\dot m^{\text{in}}_{P,e} = -\sigma_{P,e}\,\dot m_e .
$$

The gas is **calorically perfect** (constant specific heats — a very good
approximation for air over moderate temperature ranges): $p = \varrho R T$,
$h = c_p T$, $\gamma = c_p/c_v$, speed of sound $c^2 = \gamma R T$, and we
use throughout

$$
\Gamma \;\equiv\; \frac{c_p}{R} \;=\; \frac{\gamma}{\gamma-1}
\qquad(\approx 3.5 \text{ for air}).
$$

---

## 3. Governing equations and network discretization

> **In plain terms:**  Physics gives us bookkeeping laws for any region of
> space: mass in = mass out, momentum changes only due to forces, energy
> in = energy out (for steady, adiabatic flow).  We apply these laws to each
> component, treating it as a "black box" so small that nothing accumulates
> inside.  What remains is a set of *jump conditions*: algebraic relations
> between the flow states at the component's ports.  For example, "the mass
> flows at the two ends of a nozzle are equal" or "the momentum change
> through a sudden expansion equals the pressure forces on it".  The whole
> network model is just the collection of these jump conditions plus the
> rule for how temperature (energy) is carried downstream.

### 3.1 Integral conservation laws

Start from the Euler equations (frictionless compressible gas dynamics) in
integral form over a control volume $\mathcal V$ with boundary $\mathcal S$:

$$
\frac{\partial}{\partial t}\int_{\mathcal V} \tilde{\mathbf q}\,\mathrm d\mathcal V
\;+\; \oint_{\mathcal S} \mathbf f_n \,\mathrm d\mathcal S
\;=\; \dot{\mathbf\Omega},
$$

with conserved quantities and their fluxes through the boundary

$$
\tilde{\mathbf q} =
\begin{bmatrix} \varrho \\ \varrho u \\ \varrho e_t \end{bmatrix}
\quad\text{(mass, momentum, energy per volume)},
\qquad
\mathbf f_n =
\begin{bmatrix} \varrho u_n \\ \varrho u\, u_n + p\, n \\ \varrho u_n h_t \end{bmatrix},
$$

where $h_t = h + \tfrac12 u^2$ is the **total enthalpy**: the energy content
of the stream per kilogram, combining thermal energy ($h = c_pT$) and
kinetic energy ($u^2/2$).  Total enthalpy is the natural "energy currency"
of steady flow: it is exactly conserved along an adiabatic stream, no matter
how the gas accelerates, decelerates, or loses pressure.

We restrict attention to port surfaces crossed perpendicular by the flow (no
swirl or cross-flow through a port), so a single signed normal velocity $u$
carries all kinetic energy.  This does **not** mean the flow inside a
component is one-dimensional — only that its *ports* are clean cross
sections.

### 3.2 Zero-volume limit: jump conditions

For a network element $P$ whose boundary consists of the port surfaces
$\mathcal S_i$ (the edges $e_i$) plus solid walls, the steady balance is

$$
\sum_{e_i} \mathbf F_i \;=\; \dot{\mathbf\Omega}_P,
\qquad
\mathbf F_i \;=\; \sigma_{P,e_i}\, \mathbf f_i\, A_i ,
$$

where $\mathbf f_i$ is the area-averaged flux at port $i$.  In the
zero-volume limit nothing is stored inside the element, and what remains is
a set of pure **jump conditions** between the port states.  One subtlety:
pressure forces on the element's *solid walls* do not disappear in this
limit — they enter the momentum balance explicitly.  That is why mass and
energy balances are universal but the momentum/pressure relation is what
*distinguishes* one element type from another (a nozzle from an orifice from
a dump diffuser): each geometry exerts different wall forces on the gas.

Two terms are absent from the steady balance above, for two *different*
reasons.  The **time-derivative (storage) term** is dropped because we seek a
steady state: $\partial/\partial t \equiv 0$ is what zeroes it, independent of
the element's size.  (It returns as the frequency-dependent storage of the
acoustic problem, §12.5, where a finite volume is reinstated.)  The
**zero-volume limit** is a separate idealization — it removes the volume
*integral* itself, leaving pure jump conditions between the port states.

**Source terms survive the limit when they are lumped, not volumetric.**  A
genuine in-element source splits by how it scales with the vanishing volume.  A
*volumetric density* — a body force, or a distributed heating per unit volume
($\propto V$) — vanishes with $V$, exactly like the storage term.  A
*concentrated / lumped* source — a finite total rate the element imposes
regardless of its shrinking size — **remains** as a finite jump term in the
matching balance: a heat-release element adds a total $\dot Q_P$ to the energy
balance, a fan or pump adds shaft work / a momentum source, a reactor adds a
species production rate.  So the jump-condition framework fully supports
in-element sources; mechanically they enter through the element's *donor*
(§6.3, the $\dot S^{\phi}_P$ term), which keeps the per-edge transport
equations and the equation count untouched.

### 3.3 Edge state closure

The flux average over a port is modeled by evaluating the flux formulas at a
single representative state per edge — the same closure used by
finite-volume CFD at a coarser scale.  Profile-shape corrections could be
added per element later without changing the structure.

---

## 4. Solution variables and state recovery

> **In plain terms:**  We must choose which three numbers per edge the
> solver works with.  The choice matters enormously for robustness.  We use:
> the mass flow rate $\dot m$ (signed: negative means "against the arrow"),
> the static pressure $p$ (what a pressure tap in the wall would read), and
> the total enthalpy $h_t$ (the stream's energy content, essentially its
> *total temperature* — what a stagnation probe would read).
>
> Given these three numbers and the port area, every other quantity —
> density, velocity, static temperature, Mach number, total pressure — can
> be computed.  The remarkable property of this particular triple is that
> this computation **always has exactly one answer**, found by a safe,
> fast iteration, whether the flow is slow, fast, supersonic, reversed, or
> exactly zero.  Alternative triples fail this test: working with total
> pressure and total temperature, for instance, gives *two* possible static
> states for the same inputs (a slow/subsonic one and a fast/supersonic
> one) and breaks down exactly at choking conditions — a trap the solver
> would constantly step into.

### 4.1 The variable set

Each edge carries three unknowns:

$$
\mathbf x_e = (\dot m_e,\; p_e,\; h_{t,e}).
$$

Motivations:

* **$\dot m$:** mass balances become *exactly linear* equations with
  coefficients $\pm 1$ — the best-behaved equations in the system.  Reversed
  flow is just a sign, with no special handling.
* **$p$:** static pressure is what boundary conditions and junction models
  naturally constrain, and (unlike $p_t$) it makes the state recovery
  single-valued (§4.3).
* **$h_t$:** total enthalpy is the quantity that is *transported* with the
  flow and conserved through every adiabatic element, so the energy
  equations become nearly linear.

### 4.2 State recovery: the implicit density equation

Given $(\dot m, p, h_t)$ and the area $A$, write the mass flux density
$m = \dot m/A$ (kg of gas crossing each m² per second).  Combining the
ideal-gas law, the definition of total enthalpy, and $u = m/\varrho$:

$$
p = \varrho R T,
\qquad
T = \frac{h_t - \tfrac12 u^2}{c_p},
\qquad
u = \frac{m}{\varrho}
\quad\Longrightarrow\quad
\boxed{\;
F(\varrho) \;=\; \varrho \;-\; \frac{p\Gamma}{H(\varrho)} \;=\; 0,
\qquad
H(\varrho) \;=\; h_t - \frac{m^2}{2\varrho^2} . \;}
$$

$H$ is the static enthalpy that a trial density $\varrho$ would imply (total
energy minus the kinetic part).  This is one equation in one unknown,
$\varrho$.

**Existence and uniqueness.**  The physically meaningful densities are
$\varrho > \varrho_{\min} = |m|/\sqrt{2h_t}$ (below that, the implied
velocity would exceed the total energy budget).  On this domain:

$$
F'(\varrho) \;=\; 1 + \frac{p\Gamma\,m^2}{\varrho^3 H^2} \;>\; 0,
\qquad
\lim_{\varrho\to\varrho_{\min}^+} F = -\infty,
\qquad
\lim_{\varrho\to\infty} F = +\infty .
$$

$F$ is strictly increasing and runs from $-\infty$ to $+\infty$, so it
crosses zero **exactly once**: the density exists and is unique for *every*
value of $\dot m$ — positive, negative, zero, subsonic or supersonic —
provided only $p > 0$ and $h_t > 0$.  Moreover, at the solution
$H = p\Gamma/\varrho > 0$, so the static temperature is automatically positive.
Since $F(p\Gamma/h_t) \le 0$, a safeguarded Newton iteration started at
$\varrho_0 = p\Gamma/h_t$ converges unconditionally.

**Worked example.**  Take air ($R = 287\,\mathrm{J/kg\,K}$, $\Gamma = 3.5$,
$\gamma = 1.4$), an edge of area $A = 0.1\,\mathrm{m^2}$ carrying $\dot m =
25\,\mathrm{kg/s}$ (so $m = 250\,\mathrm{kg/m^2 s}$), static pressure
$p = 1.0\times10^5\,\mathrm{Pa}$, and total enthalpy $h_t = 3.0\times10^5\,
\mathrm{J/kg}$.  The floor is $\varrho_{\min} = m/\sqrt{2h_t}\approx
0.32\,\mathrm{kg/m^3}$; Newton from $\varrho_0 = p\Gamma/h_t\approx 1.17$
converges in a few steps to the single root $\varrho\approx
1.13\,\mathrm{kg/m^3}$, giving $u = m/\varrho\approx 221\,\mathrm{m/s}$,
$T = (h_t-\tfrac12u^2)/c_p\approx 274\,\mathrm{K}$, and $M = u/\sqrt{\gamma RT}
\approx 0.67$.  There is no second admissible root.

**Why this is the variable choice that has exactly one answer.**  The
uniqueness above is a property of using the *static* pressure $p$ as the
unknown.  Had we instead carried *total* pressure $p_t$ — asking "which state
has this $\dot m$, $p_t$, $h_t$?" — the answer is generically *two* states, one
subsonic and one supersonic: the classic compressible-flow branch ambiguity,
since a given mass flux and stagnation condition is realized on both sides of
$M = 1$.  A solver that must *discover* the regime cannot be handed that fork at
the innermost state-recovery step.  Choosing $(\dot m, p, h_t)$ — static $p$,
not $p_t$ — collapses the fork to the single monotone root proved above, which
is precisely why these variables are used rather than the seemingly natural
$(p_t, T_t)$ (§4.3).

*(tests: `test_roundtrip` including supersonic/reversed/quiescent states,
`test_nonphysical_raises`)*

### 4.3 Why not total pressure and total temperature as unknowns?

With $(\dot m, p_t, T_t)$ — the variables of classical compressible-flow
tables — recovering the static state requires inverting the mass-flux
function

$$
\frac{\dot m\sqrt{T_t}}{A\,p_t} \;=\;
\sqrt{\tfrac{\gamma}{R}}\; M\left(1+\tfrac{\gamma-1}{2}M^2\right)^{-\frac{\gamma+1}{2(\gamma-1)}} .
$$

This function rises with Mach number $M$, peaks at $M = 1$, and falls again:
it is **not invertible**.  For a given mass flux there are two Mach numbers
(one subsonic, one supersonic), no solution at all above the peak, and an
infinite-slope inverse exactly at $M=1$.  Inside an iterative solver, whose
intermediate guesses roam freely, every one of these is a trap.  The
$(\dot m, p, h_t)$ recovery of §4.2 has none of them.

### 4.4 Derived state

With $\varrho$ recovered:

$$
u = \frac{m}{\varrho},\quad
T = \frac{H}{c_p},\quad
c = \sqrt{\gamma R T},\quad
M = \frac{u}{c}\ \text{(signed)},
$$

$$
T_t = \frac{h_t}{c_p},
\qquad
p_t = p\left(1+\tfrac{\gamma-1}{2}M^2\right)^{\Gamma},
\qquad
s_{\text{inv}} = \frac{p}{\varrho^\gamma}.
$$

For readers less used to these quantities: the **Mach number** $M$ compares
the flow speed with the speed of sound; compressibility effects grow with
$M^2$ and become dominant above $M \approx 0.3$.  The **total pressure**
$p_t$ is the pressure the stream would reach if brought to rest without
losses; the difference $p_t - p$ is the *dynamic head*, the "pressure
currency" the stream can spend.  Losses always show up as a drop in $p_t$,
never a gain — this is the second law of thermodynamics in network terms.
Note $p_t$ and $T_t$ depend on $M^2$: they are **independent of the flow
direction**, as they must be *(test: `test_signed_quantities`)*.

**Entropy lemma** (used for the isentropic elements): for a perfect gas

$$
s - s_{\text{ref}} \;=\; c_p \ln\frac{T}{T_{\text{ref}}} - R\ln\frac{p}{p_{\text{ref}}}
\;=\; c_p \ln\frac{T_t}{T_{\text{ref}}} - R\ln\frac{p_t}{p_{\text{ref}}},
$$

the second equality because bringing a stream to rest losslessly does not
change its entropy.  Hence **entropy is a function of $(p_t, T_t)$ alone**:
if total pressure and total temperature are continuous across an element,
entropy is automatically continuous too.  This is why no separate entropy
equation appears anywhere in the framework.

---

## 5. Equation structure: counting and assignment

> **In plain terms:**  With three unknowns per edge, a network with $E$
> edges needs exactly $3E$ equations — no more, no fewer.  The question is
> *who provides which equation*.  The classical answer, based on tracking
> which way information travels in the gas, has a fatal flaw: the number of
> equations a junction must provide *changes when the flow through one of
> its ports reverses*.  A solver cannot work with a system whose very shape
> flips while it iterates.
>
> The fix is a simple, fixed rule: **every element always provides exactly
> one equation per port** (one mass balance plus pressure-type relations),
> **and every edge provides one equation** saying where its temperature
> comes from (from whichever side is upstream — expressed smoothly, so
> "upstream" can change continuously during the iteration).  Count it up
> and the books always balance, no matter which way anything flows.

### 5.1 The information-flow count — and its flaw

Small disturbances in a subsonic gas stream travel along three families of
paths ("characteristics", §9): pressure waves running downstream at speed
$u + c$, pressure waves running upstream at $c - u$, and temperature/entropy
patterns simply carried with the flow at speed $u$.  Boundary-condition
theory says an element must supply exactly as many conditions as wave
families *leaving it* into its edges:

$$
n_{\text{eq}}(P) \;=\; \underbrace{n}_{\text{one outgoing pressure wave per port}}
\;+\; \underbrace{n_{\text{out}}(P)}_{\text{one carried wave per \emph{outflow} port}} .
$$

Summed over the network: $\sum_P n_{\text{eq}} = 2E + E = 3E$ (each edge is
the outflow of exactly one of its two end elements) — globally correct.
Familiar special cases: a through-flow component supplies $2+1=3$ jump
conditions (mass, energy, entropy/momentum — the classical triple); an
inlet 2; an outlet 1.

The flaw: $n_{\text{out}}(P)$ **changes discretely when a port reverses**.
A three-port junction owes 5 equations with one inflow and two outflows,
but only 4 with two inflows and one outflow.  Any solver iterating across a
flow reversal would see the system change size mid-flight — a structural
discontinuity that no amount of numerical care can absorb.

### 5.2 The fixed split

The direction-dependent conditions are exactly the *carried* (convective)
ones.  The resolution: move them from the elements onto the **edges**, in a
smoothly upwinded form (§6).  The bookkeeping becomes:

| owner   | count per owner | content |
|---------|-----------------|---------|
| element, interior, $n$ ports | $n$ | 1 mass balance + $(n-1)$ pressure-type relations (total-pressure equality, momentum balance, static-pressure equality, loss correlation, …) |
| element, boundary (1 port)   | $1$ | one specification ($\dot m$, $p$, or $p_t$) |
| edge    | $1$ | smoothly upwinded total-enthalpy transport (§6) |

Squareness is now *unconditional*:

$$
\sum_P n_{\text{ports}}(P) + E \;=\; 2E + E \;=\; 3E
\;=\; \text{number of unknowns},
$$

**independent of the flow directions**.  Nothing of the physics is lost
relative to §5.1 — each edge still effectively receives two pressure-wave
conditions (one from each end) and one carried condition (from its upstream
side); the carried condition simply lives on the edge and *selects* its
upstream side smoothly rather than being reassigned discretely.

The network class enforces the one-equation-per-port rule at assembly time.

---

## 6. Total-enthalpy transport on edges

> **In plain terms:**  Temperature is not negotiated between components the
> way pressure is — it is *carried* by the gas.  The gas in a pipe is as hot
> as wherever it came from.  So each edge gets one equation that says
> exactly that: "my total temperature equals that of my upstream neighbor."
> Two refinements make this solver-friendly:  (1) what counts as "upstream"
> is expressed by a smooth weighting instead of an if/else, so the equation
> stays valid while the solver is still discovering the flow direction;
> (2) when several streams meet in a junction, the donated temperature is
> the mass-weighted mixture of the incoming streams — which is just energy
> conservation rearranged.

### 6.1 Donor enthalpy of an element

Steady energy conservation at element $P$ in flux form reads
$\sum_e \sigma_{P,e}\dot m_e h_{t,e} = 0$.  Combined with mass conservation
this is equivalent (for nonzero flow) to: *every outflow port carries the
mass-weighted mix of the inflow enthalpies*:

$$
H_P \;=\;
\frac{\sum_e \dot m^{\text{in}}_{P,e}\,h_{t,e}}
     {\sum_e \dot m^{\text{in}}_{P,e}}
\qquad\text{(sum over inflow ports)}.
$$

**Derivation, and the "same enthalpy at every outflow" reading.**  This is the
steady energy balance rearranged — and yes, it does say that **all outflow
ports of one element carry the same total enthalpy**.  That is the lumped
*perfect-mixing* assumption: the element has no resolved interior, so whatever
mixed stream forms inside it leaves at one common value $h_t = H_P$ through
every outlet.  To see it, split the balance $\sum_e\sigma_{P,e}\dot m_e
h_{t,e}=0$ into inflow ports (where $h_{t,e}$ is the incoming gas's, set by the
upstream neighbour) and outflow ports (where the gas leaves carrying the
element's donated $H_P$):

$$
\underbrace{\sum_{\text{in}}\dot m^{\text{in}}_{P,e}\,h_{t,e}}_{\text{energy in}}
\;=\;
\underbrace{\Big(\textstyle\sum_{\text{out}}\dot m^{\text{out}}_{P,e}\Big)\,H_P}_{\text{energy out}} ,
$$

and since mass conservation makes the two rate sums equal, $H_P$ is exactly the
inflow-mass-weighted average above.  The per-edge transport equation (§6.2)
then *delivers* that common $H_P$ onto each outflow edge.

**Worked example (a tee).**  Two streams mix and leave through one outlet:
inflows $\dot m_1 = 3\,\mathrm{kg/s}$ with $T_{t,1} = 300\,\mathrm K$ and
$\dot m_2 = 1\,\mathrm{kg/s}$ with $T_{t,2} = 500\,\mathrm K$ (so $h_{t,i} =
c_pT_{t,i}$).  The donor is

$$
H_P = \frac{3\,(c_p\cdot300) + 1\,(c_p\cdot500)}{3+1} = c_p\cdot350\,\mathrm K,
$$

i.e. $T_{t,\text{out}} = 350\,\mathrm K$ — pulled toward the larger stream, as
mixing demands.  If the same element had *two* outlets, both would leave at
$T_t = 350\,\mathrm K$; how the mass *splits* between them is set by the
pressure relations (mass/momentum), not by energy — the thermodynamic state of
the mixed stream is single-valued.

The smooth, regularized form used in the prototype replaces the inflow
rates by smooth positive parts (Appendix A):

$$
\boxed{\;
H_P \;=\;
\frac{\sum_e w_e\, h_{t,e}}{\sum_e w_e},
\qquad
w_e = \operatorname{spos}\!\big(\dot m^{\text{in}}_{P,e};\,\varepsilon\big).
\;}
$$

Because $w_e \ge \varepsilon/2 > 0$, the mixture is well defined for any
flow state (at complete stagnation it degrades gracefully to a plain
average), while at converged flow the weight of an *outflow* port decays
like $\varepsilon^2/(4|\dot m_e|)$ — a quadratically small contamination.

Boundary elements override the donor with their specification: inlets offer
$c_p T_t^{\text{spec}}$; pressure outlets offer $c_p T_t^{\text{backflow}}$,
the temperature a *re-entering* stream would carry (the same convention CFD
codes use at pressure outlets).

### 6.2 The edge transport equation

Each edge contributes one equation:

$$
\boxed{\;
h_{t,e} \;=\; \theta(\dot m_e)\, H_{\text{tail}(e)}
\;+\; \big(1-\theta(\dot m_e)\big)\, H_{\text{head}(e)},
\qquad
\theta(x) = \operatorname{sstep}(x;\varepsilon),
\;}
$$

where $\theta$ is a smoothed 0-to-1 step (Appendix A): for clearly positive
$\dot m_e$ the edge takes its enthalpy from the tail element (the gas comes
from there), for clearly negative from the head, and near zero it blends —
keeping the equation perfectly well-behaved at the one state where physics
genuinely cannot decide.

Three properties carry the whole design:

1. **No degeneracy.**  The equation always has a coefficient of 1 on
   $h_{t,e}$ itself; it cannot collapse onto another equation when the flow
   tends to zero (contrast §8.1).
2. **Smoothness.**  No kinks anywhere; the solver can iterate *through* a
   flow reversal.
3. **Asymptotic exactness.**  At a converged state with
   $|\dot m_e| \gg \varepsilon$ the smooth weights differ from the exact
   upwind values only by $O\!\big((\varepsilon/\dot m)^2\big)$; with the
   default $\varepsilon = 10^{-4}\,\dot m_{\text{ref}}$ the imprint on the
   solution is at the $10^{-8}$ relative level — far below any engineering
   tolerance.  *(tests: `conservation_ok` checks,
   `test_energy_conservation_with_mixed_temperatures`)*

No separate entropy transport is needed: entropy is determined by
$(p_t, T_t)$ (the lemma of §4.4), which are already covered by the pressure
relations and the enthalpy transport.

### 6.3 Generalization: any carried scalar — and why mass is not one

> **In plain terms:**  Total enthalpy is the first member of a family.
> *Anything the gas simply carries along with it* — and later, for reacting
> or multi-species flow, that will include species mass fractions, a mixture
> fraction, a progress variable, a passive tracer — gets exactly the same
> treatment: a donor at each element (mass-weighted mixture of what flows
> in, optionally with a source) and one smoothly-upwinded transport equation
> per edge.  Mass itself is the one thing that does *not* get this treatment,
> for a precise reason: mass flux is not a carried passenger, it is the
> vehicle.

**The pattern.**  Total enthalpy is treated this way because it is an
*intensive scalar passively advected with the flow*: its value on an edge is
inherited from whichever element is upstream, and at a junction the outflow
inherits the mass-weighted mixture of the inflows. Nothing in §6.1–6.2 used
a property of enthalpy beyond *being such a scalar*. So for any carried
scalar $\phi$ (per-unit-mass: a species mass fraction $Y_k$, a mixture
fraction $Z$, a reaction progress variable, a passive tracer, …) the
construction is identical:

$$
\boxed{\;
\Phi_P \;=\;
\frac{\sum_e w_e\,\phi_e \;+\; \dot S^{\phi}_P}{\sum_e w_e},
\qquad
\phi_e \;=\; \theta(\dot m_e)\,\Phi_{\text{tail}(e)}
\;+\;\big(1-\theta(\dot m_e)\big)\,\Phi_{\text{head}(e)},
\;}
$$

with $w_e=\operatorname{spos}(\dot m^{\text{in}}_{P,e};\varepsilon)$ and
$\theta=\operatorname{sstep}(\cdot;\varepsilon)$ exactly as before. Each new
scalar adds **one unknown per edge and one transport equation per edge**, so
squareness ($\,$unknowns $=$ equations$\,$) is preserved automatically — the
$3E$ count of §5.2 simply becomes $(2+n_\phi)E$ for $n_\phi$ carried scalars
beyond mass. Total enthalpy is the $n_\phi=1$ instance with $\phi=h_t$ and
$\Phi=H$; the entropy lemma (§4.4) is what keeps $n_\phi$ at 1 for a
single-component gas (entropy is *not* an independent carried scalar — it is
fixed by $p_t,T_t$). Reacting/multi-species flow is exactly where $n_\phi$
grows.

**Source terms drop in through the donor, not the edge.**  The $\dot
S^{\phi}_P$ above is the general element source of $\phi$ (a heat input for
$\phi=h_t$: $\dot S = \dot Q_P$; a production/consumption rate for a species;
zero for a passive tracer). It enters *only* the donor — the mass-weighted
mixture becomes "what flowed in, **plus** what the element added, spread over
the throughflow mass" — while the per-edge transport equation is untouched.
This is the same hook by which boundary elements override the donor (§6.1):
the element owns its donor, so sources, specifications, and overrides all
live in one place and never change the equation count. *Caveat:* a fixed
source with vanishing net throughflow ($\sum_e w_e\to$ its
regularization floor) has no bounded steady donor — $\dot S/\sum w$ grows
large; the $\operatorname{spos}$ floor keeps it finite (not a NaN) but
$\varepsilon$-dependent. That regime is genuinely ill-posed physics
(steady heating of a stagnant junction), not a solver artifact.

**Why mass is excluded.**  Mass conservation deliberately stays a nodal
balance (§7.1) and must *not* be recast as an edge donor/transport pair,
for two independent reasons:

1. *Nothing to carry.*  The donor/upwind device transports an **intensive,
   per-unit-mass scalar** whose edge value is inherited from upstream. Mass
   flux $\dot m_e$ is the **extensive flux itself** — it *is* the edge
   unknown, the carrier rather than the carried. There is no separate
   "amount of mass per unit mass" to upwind; the analogue of $\phi$ for mass
   is the constant $1$, whose transport equation is vacuous.
2. *No defect to fix.*  The edge split exists to remove the
   direction-dependent equation *count* of the carried conditions (§5.1).
   The mass balance is already direction-invariant in count — one per
   element, with $\sigma_{P,e}\dot m_e$ unchanged under arrow flips (§7.1) —
   so there is no structural discontinuity to absorb. Recasting it would
   only forfeit exact global conservation and risk degeneracy at
   $\dot m\to0$, the very pathology §6.2 was built to avoid.

So the clean statement of the framework is: **mass is conserved nodally; every
intensive carried scalar — total enthalpy today, species/progress variables
tomorrow — is transported on edges by the §6.1–6.2 donor/upwind pair.**

---

## 7. Element formulations

> **In plain terms:**  This section is the component library.  Every
> interior element contributes: (1) "mass in = mass out", and (2) one or
> more *pressure relations* that encode its geometry — a smooth nozzle
> conserves total pressure; a sudden expansion destroys a calculable amount
> of it (a momentum balance tells how much); an orifice/valve destroys an
> amount set by its loss coefficient; a junction ties its ports to a common
> pressure.  Boundary elements pin one quantity each (a flow rate or a
> pressure).  All formulas are written so that they remain valid and smooth
> whichever way the gas flows.

Conventions: `ports` are in connection order; residuals must be smooth in
the unknowns (Appendix A) and complex-analytic (§10.2).  Every interior
pressure-type row also carries the *vanishing-friction* term
$-\,\kappa\,\dot m^{\text{out}}_{\text{port}}$ (§7.7, §10.4) which is
exactly zero in the final solver stage; it is omitted below for clarity.

### 7.1 Mass balance (all interior elements, row 1)

$$
R_{\text{mass}} \;=\; \sum_{e}\sigma_{P,e}\,\dot m_e \;=\; 0 .
$$

Exact, linear, and direction-convention safe: flipping an edge's arrow flips
both $\sigma_{P,e}$ and the sign of $\dot m_e$ at the solution, so the
physical outflow $\sigma_{P,e}\dot m_e$ is unchanged.

### 7.2 Isentropic area change (2 ports)

A smooth, internally monotone contraction or diffuser: lossless — *until
its small port chokes*.  Equations:

$$
R_1 = \sigma_0\dot m_0 + \sigma_1\dot m_1,
\qquad
R_2 = \varphi_\epsilon\Big(1 - M^{\text{in}}_{\text{small}},\;
\tfrac{p_{t,\text{small}} - p_{t,\text{large}}}{p_{t,\text{small}}}\Big)\,
p_{t,\text{small}},
$$

where $\varphi_\epsilon$ is the smoothed Fischer–Burmeister complementarity
(Appendix A, §11.2): while the small port is subsonic — either flow
direction — the row reduces to total-pressure equality, the classical
isentropic element; when the small port reaches exactly $M = 1$ in the
diverging direction, the element is choked and a total-pressure drop (the
lumped internal normal shock) becomes admissible.  Energy continuity
($h_{t,0}=h_{t,1}$) is delivered by the edge transport equations through the
element's donor (§6), so it is not duplicated here.  Entropy continuity in
the lossless regime then follows automatically from the entropy lemma:
continuous $p_t$ and $T_t$ imply continuous $s$.  So this two-equation
element, together with the edge transport, reproduces the classical
isentropic jump triple in subsonic operation — valid for either flow
direction, regular at $\dot m = 0$ — while the choked and shocked regimes
emerge from the same row (§11).

*(tests: `test_single_iac_vs_exact` against the analytic compressible-flow
relations; `test_multi_iac_chain_from_quiescent`)*

### 7.3 Sudden area change (2 ports)

**Expansion (flow from the small pipe into the large one) — the
Borda–Carnot analysis.**  When a jet exits a small pipe into a larger one,
it cannot follow the abrupt corner; it separates, and mixes back out to the
full area further downstream with turbulent losses.  Remarkably, the
*amount* of loss needs no empirical constant — a momentum balance determines
it, because the separated "dead-water" corner region holds the small-pipe
static pressure $p_s$ against the annular back wall.  Steady momentum on
the control volume between the small section ($A_s$) and the large section
($A_l$):

$$
\underbrace{\dot m u_l - \dot m u_s}_{\text{momentum change}}
\;=\;
\underbrace{p_s A_s}_{\text{inlet}} + \underbrace{p_s (A_l - A_s)}_{\text{back wall}}
- \underbrace{p_l A_l}_{\text{outlet}}
\quad\Longrightarrow\quad
\boxed{\;\dot m\,(u_l - u_s) + A_l\,(p_l - p_s) \;=\; 0\;}
$$

together with mass and $h_t$ continuity.  (In the low-speed limit this
yields the familiar loss $\Delta p_t = \tfrac12\varrho(u_s - u_l)^2$.)  Note
that the *static* pressure actually rises through a sudden expansion, while
the *total* pressure drops — the entropy production comes out of the
momentum balance, it is not put in by hand.
*(test: `test_sudden_expansion_borda_carnot` verifies the balance and the
entropy rise.)*

**Contraction (flow from large to small).**  The same algebra applied to a
contraction would predict an entropy *decrease* — physically impossible.  A
real sudden contraction is nearly loss-free up to the vena contracta; the
prototype models it as exactly loss-free ($p_{t,0} = p_{t,1}$).  An
empirical contraction-loss correlation can be substituted without
structural change.  *(test: `test_sudden_contraction_is_lossless`)*

**Direction-invariant assembly.**  Written along the port-0 → port-1 axis:

$$
R_{\text{mom}} \;=\;
\big(\dot m u + pA\big)_1 - \big(\dot m u + pA\big)_0
- p_{\text{small}}\,(A_1 - A_0).
$$

A subtlety worth recording: the convective momentum flux
$\dot m u = \dot m^2/(\varrho A)$ is **even** under an edge-arrow flip (both
factors change sign together), so no $\sigma$ may multiply it.  Writing the
momentum balance as $\sum\sigma(\dot m u + pA)$ — by analogy with the mass
and energy balances — silently breaks the arrow-independence of the
element.  (This exact mistake occurred transiently during development and
was caught by `test_edge_direction_flip_invariance`; the test exists
because the mistake is so easy to make.)

**Smooth regime blend.**  With
$\xi = \operatorname{sstep}\!\big(\dot m^{\text{in}}_{\text{small port}};\varepsilon\big)$
(→ 1 when the flow enters through the small port, i.e. expansion):

$$
R_2 \;=\; \xi\,\frac{-R_{\text{mom}}}{A_l} \;+\; (1-\xi)\,\big(p_{t,0}-p_{t,1}\big).
$$

The momentum residual is scaled to pressure units and sign-normalized so
that near $\dot m = 0$ its pressure content is $(p_0 - p_1)$, the same as
the isentropic branch — otherwise the two halves of the blend could cancel
each other near zero flow and leave the element without an effective
equation there.

### 7.4 Concentrated loss element (2 ports)

A valve, orifice, filter, or any device characterized by a loss coefficient
$K_L$ referenced to a dynamic head:

$$
R_2 \;=\; p_{t,0} - p_{t,1} - K_L\, q_{\text{signed}},
\qquad
q_{\text{signed}} = \tfrac12\,\bar\varrho\, u_{\text{ref}}\,
\sqrt{u_{\text{ref}}^2 + u_\varepsilon^2},
$$

with $u_{\text{ref}} = \dot m_0/(\bar\varrho A_0)$ and $\bar\varrho$ the
port-average density.  $q_{\text{signed}}$ is a smooth version of
$\tfrac12\bar\varrho\,u|u|$, so the loss always *opposes* the flow,
whichever way it runs — the second law holds in both directions.

### 7.5 Junctions and splitters ($n$ ports) — and how to choose between them

Both supply one mass balance plus $(n-1)$ pressure couplings against
port 0:

$$
\text{static-pressure junction:}\quad R_{1+i} = p_0 - p_i ,
\qquad
\text{lossless splitter:}\quad R_{1+i} = p_{t,0} - p_{t,i},
\qquad i = 1,\dots,n-1 .
$$

* **`JunctionStaticP`** ties all ports to a common *static* pressure — the
  classical "header" or "manifold node" simplification, appropriate when
  all port velocities are low (the kinetic terms it ignores are then
  negligible).  Enthalpy mixing of multiple inflows is automatic through
  the donor mechanism of §6 *(test:
  `test_energy_conservation_with_mixed_temperatures`)*.
* **`LosslessSplitter`** ties all ports to a common *total* pressure — an
  isentropic distribution plenum.  With $h_t$ delivered by the edge
  transport and $p_t$ common, entropy is continuous into every outflow
  branch (the lemma of §4.4): this is exactly the classical "lossless
  splitter" (mass + energy + constant entropy).

**Selection rule (important).**  Use the static-pressure junction **only
where every port runs at low Mach number**.  At a fast port, equal static
pressure plus the port's velocity head means the junction hands the branch
$p_t \approx p + \tfrac12\varrho u^2$ — *more total pressure than the feed
possesses*.  That is free energy (a second-law violation), and its
consequences are not merely cosmetic: the surplus must be destroyed
somewhere downstream, and if no element can do so, **the network has no
steady solution at all** and the solver can only stall.  This was observed
directly while constructing `examples/ex4_manifold.py`: a static-pressure
manifold feeding a small high-speed port made the system unsolvable, while
the physically appropriate `LosslessSplitter` converged immediately.  Rule
of thumb: plenum feeding fast branches → splitter ($p_t$); low-speed header
collecting/merging comparable streams → static-$p$ junction.

A momentum-conserving junction (for ports in line, as in your original
element list) is the $n$-port generalization of §7.3 and follows the same
pattern: $n$ equations, smooth blends, no $\sigma$ on convective fluxes.

### 7.6 Boundary elements (1 port)

Boundary elements terminate one edge and supply exactly one equation; their
donor enthalpy becomes active only if the flow actually enters the network
there.

**Mass-flow inlet** ($\dot m^{\text{spec}}, T_t^{\text{spec}}$):

$$
R = \dot m^{\text{out}}_{P,e} - \dot m^{\text{spec}},
\qquad
H_P = c_p T_t^{\text{spec}} .
$$

**Total-pressure inlet (reservoir)** ($p_t^{\text{spec}}, T_t^{\text{spec}}$),
with the blend weight
$\xi = \operatorname{sstep}(\dot m^{\text{out}}_{P,e};\varepsilon)$:

$$
R = \xi\,\big(p_t - p_t^{\text{spec}}\big) + (1-\xi)\,\big(p - p_t^{\text{spec}}\big).
$$

Why the blend is *necessary* and not a convenience: drawing gas from a
reservoir is a lossless acceleration, so the stream's total pressure equals
the reservoir pressure ($\xi = 1$ branch).  But if the network turns around
and discharges *into* the reservoir, demanding $p_t = p_t^{\text{spec}}$ is
impossible — an arriving stream with surplus total pressure cannot shed it
losslessly, and **no steady solution would exist**.  Physically, the jet
dumps its velocity head into the reservoir by turbulent mixing *outside*
the network, and the correct condition is on the *static* pressure
($\xi = 0$ branch).  The smooth blend lets the solver pass continuously
between the regimes.

**Pressure outlet** ($p^{\text{spec}}, T_t^{\text{backflow}}$), with
$\xi = \operatorname{sstep}(\dot m^{\text{in}}_{P,e};\varepsilon)$:

$$
R = \xi\;\varphi_\epsilon\Big(1 - M^{\text{in}},\;
\tfrac{p - p^{\text{spec}}}{p^{\text{spec}}}\Big)\,p^{\text{spec}}
\;+\; (1-\xi)\,\big(p_t - p^{\text{spec}}\big),
\qquad
H_P = c_p T_t^{\text{backflow}} .
$$

Discharging subsonically: static pressure matched (the complementarity
reduces to $p = p^{\text{spec}}$).  Discharging at the choking limit: the
exit pins at $M = 1$ and the exit pressure detaches *upward* from the
specification — the underexpanded choked-orifice discharge (§11.2).
Backflow: the specification acts as the total pressure of the returning
stream, which carries the prescribed backflow temperature — the same
convention as CFD pressure outlets.
*(test: `test_reversed_flow_through_boundaries` converges to the exact
analytic reversed solution; example: `ex3_reverse_flow.py`)*

### 7.7 The stabilization (vanishing-friction) term

During the early solver stages only, every interior pressure-type row
carries an extra linear resistance

$$
R_{1+i} \;\mathrel{-}= \; \kappa\, \dot m^{\text{out}}_{P,\,\text{port } i},
\qquad
\kappa = \text{stab}\cdot \frac{p_{\text{ref}}}{\dot m_{\text{ref}}},
$$

i.e. a small fictitious friction between port 0 and port $i$, with the sign
the second law dictates.  Why it is *necessary* is §8.3; why it is
*harmless* is that the final solver stage sets $\text{stab} = 0$, so the
equations actually satisfied at convergence are the exact ones
*(test: `test_final_stage_solves_exact_equations`)*.

---

## 8. Why naive formulations fail

> **In plain terms:**  The earlier prototypes wrote physically correct
> equations and still could not be made robust.  This section explains the
> three reasons — they are worth understanding because each one is a
> *structural* property of compressible network equations, not a tuning
> problem, and each one dictated a specific design feature above.  The key
> concept needed here is the **Jacobian**: the sensitivity matrix that
> tells Newton's method how each equation responds to each unknown.  When
> two equations respond *identically* to all unknowns (the matrix becomes
> "singular"), the solver receives contradictory or empty guidance and
> stalls — no step size or damping can fix missing information.

### 8.1 Root cause A: flux-form energy equations degenerate at zero flow

Write the energy balance of a two-port element in flux form,
$R_E = \sum_i \sigma_i \dot m_i h_{t,i}$, and differentiate:

$$
\frac{\partial R_E}{\partial x}
= \sum_i h_{t,i}\,\frac{\partial(\sigma_i\dot m_i)}{\partial x}
\;+\; \sum_i \sigma_i\dot m_i\,\frac{\partial h_{t,i}}{\partial x}.
$$

As $\dot m_i \to 0$ the second sum vanishes, and if the enthalpy field is
uniform ($h_{t,i} = h_t$, the natural cold start) the energy row becomes
**exactly $h_t$ times the mass row** — the Jacobian is singular, no
information about temperature remains.  This is physical: at zero flow,
"energy carried in = energy carried out" is satisfied by *any* temperature
field; the equation genuinely says nothing.  The same happens in
characteristic variables (it must — see §9.3): the $M\to0$ limits of the
characteristic energy-flux coefficients are
$\pm\varrho c^2/(\gamma-1) = \pm \varrho h$, i.e. $h$ times the mass-flux
coefficients.  Every chain of isentropic elements started from rest sits on
this singularity — which is precisely why the earlier prototypes failed on
multi-element chains.  The structural fix is the *transport* form of §6,
whose equations never lose their diagonal.
*(test: `test_multi_iac_chain_from_quiescent` — solved from exactly zero
flow)*

### 8.2 Root cause B: hard switches

The `00/` code multiplied entropy rows by $\operatorname{sign}(u)$: the row
vanishes identically at $u = 0$ (singular exactly where the cold start
lives) and jumps across reversal (Newton's method assumes smoothness; a
jump stops it cold).  And at the structural level, the classical equation
assignment itself switches its count at reversal (§5.1).  Both disappear
under the fixed split (§5.2) and smooth upwinding (§6).

### 8.3 Root cause C: zero flow is a stationary point of pressure-driven networks

All pressure-type physics couples to the flow through the dynamic head:

$$
p_t - p \;=\; \frac{\gamma}{2}\,p\,M^2 + O(M^4),
\qquad
M^2 \propto \dot m^2 .
$$

The sensitivity $\partial(p_t - p)/\partial\dot m \propto \dot m$ vanishes
at $\dot m = 0$.  In a network driven *only* by pressure boundary conditions
(no mass-flow specification anywhere), the residuals therefore have **zero
first-order sensitivity to all the flow unknowns at the quiescent state**:
Newton's method, which works entirely off first-order sensitivities, sees a
flat landscape and no reason to move the flows.  Zero flow is a stationary
point — descent-type methods stall on it or near it (observed directly
during development).

The cure must inject first-order flow sensitivity *without changing the
final answer*: the vanishing-friction homotopy (§7.7, §10.4).  With the
fictitious friction active, the pressure relations behave like a
pipe-resistance network — pressure differences then *do* push directly on
the flows, the way voltage pushes current through a resistor — and the
solver finds the correct flow pattern easily.  The friction is then reduced
to exactly zero in stages, each starting from the previous solution.

A related, milder case: in a perfectly symmetric branching network at rest,
the *split* between the branches is undetermined at first order (any split
satisfies the loop equation when nothing flows), so the Jacobian is
singular there even with a mass-flow inlet.  The solver's
Levenberg–Marquardt damping (§10.3) handles this automatically.
*(test: `test_quiescent_symmetric_branching_start`)*

---

## 9. Characteristic variables

> **In plain terms:**  Disturbances travel through a flowing gas in three
> ways: as pressure (sound) waves running downstream, as sound waves
> running upstream, and as temperature/composition patterns simply carried
> along with the gas.  The "characteristic variables" $f$, $g$, $h$ are the
> bookkeeping quantities for these three carriers.  They are the natural
> language of duct acoustics and thermoacoustics — and the original
> motivation of this project was the hope that solving the *mean flow* in
> this language would converge better.  This section derives the exact
> translation between the wave language and the network unknowns, and then
> proves the central negative result: the translation is just a change of
> coordinates, and Newton's method is *blind to changes of coordinates* —
> the iterates come out identical.  The waves matter enormously, but for
> the acoustic stage (§12), not for mean-flow convergence.

### 9.1 Decomposition of the 1-D Euler system

In primitive variables $\mathbf q = (\varrho, u, p)$ the 1-D Euler equations
read $\partial_t \mathbf q + \mathbf A_q\,\partial_x \mathbf q = 0$ with

$$
\mathbf A_q =
\begin{bmatrix}
u & \varrho & 0\\
0 & u & 1/\varrho\\
0 & \varrho c^2 & u
\end{bmatrix},
\qquad
\text{eigenvalues } \lambda \in \{\,u+c,\; u-c,\; u\,\}.
$$

The eigenvalues are the three propagation speeds.  The characteristic
perturbation amplitudes, in this project's convention:

$$
\begin{aligned}
f &= \tfrac12\Big(u' + \tfrac{p'}{\varrho c}\Big) &&\text{downstream acoustic wave } (\lambda = u+c),\\[2pt]
g &= \tfrac12\Big(-u' + \tfrac{p'}{\varrho c}\Big) &&\text{upstream acoustic wave } (\lambda = u-c),\\[2pt]
h &= \varrho' - \tfrac{p'}{c^2} &&\text{entropy (convected) wave } (\lambda = u),
\end{aligned}
\qquad\Longleftrightarrow\qquad
\begin{aligned}
u' &= f - g,\\
p' &= \varrho c\,(f+g),\\
\varrho' &= h + p'/c^2 .
\end{aligned}
$$

### 9.2 Exact maps to the network variables

The network perturbations are
$\delta\mathbf x = (\delta\dot m, \delta p, \delta h_t)$.  From
$\dot m = \varrho u A$ and $h_t = \Gamma p/\varrho + u^2/2$:

$$
\delta \dot m = A\,(u\,\delta\varrho + \varrho\,\delta u),
\qquad
\delta h_t = -\frac{\Gamma p}{\varrho^2}\,\delta\varrho + u\,\delta u + \frac{\Gamma}{\varrho}\,\delta p .
$$

Composing with §9.1 gives the per-edge map
$\delta\mathbf x = \mathbf T\,\mathbf w$, $\mathbf w = (f,g,h)$, with the
remarkably clean closed form (verified to machine precision against the
implementation):

| per unit amplitude of: | $f$ | $g$ | $h$ |
|---|---|---|---|
| $\delta\dot m$ | $A\varrho\,(M+1)$ | $A\varrho\,(M-1)$ | $A u$ |
| $\delta p$     | $\varrho c$       | $\varrho c$       | $0$   |
| $\delta h_t$   | $c + u$           | $c - u$           | $-h/\varrho$ |

($h = c_pT$ in the last column).  $\mathbf T$ is invertible at every
recoverable state — including $M = 0$, reversed and supersonic flow:

$$
\det \mathbf T \;=\; -\,2\,A\,\varrho\,c\,\big(u^2 + h\big) \;<\; 0
\qquad\text{whenever } p, T > 0 .
$$

The inverse map and the primitive-variable versions are provided in
`fns/characteristics.py` *(tests: `test_char_maps_are_inverses`,
`test_char_to_dx_against_complex_step`)*.

### 9.3 The Newton-invariance theorem

**Claim.**  Solving the Newton system in characteristic amplitudes and
mapping back yields the identical update as solving directly in the network
variables.

**Proof.**  Let $\mathbf J$ be the Jacobian of the residuals at the current
iterate and $\mathbf T$ the (invertible, block-diagonal per edge) map
$\delta\mathbf x = \mathbf T\mathbf w$.  The Newton system posed in wave
amplitudes is $(\mathbf J\mathbf T)\,\mathbf w = -\mathbf R$, whence

$$
\delta\mathbf x = \mathbf T\,(\mathbf J\mathbf T)^{-1}(-\mathbf R)
= \mathbf T\,\mathbf T^{-1}\mathbf J^{-1}(-\mathbf R)
= -\mathbf J^{-1}\mathbf R . \qquad\blacksquare
$$

The update is the same.  A change of unknowns is a similarity
transformation of the linear system, and Newton's method is invariant under
it.  **Casting the mean-flow correction in characteristic variables
therefore cannot, by itself, improve convergence** — the original project
hypothesis fails at this fence, in two lines.  What can differ is
floating-point conditioning and the choice of relaxation metric —
second-order effects.  The genuine robustness levers are the equation
structure (§5–6), scaling, and globalization (§10).
*(test: `test_newton_in_characteristics_equals_newton_in_primitives`)*

### 9.4 What characteristics are actually for

1. **Counting and well-posedness** of boundary and jump conditions (§5.1) —
   the fixed split of §5.2 is its smooth realization.
2. **Diagnostics:** any Newton update or residual can be read per edge as
   wave amplitudes; at the converged mean state the residual-driven wave
   amplitudes vanish identically *(test:
   `test_characteristic_amplitudes_of_converged_solution`)*.
3. **The acoustic network** (§12): linearizing the very same element
   equations at the converged mean state and transforming with $\mathbf T$
   yields the acoustic scattering relations — the consistency goal of the
   whole framework, fully preserved.

---

## 10. Numerical solution

> **In plain terms:**  The network equations form a system of $3E$
> nonlinear equations in $3E$ unknowns.  The standard tool is **Newton's
> method**: at the current guess, compute the *residuals* (how far each
> equation is from being satisfied) and the *Jacobian* (how each residual
> responds to each unknown), then solve a linear system for the correction
> that would zero the residuals if the world were linear.  Repeat; near the
> answer this converges extremely fast (the error roughly squares each
> iteration).
>
> Plain Newton, however, only works near the answer.  Four safeguards make
> it work from a cold start on an arbitrary network: (1) put all equations
> and unknowns on comparable scales, so "small" means the same thing
> everywhere; (2) compute the Jacobian exactly — by a trick that costs
> nothing in accuracy — so the search directions are never corrupted;
> (3) damp the steps in the Levenberg–Marquardt manner, which blends
> Newton's fast steps with cautious downhill steps and handles the
> momentarily singular states of §8; (4) sneak up on the exact problem
> through a short sequence of slightly-frictioned problems (the homotopy),
> which removes the zero-flow trap and settles all flow directions before
> the exact equations are solved.

### 10.1 Nondimensionalization

Variables and residuals are scaled by reference quantities:

$$
\hat{\dot m} = \frac{\dot m}{\dot m_{\text{ref}}},\quad
\hat p = \frac{p}{p_{\text{ref}}},\quad
\hat h_t = \frac{h_t}{c_p T_{\text{ref}}},
\qquad
\hat R_{\text{mass}} = \frac{R_{\text{mass}}}{\dot m_{\text{ref}}},\quad
\hat R_{\text{press}} = \frac{R}{p_{\text{ref}}},\quad
\hat R_{\text{enth}} = \frac{R}{c_p T_{\text{ref}}} .
$$

Without this, equation magnitudes span $\sim 1$ (mass, kg/s) to
$\sim 3\times10^5$ (enthalpy, J/kg): a convergence tolerance is meaningless
(the mass errors drown) and the linear algebra is needlessly
ill-conditioned.  This alone accounted for much of the earlier prototypes'
crawling behavior.

### 10.2 Exact Jacobians by complex-step differentiation

For any analytically programmed residual $R(x)$, evaluating it with a tiny
*imaginary* perturbation gives the derivative without subtracting nearly
equal numbers:

$$
R(x + i\eta) = R(x) + i\eta R'(x) + O(\eta^2)
\quad\Longrightarrow\quad
R'(x) = \frac{\operatorname{Im} R(x+i\eta)}{\eta} + O(\eta^2),
$$

and with $\eta = 10^{-30}$ the result is exact to machine precision — unlike
finite differences, there is no cancellation error to balance against
truncation error.  Two requirements, both designed in:

* every residual is built from smooth, analytic operations (Appendix A
  replaces `abs/sign/max`);
* the implicit density solve (§4.2) propagates imaginary parts through the
  implicit function theorem: solve $F(\varrho; m, p, h_t) = 0$ in reals,
  then attach

$$
\operatorname{Im}\varrho
= -\frac{F_m \operatorname{Im} m + F_p \operatorname{Im} p + F_{h_t}\operatorname{Im} h_t}{F_\varrho},
\qquad
\begin{aligned}
F_\varrho &= 1 + p\Gammam^2/(\varrho^3H^2),\\
F_m &= -p\Gammam/(\varrho^2H^2),\\
F_p &= -\Gamma/H,\qquad F_{h_t} = p\Gamma/H^2 .
\end{aligned}
$$

This eliminates the entire class of hand-derived-Jacobian bugs (several of
which existed in the earlier prototypes); analytic Jacobian blocks can be
added later purely for speed.
*(tests: `test_density_complex_step_matches_fd`,
`test_complex_step_jacobian_matches_fd`)*

### 10.3 Newton with Levenberg–Marquardt damping

Each stage iterates, in scaled variables,

$$
\big(\mathbf J^{\!\top}\mathbf J + \lambda \mathbf I\big)\,\delta\mathbf y
= -\,\mathbf J^{\!\top}\hat{\mathbf R} .
$$

For $\lambda \to 0$ this is exactly the Newton step (when $\mathbf J$ is
square and invertible); for large $\lambda$ it turns toward the steepest
descent direction of the merit function $\|\hat{\mathbf R}\|^2$ — so every
step, damped or not, is a guaranteed "downhill" direction.  A step is
accepted if the residual norm decreases and all edge states remain physical
($p>0$, $h_t>0$ — under which the state recovery of §4.2 cannot fail);
otherwise $\lambda$ is increased and the step recomputed.  Accepted steps
shrink $\lambda$, recovering quadratic Newton convergence near the
solution.  The damping also transparently regularizes the momentarily
singular Jacobians of §8.3.

(Design note: an earlier version damped the Newton system directly,
$(\mathbf J + \lambda\mathbf I)\delta\mathbf y = -\hat{\mathbf R}$, in the
pseudo-transient-continuation style.  Such steps are *not* guaranteed
descent directions for $\|\hat{\mathbf R}\|$, and on larger branching
networks the iteration was observed to deadlock in curved valleys of the
residual landscape.  Levenberg–Marquardt removes that failure mode.)

### 10.4 The vanishing-friction homotopy

The solve is staged over the stabilization parameter of §7.7, by default

$$
\text{stab} \in \{\,0.1,\; 0.01,\; 0\,\},
$$

warm-starting each stage with the previous stage's solution.  The first
stage behaves like a resistive (almost incompressible) network — easily
solvable from any start, and it settles the *flow routing* (which way each
branch flows).  The later stages remove the fictitious friction gently —
the perturbation of the solution scales with the removed friction, so
modest steps keep each stage inside Newton's fast-convergence basin.  The
final stage solves the **exact** equations.

The smoothing widths follow the same schedule,

$$
\varepsilon = \max(0.3\,\text{stab},\; 10^{-4})\cdot \dot m_{\text{ref}} :
$$

wide while the iterate may still cross $\dot m = 0$ (regime switches are
then gentle slopes rather than cliffs), sharp in the final stage, by which
point all flows sit far from the switch points and the wide and sharp forms
agree to $O((\varepsilon/\dot m)^2)$.

Typical performance: 10–25 total Newton iterations to
$\|\hat{\mathbf R}\| \sim 10^{-12}$, including quiescent cold starts,
unknown interior flow directions, and full reversal.
*(tests: `test_converges_from_poor_starts`, the whole of
`test_networks.py`)*

---

## 11. Compressible-flow behavior: emergent choking and shocks

> **In plain terms:**  A gas passage cannot carry an arbitrarily large mass
> flow.  As the back pressure is lowered, the flow in the narrowest section
> speeds up until it reaches the local speed of sound ($M = 1$) — and then
> the flow rate stops increasing, no matter how hard you suck on the
> outlet.  This is **choking**, the most important compressible-flow
> phenomenon in network analysis (it is also why rocket nozzles and orifice
> flow meters work).  In this framework choking is *emergent*: it is encoded
> once, as an either/or ("complementarity") condition inside the area-change
> and outlet rows, and the network then discovers by itself which passage
> chokes, caps the mass flow there, and even where the resulting shock
> stands — with no regime declarations and no switching logic.

The maximum (choking) mass flux for given upstream total conditions is

$$
\left(\frac{\dot m\sqrt{T_t}}{A\,p_t}\right)_{\max}
= \sqrt{\tfrac{\gamma}{R}}\,
\Big(\tfrac{2}{\gamma+1}\Big)^{\frac{\gamma+1}{2(\gamma-1)}},
\qquad
\text{reached at the critical pressure ratio }
\frac{p}{p_t}\Big|_{M=1} = \Big(\tfrac{2}{\gamma+1}\Big)^{\frac{\gamma}{\gamma-1}}
\approx 0.528 \ (\gamma = 1.4).
$$

### 11.1 The throat is always an edge

All geometry in the model lives on edges; elements are **internally
monotone** in area (a modeling convention: an area change interpolates
between its port areas and hides no interior extremum).  Consequently the
minimum area of any flow path is an edge area, any throat is an edge, and
the natural home of the sonic state is that edge — which the state recovery
(§4.2) supports without any singularity.  Choking constraints are therefore
attached to the *existing elements adjacent to their small-port edges*, not
to a special "nozzle" element with hidden internal geometry.

### 11.2 Emergent choking via complementarity

Two facts make choking invisible to the plain equations: the choking
inequality $\Phi(M) \le \Phi(1)$ holds *identically* on the edge-state
manifold (it can never be "violated", so it cannot act as a limiter), and at
the choked state the total-pressure coupling becomes tangent to zero — a
**fold**, i.e. a double root with a singular Jacobian (measured
$\sigma_{\min} \sim 5\times10^{-5}$; over-driven cases stalled pinned at
$M = 1.000$).  The missing physics is the either/or:

* **Area change** (`IsentropicAreaChange`, §7.2): the pressure row is the
  smoothed Fischer–Burmeister complementarity

$$
0 \;\le\; \big(1 - M^{\text{in}}_{\text{small}}\big)
\;\perp\;
\frac{p_{t,\text{small}} - p_{t,\text{large}}}{p_{t,\text{small}}} \;\ge\; 0 :
$$

  *either* the small port is subsonic (any flow direction, including
  reversed) and the element is lossless — the classical isentropic element —
  *or* the small port is exactly sonic in the diverging direction and a
  total-pressure **drop** becomes admissible: the lumped normal shock
  standing in the diverging continuation.  Crucially, the "distance from
  choking" must be measured as $1 - M$ (a regular, linear zero), *not* as
  $\dot m^* - \dot m$ evaluated from the same edge — the latter vanishes
  quadratically (it is the fold in disguise) and would re-create the
  singularity.  With the complementarity, choking is an explicit regular
  equation: the measured $\sigma_{\min}$ at a choked solution improves from
  $\sim 5\times10^{-5}$ to $> 10^{-3}$ *(test:
  `test_choked_solution_is_well_conditioned`)*.

* **Pressure outlet** (§7.6): the outflow branch carries the analogous row
  $0 \le (1 - M^{\text{in}}) \perp (p - p_b)/p_b \ge 0$: a converging exit
  chokes at exactly $M = 1$ and its pressure detaches *upward* from the
  specification — the underexpanded discharge of a choked orifice, whose
  expansion to ambient happens outside the network.  Supersonic exit edges
  ($M^{\text{in}} > 1$) are intentionally infeasible at this boundary: a
  fixed-geometry exit chokes at $M = 1$ (see §11.4 for supersonic exits).

The Fischer–Burmeister function $\varphi(a,b) = a + b - \sqrt{a^2+b^2+
\epsilon^2}$ encodes the either/or as one $C^\infty$ residual; on its root
manifold $2ab = \epsilon^2$, so the "off" branch is pinned to a
quadratically small bias.  $\epsilon$ must be small and **fixed**
($10^{-5}$): the bias $\tfrac{\epsilon^2}{2} p_t$ is a fictitious
total-pressure loss and must stay far below the smallest driving pressure
difference in the network (widening $\epsilon$ during the friction homotopy
was tried and *broke* weakly driven cases).

The shock is not a solver unknown: a normal shock conserves $\dot m$ and
$h_t$, so the converged loss determines it in post-processing
(`fns.shock`): invert the Rankine–Hugoniot total-pressure ratio for the
pre-shock Mach $M_s$, then the area–Mach relation (with the sonic small port
as $A^*$) for the shock area $A_s$ — **the shock location is deduced, not
imposed**, with a validity check $A_s$ within the element's port-area range.

### 11.3 Validated operating map

For the three-edge C–D nozzle (feed → throat edge → exit), all from the same
equations with no regime input *(tests: `test_choking.py`)*:

| back pressure | behavior (all emergent) | validation |
|---|---|---|
| above first critical | venturi: subsonic, lossless, $\dot m < \dot m^*$ | exact isentropic match |
| at/below first critical | throat edge $\to M = 1$, $\dot m \to \dot m^*$, continuous monotone saturation, never exceeded | sweep `test_mass_flow_saturation_continuity` |
| shock regime | $\dot m = \dot m^*$, internal shock in the diverging element; exit matches $p_b$ | full Rankine–Hugoniot construction at $M_s \in \{1.3, 1.8, 2.15\}$: $p_t$ loss, exit state, deduced $M_s$ and $A_s$ all match *(test: `test_shock_in_nozzle_vs_analytic`)* |
| converging exit, below critical | choked orifice: $M = 1$ exactly, $\dot m = \dot m^*$ to 7 digits, exit $p$ detaches to $p^*$ | `test_choked_orifice_below_critical_pr` |
| over-driven networks (e.g. lossless paths between unequal reservoirs) | the limiting edge chokes, the transfer flow caps at its choking value, the case **converges** | `test_overdriven_transfer_loop_capped` (the former `test_1.yaml` failure class) |

### 11.4 What remains non-emergent, and why

**Supersonic exit edges** (over-/under-expanded and design operation of a
C–D nozzle discharging through a boundary).  At a supersonic exit all three
characteristics leave the domain: the back pressure physically imposes
*nothing*.  No smooth row can express "impose nothing over a finite
parameter range": scaling a residual row never weakens it (the root is
unchanged), and fading it to exactly zero leaves a structurally singular
system.  Equivalently — in the overexpanded regime the same internal
solution holds for a *continuum* of back pressures, which no square smooth
system with an active $p_b$-row can reproduce.

The precise statement is worth stating carefully, because it is narrower than
"impossible."  An emergent supersonic outlet through any smooth, *outlet-local*
row within the current degree-of-freedom set is impossible — that is the claim
above, and it stands.  But an emergent supersonic *converging–diverging nozzle*
is **not** impossible in principle: a nozzle fed from a plenum has a unique
steady solution per back pressure (no multistability), and the real obstacle is
that the regime transition is a *discontinuity in the exit-edge state* the
present element set has no freedom to absorb.  Supplying an **internal
shock-position degree of freedom** in the diverging element — the exact dual of
the `SupersonicInlet` self-vacating row below — removes that obstacle.  It is
deferred (§1 scope), not refuted, and it is the same enrichment finite-frequency
acoustics of shocked nozzles requires.

An internal-DOF discharge element (plume slack $\sigma$ absorbing the
$p_b$-sensitivity) was considered and analyzed during implementation; it
resolves the back-pressure freedom but **not** the whole problem: in the
supersonic regime the system is still one condition short — the *sonic
throat anchor* ($M = 1$ at the throat, equivalently $\dot m = \dot m^*$) —
and the resulting one-parameter solution family contains unphysical
branch-jumped members.  No outlet-local smooth row can supply the anchor,
because it lives at the throat.  The supersonic-exit configuration therefore
remains a **declaration** (the only one: "this exit runs supersonic", the
same kind of design statement as the supersonic inlet), implemented as the
`SupersonicOutlet` boundary: a plain static-pressure row at the design exit
pressure, valid only for supersonic exits at design-consistent conditions
(`run_ui_case.py` warns when the converged exit is subsonic).  At the
declared exit, the throat sits at the exact corner of the choking
complementarity; the smoothed corner shifts it by $O(\epsilon^{2/3})
\approx 4\times10^{-4}$ in Mach with a $\sim 10^{-7}$ relative fictitious
$p_t$ loss — negligible, but worth knowing.  Demonstrated:
`examples/ui_showcase/cd_nozzle_supersonic.yaml` (design point, exit
$M = 2.197$, choked mass flow, shock-free).  Off-design back pressures
(reporting the external over-/under-expansion via a $p_b$-independent
throat-area anchor plus a diagnostic mismatch) are a designed later step.  The same internal-DOF principle ("every regime freedom
gets a physical unknown; regime boundaries become bound complementarities on
it") extends to supersonic intakes — and the first stage is implemented: the
`SupersonicInlet` boundary (a declared flight condition supplying **two**
equations, $M$ and $p_t$, since all characteristics enter the domain there).
The resulting formal over-determination by one row is absorbed by a neat
mechanism: the Fischer–Burmeister row of the sonic-fed diverging element
(the one holding the terminal shock) **self-vacates** — fed at exactly
$M = 1$ its residual collapses to the $O(\epsilon^2)$ regularization floor —
which is precisely the smooth realization of the characteristic count
redistribution.  Started-intake operation at critical and supercritical
conditions (terminal shock anywhere within the first diverging segment) is
validated against the full quasi-1D construction: supersonic compression,
sonic throat edge, deduced shock Mach/position, pressure recovery
*(tests: `test_supersonic_inlet.py`; cases:
`examples/ui_showcase/intake_*.yaml`)*.  Solve such cases from the
supersonic-branch initial guess (`ui_bridge.supersonic_chain_guess`,
automatic in `run_ui_case.py`) with tolerance at the FB floor
($\sim 10^{-9}$).  Deeper supercritical operation (shock standing in a
*supersonic-fed* segment) still requires the shock-position element DOF —
the remaining designed step, and exactly the extra degree of freedom that
finite-frequency acoustics of shocked nozzles/intakes requires.

Genuine multistability (intake start/unstart hysteresis) is physics, not
formulation: both branches are valid steady solutions and history selects —
a steady solver can find each basin from appropriate warm starts and report
the Kantrowitz margin as a diagnostic.

## 12. The acoustic / perturbation network

> **In plain terms:**  The founding idea of the project: solve the steady
> operating point and the acoustic (pulsation) behavior with *one*
> consistent set of element equations, rather than maintaining two codes
> that can drift apart.  This goal is fully realized, and the reason is
> structural: every steady element equation is an *instantaneous algebraic*
> relation (no time derivative — §3), so linearizing it at the operating
> point *is* the acoustic jump condition.  The solver already computes that
> linearization as a by-product of convergence — the Jacobian.  This chapter
> builds the frequency-domain network on top of it.
>
> A user comes to the acoustic stage with one of three questions.  **(i) A
> transfer / scattering matrix** between chosen stations — obtained from two
> linearly independent forced solutions (e.g. upstream- and
> downstream-excited).  **(ii) Stability** — the complex frequencies at which
> the system sustains a free oscillation, i.e. the determinant of the system
> matrix vanishes; an eigenvalue problem whose roots give modal frequencies
> and growth rates.  **(iii) Black-box identification** — the user has a
> *measured* global response and models everything except one element (say a
> flame), recovering that element's frequency response as the unknown.  All
> three are operations on a single assembled operator $\mathbf A(\omega)$,
> developed below.

The *engineering* fit of this chapter into the implementation — what is
reused, the assembly routines, the drivers — is `implementation-plan.md` §8; the acoustic
layer adds no new JIT code.  Here we develop the theory.

### 12.1 Linearization about the mean flow

Let $\mathbf x^\ast$ solve the steady residual $\mathbf R(\mathbf x^\ast)=0$
to tolerance, and split a small unsteady motion into a mean plus a
time-harmonic fluctuation,

$$
\mathbf x(t) = \mathbf x^\ast + \mathbf x'(t),
\qquad
\mathbf x'(t) = \Re\!\big\{\hat{\mathbf x}\,e^{\mathrm i\omega t}\big\},
\quad \hat{\mathbf x}\in\mathbb C^{3E}.
$$

Because $\mathbf R$ is algebraic and differentiable at $\mathbf x^\ast$, it
expands as $\mathbf R(\mathbf x^\ast+\mathbf x') = \mathbf J\,\mathbf x' +
O(\|\mathbf x'\|^2)$ with $\mathbf J = \partial\mathbf R/\partial\mathbf
x|_{\mathbf x^\ast}$.  To first order the algebraic rows obey

$$
\boxed{\;\mathbf J\,\mathbf x' = 0\;}
$$

and **$\mathbf J$ is exactly the converged Newton Jacobian** the steady solver
already assembles by complex-step differentiation (§10),
$J_{ij}=\Im\,R_i(\mathbf x^\ast+\mathrm i h\,\mathbf e_j)/h$.  No new
linearization machinery is needed for the algebraic content of the acoustic
operator; it is a by-product of convergence.  This is the **zero-frequency
acoustic network**: the steady jump conditions, differentiated, *are* the
acoustic jump conditions.

> **Basis invariance (why the mean flow need not use characteristics).**  For
> any invertible block-diagonal $\mathbf T$, solving $\mathbf J\mathbf x'=0$ in
> variables $\mathbf w=\mathbf T^{-1}\mathbf x'$ gives $\mathbf J\mathbf T\,
> \mathbf w=0$ — a similarity transform with identical solutions (the same
> fact as §10: Newton is invariant under a change of solution basis).  So
> expressing the *mean flow* in characteristics buys nothing; characteristics
> earn their place only in the *propagation* layer below.

### 12.2 Characteristic variables for propagation

Linearizing the recovered state (§4) about the mean, the primitive
perturbations $(\delta\varrho,\delta u,\delta p)$ map to the
solution-variable perturbations through a per-edge matrix $\mathbf D(\mathbf
x^\ast_e)$ (from $\dot m=\varrho uA$ and $h_t=\Gamma p/\varrho+\tfrac12u^2$;
§9.2, `dq_to_dx`).  The **characteristic amplitudes** $\mathbf
w_e=(f,g,h)^\top$ — the eigen-amplitudes of the 1-D Euler flux Jacobian — are

$$
u' = f-g,\qquad p' = \varrho c\,(f+g),\qquad \varrho' = h + p'/c^2,
$$

the downstream acoustic, upstream acoustic, and entropy waves.  Composing the
two maps gives the per-edge change of basis

$$
\boxed{\;\mathrm d\mathbf x_e = \mathbf T_e\,\mathbf w_e,\qquad
\mathbf T_e = \mathbf D(\mathbf x^\ast_e)\,\mathbf R(\mathbf x^\ast_e)\;}
$$

(`characteristics.char_to_dx`), with inverse $\mathbf L_e=\mathbf T_e^{-1}$
(`dx_to_char`); these are the §9.2 `transformation_blocks`.  Both $\mathbf
D$ and $\mathbf R$ are nonsingular at every physical state (their determinants
are $-A(u^2+c^2/(\gamma-1))$ and $2\varrho c$), so $\mathbf T_e$ never
degenerates — the basis change is always well defined, including at $M=1$.
The propagation operator next is **diagonal in $\mathbf w$ and full in
$\mathbf x$**: the mean-flow layer and the propagation layer live most
naturally in different bases, bridged by $\mathbf T$.

### 12.3 The lossless duct: origin of frequency dependence

The elements of §7 are *lengthless* — area changes and losses are concentrated
jumps.  Wave propagation needs the one element they lack: a **length-bearing,
lossless, constant-area duct** of length $L$ carrying a uniform mean state.
On it the linearized Euler equations diagonalize, in $\mathbf w$, into three
scalar advection equations at speeds $\bar u+\bar c$, $\bar u-\bar c$, $\bar
u$.  With the harmonic ansatz each is a pure phase delay; defining the three
transit times

$$
\tau_+ = \frac{L}{\bar u+\bar c},\qquad
\tau_- = \frac{L}{\bar c-\bar u},\qquad
\tau_0 = \frac{L}{\bar u},
$$

the end-to-end relations between the duct's tail ($x=0$) and head ($x=L$) are

$$
\boxed{\;
\hat f_{\mathrm{head}} = e^{-\mathrm i\omega\tau_+}\hat f_{\mathrm{tail}},\quad
\hat g_{\mathrm{tail}} = e^{-\mathrm i\omega\tau_-}\hat g_{\mathrm{head}},\quad
\hat h_{\mathrm{head}} = e^{-\mathrm i\omega\tau_0}\hat h_{\mathrm{tail}}
\;}
$$

(the upstream wave $g$ is carried head→tail).  This is the **only** place a
bare $e^{\mathrm i\omega\tau}$ factor enters, and it is diagonal in $\mathbf
w$ — hence the basis of §12.2.  The duct is **mean-flow-transparent**: at
$\omega=0$ all phases are unity and the relations reduce to continuity
$\mathbf x_{\mathrm{tail}}=\mathbf x_{\mathrm{head}}$ (the equal-area limit of
an isentropic area change), so the duct does not perturb $\mathbf x^\ast$; its
length $L$ is metadata read only by the acoustic assembly.  Modeled as a
network entity it is a **two-port element (node)**, its two end-stations being
the two incident edge states — preserving the "equations at nodes, state at
edges" convention.

### 12.4 The perturbation system matrix

Collect the fluctuation amplitudes and assemble, at each frequency, a complex
system matrix

$$
\boxed{\;
\mathbf A(\omega)\,\hat{\mathbf x} = \hat{\mathbf b},
\qquad
\mathbf A(\omega) = \underbrace{\mathbf J_{\mathrm{alg}}}_{\text{§12.1}}
                  + \underbrace{\mathrm i\omega\,\mathbf M}_{\text{storage}}
                  + \underbrace{\mathbf P(\omega)}_{\text{propagation}}
                  + \underbrace{\mathbf S(\omega)}_{\text{sources}}
\;}
$$

with $\hat{\mathbf b}$ a forcing (zero for the stability problem).  The four
contributions:

- **Algebraic $\mathbf J_{\mathrm{alg}}$** — all element jump/conservation rows
  and the edge advection rows, taken directly from the converged complex-step
  Jacobian of §12.1.
- **Propagation $\mathbf P(\omega)$** — for each duct, its continuity rows are
  replaced by the phase relations of §12.3, built diagonally in $\mathbf w$ and
  mapped to solution-variable rows through $\mathbf L_e,\mathbf T_e$.
- **Storage $\mathbf M$** — elements with finite volume retain the
  $\partial_t$ terms dropped at steady state (§12.5).  $\mathbf M$ is
  frequency-independent.
- **Sources $\mathbf S(\omega)$** — a heat-release element couples a downstream
  total-enthalpy fluctuation to an upstream velocity fluctuation through a flame
  transfer function $\mathcal F(\omega)$ (e.g. the $n$–$\tau$ closure $\mathcal
  F=n\,e^{-\mathrm i\omega\tau}$).  This term makes $\mathbf A$ non-self-adjoint
  and drives thermoacoustic instability.

**Boundary conditions** are reflection coefficients $\hat g=\mathcal
R(\omega)\hat f$ at terminal edges, diagonal in $\mathbf w$.  The choked /
supersonic limit $\mathcal R=0$ for characteristics that cannot travel
upstream is *already* present in $\mathbf J_{\mathrm{alg}}$: the converged
complementarity blocks at a sonic throat (§11) make precisely those
characteristics one-way, so the steady regime logic and the acoustic boundary
behaviour coincide (with the caveat of §12.6).  Continuity with the steady
solution holds by construction: at $\omega=0$, $\mathrm i\omega\mathbf M\to0$,
$\mathbf P\to$ identity, and $\mathbf A(0)=\mathbf J_{\mathrm{alg}}$.

### 12.5 Storage terms: finite volume and effective length

> **In plain terms:**  A lengthless jump element responds to the *instant*.
> A real component has volume (it can store mass and energy) and length (its
> contained fluid has inertia).  These are exactly the time-derivative terms
> the steady solve discarded; restored, they are the element's acoustic
> *compliance*, *inertance*, and *thermal storage*.

Integrating the 1-D compressible Euler equations over an element's control
volume $V$ and applying the divergence theorem, the flux sum is precisely the
steady element residual of §3, and the **new** object is the volume integral
$\frac{\mathrm d}{\mathrm dt}\int_V\mathbf U\,\mathrm dV$.  Linearized and
harmonic, it produces $\mathrm i\omega V\,\widehat{\mathbf U}'$ — **three**
storage terms, one per conservation law:

$$
\begin{array}{lll}
\textbf{mass} &:& \mathrm i\omega\,V\,\hat\varrho', \\[2pt]
\textbf{momentum} &:& \mathrm i\omega\,V\,(\bar\varrho\,\hat u'+\bar u\,\hat\varrho'), \\[2pt]
\textbf{energy} &:& \mathrm i\omega\,V\,[\,\hat p'/(\gamma-1)+\bar\varrho\bar u\,\hat u'+\tfrac12\bar u^2\hat\varrho'\,].
\end{array}
$$

They populate $\mathbf M$, which is zero except in the conservation rows of
volumetric elements.  A **lengthless jump element has $V=0$ and hence no
storage** — it is the genuinely instantaneous $V\to0$ limit; this is exactly
what distinguishes a volumetric element from a jump.

In the compact lumped limit two of the three are independent and the third is
slaved:

- **Mass storage → compliance.** With the isentropic closure $\hat p'=\bar
  c^2\hat\varrho'$, the mass term becomes $\mathrm i\omega C\hat p'$ with
  $C=V/(\bar\varrho\bar c^2)$ — the acoustic compliance of a cavity.
- **Momentum storage → inertance.** The low-Mach momentum term is $\mathrm
  i\omega\,\mathcal L\,\widehat{\dot m}'$ with $\mathcal L=L_{\mathrm
  eff}/A$ — the inertance of a neck/orifice.  The relevant length is the
  **effective length** $L_{\mathrm{eff}}=\ell_{\mathrm{geom}}+\sum_{\text
  faces}\delta_{\mathrm{end}}$: a geometrically thin compact element (an
  orifice, a sudden area change) has $\ell_{\mathrm{geom}}\to0$ yet
  $L_{\mathrm{eff}}\not\to0$ — its inertance survives entirely through the
  **end corrections** $\delta_{\mathrm{end}}$ (the entrained near-field mass,
  e.g. $\approx0.85a$ flanged, $0.61a$ unflanged, $\approx0.79R$ for a thin
  orifice).  Using the geometric length here would wrongly annihilate the
  inertance.
- **Energy storage** is *not* independent in the adiabatic compact limit (the
  isentropic relation used for the compliance is the linearized energy
  equation); it becomes a free degree of freedom only when heat is exchanged
  inside the volume — precisely the row that couples to the flame source
  $\mathbf S(\omega)$.

So: a cavity (large $V$, negligible $L_{\mathrm{eff}}$) is a pure compliance,
a neck/orifice (small $V$, finite $L_{\mathrm{eff}}$) a pure inertance, and a
**Helmholtz resonator** their series pairing, $\omega_0=\bar c\sqrt{A/(L_{\mathrm
eff}V)}$.

**Compact vs. distributed — a clean dichotomy.**  The lumped terms are the
leading compact truncation of the duct phases: expanding $e^{-\mathrm
i\omega\tau}=1-\mathrm i\omega\tau+\cdots$, the $-\mathrm i\omega\tau$
correction *is* the storage.  An element is treated lumped while it is
acoustically compact, $\mathrm{He}=\omega L/\bar c\ll1$; where it is not, it is
promoted to a duct (§12.3) and resolved with the full phase.  The two agree in
their overlap.  (The interior of a non-compact element is boundary-determined —
a Helmholtz BVP with no independent interior DOF — so it always closes on the
edge amplitudes; the only genuine breakdown of the edge-scalar description is
transverse cut-on, $\mathrm{He}\gtrsim\pi$, which requires multimodal ports.)

### 12.6 Regularization at the two singular operating points

$\mathbf J_{\mathrm{alg}}$ was taken "directly from the converged Jacobian."
That is exact for the *physical* residual, but the residual the steady solver
actually converges is **regularized** (the smooth upwind switch $\theta_
\varepsilon$ of the advection rows, the smoothed Fischer–Burmeister
complementarity of choking; Appendix A).  Strictly subsonic and flowing, these
deviate from the physical jump conditions by $O(\varepsilon^2)$, so $\mathbf
J_{\mathrm{alg}}$ is the physical operator to solver accuracy.  The deviation
is **not** small at the two operating points where the regularized functions
lose smoothness — and those are the acoustically interesting ones.

- **Choking, $\bar M=1$.**  The smoothed complementarity is exact *within* a
  regime (strictly choked or strictly subsonic), where its linearization is the
  true one-sided jump and the $\mathcal R=0$ one-way claim of §12.4 holds
  cleanly.  It fails only at the exact sonic corner, where the function is
  non-differentiable; and there is in addition a genuine *physical*
  degeneracy, $\tau_-=L/(\bar c-\bar u)\to\infty$ (the upstream wave stalls).
  Near choking, impose the analytic one-way boundary $\mathcal R=0$ directly
  rather than inheriting the linearized complementarity block.

- **Stagnant edges, $\bar M=0$.**  The advection-row linearization carries a
  term $-\theta_\varepsilon'(\bar{\dot m}_e)\,(H_{\mathrm{tail}}-H_{\mathrm
  head})$ with $\theta_\varepsilon'(0)=1/(2\varepsilon)$ — apparently
  divergent.  But it multiplies the **difference of donor total enthalpies
  across the edge**.  A truly zero-flow edge is never a mean-flow solution
  (it is under-determined); $\bar M=0$ arises only as a deliberate *quiescent*
  analysis (acoustics in a still, uniform medium), where every element shares
  the same stagnation enthalpy, $H_{\mathrm{tail}}=H_{\mathrm{head}}$, and the
  divergent term **vanishes identically for any $\varepsilon$**.  So feeding
  $\bar M=0$ into the assembled Jacobian is well posed for quiescent acoustics
  with no special treatment; moreover at $\bar u=0$ the entropy wave decouples
  ($\partial_t h=0$), so the residual advection row reduces to a benign
  interpolation that cannot pollute the $(f,g)$ spectrum.  The single
  configuration that *does* break — a near-stagnant edge bridging a real mean
  enthalpy gap (a dead-end leg off a hot main, an edge abutting a flame) — is
  not a well-posed steady state to begin with (no conduction in the model), so
  it is **guarded**, not solved.

**An un-regularized assembly option.**  Because the regularization scales live
in one place (Appendix A), the residual can be re-evaluated with them switched
off and the same complex-step driver run at $\mathbf x^\ast$ to obtain a
physical $\mathbf J_{\mathrm{alg}}$ — recovering the true one-sided block at a
choked throat and dropping the $O(\varepsilon^2)$ constitutive bias.  It cannot
manufacture a derivative at a genuine switch point (a hard Heaviside or a
corner has none).  Since $\mathbf x^\ast$ is a root of the *regularized*
residual, exact $\omega\to0$ consistency wants a few un-regularized Newton
*polish* steps before forming $\mathbf J_{\mathrm{alg}}$: the clean pipeline is
*converge-with-regularization → polish-without → assemble without*.

### 12.7 The three analyses

All three target computations are operations on $\mathbf A(\omega)$.

- **Transfer / scattering matrix.**  Apply two linearly independent forcings
  (upstream- and downstream-excited), solve $\mathbf A(\omega)\hat{\mathbf
  x}=\hat{\mathbf b}^{(k)}$, and read the wave amplitudes $\mathbf w=\mathbf
  L_e\hat{\mathbf x}$ at the chosen ports to fill the $2\times2$ scattering
  matrix $\mathbf S(\omega)$ (equivalently the transfer matrix $\mathbf T$).
  Sweep $\omega$ for the spectrum; two complex solves per frequency.

- **Stability — a nonlinear eigenvalue problem.**  Set $\hat{\mathbf b}=0$; a
  nontrivial mode exists iff

  $$
  \boxed{\;\det\mathbf A(\omega)=0,\qquad \omega=\omega_r+\mathrm i\omega_i\in\mathbb C.\;}
  $$

  Because $\mathbf A$ depends on $\omega$ through $\mathrm i\omega$, the duct
  phases $e^{-\mathrm i\omega\tau}$, and $\mathcal F(\omega)$, this is
  *nonlinear* in $\omega$.  Each root gives the modal frequency $\omega_r$, the
  growth rate $\omega_i$, and the eigenmode as the null vector.  Solve by
  Newton on the determinant from acoustic-mode seeds ($\omega\approx n\pi\bar
  c/L$ with sources off), or by Beyn's contour-integral method to capture all
  modes in a region without seeds.  (The phases $e^{-\mathrm i\omega\tau}$
  over/underflow for complex $\omega$, so a robust driver scales them.)

- **Black-box element identification (inverse problem).**  Partition the
  network into known elements plus one unmodeled $k$-port $B$ (e.g. a flame
  transfer matrix).  Given a *measured* global response $\mathbf S_{\mathrm
  meas}(\omega)$, the assembled relation with $B$'s block left symbolic
  produces, at each frequency, a **linear matrix equation** for $B(\omega)$ —
  the formal inverse of the scattering computation (solvable exactly when the
  independent measured columns suffice, least-squares otherwise, the residual
  then reporting model–measurement consistency).

The division of labour is clean: **the mean-flow solver does not need
characteristic variables; the acoustic stage gets them — and the storage,
propagation, and source structure — exactly, from the same residuals.**

---

## 13. Worked examples

All in `examples/`, runnable directly; each prints a full solution table.

| example | what it demonstrates |
|---|---|
| `ex1_nozzle_chain.py` | five isentropic elements in series, solved from an exactly quiescent cold start (the historically failing case) |
| `ex2_split_fractions.py` | flow split between a clean branch and a lossy branch, with junction mixing |
| `ex3_reverse_flow.py` | boundary data that drive the flow *against* both edge arrows; backflow temperature advection |
| `ex4_manifold.py` | 13-element network: cold mass-flow source + hot pressurized source, mixing chamber, feed, lossless distribution manifold, three dissimilar branches (loss / sudden dump / nozzle+loss) at three different back pressures; max Mach ≈ 0.8; includes the junction-vs-splitter selection lesson of §7.5 |
| `ex5_compressibility.py` | pressure-ratio sweep against the exact isentropic solution up to $M = 0.99$; mass-flow saturation at choking; the supersonic branch below the critical ratio |
| `ex6_bridge.py` | compressible Wheatstone bridge: an interior edge whose flow direction is decided by the resistance arrangement — positive, exactly antisymmetric on mirroring, exactly zero when balanced |

---

## 14. Validation map

| Claim / feature | Test |
|---|---|
| State recovery unique & smooth (incl. reversed, supersonic, quiescent) | `test_state.py::test_roundtrip` |
| Stagnation quantities direction-independent | `test_state.py::test_signed_quantities` |
| Implicit-density derivative propagation exact | `test_state.py::test_density_complex_step_matches_fd` |
| Complex-step Jacobian exact | `test_solver.py::test_complex_step_jacobian_matches_fd` |
| Isentropic element vs analytic relations | `test_networks.py::test_single_iac_vs_exact` |
| Multi-element chain from exact quiescent cold start | `test_networks.py::test_multi_iac_chain_from_quiescent` |
| Edge-direction flip invariance | `test_networks.py::test_edge_direction_flip_invariance` |
| Borda–Carnot balance, entropy rise | `test_networks.py::test_sudden_expansion_borda_carnot` |
| Contraction regime lossless | `test_networks.py::test_sudden_contraction_is_lossless` |
| Pressure-driven forward flow vs exact | `test_networks.py::test_pressure_driven_forward_vs_exact` |
| Full flow reversal through boundaries vs exact | `test_networks.py::test_reversed_flow_through_boundaries` |
| Split fractions, lossless diamond | `test_networks.py::test_lossless_diamond_split_fractions` |
| Loss-driven split shift + conservation | `test_networks.py::test_diamond_with_loss_branch` |
| Junction enthalpy mixing exact (to regularization) | `test_networks.py::test_energy_conservation_with_mixed_temperatures` |
| Near-choking accuracy ($M = 0.99$) vs exact | `test_networks.py::test_high_subsonic_vs_exact` |
| Choked orifice below critical PR (M = 1, capped, p detaches) | `test_networks.py::test_choked_orifice_below_critical_pr` |
| Venturi regime exact (lossless, below choking) | `test_choking.py::test_venturi_regime_matches_isentropic` |
| Shock-in-nozzle vs Rankine–Hugoniot construction (3 shock Machs) | `test_choking.py::test_shock_in_nozzle_vs_analytic` |
| Mass-flow saturation continuous, monotone, never exceeds choking | `test_choking.py::test_mass_flow_saturation_continuity` |
| Complementarity removes the choking fold (conditioning) | `test_choking.py::test_choked_solution_is_well_conditioned` |
| Over-driven transfer loop caps at choking and converges | `test_choking.py::test_overdriven_transfer_loop_capped` |
| FB rows reduce to pt-equality in subsonic operation | `test_choking.py::test_subsonic_cases_unaffected_by_fb_rows` |
| Started intake (supersonic inlet, sonic throat, terminal shock) vs quasi-1D | `test_supersonic_inlet.py::test_started_intake_vs_analytic` |
| Terminal shock swallowed deeper with falling back pressure | `test_supersonic_inlet.py::test_supercritical_shock_moves_with_back_pressure` |
| Wheatstone bridge: reversal antisymmetry & balance | `test_networks.py::test_wheatstone_bridge_reversal_and_balance` |
| Large heterogeneous network with mixing (ex4) | `test_networks.py::test_manifold_network_with_mixing` |
| Newton invariance under characteristic basis change | `test_characteristics.py::test_newton_in_characteristics_equals_newton_in_primitives` |
| Characteristic maps exact / inverse | `test_characteristics.py` (map tests) |
| LM damping regularizes indeterminate symmetric split | `test_solver.py::test_quiescent_symmetric_branching_start` |
| Homotopy final stage solves exact equations | `test_solver.py::test_final_stage_solves_exact_equations` |
| Robustness to poor starts (zero, tiny, wrong-sign, large) | `test_solver.py::test_converges_from_poor_starts` |

---

## 15. Limitations and the development path

> **In plain terms:**  This is the honest map.  It records what the method
> cannot do, the harder approaches we built and tested to push past those
> limits, and why the first version deliberately stops where it does.  The
> point is to spare the next person from re-walking dead ends, and to mark the
> deferred work clearly enough to pick up.

### 15.1 A change of solution basis cannot improve convergence

The project began with the hypothesis that casting the Newton correction in
characteristic variables $(f,g,h)$ would converge better than the primitive
$(\dot m, p, h_t)$.  It does not: a change of solution basis is a similarity
transform of the linear system, and Newton's iterates are invariant under it
(§10.1, and the boxed argument of §12.1).  Only the linear-solve conditioning
and the relaxation metric change, not the path.  The characteristic machinery
was nonetheless *kept* — not for the mean flow, but because it is exactly the
basis the acoustic stage needs (§12.2).  Lesson: the win we expected was
illusory, but the tool found its real home one layer up.

### 15.2 Emergent supersonic / shock branches: the fold problem

This is the central limitation, and the one that most shaped the scope (§1).  A
converging–diverging nozzle's steady solution is *unique per back pressure*,
yet a cold-start Newton solve cannot reach the supersonic branch: a genuine
**fold** in solution space (the $p_{\text{floor}}$ crossing, §11.4) sits
between the start and the answer.  We explored, in order:

- **A regime classifier** (compute $p_{\text{floor}}$, pick the branch, build a
  matching guess).  Equivalent to the hand calculation; automatable for one
  nozzle, but it does **not scale** to arbitrary networks.  Rejected.
- **Pseudo-time-stepping the steady residual** ($\dot{\mathbf x}=-\mathbf
  R(\mathbf x)$).  A trap: gradient-like flow stays confined to the residual's
  basins and **cannot tunnel through a fold** — it converges to roots of the
  *same* $\mathbf R$, so the same cold start gives the same failure.
- **An internal shock-position degree of freedom.**  Implemented (the
  `SupersonicInlet` self-vacating Fischer–Burmeister row, §11.4): it gives the
  unresolved shock a coordinate and an equation, and it works for a *declared,
  single, anticipated* shock.  But it is a **lumping artifact, not a fix**: it
  must anticipate each shock, so it does not generalize to unexpected,
  multiple, interacting shocks or shock trains; and it sits outside the
  per-edge characteristic transform (which asserts no internal DOF), so it does
  not fit the acoustic framework cleanly.
- **A genuine transient (time-domain) solver** — the principled cure.  The two
  states across the fold are disconnected as roots of $\mathbf R$ but
  *connected by a continuous path in time* (the shock physically blows out of
  the nozzle over milliseconds).  Restoring real storage terms
  ($\partial_t(\varrho V)$, finite-speed evolution) lets a march traverse that
  path with **no seed and no classifier**.  In a *resolved* (or hybrid:
  resolved nozzles/intakes, lumped plenums) quasi-1-D unsteady Euler model the
  shock is simply *captured* across a few cells — emergent, no DOF — and each
  cell still carries the ordinary $(\dot m, p, h_t)$ triple, so it stays
  compatible with the acoustic stage.  This was designed but **deferred**: it is
  a real re-architecture (state on cells, stiffness, implicit stepping) and the
  v1 scope does not need it.

**Where v1 lands.**  Supersonic exits and intakes are a **declaration**, not an
emergence — the `SupersonicOutlet` / `SupersonicInlet` boundaries — and the
supersonic branch is reached by **seeding**.  The scope line (§1) is drawn at
smooth, subsonic mean flow, where the steady solver is robust.  Note that
genuine multistability (intake start/unstart hysteresis) is *physics, not a
formulation defect*: both branches are valid steady states and history selects;
a steady solver can find each basin from an appropriate warm start.

### 15.3 Acoustics at the singular operating points

The reuse of the converged Jacobian as the acoustic operator is exact only
where the mean flow is strictly subsonic and flowing.  At the two operating
points where the steady regularizations lose smoothness — $\bar M = 1$
(choking) and $\bar M = 0$ (stagnant / quiescent) — care is required; these are
treated in full in §12.6 (the choked boundary by its analytic one-way limit;
the quiescent case automatically, by a cancellation; a near-stagnant edge
bridging a real enthalpy gap is *guarded*, not solved).  Duct propagation
(§12.3) is likewise formulated for subsonic mean flow; supersonic acoustic
propagation is deferred together with the supersonic mean flow.

### 15.4 The unifying view, and the shared enrichment

The steady solve **S**, the acoustic network **P** (§12), and the deferred
transient solver **T** (§15.2) are three faces of one storage-augmented
physics over the *same* element residuals and the same complex-step Jacobian:
the transformed mean Jacobian is the zero-frequency limit of P; finite-frequency
P is the linearization of T about the mean with an $e^{\mathrm i\omega t}$
ansatz; T is P taken to full amplitude.  Their *one shared enrichment* is
**length + storage per element** — the duct length carrying $e^{-\mathrm
i\omega\tau}$ and the volume terms reinstated in §12.5.  Model that storage
once and both faces follow (frequency-domain $\mathrm i\omega C$, $\mathrm
i\omega\mathcal L$ for P; actual time derivatives for T).  This is the long-term
direction; v1 implements S and the assembly for P, and leaves T as designed
future work.

---

## Appendix A: smooth regularized functions

> **In plain terms:**  Newton's method differentiates the equations, so the
> equations must not contain kinks or jumps.  Wherever physics suggests an
> `if` (which side is upstream? which way does the loss act?), we use a
> smooth function that behaves like the `if` away from the switch point and
> rounds it off within a narrow band of width $\delta$.  The price is a tiny
> imprint on the converged solution — and the key fact is that this imprint
> shrinks with the *square* of $\delta$, so a modest band buys full
> smoothness at negligible cost in accuracy.

All switching constructs use $C^\infty$, complex-analytic regularizations
with width $\delta > 0$ (`fns/smooth.py`):

$$
\begin{aligned}
\operatorname{sabs}(x;\delta) &= \sqrt{x^2+\delta^2}
&&\to |x| + \frac{\delta^2}{2|x|} + \dots\\[4pt]
\operatorname{spos}(x;\delta) &= \tfrac12\big(x + \sqrt{x^2+\delta^2}\big)
&&\to
\begin{cases}
x + \dfrac{\delta^2}{4x}, & x \gg \delta\\[6pt]
\dfrac{\delta^2}{4|x|}, & x \ll -\delta
\end{cases}\\[4pt]
\operatorname{sstep}(x;\delta) &= \tfrac12\Big(1 + \dfrac{x}{\sqrt{x^2+\delta^2}}\Big)
&&\to
\begin{cases}
1 - \dfrac{\delta^2}{4x^2}, & x \gg \delta\\[6pt]
\dfrac{\delta^2}{4x^2}, & x \ll -\delta
\end{cases}
\end{aligned}
$$

$$
\varphi_\epsilon(a,b) = a + b - \sqrt{a^2 + b^2 + \epsilon^2}
\qquad\text{(smoothed Fischer–Burmeister)}
$$

vanishes (for $\epsilon \to 0$) exactly when $a \ge 0$, $b \ge 0$ and
$ab = 0$: an either/or regime switch as a single smooth residual.  On the
smoothed root manifold $2ab = \epsilon^2$, so within a regime the "off"
variable is pinned to $\epsilon^2/(2\,\text{active})$ — a quadratically
small bias.  Used for the emergent choking rows (§7.2, §7.6, §11.2).

Facts used throughout: all tails are **quadratically** small in
$\delta/|x|$ (converged solutions satisfy the exact equations to
$O((\delta/\dot m)^2)$); all functions are analytic near the real axis
(complex-step differentiation stays exact); and
$\operatorname{spos}(0) = \delta/2 > 0$ (donor-mix denominators never
vanish).

## Appendix B: symbols and terms

| symbol | meaning |
|---|---|
| $\varrho, u, p, T$ | density, signed normal velocity, static pressure, static temperature |
| $h,\ h_t$ | static / total specific enthalpy; $h_t = h + u^2/2$ |
| $p_t,\ T_t$ | total (stagnation) pressure / temperature |
| $c,\ M$ | speed of sound, signed Mach number $u/c$ |
| $s$ | entropy; invariant form $p/\varrho^\gamma$ |
| $\dot m,\ m$ | mass flow rate (signed along the edge arrow), mass flux density $\dot m/A$ |
| $A_e$ | edge (port) area |
| $\sigma_{P,e}$ | orientation of edge $e$ at element $P$ ($+1$ tail, $-1$ head) |
| $\Gamma$ | $c_p/R = \gamma/(\gamma-1)$ ($\approx 3.5$ for air) |
| $H$ | static enthalpy implied by a trial density, $h_t - m^2/(2\varrho^2)$ |
| $f, g, h$ | characteristic wave amplitudes (downstream acoustic, upstream acoustic, entropy) |
| $\mathbf x, \mathbf R, \mathbf J$ | unknown vector, residual vector, Jacobian matrix |
| $\theta, w, \xi$ | smooth upwind / donor / boundary-regime weights |
| $\varepsilon$ | smoothing width (mass-flow units) |
| $\kappa$, stab | vanishing-friction coefficient / dimensionless homotopy stage |
| $\lambda$ | Levenberg–Marquardt damping parameter |
| $E$ | number of edges |

| term | meaning |
|---|---|
| **residual** | how far an equation is from being satisfied at the current guess; all zeros = solved |
| **Jacobian** | the matrix of sensitivities $\partial R_i/\partial x_j$; Newton's "map" of the system |
| **singular (matrix)** | a Jacobian that has lost information (two equations responding identically); Newton stalls there |
| **Newton's method** | iterative root-finder: repeatedly solve the linearized system for a correction |
| **Levenberg–Marquardt** | damped Newton variant whose every step is guaranteed downhill for the residual norm |
| **homotopy / continuation** | solving a sequence of progressively less-modified problems, each warm-started from the last |
| **upwinding** | taking a transported quantity from the side the flow comes from |
| **stagnation (total) state** | the state the gas would reach if brought to rest losslessly |
| **dynamic head** | $p_t - p \approx \tfrac12\varrho u^2$: the convertible "pressure budget" of the moving gas |
| **choking** | mass-flow saturation when the narrowest section reaches $M = 1$ |
| **jump condition** | algebraic relation between the states on the two sides of a zero-volume element |
| **well-posed** | a problem with exactly as many independent conditions as unknowns — solvable and unambiguous |
