# Representing Standard Catalogue Elements (Orifices, Valves, Nozzles)

How do we take a "standard" restriction — a sharp-edged orifice with a known
discharge coefficient, a control valve with a catalogue $C_v$, a flow nozzle
from ISO 5167 — and build its faithful equivalent inside the `fns`
network, using only the elements we already have?

This document works that out. It is written in the same two voices as
`theory.md`:

> **In plain terms:** these blocks carry the intuition. If you read only
> them, you will understand the argument and the conclusions.

The mathematics after each block makes the statement precise. The companion
references are `theory.md` (the element library, especially §7.2 *isentropic
area change* and §7.3 *sudden area change*) and the two De Domenico papers in
`resources/` on the `dev` branch, whose "generalised nozzle" is the physical
basis for everything below.

---

## Contents

1. [The two questions a restriction answers](#1-the-two-questions-a-restriction-answers)
2. [Static vs. total pressure: the one picture to keep](#2-static-vs-total-pressure-the-one-picture-to-keep)
3. [The sudden-expansion model of an orifice](#3-the-sudden-expansion-model-of-an-orifice)
4. [Where the pressure is measured changes the number you read](#4-where-the-pressure-is-measured-changes-the-number-you-read)
5. [When a single $C_d$ is enough — and the algebra that proves it](#5-when-a-single-c_d-is-enough--and-the-algebra-that-proves-it)
6. [Compressibility and choking](#6-compressibility-and-choking)
7. [The configurations, and which catalogue data describe them](#7-the-configurations-and-which-catalogue-data-describe-them)
8. [Can our strategy represent these? — the verdict](#8-can-our-strategy-represent-these--the-verdict)
9. [Axial swirlers and fuel injectors: where the orifice analogy ends](#9-axial-swirlers-and-fuel-injectors-where-the-orifice-analogy-ends)
10. [Practical recipe: catalogue datum → network element](#10-practical-recipe-catalogue-datum--network-element)
11. [Summary](#11-summary)

---

## 1. The two questions a restriction answers

> **In plain terms:** Any time you put a restriction (a hole, an orifice, a
> half-open valve) into a pipe, an engineer asks two separate questions:
>
> 1. **How much flow do I get for a given pressure drop?** — the *flow
>    capacity*. This is what a flow meter exploits: read the pressure drop,
>    infer the flow.
> 2. **How much pressure have I lost for good?** — the *permanent loss*,
>    the energy turned irreversibly into heat and turbulence that you will
>    never get back.
>
> These sound like the same question. They are not. A flow meter can show a
> large pressure drop while wasting very little energy, because most of that
> drop is *recoverable* — the fluid sped up, and it will slow down again and
> hand the pressure back. The two questions need, in general, **two
> numbers** to answer. The whole subtlety of this document is keeping them
> apart.

In the network's variables (`theory.md` §4) the two questions map cleanly:

* **Flow capacity** is a relation between mass flow $\dot m$ and *static*
  pressures — it is about how fast the gas has to move to squeeze through the
  smallest area.
* **Permanent loss** is the drop in *total* pressure $p_t$ between the
  upstream and downstream ducts — and, by the entropy lemma of §4.4, it is
  exactly the entropy the element generates.

A model that reproduces only the first (a pure flow-capacity curve) gets the
operating point right but carries no information about losses or — crucially
for this project's acoustic goal — about entropy-to-sound conversion. A model
that reproduces both is a faithful element.

---

## 2. Static vs. total pressure: the one picture to keep

> **In plain terms:** Pressure in a moving fluid comes in two parts. The
> **static** pressure is what a flush tap in the pipe wall feels — the
> "squeeze." The **dynamic** pressure, $\tfrac12\rho u^2$, is the "push" the
> motion carries. Their sum is the **total** pressure: what the fluid would
> reach if you brought it gently to rest.
>
> The key fact: **total pressure only ever goes down.** It is the honest
> energy account. Static pressure, by contrast, can go down *and back up* —
> when the fluid speeds up, static converts into dynamic (a loan); when it
> slows down again, dynamic converts back into static (the loan repaid).
> Losses are the part of the *total* pressure that never comes back.

$$
p_t \;=\; \underbrace{p}_{\text{static}} \;+\; \underbrace{\tfrac12\rho u^2}_{\text{dynamic}}
\qquad(\text{low-speed form; the exact compressible } p_t \text{ is in §4.4}).
$$

A flush wall tap reads $p$ (static) only. It never sees the dynamic part
directly. This is why *where you put the tap* matters so much — explained in
§4.

---

## 3. The sudden-expansion model of an orifice

> **In plain terms:** A sharp orifice does two things in sequence. First the
> flow **accelerates** through the hole and keeps contracting a little past
> it, to a narrowest point called the **vena contracta** — narrower than the
> hole itself, because the streamlines cannot turn the sharp corner
> instantly. Up to here almost nothing is lost; the fluid simply traded
> static pressure for speed. Then the fast jet **dumps** into the slower,
> wider pipe downstream: it cannot decelerate smoothly, so it breaks into
> turbulence and mixes back out to fill the pipe. *That* mixing is where the
> permanent loss happens.
>
> So an orifice is, physically, a **smooth contraction followed by a sudden
> expansion** — and those are two elements we already have.

Define the stations (constant-area pipe, $A_1 = A_2 = A_p$ up- and
downstream):

```
        plate
   A_p   ║                                     A_p
 ========║▓▓                                 ========
 flow →  ║  ▓▓▓▓·····jet·····~~~~~~~~~~~~~~   →
 ========║▓▓                                 ========
   [1]   ║   [c]                             [2]
       bore A0   vena contracta A_c = Cc·A0
                 (smallest area, fastest, lowest static p)
```

* **[1]** upstream pipe, area $A_p$, velocity $u_1$, static $p_1$.
* **[c]** vena contracta, the effective minimum area
  $A_c = C_c A_0$ where $A_0$ is the geometric bore and $C_c \le 1$ is the
  **contraction coefficient**. Fastest flow, lowest static pressure $p_c$.
* **[2]** downstream pipe, area $A_p$, velocity $u_2 = u_1$, static $p_2$.

In `fns` terms this is exactly:

$$
\underbrace{\text{isentropic area change (§7.2)}}_{A_1=A_p \;\to\; A_c}
\quad\oplus\quad
\underbrace{\text{sudden expansion / Borda–Carnot (§7.3)}}_{A_c \;\to\; A_2=A_p}.
$$

The contraction is loss-free (§7.3 notes that a contraction modelled by the
same momentum balance would imply an *impossible* entropy decrease, so we
take it as isentropic). All the loss lives in the expansion, where the
Borda–Carnot momentum balance *derives* the loss with no empirical constant.

This is precisely the De Domenico "generalised nozzle": their isentropic
inlet-to-detachment region ($A_1\to A_j$) is our §7.2; their non-isentropic
jet region ($A_j\to A_2$) is our §7.3. For a bare orifice the detachment area
$A_j$ **is** the vena contracta $A_c$, so the two coincide — a simplification
that does *not* hold for a smooth nozzle (see §7).

---

## 4. Where the pressure is measured changes the number you read

> **In plain terms:** Walk a pressure tap along the pipe and watch the
> static reading. It starts at $p_1$, **dives** to a minimum $p_c$ at the
> vena contracta (the fluid is fastest there), then **climbs back** as the
> jet expands and slows, settling at $p_2$, a little below where it started.
> The leftover gap $p_1 - p_2$ is the permanent loss. But the *dive* down to
> $p_c$ is far deeper than that gap — and most of it is recoverable.
>
> So the orifice does not have "a" pressure drop. It has a **recoverable
> dip** (large) and a **permanent loss** (small), and which one your gauge
> reports depends entirely on **where the taps are.**

```
 static
 pressure
   p1 ●────╮
            ╲                    ╭──────────● p2     ← permanent loss = p1 − p2
             ╲                  ╱                       (the honest total-pressure loss)
              ╲                ╱
               ╲____________╱
                    ● p_c                            ← vena contracta minimum
            └────────┬────────┘
              metered dip  Δp_meas = p1 − p_c   (mostly recoverable, NOT loss)
        axial position →
```

This is the heart of the matter, and the reason tap conventions exist in the
standards:

| Tap convention (ISO 5167) | Roughly where it sits | What its $\Delta p$ mainly is |
|---|---|---|
| **Corner taps** / **D and D/2 taps** / **flange taps** | straddling the plate, near the vena contracta | close to the **dip** $p_1-p_c$ → used for **flow metering** |
| **Far-downstream tap** (≳ 6 pipe diameters) | after full reattachment | the **permanent loss** $p_1-p_2$ |

Two practical consequences:

1. **The discharge coefficient $C_d$ is defined against the metering taps**,
   i.e. against the *recoverable dip*. It answers question 1 (flow capacity).
   It is **not** the loss.
2. **The permanent-loss coefficient $K$ (or ISO's "permanent pressure loss")
   is defined against the recovered downstream pressure.** It answers
   question 2 (loss). When the up- and downstream pipes are the same size,
   $u_1=u_2$ and so the permanent *static* drop equals the *total*-pressure
   loss exactly:
   $$
   \underbrace{p_1 - p_2}_{\text{permanent static drop}}
   \;=\;
   \underbrace{p_{t,1} - p_{t,2}}_{\text{total-pressure loss}}
   \qquad (\text{since } \tfrac12\rho u_1^2 = \tfrac12\rho u_2^2).
   $$

> **Why this matters for our model.** The network tracks $p$ on each edge at
> a *uniform-flow* station. Stations [1] and [2] are clean (fully developed
> pipe). But the **metering taps are not at a uniform-flow station** — they
> sit in the contracting/vena-contracta zone, which our compact element does
> not resolve spatially. So a measured $C_d$ cannot be fed in as if it were a
> static-pressure relation between edges. It must first be *interpreted*
> (§5) into the two quantities the model does understand: an **effective
> area** and a **total-pressure loss**. Feeding the metering $\Delta p$
> straight in as a loss would overstate the loss by a large factor.

---

## 5. When a single $C_d$ is enough — and the algebra that proves it

> **In plain terms:** For a sharp orifice, the *one* geometric fact that
> controls everything is the contraction coefficient $C_c$ — how much the
> jet necks down. It sets the smallest area (hence the flow capacity) **and**
> it sets how violent the subsequent dumping is (hence the loss). Because a
> single number controls both, **for a sharp orifice one catalogue value is
> enough** to fix both answers. The catch is that this only works because we
> *assumed* the contraction is loss-free and the jet expands from exactly the
> vena contracta — true for an orifice, false for a smooth nozzle.

Take the incompressible core (compressibility added in §6). Let
$\beta \equiv A_c/A_p = C_c\,(A_0/A_p)$ be the **effective jet-area ratio**
(De Domenico's loss parameter — *not* the ISO diameter ratio; see the warning
in §9). With $u_2 = \dot m/(\rho A_p)$:

**Contraction [1]→[c], loss-free (Bernoulli):**
$$
p_1 - p_c \;=\; \tfrac12\rho\big(u_c^2 - u_1^2\big)
\;=\; \tfrac12\rho u_2^2\Big(\tfrac{1}{\beta^2} - 1\Big).
$$
This is the **metering dip** — it gives the flow-capacity relation. Inverting
it is exactly the flow-meter equation; the catalogue wraps the constants
(plus the small tap-position correction) into $C_d$, with $C_d \approx C_c$
in the ideal limit.

**Expansion [c]→[2], Borda–Carnot (momentum, §7.3):** the static pressure
*rises* by
$$
p_2 - p_c \;=\; \rho u_2^2\Big(\tfrac{1}{\beta} - 1\Big).
$$

**Permanent loss** = dip minus recovery:
$$
\boxed{\;
p_1 - p_2 \;=\; \tfrac12\rho u_2^2\Big(\tfrac{1}{\beta} - 1\Big)^2
\;\;\Longrightarrow\;\;
K_{\text{pipe}} = \Big(\tfrac{1}{\beta} - 1\Big)^2 .
\;}
$$

This is the De Domenico result $C_{p0} = (1-\beta)^2$ (their Eq. 8) rewritten
against the pipe dynamic head. Two illuminating ratios fall straight out:

$$
\underbrace{\frac{p_1 - p_2}{p_1 - p_c}}_{\text{loss / metered dip}}
= \frac{1-\beta}{1+\beta},
\qquad
\underbrace{\frac{p_2 - p_c}{p_1 - p_c}}_{\text{fraction recovered}}
= \frac{2\beta}{1+\beta}.
$$

> **Read these two fractions.** For a strong restriction ($\beta \to 0$,
> small hole): almost nothing is recovered, the dip is *almost all* permanent
> loss. For a mild one ($\beta \to 1$, nearly full bore): the recovered
> fraction $\to 1$ and the loss $\to 0$ — the dip was a loan, fully repaid.
> The same shape is the ISO 5167 "ratio of permanent pressure loss to
> differential pressure," which is why ISO can publish a loss curve as a
> function of geometry alone.

**The sufficiency claim, stated carefully.** Everything above is a function
of the single parameter $\beta = C_c\,(A_0/A_p)$. Since the catalogue $C_d$
fixes $C_c$ (the rest of $C_d$'s job is the tap-position correction), and the
geometry fixes $A_0/A_p$, **one number $C_d$ + the geometry determines both
the flow capacity and the loss — for a sharp orifice.** This is the case the
title of §5 promised.

It works **only** because two assumptions held: (i) the contraction was
loss-free, and (ii) the jet expands from exactly the vena contracta with no
geometric diffuser helping it recover. Break either — a thick orifice with
internal friction, or a smooth nozzle with a gentle diffuser — and $\beta$ is
no longer tied to $C_c$. Then $C_d$ still gives you the flow capacity, but the
loss needs a **second, independent** number (§7).

---

## 6. Compressibility and choking

> **In plain terms:** Everything above was written for low-speed
> (incompressible) flow to keep the algebra clean. Two things change at high
> speed, and our elements already handle both.

1. **Density changes through the dip.** Catalogue practice folds this into an
   **expansibility factor** $\varepsilon \le 1$ multiplying the flow
   equation. ISO 5167 publishes $\varepsilon$ for orifices; for our model the
   compressible state recovery (§4.2) handles it natively — we do not need
   $\varepsilon$, we need the *area*. (The De Domenico 2019 paper is explicit
   that there is no universal closed-form compressible loss law; the loss
   parameter is calibrated, not derived.)

2. **The effective throat can choke.** When the vena contracta reaches
   $M=1$, the orifice chokes and the discharge becomes underexpanded — the
   exit pressure detaches upward from the back pressure. This is exactly the
   emergent behaviour the §7.2 element produces through its Fischer–Burmeister
   complementarity (`theory.md` §11.2). Important consequence: it is the
   **vena-contracta area $A_c$** that chokes, not the geometric bore $A_0$ —
   so a real orifice chokes at a *lower* mass flow than a hole of area $A_0$
   would. The 2019 paper makes the same point (the non-isentropic nozzle
   chokes at lower flow than an equivalent orifice plate of area $A_j$).

So the choking element must be sized to the **effective** area $A_c = C_c A_0$,
not the geometric one — another reason the $C_d$-to-area interpretation
matters.

---

## 7. The configurations, and which catalogue data describe them

> **In plain terms:** "Sharp orifice in a pipe" is the easy case where one
> number does everything. Real catalogues cover a spectrum, and as we move
> along it the single number stops being enough. Here is the spectrum, with
> what data each end actually needs.

Throughout, **P1** = "effective flow/choking area" (the question-1 parameter,
$A_c = C_c A_0$) and **P2** = "permanent-loss parameter" (the question-2
parameter, $\beta$ or $K$ or $C_{p0}$).

| # | Configuration | Catalogue data usually available | P1 from | P2 from |
|---|---|---|---|---|
| A | **Sharp-edged orifice, equal pipe up/down** (classic ISO 5167) | $C_d(\beta_d, Re)$, expansibility $\varepsilon$, permanent-loss curve | $C_d$ | derivable from $C_d$ (Borda–Carnot) **or** ISO permanent-loss curve |
| B | **Orifice / hole discharging to a large plenum or atmosphere** ($A_2 \to \infty$) | $C_d$ of the hole | $C_d$ | *fixed by geometry*: full dynamic-head loss ($\beta \to \beta_{\min}$) |
| C | **Smooth flow nozzle / venturi** (ISO 1932, venturi) | $C_d \approx 0.98\text{–}0.99$, separately quoted permanent loss (~10–20% of $\Delta p$) | $C_d$ (≈ geometric throat) | **separate** small loss — *not* derivable from $C_d$ |
| D | **Thick / finite-length orifice** | $C_d$ (geometry-specific) | $C_d$ | needs separate loss; internal friction breaks Borda–Carnot |
| E | **Control valve** at a given opening | $C_v$ / $K_v$ (flow coeff.), $F_L$ (pressure-recovery factor) | $C_v$/$K_v$ | $F_L$ — *exactly* a P2 parameter |
| F | **Perforated / multi-hole plate** | open-area ratio, per-hole $C_d$ | aggregate open area × $C_d$ | per-hole Borda–Carnot **plus** jet-merging correction |
| G | **Axial swirler / injector** dumping to combustor | effective area / flow number, measured $\Delta p_t$, vane angle $\alpha$ | flow number ($C_d A$) | **measured** loss (≈ $\sec^2\alpha$ of axial head); see §9 |

Notes that decide the verdict:

* **A** is the self-similar sweet spot of §5: P1 and P2 collapse onto one
  $C_c$. ISO is the richest source because it *also* publishes the permanent
  loss directly, so you need not even rely on the Borda–Carnot link — you can
  cross-check it.
* **B** is De Domenico's "convergent nozzle terminating a duct" limit
  ($A_2\to\infty$, Fig. 2c of either paper). The jet's entire dynamic head is
  dissipated in the plenum, so P2 is pinned by geometry — one number ($C_d$)
  again suffices for the mean flow.
* **C** is where the single-number story **fails**: $C_d\approx1$ tells you
  the throat is nearly the geometric area, but the gentle diffuser recovers
  most of the dynamic head, so the loss is small and set by the diffuser
  quality, *independently* of $C_d$. You must have the separately quoted
  loss. In De Domenico language: $A_j \to A_2$ (high recovery), well away
  from the vena contracta.
* **E** is the pleasant surprise: valve catalogues already ship the
  **two-parameter** description — $C_v$ for capacity, $F_L$ for recovery.
  This is independent evidence that two scalars is the right abstraction, not
  an artefact of the nozzle papers.
* **F** is the one genuine question mark: a bank of jets may merge and
  recover differently than a single equivalent jet, so the aggregate $\beta$
  is not simply the open-area ratio. May need a third descriptor or an
  empirical correction.

---

## 8. Can our strategy represent these? — the verdict

> **In plain terms:** The strategy is: build every standard element as
> *isentropic contraction (§7.2) followed by sudden expansion (§7.3)*, sized
> by two numbers — an effective throat area and a loss parameter. The
> question is whether that shape is rich enough for the whole catalogue. The
> answer is **yes for A, B, D, E without reservation; yes for C provided the
> loss is supplied separately; and yes for F with a caveat on multi-jet
> recovery.**

| Config | Representable by §7.2 ⊕ §7.3? | Parameters needed | Caveat |
|---|---|---|---|
| A — sharp orifice | **Yes** | $C_d$ alone | none; this is the exact case the model was derived for |
| B — discharge to plenum | **Yes** | $C_d$ alone | use §7.6 pressure outlet for the dump; $\beta$ pinned by $A_2\to\infty$ |
| C — smooth nozzle/venturi | **Yes** | $C_d$ **and** a loss datum | single $C_d$ is *insufficient*; must supply the recovery/loss |
| D — thick orifice | **Yes** | $C_d$ **and** a loss datum | Borda–Carnot link broken by internal friction |
| E — control valve | **Yes** | $C_v/K_v$ **and** $F_L$ | maps one-to-one; already the catalogue's own two parameters |
| F — perforated plate | **Mostly** | open area, per-hole $C_d$, (+?) | aggregate $\beta$ ≠ open-area ratio if jets merge |
| G — axial swirler | **Mean flow yes; acoustics partial** | effective area **and** measured loss | swirl field and acoustic-to-swirl conversion are outside the 1-D model; see §9 |

Three structural points underpin every "yes":

1. **The shape is exactly the validated physics.** §7.2 ⊕ §7.3 *is* the De
   Domenico generalised nozzle. We are not approximating their model; we are
   instantiating it from our primitives.

2. **The acoustics come for free.** Because the element is built from the
   real conservation primitives (not a lumped loss coefficient), linearising
   it at the operating point — the framework's §9/§12 machinery — reproduces
   the De Domenico acoustic and entropic transfer functions ($R, T, S_R,
   S_T$) automatically. A pure loss-coefficient element (§7.4) would match
   the mean $\Delta p$ but carry **no** valid entropy-to-sound conversion, so
   it is a *mean-flow-only* fast path, not a substitute.

3. **Two scalars, calibrated not derived.** The only inputs are an effective
   area (P1) and a loss parameter (P2). For config A they fuse into one; for
   C/D/E they stay distinct. There is no compressible loss law to chase — the
   loss is a measured/catalogue number, which is precisely how De Domenico
   2019 says it must be supplied.

The one place to keep eyes open is **F** (multi-hole), where the per-hole
composition may not aggregate linearly — worth a dedicated check before
trusting it.

---

## 9. Axial swirlers and fuel injectors: where the orifice analogy ends

> **In plain terms:** A swirler is a ring of angled vanes that spins the
> incoming air before it dumps into the combustor. The spin is deliberate —
> it creates a central recirculation zone that anchors the flame. From a
> plumbing standpoint a swirler still "drops pressure for a given flow," so
> it looks like one more restriction. But the *reason* it drops pressure is
> different from an orifice, and that difference decides how faithfully a 1-D
> network can stand in for it. (The liquid-fuel port is irrelevant to the
> air-side pressure drop and is ignored here.)

### 9.1 Why a swirler loses pressure (and why it isn't Borda–Carnot)

An orifice loses total pressure because a purely *axial* jet dumps into a
wider space and the axial dynamic head mixes out (§3, §5). A swirler loses
total pressure for two further reasons that have **no orifice analogue**:

1. **Vane turning.** The vanes turn the flow from axial to a helix at vane
   angle $\alpha$. Turning a flow costs total pressure — profile drag on the
   vanes, secondary flows in the passage corners, and (at high $\alpha$)
   separation on the vane suction surfaces. This is a *cascade / blade-row*
   loss, closer to a turbomachinery row than to a sudden expansion.

2. **Unrecovered swirl kinetic energy.** At exit the flow carries a
   tangential velocity $u_\theta \approx u_x\tan\alpha$ on top of its axial
   velocity $u_x$, so the exit dynamic head is
   $$
   \tfrac12\rho\big(u_x^2 + u_\theta^2\big)
   = \tfrac12\rho u_x^2\big(1+\tan^2\alpha\big)
   = \tfrac12\rho u_x^2\,\sec^2\alpha .
   $$
   The tangential part is **by design not recovered** — it is spent driving
   the recirculation zone in the combustor. So the swirler throws away
   roughly $\sec^2\alpha$ times the dynamic head a purely axial jet of the
   same axial velocity would lose. A $45^\circ$ swirler: $\times 2$; a
   $60^\circ$ swirler: $\times 4$.

This is the central physical difference: **an orifice loses (mostly) the
axial dynamic head it cannot help losing; a swirler additionally loses the
tangential dynamic head it was built to create.** The loss is larger, and it
is *deliberate*, so it cannot be derived from a contraction coefficient the
way the orifice loss was in §5. It must be measured.

### 9.2 The link between the two questions breaks

For a sharp orifice, one number ($C_c$) fixed both flow capacity and loss
(§5). For a swirler that link is gone:

* **Flow capacity** is set by the smallest passage area (the vane throat or
  the annular gap), expressed in gas-turbine practice as an **effective area**
  or **flow number** $\mathrm{FN}\propto \dot m\sqrt{T_t}/p_t$ — the same
  quantity as $C_d A$.
* **Loss** is set independently by the vane angle and aerodynamics
  ($\sim\sec^2\alpha$ plus turning losses), and is *much larger* than a
  Borda–Carnot expansion from that same effective area would predict.

So a swirler is firmly a **two-parameter** device — effective area *and* a
measured loss — with the two no longer derivable from one another. In the §7
table it resembles the "dump to plenum" case **B**, but with the dynamic head
inflated by the swirl factor.

### 9.3 The orifice-with-swirler-dumping-to-a-combustor setup

The concrete arrangement: a passage (the "orifice") fitted with a swirler,
discharging into the combustor. Seen from the network this is a single series
path:

```
 feed plenum     metering passage      swirl vanes        combustor
 (p_t, T_t)  →   (sets effective area) → (adds swirl,    → (large volume:
                                          turning loss)     jet + tangential
                                                            KE dissipate in
                                                            the recirculation)
```

* The **metering passage** sets the effective area → an isentropic
  contraction to $A_{\text{eff}}$ (§7.2), which also fixes **choking** (the
  vane throat can choke at $M=1$; §6).
* The **swirl + dump** is the loss. Physically it is *not* a clean
  reattaching jet (§3): the swirling jet spreads wide and, above a swirl
  number $S\approx 0.6$, undergoes **vortex breakdown** into a central
  toroidal recirculation zone. The network cannot resolve any of that; it
  sees only the lumped total-pressure loss.

So in `fns` terms the faithful build is:

$$
\underbrace{\text{§7.2 contraction to } A_{\text{eff}}}_{\text{capacity + choking}}
\;\oplus\;
\underbrace{\text{a loss with the \emph{measured} } \Delta p_t}_{\text{not the Borda–Carnot value}} .
$$

Practically: keep the §7.2 element for the area, and impose the measured loss
either through the §7.4 concentrated-loss element (a measured $K$) or by
inflating the §7.3 expansion's discharged dynamic head by $\sec^2\alpha$ when
the vane angle is known. The Borda–Carnot *derivation* of §5 is **overridden
by measurement** here — that is the key procedural difference from the
orifice.

### 9.4 How well it is represented — and the hard limits

**What the network captures well:**

* The **mean** mass-flow-vs-pressure-drop characteristic — exactly, given the
  measured effective area + loss. This is usually the engineering quantity of
  interest for a flow-split / pressure-balance network.
* **Choking** at the vane throat.
* The **longitudinal acoustic** scattering (reflection/transmission of plane
  acoustic waves), to the same fidelity as any loss element — often adequate
  for the bulk combustor acoustic network.

**What it fundamentally cannot capture:**

1. **The swirl field itself.** The network is 1-D and carries no angular
   momentum. The tangential velocity, the swirl number, the recirculation
   zone the flame actually sits in — all invisible. The swirler is, to the
   network, a scalar resistance. It can *reproduce* a measured loss; it can
   never *predict* one, because the loss is a 3-D aerodynamic outcome.

2. **Acoustic-to-swirl conversion — a fourth wave the framework has no slot
   for.** This is the deepest limitation, and the one most relevant to the
   project's thermoacoustic goal. The framework carries three characteristic
   waves: downstream acoustic, upstream acoustic, and convected entropy
   (`theory.md` §9). A swirler converts incoming acoustic perturbations into
   **convected azimuthal-velocity (vorticity) fluctuations** that travel to
   the flame with a convective time lag and modulate the heat release — a
   well-established contributor to the flame transfer function in swirl
   combustors. This swirl wave is a *fourth* degree of freedom, outside the
   $(f,g,h)$ triple. The De Domenico machinery extends acoustics to entropy
   waves but **not** to vorticity waves, so the swirler's most important
   thermoacoustic role is structurally beyond the current model.

3. **Operating-point-dependent effective area.** Vortex breakdown can set in,
   shift, or show hysteresis with flow rate, so a swirler's effective area is
   not always a clean constant — the single-curve assumption degrades near
   breakdown onset.

### 9.5 Differences against the orifice, at a glance

| | Sharp orifice (§3–5) | Axial swirler / injector |
|---|---|---|
| Exit velocity | axial only | axial **+** tangential ($\sec^2\alpha$ more dynamic head) |
| Loss mechanism | Borda–Carnot from vena contracta | vane turning + deliberate swirl dissipation |
| Capacity ↔ loss link | one number ($C_c$) fixes both | **decoupled**; loss must be measured |
| Downstream state | reattaching jet → uniform pipe | vortex breakdown → 3-D recirculation zone |
| Conserved quantity ignored | none of consequence | **angular momentum** |
| Extra wave generated | entropy (captured) | **swirl / vorticity (not captured)** |
| Predict from geometry? | yes (sharp orifice) | **no** — calibrate from measured $\Delta p_t$ |

**Net:** a swirler is representable as a network element *for what a network
is for* — mass flow, pressure split, choking, and bulk longitudinal
acoustics — using a measured effective area plus a measured loss. It is **not**
representable where its purpose lives: the swirl field, the recirculation
zone, and the acoustic-to-swirl flame-forcing mechanism. Use it as a
calibrated two-parameter resistance, and treat its flame-coupling acoustics as
out of scope for this framework.

---

## 10. Practical recipe: catalogue datum → network element

A thin synthesis layer would perform this mapping. No code here — just the
contract.

**Inputs (one of):**

* sharp orifice: $C_d$, bore $d$, pipe $D$ (→ ISO diameter ratio
  $\beta_d = d/D$), fluid state;
* nozzle/venturi: $C_d$, throat & pipe diameters, **permanent-loss fraction**;
* valve: $C_v$ or $K_v$, $F_L$, port size;
* swirler/injector: effective area / flow number, **measured** $\Delta p_t$
  (and vane angle $\alpha$ if available) — the loss is *not* derived (§9);
* generic: effective area + a loss coefficient $K$ (any reference) + its
  reference area.

**Mapping:**

1. **Effective throat area** $A_c = C_c A_0$. For an orifice, recover $C_c$
   from $C_d$ (with the velocity-of-approach factor $1/\sqrt{1-\beta_d^4}$);
   for a nozzle, $A_c \approx$ geometric throat. → sizes the §7.2 element and
   sets the **choking** area.
2. **Loss parameter** $\beta = A_c/A_p$ (orifice) or from the supplied loss:
   invert $C_{p0}=(1-\beta)^2$, or convert $K$, or read $F_L$. → sizes the
   §7.3 sudden expansion.
3. **Assemble** §7.2 ($A_p \to A_c$) ⊕ §7.3 ($A_c \to A_p$, or $\to$ §7.6
   outlet for a plenum discharge).

**Two cautions that will otherwise cause silent errors:**

* **The two $\beta$'s.** ISO $\beta_d = d/D$ is a *diameter* ratio and keys
  the $C_d$ correlation. De Domenico $\beta = A_j/A_2$ is an *area* ratio and
  keys the loss. They differ by roughly a square and a $C_c$. Name them apart
  (`beta_iso`, `beta_loss`) everywhere.
* **Reference-area bookkeeping.** $C_d$, $K$, $C_{p0}$, $F_L$ are each
  referenced to a specific area or dynamic head (bore vs. pipe vs. jet). Every
  conversion is a continuity-based area-ratio step — individually trivial,
  collectively the most likely place to drop a factor. Fix one internal
  convention (jet dynamic head, as De Domenico) and convert everything to it
  on the way in.

---

## 11. Summary

* A restriction answers **two** questions — *flow per pressure drop* and
  *permanent loss* — and in general needs **two** numbers.
* Physically an orifice is an **isentropic contraction followed by a sudden
  expansion**: our §7.2 ⊕ §7.3, which *is* the De Domenico generalised
  nozzle.
* **Where you tap the pressure** decides which number you read: metering taps
  (corner/flange/D-D/2) catch the large *recoverable dip* and define $C_d$;
  far-downstream taps catch the small *permanent loss* and define $K$. They
  are not the same drop, and confusing them overstates the loss badly.
* For a **sharp orifice**, a single contraction coefficient $C_c$ controls
  both questions, so **one catalogue $C_d$ is sufficient** to fix capacity and
  loss together. This breaks for smooth nozzles, thick orifices and valves,
  which need a second, independent loss/recovery datum.
* The strategy represents the **entire** catalogue spectrum (A–F) with the
  same two-element composition; the only varying input is the parameter pair,
  which fuses to one for sharp orifices and stays two elsewhere.
* Because the element is built from real primitives, the **acoustic and
  entropic transfer functions are inherited for free** — the reason to prefer
  this composition over a lumped loss coefficient.
* **Axial swirlers** (§9) extend the scheme but mark its boundary: their loss
  must be *measured*, not derived (it includes the deliberately unrecovered
  swirl kinetic energy, $\sim\sec^2\alpha$); their mean flow, choking and bulk
  acoustics are representable, but the swirl field and the **acoustic-to-swirl
  flame-forcing** mechanism are a fourth wave outside the 1-D framework.
