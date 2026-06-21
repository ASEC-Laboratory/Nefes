"""Network container and residual assembly.

Unknown vector layout: x = [mdot_0, p_0, ht_0, mdot_1, p_1, ht_1, ...]
(three unknowns per edge, mdot signed along the edge direction).

Equations, in order:
  1. element equations (each element contributes exactly n_ports residuals),
  2. one total-enthalpy advection equation per edge:

         ht_e = theta(mdot_e) * H_donor(tail) + (1 - theta(mdot_e)) * H_donor(head)

     with theta a smooth step.  This is first-order upwinding of the advected
     scalar on the graph; at converged unidirectional flow it reduces to exact
     upwind transport (errors O((eps/mdot)^2)), at mdot = 0 it stays regular.
"""

import numpy as np

from .state import recover_state
from .smooth import smooth_step
from .elements import Port, KIND_MASS, KIND_PRESSURE, KIND_ENTHALPY


class Edge:
    __slots__ = ("index", "tail", "head", "area", "name")

    def __init__(self, index, tail, head, area, name=""):
        self.index = index
        self.tail = tail  # element index
        self.head = head  # element index
        self.area = area
        self.name = name or f"e{index}"


class Network:
    def __init__(self, gas, p_ref: float = 101325.0, T_ref: float = 300.0, mdot_ref: float = None):
        self.gas = gas
        self.p_ref = p_ref
        self.T_ref = T_ref
        self._mdot_ref = mdot_ref
        # Fischer-Burmeister corner width for the choking complementarities.
        # Default matches the documented 1e-5; exposed so it can be swept
        # (it controls the O(eps_fb^{2/3}) sonic-corner offset).
        self.eps_fb = 1e-5
        self.elements = []
        self.edges = []
        # per element: list of (edge_index, sigma) in attachment order
        self._ports = []

    # -- construction -------------------------------------------------------

    def add(self, element) -> int:
        self.elements.append(element)
        self._ports.append([])
        return len(self.elements) - 1

    def connect(self, tail: int, head: int, area: float, name: str = "") -> int:
        """Add an edge directed from element `tail` to element `head`."""
        e = Edge(len(self.edges), tail, head, area, name)
        self.edges.append(e)
        self._ports[tail].append((e.index, +1))
        self._ports[head].append((e.index, -1))
        return e.index

    # -- references / scales -------------------------------------------------

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def n_unknowns(self):
        return 3 * self.n_edges

    @property
    def h_ref(self):
        return self.gas.cp * self.T_ref

    @property
    def mdot_ref(self):
        if self._mdot_ref is not None:
            return self._mdot_ref
        # largest specified mass flow, else a Mach ~ 0.3 guess on the median area
        specs = [abs(el.mdot) for el in self.elements if hasattr(el, "mdot")]
        if specs and max(specs) > 0.0:
            return max(specs)
        rho = self.p_ref / (self.gas.R * self.T_ref)
        c = np.sqrt(self.gas.gamma * self.gas.R * self.T_ref)
        a_med = float(np.median([e.area for e in self.edges]))
        return 0.3 * rho * c * a_med

    @property
    def eps_mdot(self):
        """Regularization scale for upwinding/mixing (see smooth.py)."""
        return 1e-3 * self.mdot_ref

    def variable_scales(self):
        s = np.empty(self.n_unknowns)
        s[0::3] = self.mdot_ref
        s[1::3] = self.p_ref
        s[2::3] = self.h_ref
        return s

    def residual_scales(self):
        kind_scale = {
            KIND_MASS: self.mdot_ref,
            KIND_PRESSURE: self.p_ref,
            KIND_ENTHALPY: self.h_ref,
            "momentum": self.p_ref,
        }
        scales = []
        for el in self.elements:
            scales.extend(kind_scale[k] for k in el.eq_kinds)
        scales.extend([self.h_ref] * self.n_edges)  # advection rows
        return np.asarray(scales)

    # -- state & residual -----------------------------------------------------

    def check_square(self):
        """Validate the equation bookkeeping.

        Standard rule: one equation per element port, plus one advection
        equation per edge -> exactly 3E rows for 3E unknowns.  Supersonic
        boundaries (SupersonicInlet) legitimately supply one extra row each
        (all characteristics enter the domain there); the resulting formal
        over-determination is absorbed by the self-vacating complementarity
        of the sonic-fed element downstream (see elements.SupersonicInlet)
        and handled natively by the least-squares Newton step.
        """
        n_eq = 0
        for el, ports in zip(self.elements, self._ports):
            expected = el.n_ports
            if expected is not None and len(ports) != expected:
                raise ValueError(
                    f"element '{el.name}' has {len(ports)} connected edges, expects {expected}"
                )
            n_rows = len(el.eq_kinds)
            if n_rows < len(ports):
                raise ValueError(
                    f"element '{el.name}' supplies {n_rows} equations "
                    f"but has {len(ports)} ports (rule: at least one equation per port)"
                )
            n_eq += n_rows
        n_eq += self.n_edges
        if n_eq < self.n_unknowns:
            raise ValueError(f"under-determined system: {n_eq} equations, {self.n_unknowns} unknowns")

    def states(self, x):
        """Recover all edge states from the unknown vector (complex-step safe)."""
        out = []
        for e in self.edges:
            i = 3 * e.index
            out.append(recover_state(x[i], x[i + 1], x[i + 2], e.area, self.gas))
        return out

    def residual(self, x, stab: float = 0.0):
        """Full residual vector (unscaled, physical units).

        ``stab`` is the dimensionless vanishing-friction homotopy parameter:
        interior pressure rows receive a linear resistance
        ``stab * p_ref / mdot_ref * mdot_through``.  stab = 0 gives the exact
        equations (the solver always finishes at stab = 0).
        """
        gas = self.gas
        # Smoothing width follows the homotopy: wide while the friction
        # stabilization is active (so regime switches/upwinding do not act as
        # near-discontinuities while the iterate may cross mdot = 0), sharp in
        # the exact final stage (stab = 0), by which point the iterate sits at
        # |mdot| >> eps and the wide/sharp forms agree.
        eps = max(stab * 0.3, 1e-4) * self.mdot_ref
        # Fischer-Burmeister corner width for the choking complementarities.
        # Fixed and small: the FB residual is globally smooth for any eps, so
        # no homotopy widening is needed -- and widening is actively harmful,
        # because the regime bias eps^2/2 * pt (a fictitious total-pressure
        # loss) must stay far below the smallest driving pressure difference
        # in the network.
        eps_fb = self.eps_fb
        stab_coeff = stab * self.p_ref / self.mdot_ref
        states = self.states(x)

        ports_of = [
            [Port(states[ei], sigma, self.edges[ei].area, ei) for ei, sigma in plist]
            for plist in self._ports
        ]

        res = []
        for el, ports in zip(self.elements, ports_of):
            res.extend(el.equations(ports, gas, eps, stab_coeff, eps_fb))

        # donor enthalpies, then one advection equation per edge
        donor = [el.donor_enthalpy(ports, gas, eps) for el, ports in zip(self.elements, ports_of)]
        for e in self.edges:
            theta = smooth_step(states[e.index].mdot, eps)
            h_up = theta * donor[e.tail] + (1.0 - theta) * donor[e.head]
            res.append(states[e.index].ht - h_up)

        return np.asarray(res)

    def initial_guess(self, mdot0: float = None, p0: float = None, Tt0: float = None):
        """Uniform initial state; a small co-directional mdot by default."""
        x = np.empty(self.n_unknowns)
        x[0::3] = 0.05 * self.mdot_ref if mdot0 is None else mdot0
        x[1::3] = self.p_ref if p0 is None else p0
        x[2::3] = self.h_ref if Tt0 is None else self.gas.cp * Tt0
        return x

    # -- reporting -------------------------------------------------------------

    def report(self, x) -> str:
        states = self.states(x)
        cols = ["edge", "tail->head", "area", "mdot", "u", "M", "p", "p_t", "T", "T_t", "rho"]
        rows = []
        for e in self.edges:
            st = states[e.index]
            rows.append(
                [
                    e.name,
                    f"{self.elements[e.tail].name}->{self.elements[e.head].name}",
                    f"{e.area:.4g}",
                    f"{st.mdot:.6g}",
                    f"{st.u:.6g}",
                    f"{st.M:.4f}",
                    f"{st.p:.7g}",
                    f"{st.pt:.7g}",
                    f"{st.T:.6g}",
                    f"{st.Tt:.6g}",
                    f"{st.rho:.5g}",
                ]
            )
        widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
        lines = ["  ".join(c.rjust(w) for c, w in zip(cols, widths))]
        for r in rows:
            lines.append("  ".join(c.rjust(w) for c, w in zip(r, widths)))
        return "\n".join(lines)

    def choking_report(self, x, threshold: float = 0.9):
        """Edges running close to (or beyond) their choking limit.

        Returns a list of (edge, |M|, flux_ratio) where flux_ratio is the
        local mass flux over the choking mass flux at the edge's total state,
        for edges with |M| >= threshold.  A non-converged solve with entries
        pinned near M = 1 is the signature of boundary data demanding more
        than a passage can carry (docs/theory.md section 11).
        """
        g = self.gas.gamma
        flux_star = np.sqrt(g / self.gas.R) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
        out = []
        for e, st in zip(self.edges, self.states(x)):
            m_choke = st.pt / np.sqrt(st.Tt) * flux_star * e.area
            ratio = abs(st.mdot) / m_choke
            if abs(st.M) >= threshold:
                out.append((e, abs(st.M), ratio))
        return out

    def conservation_report(self, x):
        """Per-element mass and energy imbalance (energy via mdot-weighted ht)."""
        states = self.states(x)
        out = []
        for el, plist in zip(self.elements, self._ports):
            dm = sum(sigma * states[ei].mdot for ei, sigma in plist)
            de = sum(sigma * states[ei].mdot * states[ei].ht for ei, sigma in plist)
            out.append((el.name, dm, de))
        return out
