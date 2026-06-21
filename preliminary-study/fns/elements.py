"""Network elements (nodes): zero-volume control volumes imposing jump conditions.

Equation counting rule (see REVIEW.md for the derivation):

    **Every element supplies exactly as many equations as it has ports.**

    * interior element with n ports:  1 mass balance + (n-1) pressure-type
      relations (total-pressure equality, momentum balance, static-pressure
      equality, loss correlation, ...)
    * boundary element (1 port):      1 specification (mdot, p, or p_t)

    The advection of total enthalpy is NOT an element equation: each *edge*
    carries one smoothly-upwinded advection equation (assembled by the
    Network).  Elements only provide the "donor" enthalpy they would hand to
    an edge that draws from them.

    Element equations + edge equations = 2E + E = 3E = number of unknowns,
    a square system *independently of the flow direction*.  (In the classical
    characteristics counting the number of jump conditions an element must
    supply changes discretely when the flow through a port reverses --
    n_acoustic + n_outflow_entropy -- which is what made flow-reversal and
    near-zero-flow states structurally singular in the earlier prototypes.)

Conventions
-----------
``sigma = +1`` if the edge is directed *away* from the element (element is the
tail), ``-1`` if directed towards it.  Hence ``sigma * mdot`` is the flow rate
*leaving* the element through that port, and mass balance reads
``sum(sigma_i * mdot_i) = 0``.

All residuals must be smooth in the solution variables (no abs/sign/max --
use fns.smooth) so that Newton with complex-step Jacobians is exact and
flow reversal does not create kinks.
"""

import numpy as np

from .smooth import smooth_pos, smooth_step, fischer_burmeister

# Equation kinds, used by the network/solver for row scaling.
KIND_MASS = "mass"  # scaled by mdot_ref
KIND_PRESSURE = "pressure"  # scaled by p_ref
KIND_ENTHALPY = "enthalpy"  # scaled by h_ref
KIND_MOMENTUM = "momentum"  # scaled by p_ref (residual divided by area internally)


class Port:
    """View of one element port: recovered edge state + orientation."""

    __slots__ = ("state", "sigma", "area", "edge")

    def __init__(self, state, sigma, area, edge):
        self.state = state
        self.sigma = sigma  # +1 edge leaves element, -1 edge enters element
        self.area = area
        self.edge = edge  # global edge index

    @property
    def mdot_out(self):
        """Signed flow rate leaving the element through this port."""
        return self.sigma * self.state.mdot

    @property
    def mdot_in(self):
        return -self.sigma * self.state.mdot


class Element:
    """Base class. Subclasses define n_ports, eq_kinds, equations(), donor_enthalpy()."""

    n_ports: int = None  # None -> variable (set at construction)
    name: str = ""

    @property
    def eq_kinds(self):
        raise NotImplementedError

    def equations(self, ports, gas, eps_mdot, stab=0.0, eps_fb=1e-5):
        """Return list of residuals, len == number of ports.

        ``stab`` [Pa/(kg/s)] is a vanishing-friction homotopy coefficient set
        by the solver.  Interior pressure-type rows must include a linear
        resistance term ``- stab * mdot_through`` (see REVIEW.md: near
        mdot = 0 the dynamic head is quadratic in mdot, making zero flow a
        stationary point of the residual landscape in purely pressure-driven
        networks; the homotopy removes the trap and is driven to exactly zero
        in the final solver stage, so the converged equations are unmodified).
        """
        raise NotImplementedError

    def donor_enthalpy(self, ports, gas, eps_mdot):
        """Total enthalpy this element offers to an edge drawing from it.

        Interior elements: mass-flow-weighted mix of the *incoming* port
        enthalpies (smoothly upwinded, regularized so it stays defined at
        zero flow).  Boundary elements override with their specification.
        """
        # Smooth-weight ratio: w = smooth_pos(inflow) >= eps/2 keeps the mix
        # defined at zero flow (-> plain average); at converged flow the
        # outflow-port weights decay like eps^2/(4 mdot), so the mixing bias
        # is only O((eps/mdot)^2).
        w_sum = 0.0
        wh_sum = 0.0
        for prt in ports:
            w = smooth_pos(prt.mdot_in, eps_mdot)
            w_sum = w_sum + w
            wh_sum = wh_sum + w * prt.state.ht
        return wh_sum / w_sum

    def _mass_balance(self, ports):
        r = 0.0
        for prt in ports:
            r = r + prt.mdot_out
        return r


# ---------------------------------------------------------------------------
# Boundary elements
# ---------------------------------------------------------------------------


class MassFlowInlet(Element):
    """Specified mass flow rate into the network and total temperature.

    mdot > 0 means flow *into* the network.
    """

    n_ports = 1

    def __init__(self, mdot: float, Tt: float, name: str = "mass-flow-inlet"):
        self.mdot = mdot
        self.Tt = Tt
        self.name = name

    @property
    def eq_kinds(self):
        return [KIND_MASS]

    def equations(self, ports, gas, eps_mdot, stab=0.0, eps_fb=1e-5):
        return [ports[0].mdot_out - self.mdot]

    def donor_enthalpy(self, ports, gas, eps_mdot):
        return gas.cp * self.Tt


class TotalPressureInlet(Element):
    """Reservoir at specified total pressure and total temperature.

    Direction-aware (smoothly blended, like a CFD pressure inlet):
      * inflow to the network: stream total pressure = pt (isentropic draw
        from the reservoir),
      * reversed flow (network discharges into the reservoir): stream static
        pressure = pt (jet dumping its dynamic head into the reservoir).

    Without the blend the reversed regime is over-determined (an arriving
    stream cannot lower its total pressure losslessly) and no steady solution
    exists.
    """

    n_ports = 1

    def __init__(self, pt: float, Tt: float, name: str = "total-pressure-inlet"):
        self.pt = pt
        self.Tt = Tt
        self.name = name

    @property
    def eq_kinds(self):
        return [KIND_PRESSURE]

    def equations(self, ports, gas, eps_mdot, stab=0.0, eps_fb=1e-5):
        prt = ports[0]
        xi = smooth_step(prt.mdot_out, eps_mdot)  # 1: feeding the network
        return [xi * (prt.state.pt - self.pt) + (1.0 - xi) * (prt.state.p - self.pt)]

    def donor_enthalpy(self, ports, gas, eps_mdot):
        return gas.cp * self.Tt


class SupersonicInlet(Element):
    """Supersonic inflow boundary (intake flight condition): M, p_t and T_t.

    At a supersonic inflow edge all three characteristics enter the network,
    so this boundary legitimately supplies TWO equations (Mach and total
    pressure) plus the donor enthalpy -- one more than a subsonic boundary.
    This is a declared problem statement (the flight condition), not a
    regime switch; see docs/theory.md section 11.4.

    Counting note: the extra row makes the global system formally
    over-determined by one.  It is absorbed by the Fischer-Burmeister row of
    the sonic-fed diverging element downstream (the element holding the
    terminal shock): fed at exactly M = 1, that row self-vacates to its
    O(eps_fb^2) regularization floor, which is the smooth realization of the
    characteristic count redistribution.  Solve such cases with
    tol >~ 1e-9 (the floor) and, since the supersonic branch must be
    selected, from an analytic initial guess.
    """

    n_ports = 1

    def __init__(self, M: float, pt: float, Tt: float, name: str = "supersonic-inlet"):
        if M <= 1.0:
            raise ValueError("SupersonicInlet requires M > 1 (use TotalPressureInlet otherwise)")
        self.M = M
        self.pt = pt
        self.Tt = Tt
        self.name = name

    @property
    def eq_kinds(self):
        return [KIND_PRESSURE, KIND_PRESSURE]

    def equations(self, ports, gas, eps_mdot, stab=0.0, eps_fb=1e-5):
        prt = ports[0]
        m_out = prt.sigma * prt.state.M  # Mach oriented into the network
        return [
            prt.state.pt - self.pt,
            (m_out - self.M) * self.pt,  # scaled to pressure units
        ]

    def donor_enthalpy(self, ports, gas, eps_mdot):
        return gas.cp * self.Tt


class SupersonicOutlet(Element):
    """Supersonic outflow boundary at the design exit pressure.

    The declared supersonic-exit configuration of docs/theory.md section 11.4: at
    a supersonic exit all characteristics leave the domain and the back
    pressure cannot influence the nozzle, so this boundary imposes the plain
    static-pressure condition WITHOUT the choking complementarity of
    PressureOutlet -- the row the full-supersonic (design-point) solution
    satisfies.

    Validity is the user's declaration: use it only where the exit edge runs
    supersonic and the specified pressure is the design-consistent exit
    pressure (p_t,upstream * isentropic ratio at the exit Mach implied by
    the area ratio).  At off-design pressures the equations admit
    physically unrealizable roots (branch jumps) -- over-/under-expanded
    external adjustment needs the plume element with an internal degree of
    freedom (roadmap).  `run_ui_case.py` warns if the converged exit is
    subsonic (then PressureOutlet is the right boundary).
    """

    n_ports = 1

    def __init__(self, p: float, name: str = "supersonic-outlet"):
        self.p = p
        self.name = name

    @property
    def eq_kinds(self):
        return [KIND_PRESSURE]

    def equations(self, ports, gas, eps_mdot, stab=0.0, eps_fb=1e-5):
        return [ports[0].state.p - self.p]

    def donor_enthalpy(self, ports, gas, eps_mdot):
        # never the upwind side in valid (supersonic outflow) use
        return ports[0].state.ht


class PressureOutlet(Element):
    """Discharge reservoir at specified static pressure, with emergent choking.

    Regimes (one smooth row, no declarations):
      * subsonic outflow: stream static pressure = p (classical outlet);
      * choked outflow (exit edge reaches M = 1): the exit pressure detaches
        UPWARD from the specification -- the underexpanded discharge of a
        choked (converging) exit, whose expansion to ambient happens outside
        the network.  Encoded as the complementarity
        0 <= (1 - M_in) perp (p_exit - p_spec) >= 0;
      * reversed flow: the specification acts as the *total* pressure of the
        backflow, which advects cp * Tt_backflow into the network (smooth
        blend, as before).

    A supersonic exit edge (M_in > 1, from a converging-diverging passage)
    is intentionally infeasible here: a fixed-geometry exit chokes at M = 1;
    supersonic exits require the dedicated supersonic-boundary treatment
    (docs/theory.md section 11).
    """

    n_ports = 1

    def __init__(self, p: float, Tt_backflow: float = 300.0, name: str = "pressure-outlet"):
        self.p = p
        self.Tt_backflow = Tt_backflow
        self.name = name

    @property
    def eq_kinds(self):
        return [KIND_PRESSURE]

    def equations(self, ports, gas, eps_mdot, stab=0.0, eps_fb=1e-5):
        prt = ports[0]
        # Mach oriented into the element (= out of the network): M carries the
        # sign of mdot, so the oriented value is -sigma * M.
        m_in = -prt.sigma * prt.state.M
        xi = smooth_step(prt.mdot_in, eps_mdot)  # 1: network discharging here
        choked_row = fischer_burmeister(1.0 - m_in, (prt.state.p - self.p) / self.p, eps_fb) * self.p
        return [xi * choked_row + (1.0 - xi) * (prt.state.pt - self.p)]

    def donor_enthalpy(self, ports, gas, eps_mdot):
        return gas.cp * self.Tt_backflow


# ---------------------------------------------------------------------------
# Interior two-port elements
# ---------------------------------------------------------------------------


class IsentropicAreaChange(Element):
    """Smooth (internally monotone) area change with *emergent choking*.

    Pressure row: smoothed Fischer-Burmeister complementarity

        0 <= (1 - M_in_small)  perp  (pt_small - pt_large)/pt_small >= 0

    where M_in_small is the Mach number at the SMALL port oriented into the
    element.  Regimes, with no switching logic:

      * subsonic anywhere, either direction (M_in_small < 1, including the
        converging direction where M_in_small < 0): the loss is pinned to
        ~0 and the row reduces to total-pressure equality -- the classical
        isentropic element;
      * small port exactly sonic with diverging flow (M_in_small = 1): the
        element is choked; a total-pressure DROP from small to large side
        becomes admissible -- the lumped normal shock standing in the
        diverging continuation.  The shock position is not an unknown; it is
        recovered from the pt loss (fns.shock).

    Because the framework keeps all geometry on edges (elements are
    internally monotone), any throat is an edge, and this row placed on every
    area change makes choking and mass-flow capping network-wide emergent
    behavior.  h_t continuity comes from the edge transport; entropy follows
    from s = s(p_t, T_t).

    Not yet covered: supersonic inflow at the small port (started-intake
    terminal shocks; needs the shock-position internal DOF) and supersonic
    *exit* edges at boundaries (see docs/theory.md section 11).
    """

    n_ports = 2

    def __init__(self, name: str = "isentropic-area-change"):
        self.name = name

    @property
    def eq_kinds(self):
        return [KIND_MASS, KIND_PRESSURE]

    def equations(self, ports, gas, eps_mdot, stab=0.0, eps_fb=1e-5):
        a0, a1 = ports[0].area, ports[1].area
        small, large = (ports[0], ports[1]) if a0 <= a1 else (ports[1], ports[0])
        # Mach at the small port, oriented INTO the element (positive when
        # the element diffuses the flow small -> large).
        m_in = -small.sigma * small.state.M
        sub_margin = 1.0 - m_in
        loss = (small.state.pt - large.state.pt) / small.state.pt
        row = fischer_burmeister(sub_margin, loss, eps_fb) * small.state.pt
        return [
            self._mass_balance(ports),
            row - stab * ports[1].mdot_out,
        ]


class SuddenAreaChange(Element):
    """Abrupt area change between its two ports.

    Flow from the *small* area to the *large* area (sudden expansion):
    Borda-Carnot momentum balance with the flange wetted by the small-side
    static pressure,

        sum sigma*(mdot*u + p*A) - p_small * sum sigma*A = 0.

    Flow from large to small (sudden contraction): modeled as loss-free
    (vena-contracta losses can be added later), i.e. total-pressure equality.
    The two regimes are blended smoothly with the expansion-direction mass
    flow, so the residual stays C-infinity through reversal and respects the
    second law in both directions (the raw Borda-Carnot balance applied
    against a contraction would predict an entropy *decrease*).
    """

    n_ports = 2

    def __init__(self, name: str = "sudden-area-change"):
        self.name = name

    @property
    def eq_kinds(self):
        return [KIND_MASS, KIND_PRESSURE]

    def equations(self, ports, gas, eps_mdot, stab=0.0, eps_fb=1e-5):
        a0, a1 = ports[0].area, ports[1].area
        small, large = (ports[0], ports[1]) if a0 <= a1 else (ports[1], ports[0])

        # Borda-Carnot momentum balance along the port0 -> port1 axis.  The
        # convective momentum flux mdot*u = mdot^2/(rho*A) is invariant under
        # edge-direction flips (both factors change sign together), so no
        # sigma appears here -- writing it with sigma would break the
        # edge-direction independence of the element.
        st0, st1 = ports[0].state, ports[1].state
        r_mom = (
            (st1.mdot * st1.u + st1.p * ports[1].area)
            - (st0.mdot * st0.u + st0.p * ports[0].area)
            - small.state.p * (ports[1].area - ports[0].area)
        )
        # Scale to pressure units and orient so the leading p-content is
        # (p_0 - p_1), matching r_isen -- otherwise the smooth blend of the
        # two regimes could cancel near mdot = 0.
        r_mom = -r_mom / large.area

        r_isen = ports[0].state.pt - ports[1].state.pt

        # Expansion regime when flow enters the element through the small port.
        xi = smooth_step(small.mdot_in, eps_mdot)
        return [
            self._mass_balance(ports),
            xi * r_mom + (1.0 - xi) * r_isen - stab * ports[1].mdot_out,
        ]


class LossElement(Element):
    """Concentrated total-pressure loss: pt_up - pt_down = K * q_dyn.

    K is referenced to the dynamic pressure of port 0.  Written in a smooth,
    direction-aware form: the loss always opposes the flow.
    """

    n_ports = 2

    def __init__(self, K: float, name: str = "loss"):
        self.K = K
        self.name = name

    @property
    def eq_kinds(self):
        return [KIND_MASS, KIND_PRESSURE]

    def equations(self, ports, gas, eps_mdot, stab=0.0, eps_fb=1e-5):
        st0, st1 = ports[0].state, ports[1].state
        rho_avg = 0.5 * (st0.rho + st1.rho)
        # 0.5 * rho * u * |u| based on port-0 area, smooth in mdot:
        u_ref = st0.mdot / (rho_avg * ports[0].area)
        u_abs = np.sqrt(u_ref * u_ref + (eps_mdot / (rho_avg * ports[0].area)) ** 2)
        q_signed = 0.5 * rho_avg * u_ref * u_abs
        # mdot > 0 (port0 -> port1): pt0 - pt1 = +K q   (loss downstream)
        return [
            self._mass_balance(ports),
            st0.pt - st1.pt - self.K * q_signed - stab * ports[1].mdot_out,
        ]


# ---------------------------------------------------------------------------
# Multi-port elements
# ---------------------------------------------------------------------------


class JunctionStaticP(Element):
    """n-port junction, momentum neglected: common static pressure.

    Equations: 1 mass balance + (n-1) static-pressure equalities.
    Outflow enthalpy mixing is handled by the edge advection equations using
    the smoothly upwinded donor mix of the base class.
    """

    def __init__(self, n_ports: int, name: str = "junction"):
        self.n_ports = n_ports
        self.name = name

    @property
    def eq_kinds(self):
        return [KIND_MASS] + [KIND_PRESSURE] * (self.n_ports - 1)

    def equations(self, ports, gas, eps_mdot, stab=0.0, eps_fb=1e-5):
        eqs = [self._mass_balance(ports)]
        for prt in ports[1:]:
            eqs.append(ports[0].state.p - prt.state.p - stab * prt.mdot_out)
        return eqs


class LosslessSplitter(Element):
    """n-port lossless (isentropic) splitter: common total pressure.

    Equations: 1 mass balance + (n-1) total-pressure equalities.
    Exactly the user's 'lossless splitter' (mass + energy + constant entropy):
    with h_t supplied by edge advection and p_t common, entropy s(p_t, T_t) is
    continuous from the feeding port into every outflow port.
    """

    def __init__(self, n_ports: int, name: str = "splitter"):
        self.n_ports = n_ports
        self.name = name

    @property
    def eq_kinds(self):
        return [KIND_MASS] + [KIND_PRESSURE] * (self.n_ports - 1)

    def equations(self, ports, gas, eps_mdot, stab=0.0, eps_fb=1e-5):
        eqs = [self._mass_balance(ports)]
        for prt in ports[1:]:
            eqs.append(ports[0].state.pt - prt.state.pt - stab * prt.mdot_out)
        return eqs
