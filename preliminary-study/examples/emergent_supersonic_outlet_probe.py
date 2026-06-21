"""Independent verification: is the *emergent* supersonic outlet impossible?

Reproduces, on the canonical e0/e1/e2 C-D nozzle
    inlet -e0- [contraction] -e1(throat)- [expansion] -e2- outlet
the four experiments behind the verdict in EMERGENT_SUPERSONIC_OUTLET.md:

  1. baseline: PressureOutlet (sweep) and the declared SupersonicOutlet;
  2. corner-exclusion: the throat sonic-offset scales as O(eps_fb^{2/3});
  3. a self-vacating ("release p_b when supersonic") outlet is under-determined
     by exactly one (singular Jacobian); the sonic-throat anchor restores rank
     but admits unphysical branch-jumped roots;
  4. a gated, over-determined construction (anchor + "supersonic => lossless",
     the SupersonicInlet self-vacating mirror) computes each branch correctly
     but only with branch-appropriate seeding;
  5. the "extrapolated pressure" outlet: (a) the flow function Phi(M) is
     double-valued (so (pt,Tt,mdot) does NOT fix the Mach), and (b) imposing the
     supersonic pressure from the edge's *own* Phi is the identity p=p (rank
     loss, collapses), while imposing it from *geometry* (area ratio) keeps full
     rank and converges -- but that is a declaration (= a live-pt SupersonicOutlet).

See EMERGENT_SUPERSONIC_OUTLET.md for the full discussion and the revised verdict
(the nozzle outlet is NOT impossible in principle; it needs an internal
shock-position DOF on the diverging element -- the dual of SupersonicInlet).

Run:  python examples/emergent_supersonic_outlet_probe.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fns import (
    AIR, Network, TotalPressureInlet, MassFlowInlet, PressureOutlet,
    SupersonicOutlet, IsentropicAreaChange, solve, complex_step_jacobian,
)
from fns.elements import Element, KIND_PRESSURE
from fns.smooth import smooth_step
from fns.state import StateError
from fns.shock import area_ratio, normal_shock_pt_ratio, normal_shock_post_mach

GAS = AIR
G, R = GAS.gamma, GAS.R
PT, TT = 2.0e5, 400.0
A_FEED, A_THROAT, A_EXIT = 0.10, 0.03, 0.06
AR = A_EXIT / A_THROAT
THR_EDGE, EXIT_EDGE = 1, 2


def mach_from_area_ratio(ar, supersonic):
    lo, hi = (1.0 + 1e-12, 50.0) if supersonic else (1e-9, 1.0 - 1e-12)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (area_ratio(mid) > ar) == supersonic:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def isentropic_p(M, pt):
    return pt * (1 + 0.5 * (G - 1) * M * M) ** (-G / (G - 1))


M_SUPER = mach_from_area_ratio(AR, supersonic=True)
P_DESIGN = isentropic_p(M_SUPER, PT)


def flow_function(M):
    """Phi = mdot*sqrt(Tt)/(A*pt); rises to a max at M=1, then falls (double-valued)."""
    return np.sqrt(G / R) * M * (1 + 0.5 * (G - 1) * M * M) ** (-(G + 1) / (2 * (G - 1)))


def flow_function_deriv(M):
    bM2 = 0.5 * (G - 1) * M * M
    k = (G + 1) / (2 * (G - 1))
    return np.sqrt(G / R) * (1 + bM2) ** (-k - 1) * (1 + bM2 * (1 - 2 * k))


def super_mach_from_phi(phi):
    """Supersonic root of flow_function(M) = phi.  Complex-step safe (IFT)."""
    pr = min(float(np.real(phi)), float(flow_function(1.0)) * (1 - 1e-12))
    M = 3.0
    for _ in range(100):
        Mn = M - (flow_function(M) - pr) / flow_function_deriv(M)
        if Mn <= 1.0:
            Mn = 0.5 * (M + 1.0)
        if abs(Mn - M) < 1e-14:
            M = Mn
            break
        M = Mn
    if isinstance(phi, complex) or np.iscomplexobj(phi):
        return M + 1j * np.imag(phi) / flow_function_deriv(M)
    return M


class ExtrapolatingOutlet(Element):
    """User's 'extrapolate the pressure' idea: target = p_b when subsonic, else a
    supersonic static pressure, blended smoothly.  source='geometry' uses the area
    ratio (Version A: real constraint, full rank); source='own' uses the edge's own
    flow function (Version B: identity p=p, rank loss)."""
    n_ports = 1

    def __init__(self, p, source="geometry", width=0.15, name="extrap-outlet"):
        self.p, self.source, self.width, self.name = p, source, width, name

    @property
    def eq_kinds(self):
        return [KIND_PRESSURE]

    def equations(self, ports, gas, eps_mdot, stab=0.0, eps_fb=1e-5):
        prt = ports[0]
        st = prt.state
        M = -prt.sigma * st.M
        w = smooth_step(1.0 - M, self.width)  # 1 subsonic, 0 supersonic
        if self.source == "geometry":
            p_super = isentropic_p(M_SUPER, st.pt)
        else:  # 'own': supersonic Mach from this edge's own (mdot, pt, Tt)
            phi = abs(st.mdot) * np.sqrt(st.Tt) / (prt.area * st.pt)
            p_super = isentropic_p(super_mach_from_phi(phi), st.pt)
        return [st.p - (w * self.p + (1.0 - w) * p_super)]

    def donor_enthalpy(self, ports, gas, eps_mdot):
        return ports[0].state.ht


def shock_floor_pb():
    """Back pressure with a normal shock standing at the exit plane: below this
    the exit is supersonic (over-/under-expanded), above it the shock is inside."""
    pt2 = normal_shock_pt_ratio(M_SUPER) * PT
    M2 = normal_shock_post_mach(M_SUPER)
    A2star = A_EXIT / area_ratio(M2)
    Mex = mach_from_area_ratio(A_EXIT / A2star, supersonic=False)
    return isentropic_p(Mex, pt2) / PT


class ReleasingOutlet(Element):
    """p = p_b when the incident edge is subsonic; the row self-vacates (the
    back pressure is released) when it is supersonic."""
    n_ports = 1

    def __init__(self, p, width=0.15, name="releasing-outlet"):
        self.p, self.width, self.name = p, width, name

    @property
    def eq_kinds(self):
        return [KIND_PRESSURE]

    def equations(self, ports, gas, eps_mdot, stab=0.0, eps_fb=1e-5):
        prt = ports[0]
        m_out = -prt.sigma * prt.state.M
        w = smooth_step(1.0 - m_out, self.width)  # 1 subsonic, 0 supersonic
        return [w * (prt.state.p - self.p)]

    def donor_enthalpy(self, ports, gas, eps_mdot):
        return ports[0].state.ht


def cd(outlet, inlet=None):
    net = Network(GAS, p_ref=PT, T_ref=TT, mdot_ref=15.0)
    inl = net.add(inlet or TotalPressureInlet(pt=PT, Tt=TT, name="res"))
    conv = net.add(IsentropicAreaChange(name="conv"))
    div = net.add(IsentropicAreaChange(name="div"))
    out = net.add(outlet)
    net.connect(inl, conv, area=A_FEED)
    net.connect(conv, div, area=A_THROAT)
    net.connect(div, out, area=A_EXIT)
    return net


def jac_svd(net, x):
    xs, rs = net.variable_scales(), net.residual_scales()
    J = complex_step_jacobian(lambda y: net.residual(y * xs) / rs, x / xs)
    return np.linalg.svd(J, compute_uv=False)


def emergent_residual(net, x, gate_w=0.15):
    """Base residual + two gated rows that activate only for a supersonic exit:
    anchor (throat sonic) and lossless (no total-pressure drop)."""
    r = list(net.residual(x))
    st = net.states(x)
    g = smooth_step(st[EXIT_EDGE].M - 1.0, gate_w)
    anchor = g * (st[THR_EDGE].M - 1.0) * PT
    lossless = g * (st[THR_EDGE].pt - st[EXIT_EDGE].pt) / st[THR_EDGE].pt * PT
    return np.concatenate([r, [anchor, lossless]])


def emergent_solve(net, x0, iters=300):
    xs = net.variable_scales()
    rs = np.concatenate([net.residual_scales(), [PT, PT]])

    def nrm(x):
        try:
            return float(np.linalg.norm(emergent_residual(net, x) / rs)), True
        except StateError:
            return np.inf, False

    x, (cur, _) = x0.copy(), nrm(x0)
    for _ in range(iters):
        J = complex_step_jacobian(lambda y: emergent_residual(net, y * xs) / rs, x / xs)
        dy, *_ = np.linalg.lstsq(J, -emergent_residual(net, x) / rs, rcond=None)
        lam = 1.0
        for _ls in range(40):
            xt = x + lam * dy * xs
            nt, ok = nrm(xt)
            if ok and nt < cur:
                break
            lam *= 0.5
        else:
            break
        x, cur = xt, nt
        if cur < 1e-10:
            break
    return x, cur


def main():
    print(f"area ratio {AR}: supersonic exit M = {M_SUPER:.4f}, "
          f"p_design = {P_DESIGN/PT:.4f}*pt, shock-at-exit floor = {shock_floor_pb():.4f}*pt\n")

    print("[1] baseline PressureOutlet sweep (no supersonic exit ever appears)")
    for frac in [0.7, 0.35, 0.2, 0.05]:
        net = cd(PressureOutlet(p=frac * PT, name="exit"))
        st = net.states(solve(net).x)
        print(f"    p_b={frac:.2f}: M_exit={st[2].M:+.3f} p_exit={st[2].p/PT:.3f}*pt")
    ref = cd(SupersonicOutlet(p=P_DESIGN, name="exit"))
    xr = solve(ref, tol=1e-12).x
    sr = ref.states(xr)
    print(f"    declared SupersonicOutlet: M_exit={sr[2].M:.4f} M_thr={sr[1].M:.5f}"
          f"  (sigma_min={jac_svd(ref, xr)[-1]:.2e}, well-posed)\n")

    print("[2] corner exclusion: throat offset ~ C * eps_fb^(2/3)")
    for efb in [1e-3, 1e-4, 1e-5, 1e-6]:
        net = cd(SupersonicOutlet(p=P_DESIGN, name="exit"))
        net.eps_fb = efb
        off = 1.0 - net.states(solve(net, tol=1e-12).x)[1].M
        print(f"    eps_fb={efb:.0e}: 1-M_thr={off:.3e}  C={off/efb**(2/3):.4f}")
    print()

    print("[3] self-vacating outlet is under-determined by exactly one")
    net = cd(ReleasingOutlet(p=0.2 * PT, name="exit"))
    sv = jac_svd(net, xr)
    print(f"    Jacobian sigma_min={sv[-1]:.2e}  (next={sv[-2]:.2e}) at the supersonic state")
    print(f"    => rank-deficient by 1; declared system was {jac_svd(ref, xr)[-1]:.2e}\n")

    print("[4] gated over-determined construction -- correct per branch, but seeded")
    print("    downward sweep (cannot cross shock->supersonic at the fold):")
    x = None
    for frac in [0.85, 0.55, 0.35, 0.08]:
        net = cd(ReleasingOutlet(p=frac * PT, name="exit"))
        x, rn = emergent_solve(net, x if x is not None else net.initial_guess())
        st = net.states(x)
        print(f"      p_b={frac:.2f}: M_exit={st[2].M:.4f} p_exit={st[2].p/PT:.4f}*pt "
              f"loss={1-st[2].pt/st[1].pt:+.4f} ||R||={rn:.1e}")
    print("    upward sweep from a supersonic seed (holds the design exit, p_b-independent):")
    seed = cd(ReleasingOutlet(p=0.04 * PT, name="exit"))
    x, _ = emergent_solve(seed, seed.initial_guess())
    for frac in [0.04, 0.20, 0.45, 0.65]:
        net = cd(ReleasingOutlet(p=frac * PT, name="exit"))
        x, rn = emergent_solve(net, x)
        st = net.states(x)
        print(f"      p_b={frac:.2f}: M_exit={st[2].M:.4f} p_exit={st[2].p/PT:.4f}*pt "
              f"loss={1-st[2].pt/st[1].pt:+.4f} ||R||={rn:.1e}")
    print("    => the gated construction holds whatever branch it is seeded on. NOTE:")
    print("       for a *nozzle* the steady solution is actually unique (no real")
    print("       multistability); the held supersonic state above the floor is the")
    print("       supersonic row ignoring p_b, not a second physical state. The true")
    print("       blocker is the exit-edge discontinuity -> needs an internal shock")
    print("       DOF on the diverging element. See EMERGENT_SUPERSONIC_OUTLET.md.\n")

    print("[5] the 'extrapolated pressure' outlet")
    phi_d = flow_function(M_SUPER)
    print(f"    flow function is double-valued: Phi={phi_d:.5f} at design gives BOTH")
    print(f"      supersonic M={M_SUPER:.4f} (p/pt={isentropic_p(M_SUPER,1.0):.4f}) and "
          f"a subsonic shadow -> (pt,Tt,mdot) does NOT fix the Mach.")
    ref = cd(SupersonicOutlet(p=P_DESIGN, name="exit"))
    xr = solve(ref, tol=1e-12).x
    for src in ("own", "geometry"):
        net = cd(ExtrapolatingOutlet(p=0.20 * PT, source=src, name="exit"))
        sv = jac_svd(net, xr)
        try:
            res = solve(net, tol=1e-10)
            st = net.states(res.x)
            outcome = (f"converged={res.converged} M_exit={st[2].M:.4f} "
                       f"p_exit={st[2].p/PT:.4f}*pt")
        except Exception as e:  # noqa: BLE001
            outcome = f"raised {type(e).__name__}"
        tag = "Version B (own Phi, identity p=p)" if src == "own" else "Version A (geometry)"
        print(f"    {tag}: sigma_min={sv[-1]:.2e} -> cold start {outcome}")
    print("    => 'own' is rank-deficient and collapses to M=1; 'geometry' is full")
    print("       rank and reaches M=2.19 -- but geometry+'supersonic branch' is a")
    print("       declaration (= a SupersonicOutlet that tracks the live pt).")


if __name__ == "__main__":
    main()
