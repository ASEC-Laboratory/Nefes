"""Example 5: compressible flow features -- pressure-ratio sweep and choking.

A reservoir (2 bar, 400 K) discharges through an isentropic nozzle (feed area
0.10 m^2, exit area 0.03 m^2).  The exit pressure is swept downward and the
computed mass flow is compared with the exact isentropic solution at every
point, up to an exit Mach number of 0.99.

This shows the defining compressible-flow behavior that incompressible
network solvers miss entirely:

  * mass flow is NOT proportional to sqrt(pressure drop) -- it saturates;
  * at the critical pressure ratio (p/pt = 0.5283 for gamma = 1.4) the flow
    chokes: the mass flow can never exceed the choking value no matter how
    low the back pressure;
  * below the critical ratio the exit stays pinned at exactly M = 1 with the
    mass flow capped at the choking value, and the exit pressure detaches
    upward from the specification (underexpanded discharge) -- emergent
    behavior of the complementarity rows, no switches, no declarations.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fns import AIR, Network, TotalPressureInlet, IsentropicAreaChange, PressureOutlet, solve

G, R = AIR.gamma, AIR.R
pt_in, Tt_in = 2.0e5, 400.0
a_feed, a_exit = 0.10, 0.03


def build(p_out):
    net = Network(AIR, p_ref=pt_in, T_ref=Tt_in, mdot_ref=15.0)
    inl = net.add(TotalPressureInlet(pt=pt_in, Tt=Tt_in, name="reservoir"))
    iac = net.add(IsentropicAreaChange(name="nozzle"))
    out = net.add(PressureOutlet(p=p_out, name="exit"))
    net.connect(inl, iac, area=a_feed)
    net.connect(iac, out, area=a_exit)
    return net


def exact(PR):
    """Exact isentropic exit state for a given pressure ratio p_exit/pt."""
    M = np.sqrt(2.0 / (G - 1.0) * (PR ** (-(G - 1.0) / G) - 1.0))
    T = Tt_in / (1.0 + 0.5 * (G - 1.0) * M * M)
    rho = PR * pt_in / (R * T)
    return rho * M * np.sqrt(G * R * T) * a_exit, M


PR_crit = (2.0 / (G + 1.0)) ** (G / (G - 1.0))
mdot_choke = (
    pt_in / np.sqrt(Tt_in) * np.sqrt(G / R)
    * (2.0 / (G + 1.0)) ** ((G + 1.0) / (2.0 * (G - 1.0))) * a_exit
)
print(f"critical pressure ratio: {PR_crit:.4f}   choking mass flow: {mdot_choke:.4f} kg/s")
print()
print("   PR      mdot [kg/s]   exact [kg/s]   M_exit   exact M   iters")

x_warm = None
for PR in (0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.55, 0.535):
    net = build(PR * pt_in)
    res = solve(net, x0=x_warm)  # warm start from the previous pressure ratio
    md_ex, M_ex = exact(PR)
    st = net.states(res.x)
    print(
        f"  {PR:.3f}   {st[1].mdot:10.5f}   {md_ex:10.5f}     {st[1].M:.4f}   {M_ex:.4f}   {res.iterations:4d}"
    )
    x_warm = res.x

print()
print(f"mass flow at PR = 0.535 has reached {100 * st[1].mdot / mdot_choke:.2f} % of the choking value.")

# Below the critical ratio: the exit chokes (emergent).
net = build(0.45 * pt_in)
res = solve(net)
st = net.states(res.x)
print()
print(f"PR = 0.450 (below critical): converged = {res.converged}")
print(f"-> exit pinned at M = {st[1].M:.4f}, mass flow capped at "
      f"{st[1].mdot:.4f} kg/s ({100 * st[1].mdot / mdot_choke:.2f} % of choking),")
print(f"   exit pressure detached upward to p* = {st[1].p:.0f} Pa "
      f"(specified {0.45 * pt_in:.0f} Pa): underexpanded choked discharge.")
