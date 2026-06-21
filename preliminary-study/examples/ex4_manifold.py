"""Example 4: a gas-turbine-style distribution network.

Two thermally different sources (a cold mass-flow-controlled stream and a hot
pressurized reservoir) merge in a mixing chamber, flow through a feed pipe
into a distribution manifold, and leave through three dissimilar branches:

                                    +--> [loss K=3]    --> outlet A (2.2 bar)
  cold (8 kg/s, 450 K) --+
                         +--> mix --> feed --> manifold +--> [sudden dump]  --> outlet B (3.4 bar)
  hot  (4.5 bar, 800 K) -+
                                    +--> [nozzle+loss] --> outlet C (2.7 bar)

Demonstrates: enthalpy mixing of streams at different temperature, a
pressure-fed source whose flow rate is an outcome (not an input), split
fractions across branches with different element types, and high-subsonic
compressible flow (max Mach ~ 0.8) -- all from a quiescent cold start.

Element-selection note: the manifold is a LosslessSplitter (total-pressure
equality), the correct model for an isentropic distribution plenum feeding
fast branch ports.  A static-pressure junction must NOT be used where a port
runs at significant Mach: equal static pressure plus the port's velocity
head would hand the branch MORE total pressure than the feed has (a
second-law violation), and the network may then have no steady solution.
JunctionStaticP is reserved for the low-speed mixing chamber, where the
distinction is negligible.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fns import (
    AIR,
    Network,
    MassFlowInlet,
    TotalPressureInlet,
    PressureOutlet,
    IsentropicAreaChange,
    SuddenAreaChange,
    LossElement,
    JunctionStaticP,
    LosslessSplitter,
    solve,
)

net = Network(AIR, p_ref=2.5e5, T_ref=500.0, mdot_ref=10.0)

cold = net.add(MassFlowInlet(mdot=8.0, Tt=450.0, name="cold-src"))
hot = net.add(TotalPressureInlet(pt=4.5e5, Tt=800.0, name="hot-src"))
mix = net.add(JunctionStaticP(3, name="mix-chamber"))
feed = net.add(IsentropicAreaChange(name="feed-pipe"))
man = net.add(LosslessSplitter(4, name="manifold"))
k1 = net.add(LossElement(K=3.0, name="brA-loss"))
outA = net.add(PressureOutlet(p=2.2e5, name="outA"))
sx = net.add(SuddenAreaChange(name="brB-dump"))
outB = net.add(PressureOutlet(p=3.4e5, name="outB"))
orf = net.add(IsentropicAreaChange(name="brC-nozzle"))
k2 = net.add(LossElement(K=1.5, name="brC-loss"))
outC = net.add(PressureOutlet(p=2.7e5, name="outC"))

e_cold = net.connect(cold, mix, area=0.06)
e_hot = net.connect(hot, mix, area=0.03)
e_m1 = net.connect(mix, feed, area=0.08)
e_m2 = net.connect(feed, man, area=0.035)
eA1 = net.connect(man, k1, area=0.02)
eA2 = net.connect(k1, outA, area=0.02)
eB1 = net.connect(man, sx, area=0.012)
eB2 = net.connect(sx, outB, area=0.03)
eC1 = net.connect(man, orf, area=0.02)
eC2 = net.connect(orf, k2, area=0.008)
eC3 = net.connect(k2, outC, area=0.008)

result = solve(net, x0=net.initial_guess(mdot0=0.0))
print(result)
print()
print(net.report(result.x))

st = net.states(result.x)
total = st[e_m2].mdot
print()
print(f"hot-source inflow (an outcome, not an input): {st[e_hot].mdot:7.3f} kg/s")
print(f"mixed total temperature:                      {st[e_m1].Tt:7.1f} K")
mix_check = (st[e_cold].mdot * 450.0 + st[e_hot].mdot * 800.0) / (st[e_cold].mdot + st[e_hot].mdot)
print(f"mass-weighted check:                          {mix_check:7.1f} K")
print(f"maximum Mach number in the network:           {max(abs(s.M) for s in st):7.3f}")
print()
for nm, e in (("A", eA1), ("B", eB1), ("C", eC1)):
    print(f"branch {nm}: {st[e].mdot:7.4f} kg/s  ({st[e].mdot / total * 100:5.1f} %)")
print()
print("element conservation check (interior elements):")
for name, dm, de in net.conservation_report(result.x):
    if name.startswith(("mix", "feed", "manifold", "br")):
        print(f"  {name:>12s}:  dm = {dm:+.2e} kg/s   dE = {de:+.2e} W")
