"""Example 2: split fractions in a branching network (gas-turbine style).

A feed is split into two parallel branches, one clean and one with a
concentrated loss, then re-merged.  The solver finds the split fractions;
this is the flow-network use case (secondary air systems etc.).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fns import (
    AIR,
    Network,
    MassFlowInlet,
    LosslessSplitter,
    IsentropicAreaChange,
    LossElement,
    JunctionStaticP,
    PressureOutlet,
    solve,
)

net = Network(AIR, p_ref=101325.0, T_ref=300.0)
inl = net.add(MassFlowInlet(mdot=20.0, Tt=400.0, name="inlet"))
spl = net.add(LosslessSplitter(3, name="splitter"))
brA = net.add(IsentropicAreaChange(name="branchA"))
brB = net.add(LossElement(K=5.0, name="branchB(K=5)"))
jun = net.add(JunctionStaticP(3, name="junction"))
out = net.add(PressureOutlet(p=101325.0, name="outlet"))

net.connect(inl, spl, area=0.5)
eA = net.connect(spl, brA, area=0.25)
eB = net.connect(spl, brB, area=0.25)
net.connect(brA, jun, area=0.30)
net.connect(brB, jun, area=0.25)
net.connect(jun, out, area=0.5)

result = solve(net, x0=net.initial_guess(mdot0=0.0))
print(result)
print()
print(net.report(result.x))

states = net.states(result.x)
print()
print(f"split fraction branch A: {states[eA].mdot / 20.0:.4f}")
print(f"split fraction branch B: {states[eB].mdot / 20.0:.4f}")
