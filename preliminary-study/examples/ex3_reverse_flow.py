"""Example 3: flow against the edge directions.

The receiving reservoir total pressure is BELOW the discharge static pressure,
so the physical flow runs against both edge directions (negative mdot).  The
solver must discover this on its own from a co-directional initial guess.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fns import AIR, Network, TotalPressureInlet, IsentropicAreaChange, PressureOutlet, solve

net = Network(AIR, p_ref=101325.0, T_ref=300.0, mdot_ref=10.0)
inl = net.add(TotalPressureInlet(pt=90000.0, Tt=300.0, name="reservoir"))
iac = net.add(IsentropicAreaChange(name="iac"))
out = net.add(PressureOutlet(p=101325.0, Tt_backflow=400.0, name="discharge"))
net.connect(inl, iac, area=0.3)
net.connect(iac, out, area=0.5)

result = solve(net, verbose=True)
print()
print(result)
print()
print(net.report(result.x))
print()
print("negative mdot: the flow runs against the edge direction, entering at the")
print("'discharge' boundary (carrying its 400 K backflow total temperature) and")
print("leaving into the low-pressure 'reservoir'.")
