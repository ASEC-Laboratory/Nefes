"""Example 1: chain of isentropic area changes, solved from a quiescent start.

This is the configuration class that defeated the earlier prototypes
(multiple isentropic elements in series).  Run:

    python examples/ex1_nozzle_chain.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fns import AIR, Network, MassFlowInlet, IsentropicAreaChange, PressureOutlet, solve

areas = [1.0, 0.5, 0.15, 0.5, 2.0, 0.12]

net = Network(AIR, p_ref=101325.0, T_ref=300.0)
elems = [net.add(MassFlowInlet(mdot=25.0, Tt=500.0, name="inlet"))]
for k in range(len(areas) - 1):
    elems.append(net.add(IsentropicAreaChange(name=f"iac{k}")))
elems.append(net.add(PressureOutlet(p=101325.0, name="outlet")))
for k, a in enumerate(areas):
    net.connect(elems[k], elems[k + 1], area=a)

# exactly zero flow, uniform cold initial state
x0 = net.initial_guess(mdot0=0.0, p0=101325.0, Tt0=300.0)
result = solve(net, x0=x0, verbose=True)

print()
print(result)
print()
print(net.report(result.x))
print()
print("element conservation check (mass, energy):")
for name, dm, de in net.conservation_report(result.x):
    print(f"  {name:>10s}:  dm = {dm:+.3e} kg/s   dE = {de:+.3e} W")
