"""Example 6: a compressible Wheatstone bridge -- interior flow reversal.

Topology (all elements between the splitter and the merge are loss elements;
the bridge edge connects the two midpoint junctions):

                 +--[K1]--> (mid-top) --[K2]--+
   src --> split |             |bridge        | merge --> sink
                 +--[K3]--> (mid-bot) --[K4]--+

The direction of the flow in the bridge edge is NOT known in advance -- it
depends on the resistance arrangement, exactly like the current direction in
the electrical Wheatstone bridge:

  * K1 < K3 (top path drops less pressure first): mid-top sits at a higher
    pressure than mid-bot, the bridge flows top -> bottom (positive, since
    the edge is directed that way);
  * mirrored resistances: the bridge flow reverses sign exactly;
  * balanced bridge (all K equal): the bridge carries exactly zero flow.

A solver that required flow to follow the edge directions, or that assigned
equations based on assumed flow directions, could not handle this network.
Here the bridge edge's mdot is simply an unknown like any other and comes
out with whichever sign the physics demands.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fns import (
    AIR,
    Network,
    TotalPressureInlet,
    PressureOutlet,
    LossElement,
    JunctionStaticP,
    LosslessSplitter,
    solve,
)


def build(K1, K2, K3, K4):
    net = Network(AIR, p_ref=2.5e5, T_ref=500.0, mdot_ref=5.0)
    inl = net.add(TotalPressureInlet(pt=3.0e5, Tt=500.0, name="src"))
    spl = net.add(LosslessSplitter(3, name="split"))
    kA1 = net.add(LossElement(K=K1, name="K1"))
    kA2 = net.add(LossElement(K=K2, name="K2"))
    kB1 = net.add(LossElement(K=K3, name="K3"))
    kB2 = net.add(LossElement(K=K4, name="K4"))
    jt = net.add(JunctionStaticP(3, name="mid-top"))
    jb = net.add(JunctionStaticP(3, name="mid-bot"))
    jun = net.add(JunctionStaticP(3, name="merge"))
    out = net.add(PressureOutlet(p=2.0e5, name="sink"))
    a = 0.02
    net.connect(inl, spl, area=2 * a)
    net.connect(spl, kA1, area=a)
    net.connect(kA1, jt, area=a)
    net.connect(spl, kB1, area=a)
    net.connect(kB1, jb, area=a)
    e_bridge = net.connect(jt, jb, area=0.5 * a)  # directed top -> bottom
    net.connect(jt, kA2, area=a)
    net.connect(kA2, jun, area=a)
    net.connect(jb, kB2, area=a)
    net.connect(kB2, jun, area=a)
    net.connect(jun, out, area=2 * a)
    return net, e_bridge


print(f"{'K1':>4} {'K2':>4} {'K3':>4} {'K4':>4} | {'bridge mdot [kg/s]':>19} {'total [kg/s]':>13} {'iters':>6}")
for K in ((2.0, 8.0, 8.0, 2.0), (8.0, 2.0, 2.0, 8.0), (4.0, 4.0, 4.0, 4.0)):
    net, eb = build(*K)
    res = solve(net, x0=net.initial_guess(mdot0=0.0))
    assert res.converged, res
    st = net.states(res.x)
    print(
        f"{K[0]:4.0f} {K[1]:4.0f} {K[2]:4.0f} {K[3]:4.0f} |"
        f" {st[eb].mdot:+19.4f} {st[0].mdot:13.3f} {res.iterations:6d}"
    )

print()
print("Note the exact antisymmetry of the first two rows and the exactly")
print("balanced (zero-flow) bridge in the third -- the network analogue of the")
print("balanced Wheatstone bridge.")
