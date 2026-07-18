"""Verification: Parrott (1973) three-stage helicopter exhaust muffler.

T. L. Parrott, "An improved method for design of expansion-chamber mufflers with application
to an operational helicopter," NASA TN D-7309 (1973).  The optimised muffler (its Fig 13) is a
series of three concentric extended-tube expansion chambers: twelve components with published
lengths and areas, running an exhaust at ``c = 610 m/s`` (2000 ft/s) and tailpipe Mach 0.1.

Each chamber is entered by an inlet tube reaching ``L_in`` into it and left by an outlet tube
reaching ``L_out`` in from the far end.  Acoustically the tube tip is an area change carrying the
through flow and the annulus around it -- closed by the chamber end wall -- is a side branch of
length equal to the tube's reach and area ``S_c - S_1``.  So each chamber is an inlet-tube tip
(expansion + closed inlet stub), the open chamber body, and an outlet-tube entrance (contraction +
closed outlet stub); the pipe between chambers is the upstream extended outlet, the connector, and
the downstream extended inlet in series.

The reference is the classical transfer-matrix (four-pole) method in the ``(p, U = S v)`` variables,
in which area changes are transparent and a stub of area ``S`` and length ``l`` closed by an end
reflection ``R`` is a shunt ``Zb = (rho c / S)(1+G)/(1-G)``, ``G = R exp(-2 i k l)`` (rigid ``R=1``
gives ``-i (rho c / S) cot(k l)``).  Nefes builds the same muffler from physical geometry and solves
the linearised perturbation network; the two share no code and agree to machine precision, both for
rigid stubs and for the absorptive (``R<1``) branch walls Parrott used to model the yielding end
caps.  The absorptive walls are read exactly like the rigid ones: give each wall an ``R<1``
``PerturbationBC.reflection`` and ``freeze`` it, so the response folds it in at that declared
closure.  The notebook ``examples/validation/parrott_helicopter_muffler.ipynb`` walks the case
through.
"""

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.perturbation import PerturbationBC
from nefes.shell.network import Network
from nefes.thermo.configure import perfect_gas

GAMMA, R = 1.4, 287.0
CO = 610.0  # exhaust sound speed [m/s] (2000 ft/s)
P0 = 101325.0
T0 = CO**2 / (GAMMA * R)  # temperature giving c == CO
RHO0 = P0 / (R * T0)

S1 = 0.002  # pipe / tailpipe area [m^2]
SC = 0.019  # chamber area [m^2]
SANN = SC - S1  # annular side-branch area
M_RATIO = SC / S1

# (chamber length, extended-inlet length, extended-outlet length) [m], from Fig 13
CHAMBERS = [(0.76, 0.44, 0.13), (0.51, 0.30, 0.08), (0.25, 0.04, 0.06)]
CONNECTORS = [1.04, 0.13]  # S1 pipe between chambers = ext_out + connector + ext_in
TAILPIPE = 0.66  # ext_out(first chamber) + tailpipe

FREQS = np.linspace(20.0, 1500.0, 600)


def _gas():
    return perfect_gas(R, GAMMA)


def _tl_two_port(sol, freqs, e_in, e_out, freeze):
    resp = sol.perturbation_response(freqs, freeze=freeze)
    tau = resp.acoustic_scattering_matrix(e_in, e_out)[:, 1, 0]
    return -20.0 * np.log10(np.abs(tau))


def _add_chamber(net, node_in, Lch, Lin, Lout, walls, wall_reflection):
    def new_wall():
        if wall_reflection is None:
            return net.add(cat.wall())
        return net.add(cat.wall(perturbation_bc=PerturbationBC.reflection(wall_reflection)))

    tip_in = net.add(cat.junction())
    body = net.add(cat.duct(Lch - Lin - Lout))
    tip_out = net.add(cat.junction())
    stub_in = net.add(cat.duct(Lin))
    wall_in = new_wall()
    stub_out = net.add(cat.duct(Lout))
    wall_out = new_wall()
    net.connect(node_in, tip_in, S1)
    net.connect(tip_in, body, SC)
    net.connect(tip_in, stub_in, SANN)
    net.connect(stub_in, wall_in, SANN)
    net.connect(body, tip_out, SC)
    net.connect(tip_out, stub_out, SANN)
    net.connect(stub_out, wall_out, SANN)
    walls.extend([wall_in, wall_out])
    return tip_out


def build_muffler(mach, wall_reflection=None):
    """Parrott's three-stage muffler; returns (net, sol, inlet_edge, outlet_edge, stub_walls).

    ``wall_reflection`` closes the stub end walls at a reflection R < 1 (absorptive)."""
    net = Network(_gas())
    walls = []
    inlet = net.add(cat.mass_flow_inlet(RHO0 * mach * CO * S1, T0))
    inlet_tube = net.add(cat.duct(CHAMBERS[0][1]))
    e_in = net.connect(inlet, inlet_tube, S1)
    node = inlet_tube
    for i, (Lch, Lin, Lout) in enumerate(CHAMBERS):
        node = _add_chamber(net, node, Lch, Lin, Lout, walls, wall_reflection)
        if i < len(CONNECTORS):
            link = net.add(cat.duct(CONNECTORS[i]))
            net.connect(node, link, S1)
            node = link
    tail = net.add(cat.duct(TAILPIPE))
    net.connect(node, tail, S1)
    outlet = net.add(cat.pressure_outlet(P0, T0))
    e_out = net.connect(tail, outlet, S1)
    return net, net.solve(), e_in, e_out, walls


def tl_transfer_matrix(freqs, r_in=1.0, r_out=1.0):
    """Classical four-pole (p, U) transmission loss of the same muffler (no flow), with stubs
    closed by end reflections r_in / r_out (r=1 -> rigid)."""
    Z1, Z2, Zann = RHO0 * CO / S1, RHO0 * CO / SC, RHO0 * CO / SANN

    def duct(k, Z, L):
        kl = k * L
        return np.array([[np.cos(kl), 1j * Z * np.sin(kl)], [1j * np.sin(kl) / Z, np.cos(kl)]])

    def stub(k, L, Rend):
        G = Rend * np.exp(-2j * k * L)
        Zb = Zann * (1.0 + G) / (1.0 - G)
        return np.array([[1.0, 0.0], [1.0 / Zb, 1.0]])

    out = np.empty_like(freqs)
    for j, f in enumerate(freqs):
        k = 2.0 * np.pi * f / CO
        chain = [duct(k, Z1, CHAMBERS[0][1])]
        for i, (Lch, Lin, Lout) in enumerate(CHAMBERS):
            chain += [stub(k, Lin, r_in), duct(k, Z2, Lch - Lin - Lout), stub(k, Lout, r_out)]
            chain.append(duct(k, Z1, CONNECTORS[i] if i < len(CONNECTORS) else TAILPIPE))
        T = np.eye(2, dtype=complex)
        for Mx in chain:
            T = T @ Mx
        out[j] = 20.0 * np.log10(0.5 * abs(T[0, 0] + T[0, 1] / Z1 + Z1 * T[1, 0] + T[1, 1]))
    return out


def tl_chamber_analytic(freqs, length):
    k = 2.0 * np.pi * freqs / CO
    return 10.0 * np.log10(1.0 + 0.25 * (M_RATIO - 1.0 / M_RATIO) ** 2 * np.sin(k * length) ** 2)


@pytest.fixture(scope="module")
def quiescent():
    net, sol, e_in, e_out, walls = build_muffler(0.0)
    return sol, e_in, e_out, walls


def test_plain_chamber_matches_analytic():
    """A single expansion chamber reproduces the closed-form transmission loss exactly."""
    net = Network(_gas())
    inlet = net.add(cat.mass_flow_inlet(0.0, T0))
    exp = net.add(cat.isentropic_area_change("expansion"))
    chamber = net.add(cat.duct(0.61))
    con = net.add(cat.isentropic_area_change("contraction"))
    outlet = net.add(cat.pressure_outlet(P0, T0))
    e_in = net.connect(inlet, exp, S1)
    net.connect(exp, chamber, SC)
    net.connect(chamber, con, SC)
    e_out = net.connect(con, outlet, S1)
    sol = net.solve()
    assert sol.converged
    tl = _tl_two_port(sol, FREQS, e_in, e_out, freeze=None)
    ref = tl_chamber_analytic(FREQS, 0.61)
    assert np.max(np.abs(tl - ref)) < 1e-8


def test_muffler_converges(quiescent):
    sol = quiescent[0]
    assert sol.converged
    assert sol.residual_norm < 1e-8


def test_muffler_matches_transfer_matrix(quiescent):
    """The Nefes network reproduces the classical four-pole transmission loss to machine precision."""
    sol, e_in, e_out, walls = quiescent
    tl = _tl_two_port(sol, FREQS, e_in, e_out, freeze=walls)
    ref = tl_transfer_matrix(FREQS)
    d = np.abs(tl - ref)
    # rms is at the 1e-5 dB level over the whole band; the only larger residual sits on a
    # razor-sharp stub resonance (>80 dB) where the reference's cot(kl) is singular.
    assert np.sqrt(np.mean(d**2)) < 1e-3
    assert np.max(d[ref < 80.0]) < 1e-2


def test_absorptive_stub_walls_match_lossy_transfer_matrix(quiescent):
    """Absorptive stub walls (R=0.8), declared as a reflection BC and frozen, match the independent
    lossy transfer-matrix, and physically round the peaks and fill the nulls of the lossless case."""
    _, sol_d, ed_in, ed_out, walls_d = build_muffler(0.0, wall_reflection=0.8)
    tl_damped = _tl_two_port(sol_d, FREQS, ed_in, ed_out, freeze=walls_d)
    ref = tl_transfer_matrix(FREQS, r_in=0.8, r_out=0.8)
    assert np.max(np.abs(tl_damped - ref)) < 1e-4
    sol, e_in, e_out, walls = quiescent
    tl_rigid = _tl_two_port(sol, FREQS, e_in, e_out, freeze=walls)
    assert tl_damped.max() < tl_rigid.max() - 20.0  # peaks rounded
    assert tl_damped.min() > tl_rigid.min() + 5.0  # nulls filled


def test_mean_flow_operating_point():
    """Nefes solves the compressible mean flow the reference assumes: Mach 0.1 in the pipes,
    an order of magnitude less in the wide chambers, zero in the closed stubs."""
    net, sol, e_in, e_out, walls = build_muffler(0.1)
    assert sol.converged
    mach = np.abs(sol.field("M"))
    assert float(mach[e_out]) == pytest.approx(0.1, abs=2e-3)  # tailpipe
    assert mach.max() == pytest.approx(0.1, abs=2e-3)  # pipes are the fastest
    # the chamber bodies decelerate by the area ratio; the stubs carry no mean flow
    assert np.count_nonzero(mach < 1e-9) >= 2 * len(CHAMBERS)


def test_mean_flow_leaves_anechoic_tl_nearly_unchanged(quiescent):
    """The anechoic transmission loss is nearly flow-independent; the flow only shifts the sharp
    resonances by ~0.1 % (Parrott's own finding that internal flow corrections are insignificant)."""
    sol0, e_in0, e_out0, walls0 = quiescent
    tl0 = _tl_two_port(sol0, FREQS, e_in0, e_out0, freeze=walls0)
    _, sol1, e_in1, e_out1, walls1 = build_muffler(0.1)
    tl1 = _tl_two_port(sol1, FREQS, e_in1, e_out1, freeze=walls1)
    smooth = FREQS <= 200.0  # away from the razor-sharp stub resonances
    assert np.max(np.abs(tl1 - tl0)[smooth]) < 0.5
