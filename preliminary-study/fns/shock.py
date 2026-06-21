"""Normal-shock relations and shock-location deduction.

When a choking-aware element (IsentropicAreaChange) is choked with an
engaged total-pressure loss, that loss is a lumped normal shock standing in
the diverging continuation downstream of the sonic (small-port) edge.  The
shock is *not* a solver unknown -- its strength and position are recovered
in post-processing from the converged loss:

    pt_out / pt_in  ->  pre-shock Mach M_s   (Rankine-Hugoniot, inverted)
    M_s             ->  shock area A_s       (isentropic area-Mach relation,
                                              with the sonic small port as
                                              the reference throat A*)

Validity: A_s must lie between the element's port areas; an implied A_s
beyond the large port means the demanded loss exceeds a shock-at-exit -- the
case is over-driven into a regime the isentropic family cannot represent.
"""

import numpy as np


def area_ratio(M, gamma=1.4):
    """Isentropic A/A* at Mach M."""
    g = gamma
    return (1.0 / M) * ((2.0 + (g - 1.0) * M * M) / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def normal_shock_pt_ratio(M, gamma=1.4):
    """Total-pressure ratio pt2/pt1 across a normal shock at pre-shock Mach M."""
    g = gamma
    t1 = ((g + 1.0) * M * M / ((g - 1.0) * M * M + 2.0)) ** (g / (g - 1.0))
    t2 = ((g + 1.0) / (2.0 * g * M * M - (g - 1.0))) ** (1.0 / (g - 1.0))
    return t1 * t2


def normal_shock_post_mach(M, gamma=1.4):
    """Post-shock Mach number for pre-shock Mach M > 1."""
    g = gamma
    return np.sqrt((1.0 + 0.5 * (g - 1.0) * M * M) / (g * M * M - 0.5 * (g - 1.0)))


def shock_mach_from_pt_ratio(sigma, gamma=1.4, M_max=50.0):
    """Invert pt2/pt1 = sigma for the pre-shock Mach (bisection; sigma in (0, 1])."""
    if sigma >= 1.0:
        return 1.0
    lo, hi = 1.0, M_max
    if normal_shock_pt_ratio(hi, gamma) > sigma:
        raise ValueError(f"pt ratio {sigma} below the M={M_max} shock limit")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if normal_shock_pt_ratio(mid, gamma) > sigma:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * hi:
            break
    return 0.5 * (lo + hi)


def shock_report(network, x, loss_threshold=1e-4):
    """Deduced lumped shocks in choked area-change elements.

    Returns a list of dicts (element, pt_ratio, M_shock, A_shock, valid) for
    every 2-port area-change element whose converged relative total-pressure
    loss exceeds ``loss_threshold``.  ``valid`` is False when the implied
    shock area falls outside the element's port-area range (over-driven
    case: the loss cannot be a standing internal shock).
    """
    from .elements import IsentropicAreaChange

    states = network.states(x)
    out = []
    for el, plist in zip(network.elements, network._ports):
        if not isinstance(el, IsentropicAreaChange) or len(plist) != 2:
            continue
        ports = [(states[ei], sigma, network.edges[ei].area) for ei, sigma in plist]
        ports.sort(key=lambda t: t[2])  # small first
        (st_s, sig_s, A_s_port), (st_l, _, A_l_port) = ports
        sigma_pt = st_l.pt / st_s.pt
        if 1.0 - sigma_pt < loss_threshold:
            continue
        M_shock = shock_mach_from_pt_ratio(sigma_pt, network.gas.gamma)
        A_shock = A_s_port * area_ratio(M_shock, network.gas.gamma)
        out.append(
            {
                "element": el.name,
                "pt_ratio": float(sigma_pt),
                "M_shock": float(M_shock),
                "A_shock": float(A_shock),
                "valid": bool(A_s_port <= A_shock <= A_l_port * (1.0 + 1e-9)),
            }
        )
    return out
