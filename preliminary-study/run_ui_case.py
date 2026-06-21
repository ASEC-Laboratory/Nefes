#!/usr/bin/env python3
"""Solve a flow network saved from the node-graph UI editor and write a UI-loadable results file.

Usage:
    python run_ui_case.py <case>.yaml [-o results.json] [--mdot0 X] [--tol T] [-v]

Build the network in the UI editor with the "FNS Flow Network" model, save it,
run this script on the save file, then load the produced JSON in the UI's
data pane to color the canvas with the solution.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fns import solve
from fns.ui_bridge import UICaseError, load_case, supersonic_chain_guess, write_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", help="UI save file (.yaml)")
    parser.add_argument("-o", "--output", help="results JSON path (default: <case>-results.json)")
    parser.add_argument("--mdot0", type=float, default=0.0,
                        help="initial mass-flow guess on every edge [kg/s] (default 0)")
    parser.add_argument("--tol", type=float, default=1e-10, help="convergence tolerance")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Newton iteration budget per homotopy stage (default 200)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print solver iterations")
    args = parser.parse_args(argv)

    try:
        net, meta = load_case(args.case)
    except (UICaseError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"loaded '{args.case}': {len(net.elements)} elements, {net.n_edges} edges")
    x0 = supersonic_chain_guess(net)
    if x0 is not None:
        # supersonic intake: select the supersonic branch via the chain guess,
        # pure Newton (no friction homotopy), tolerance at the FB floor
        print("supersonic inlet detected: using supersonic-branch initial guess")
        result = solve(net, x0=x0, tol=max(args.tol, 2e-9), stab_stages=(0.0,),
                       max_iter=args.max_iter, verbose=args.verbose)
    else:
        result = solve(net, x0=net.initial_guess(mdot0=args.mdot0), tol=args.tol,
                       max_iter=args.max_iter, verbose=args.verbose)
    print(result)
    if not result.converged:
        print("solver did NOT converge", file=sys.stderr)
        choked = net.choking_report(result.x)
        if choked:
            print("\nedges at or near their choking limit (the likely cause -- the",
                  file=sys.stderr)
            print("boundary data demand more flow than these passages can carry;",
                  file=sys.stderr)
            print("see docs/theory.md section 11):", file=sys.stderr)
            for edge, M, ratio in choked:
                tail = net.elements[edge.tail].name
                head = net.elements[edge.head].name
                print(f"  edge '{edge.name}' ({tail} -> {head}, area {edge.area:g}): "
                      f"|M| = {M:.3f}, flux at {100 * ratio:.1f} % of choking",
                      file=sys.stderr)
            print("remedies: lower the driving pressure difference, enlarge the",
                  file=sys.stderr)
            print("flagged areas, or add a LossElement to throttle the path.",
                  file=sys.stderr)
        else:
            print("no edge is near choking -- check for inconsistent boundary data "
                  "or junction model selection (docs/theory.md sections 7.5 and 11)",
                  file=sys.stderr)

    print()
    print(net.report(result.x))

    if result.converged:
        from fns import shock_report

        shocks = shock_report(net, result.x)
        flagged = net.choking_report(result.x, threshold=0.999)
        choked = [(e, M, r) for e, M, r in flagged if M < 1.01]
        supersonic = [(e, M, r) for e, M, r in flagged if M >= 1.01]
        if choked:
            print("\nchoked edges (flow capped at the local choking limit):")
            for edge, M, ratio in choked:
                print(f"  edge '{edge.name}' (area {edge.area:g}): |M| = {M:.4f}")
        if supersonic:
            print("\nsupersonic edges:")
            for edge, M, ratio in supersonic:
                print(f"  edge '{edge.name}' (area {edge.area:g}): |M| = {M:.4f}")
        from fns import SupersonicOutlet
        states = net.states(result.x)
        for ei, el in enumerate(net.elements):
            if isinstance(el, SupersonicOutlet):
                edge_idx = net._ports[ei][0][0]
                if abs(states[edge_idx].M) < 1.0:
                    print(f"warning: SupersonicOutlet '{el.name}' has a SUBSONIC exit "
                          f"(M = {states[edge_idx].M:.3f}); use PressureOutlet instead",
                          file=sys.stderr)
        for sh in shocks:
            note = "" if sh["valid"] else "  [implied shock outside the element: over-driven case]"
            print(
                f"  lumped shock in '{sh['element']}': pre-shock M = {sh['M_shock']:.3f}, "
                f"shock area = {sh['A_shock']:.4g} m^2, pt ratio = {sh['pt_ratio']:.4f}{note}"
            )

    out = args.output or os.path.splitext(args.case)[0] + "-results.json"
    write_results(net, meta, result.x, out)
    print(f"\nresults written to '{out}' (load it in the UI data pane)")
    return 0 if result.converged else 1


if __name__ == "__main__":
    sys.exit(main())
