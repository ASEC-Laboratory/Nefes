"""Transfer / scattering matrices over a converged mean flow (theory.md s12.7).

The workflow is **force once, extract many**:

1. ``acoustic_response`` solves the perturbation system ``A(omega) x = b`` for
   **two linearly independent forcings** (an incoming wave from each terminal)
   over a user-supplied frequency array, and **stores the full perturbation
   fields** ``X1, X2`` plus the per-edge characteristic maps ``L_e``.
2. The returned ``AcousticResponse`` reconstructs the 2x2 transfer or scattering
   matrix between **any** edge pair the user later asks for -- no further solve,
   because the two forced fields span the 2-D acoustic response.

Forcing is injected by overwriting each terminal's single boundary row with a
"prescribe the incoming characteristic" row (the homogeneous reflection closure
is replaced by an imposed incoming wave).  Only the acoustic ``(f, g)`` pair is
reconstructable (two terminal forcings span a 2-D space); the entropy ``h`` is
read but not part of the 2x2.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .operator import build_acoustic_blocks, assemble_acoustic
from .characteristics import dx_to_char
from ..solver.control import states_table
from ..derive import ES_RHO, ES_C, ES_U, ES_P, ES_AREA
from ..elements.ids import MASS_FLOW_INLET, PT_INLET, P_OUTLET

_BOUNDARY_RIDS = (MASS_FLOW_INLET, PT_INLET, P_OUTLET)


@dataclass
class Terminal:
    """A 1-port boundary edge where an incoming wave can be injected/read."""

    node: int  # the boundary element
    rid: int  # its residual id (one of _BOUNDARY_RIDS)
    edge: int  # the single incident edge
    at_tail: bool  # True if the boundary is the edge's tail (wave enters as f)
    row: int  # the boundary element's single equation row
    incoming: int  # wave index injected here: 0 (f) if at_tail else 1 (g)
    outgoing: int  # the reflected/transmitted wave index read here


def find_terminals(prob) -> List[Terminal]:
    """All 1-port boundary terminals of the network (edges at a boundary node)."""
    terms = []
    for n in range(prob.n_nodes):
        rid = int(prob.node_rid[n])
        if rid not in _BOUNDARY_RIDS:
            continue
        base = int(prob.row_ptr[n])
        deg = int(prob.row_ptr[n + 1]) - base
        if deg != 1:
            raise ValueError(f"boundary node {n} has degree {deg}; a 1-port must have one edge")
        edge = int(prob.col_edge[base])
        at_tail = int(prob.tail_node[edge]) == n
        incoming = 0 if at_tail else 1
        terms.append(
            Terminal(
                node=n,
                rid=rid,
                edge=edge,
                at_tail=at_tail,
                row=int(prob.node_row_ptr[n]),
                incoming=incoming,
                outgoing=1 - incoming,
            )
        )
    return terms


def _edge_transforms(prob, x_bar, K):
    """Per-edge L_e = dx_to_char at the frozen mean state."""
    est = states_table(prob, x_bar)
    L = []
    for e in range(prob.n_edges):
        L.append(
            dx_to_char(
                float(est[ES_RHO, e]),
                float(est[ES_C, e]),
                float(est[ES_U, e]),
                float(est[ES_P, e]),
                float(est[ES_AREA, e]),
                K,
            )
        )
    return L


def _select_forcing(terms: List[Terminal], forcing: Optional[Sequence[int]]) -> List[Terminal]:
    if forcing is None:
        sel = list(terms)
    else:
        by_node = {t.node: t for t in terms}
        sel = []
        for nd in forcing:
            if nd not in by_node:
                raise ValueError(f"forcing location {nd} is not a 1-port terminal")
            sel.append(by_node[nd])
    if len(sel) != 2:
        raise ValueError(
            f"v1 scattering forces exactly 2 terminals; got {len(sel)} " "(pass `forcing=(node_a, node_b)`)"
        )
    return sel


def acoustic_response(prob, x_bar, omegas, forcing=None, *, eps=None, eps_fb=1e-6, u_floor=1e-8):
    """Force two independent cases per frequency and store the perturbation fields.

    ``omegas`` is the user frequency array; ``forcing`` is the pair of terminal
    node ids to excite (default: the network's two terminals).  Returns an
    ``AcousticResponse`` from which transfer/scattering matrices between any edge
    pair are extracted without re-solving.
    """
    omegas = np.asarray(omegas, dtype=float)
    blocks = build_acoustic_blocks(prob, x_bar, eps=eps, eps_fb=eps_fb, u_floor=u_floor)
    K = float(prob.tf[0]) / float(prob.tf[1])
    L = _edge_transforms(prob, x_bar, K)
    terms = find_terminals(prob)
    sel = _select_forcing(terms, forcing)
    ns = int(prob.n_solve)
    n = int(prob.n_col)

    X1 = np.zeros((omegas.size, n), dtype=np.complex128)
    X2 = np.zeros((omegas.size, n), dtype=np.complex128)
    for i, omega in enumerate(omegas):
        A = assemble_acoustic(omega, blocks).tolil()
        b = np.zeros((n, 2), dtype=np.complex128)
        for k, t in enumerate(sel):
            cols = [ns * t.edge + v for v in range(3)]
            row = L[t.edge][t.incoming, :]
            A.rows[t.row] = []
            A.data[t.row] = []
            for c, val in zip(cols, row):
                A[t.row, c] = val
            b[t.row, k] = 1.0
        lu = spla.splu(sp.csc_matrix(A))
        sol = lu.solve(b)
        X1[i, :] = sol[:, 0]
        X2[i, :] = sol[:, 1]

    return AcousticResponse(omegas=omegas, X1=X1, X2=X2, L=L, n_solve=ns, forcing=sel)


@dataclass
class AcousticResponse:
    """Stored two-case perturbation fields; extracts 2x2 matrices on demand."""

    omegas: np.ndarray  # (n_omega,)
    X1: np.ndarray  # (n_omega, 3E) -- case 1 (terminal 0 forced)
    X2: np.ndarray  # (n_omega, 3E) -- case 2 (terminal 1 forced)
    L: List[np.ndarray]  # per-edge dx_to_char (3x3) at the mean state
    n_solve: int
    forcing: List[Terminal]

    def _waves(self, X, edge):
        """Wave amplitudes (f, g, h) at ``edge`` for every omega: (n_omega, 3)."""
        ns = self.n_solve
        Xe = X[:, ns * edge : ns * edge + 3]
        return Xe @ self.L[edge].T

    def transfer_matrix(self, a, b):
        """Wave transfer matrix ``T_ba`` mapping (f,g)_a -> (f,g)_b: (n_omega, 2, 2).

        Read along each edge's own arrow.
        """
        Wa = np.stack([self._waves(self.X1, a)[:, :2], self._waves(self.X2, a)[:, :2]], axis=2)
        Wb = np.stack([self._waves(self.X1, b)[:, :2], self._waves(self.X2, b)[:, :2]], axis=2)
        return Wb @ np.linalg.inv(Wa)

    def scattering_matrix(self, a, b):
        """Scattering matrix mapping incoming (f_a, g_b) -> outgoing (g_a, f_b)."""
        wa1, wa2 = self._waves(self.X1, a), self._waves(self.X2, a)
        wb1, wb2 = self._waves(self.X1, b), self._waves(self.X2, b)
        fa = np.stack([wa1[:, 0], wa2[:, 0]], axis=1)
        ga = np.stack([wa1[:, 1], wa2[:, 1]], axis=1)
        fb = np.stack([wb1[:, 0], wb2[:, 0]], axis=1)
        gb = np.stack([wb1[:, 1], wb2[:, 1]], axis=1)
        In = np.stack([fa, gb], axis=1)
        Out = np.stack([ga, fb], axis=1)
        return Out @ np.linalg.inv(In)
