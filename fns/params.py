"""Parse-time packing of per-entity model parameters into flat @njit buffers.

Heterogeneous, named, per-element-type parameters are packed CSR-style per
dtype (float/int/bool), so the @njit assembly loop reads a parameter by a single
indexed load ``buf[ptr[entity] + offset]`` with ``offset`` a compile-time
constant from the name->slot map (implementation-plan.md section 2.5).

A parameter present on *every* entity is promoted out as uniform field data
(returned separately) and dropped from the packed store.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class PackedParams:
    """Flat CSR-packed parameter buffers for one entity kind (node or edge)."""

    fbuf: np.ndarray
    fptr: np.ndarray
    ibuf: np.ndarray
    iptr: np.ndarray
    bbuf: np.ndarray
    bptr: np.ndarray
    # (entity_index, name) -> offset within that entity's dtype block
    slot: Dict[Tuple[int, str], int] = field(default_factory=dict)
    promoted: Dict[str, np.ndarray] = field(default_factory=dict)

    def get_float(self, entity: int, name: str) -> float:
        return float(self.fbuf[self.fptr[entity] + self.slot[(entity, name)]])

    def get_int(self, entity: int, name: str) -> int:
        return int(self.ibuf[self.iptr[entity] + self.slot[(entity, name)]])

    def get_bool(self, entity: int, name: str) -> bool:
        return bool(self.bbuf[self.bptr[entity] + self.slot[(entity, name)]])


def _dtype_of(v) -> str:
    if isinstance(v, bool):
        return "b"
    if isinstance(v, (int, np.integer)):
        return "i"
    return "f"


def pack_params(entity_dicts: List[Dict[str, object]], promote_uniform: bool = True) -> PackedParams:
    """Pack a list of per-entity ``{name: value}`` dicts into CSR buffers.

    Promotes parameters present on *every* entity (with a consistent dtype) into
    ``promoted`` dense arrays and drops them from the packed store.
    """
    n = len(entity_dicts)
    dicts = [dict(d) for d in entity_dicts]

    promoted: Dict[str, np.ndarray] = {}
    if promote_uniform and n > 0:
        common = set(dicts[0])
        for d in dicts[1:]:
            common &= set(d)
        for name in sorted(common):
            promoted[name] = np.array([d[name] for d in dicts])
            for d in dicts:
                del d[name]

    fbuf, ibuf, bbuf = [], [], []
    fptr = np.zeros(n + 1, dtype=np.int64)
    iptr = np.zeros(n + 1, dtype=np.int64)
    bptr = np.zeros(n + 1, dtype=np.int64)
    slot: Dict[Tuple[int, str], int] = {}

    for e, d in enumerate(dicts):
        nf = ni = nb = 0
        for name in d:  # preserve insertion order
            v = d[name]
            kind = _dtype_of(v)
            if kind == "f":
                slot[(e, name)] = nf
                fbuf.append(float(v))
                nf += 1
            elif kind == "i":
                slot[(e, name)] = ni
                ibuf.append(int(v))
                ni += 1
            else:
                slot[(e, name)] = nb
                bbuf.append(bool(v))
                nb += 1
        fptr[e + 1] = fptr[e] + nf
        iptr[e + 1] = iptr[e] + ni
        bptr[e + 1] = bptr[e] + nb

    return PackedParams(
        fbuf=np.array(fbuf, dtype=np.float64),
        fptr=fptr,
        ibuf=np.array(ibuf, dtype=np.int64),
        iptr=iptr,
        bbuf=np.array(bbuf, dtype=np.bool_),
        bptr=bptr,
        slot=slot,
        promoted=promoted,
    )
