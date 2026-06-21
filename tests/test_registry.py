"""Phase 2 validation: field registry layout and parameter packing."""

import pytest

from fns.registry import (
    perfect_gas_layout,
    FieldRegistry,
    MDOT,
    P,
    HT,
    FL_N_ELEM,
    FL_EDGE_N_VARS,
)
from fns.params import pack_params


def test_perfect_gas_layout_indices():
    lay = perfect_gas_layout(2.0, 101325.0, 3.0e5)
    assert (MDOT, P, HT) == (0, 1, 2)
    assert lay.index == {
        "mdot": 0,
        "p": 1,
        "h_t": 2,
        "T": 3,
        "rho": 4,
        "c": 5,
        "W": 6,
        "u": 7,
        "M": 8,
        "q": 9,
    }
    assert lay.n_solve == 3
    assert lay.n_elem == 0
    assert lay.edge_n_vars == 10
    assert lay.thermo_slice() == slice(3, 7)
    assert lay.flow_slice() == slice(7, 10)


def test_perfect_gas_layout_scale_and_flexoffsets():
    lay = perfect_gas_layout(2.0, 101325.0, 3.0e5)
    assert list(lay.scale) == [2.0, 101325.0, 3.0e5]
    assert lay.field_layout[FL_N_ELEM] == 0
    assert lay.field_layout[FL_EDGE_N_VARS] == 10


def test_registry_rejects_bad_band1_order():
    reg = FieldRegistry().add_unknowns("p", "mdot", "h_t")
    with pytest.raises(ValueError):
        reg.finalize({"p": 1, "mdot": 1, "h_t": 1})


def test_registry_with_composition():
    reg = FieldRegistry()
    reg.add_unknowns("mdot", "p", "h_t", "Z_C", "Z_H")
    reg.add_thermo("T", "rho", "c", "W")
    reg.add_flow("u", "M", "q")
    lay = reg.finalize({"mdot": 1, "p": 1, "h_t": 1, "Z_C": 1, "Z_H": 1})
    assert lay.n_solve == 5
    assert lay.n_elem == 2
    assert lay.index["T"] == 5


def test_pack_params_worked_example():
    # implementation-plan.md section 2.5 worked example.
    nodes = [{"npf0": 2.0, "npf1": 3.5}, {"npf2": 7.0, "npi0": 4}, {"npf3": 1.5}]
    edges = [{"epi0": 1}, {"epi0": 0, "epf0": 9.0}]

    pn = pack_params(nodes)
    assert list(pn.fbuf) == [2.0, 3.5, 7.0, 1.5]
    assert list(pn.fptr) == [0, 2, 3, 4]
    assert list(pn.ibuf) == [4]
    assert list(pn.iptr) == [0, 0, 1, 1]
    assert pn.promoted == {}  # nothing universal on nodes
    assert pn.get_float(1, "npf2") == 7.0
    assert pn.get_int(1, "npi0") == 4

    pe = pack_params(edges)
    # epi0 is on every edge -> promoted out; only epf0 remains packed.
    assert list(pe.fbuf) == [9.0]
    assert list(pe.fptr) == [0, 0, 1]
    assert list(pe.ibuf) == []
    assert "epi0" in pe.promoted
    assert list(pe.promoted["epi0"]) == [1, 0]
    assert pe.get_float(1, "epf0") == 9.0


def test_pack_params_bool_separated():
    ents = [{"flag": True, "x": 1.0}, {"flag": False}]
    p = pack_params(ents, promote_uniform=False)
    assert p.get_bool(0, "flag") is True
    assert p.get_bool(1, "flag") is False
    assert p.get_float(0, "x") == 1.0
    assert list(p.bptr) == [0, 1, 2]
