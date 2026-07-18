"""Scalar-parameter protocol for objects attached to elements.

An object stored as an element parameter (a dynamic source, a perturbation boundary
condition, a transfer function) can expose its own scalar knobs by implementing three
hooks:

* ``param_descriptors()`` -- the knobs as :class:`~nefes.elements.parameters.ParamDescriptor` rows,
* ``get(name)`` -- the current value of one knob,
* ``with_value(name, value)`` -- a modified copy (functional, never in-place).

The parameter inventory recurses into any element field whose value implements the
protocol, so the knobs join the network's address space (``flame.dynamic_source.gain``,
``inlet.perturbation_bc.magnitude``) and every driver built on ``get``/``set``/
``with_params`` -- sweeps, trajectories, eigenvalue sensitivities -- picks them up
unchanged.  Ownership of "which knobs exist" stays with the object: a custom transfer
exposes its own scalars without touching any element schema.

Main exports: :func:`is_parametric`, :class:`AttributeParams`.

See also
--------
nefes.elements.parameters : the descriptor schema the hooks reuse.
nefes.shell.params : the inventory and write paths that consume the protocol.
"""

import copy
from typing import Tuple

from .parameters import ParamDescriptor

_HOOKS = ("param_descriptors", "get", "with_value")


def is_parametric(obj) -> bool:
    """Whether ``obj`` implements the scalar-parameter protocol (all three hooks)."""
    return obj is not None and all(callable(getattr(obj, h, None)) for h in _HOOKS)


class AttributeParams:
    """Protocol implementation for objects whose knobs are plain instance attributes.

    A subclass lists its knobs in ``_PARAM_SPEC`` (descriptors whose ``name`` doubles as
    the attribute name); ``get`` reads the attribute (as a real scalar) and ``with_value``
    returns a shallow copy with the validated value set.  Suits immutable-by-convention
    value objects; anything with derived internal state overrides :meth:`with_value`.
    """

    _PARAM_SPEC: Tuple[ParamDescriptor, ...] = ()

    def param_descriptors(self) -> Tuple[ParamDescriptor, ...]:
        """The scalar knobs this object exposes."""
        return tuple(self._PARAM_SPEC)

    def _descriptor(self, name: str) -> ParamDescriptor:
        for d in self._PARAM_SPEC:
            if d.name == name:
                return d
        known = [d.name for d in self._PARAM_SPEC]
        raise KeyError(f"{type(self).__name__} has no parameter {name!r}; it has: {known or 'none'}")

    def get(self, name: str) -> float:
        """Current value of knob ``name`` (the real part, for a real-valued complex store)."""
        v = complex(getattr(self, self._descriptor(name).name))
        return float(v.real)

    def with_value(self, name: str, value):
        """A copy of this object with knob ``name`` set to the validated ``value``."""
        d = self._descriptor(name)
        v = d.validate(value, where=type(self).__name__)
        out = copy.copy(self)
        setattr(out, d.name, v)
        return out
