# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import operator
from copy import deepcopy
from pprint import pformat

from jax import numpy as jnp
from jax.tree_util import (
    register_pytree_node_class, tree_leaves, tree_map, tree_structure
)

from .vector_math import matmul, max, min, size, sum


def _copy(obj):
    return obj.copy() if hasattr(obj, "copy") else deepcopy(obj)


def _value_op(op, name=None):
    def value_call(lhs, *args, **kwargs):
        return op(lhs.tree, *args, **kwargs)

    name = op.__name__ if name is None else name
    value_call.__name__ = f"__{name}__"
    return value_call


def _unary_op(op, name=None):
    def unary_call(lhs):
        return tree_map(op, lhs)

    name = op.__name__ if name is None else name
    unary_call.__name__ = f"__{name}__"
    return unary_call


def _broadcast_binary_op(op, lhs, rhs):
    from itertools import repeat

    ts_lhs = tree_structure(lhs)
    ts_rhs = tree_structure(rhs)
    # Catch non-objects scalars and 0d array-likes with a `ndim` property
    if jnp.isscalar(lhs) or getattr(lhs, "ndim", -1) == 0:
        lhs = ts_rhs.unflatten(repeat(lhs, ts_rhs.num_leaves))
    elif jnp.isscalar(rhs) or getattr(rhs, "ndim", -1) == 0:
        rhs = ts_lhs.unflatten(repeat(rhs, ts_lhs.num_leaves))
    elif ts_lhs.num_nodes != ts_rhs.num_nodes:
        ve = f"invalid binary operation {op} for {ts_lhs!r} and {ts_rhs!r}"
        raise ValueError(ve)
    return tree_map(op, lhs, rhs)


def _binary_op(op, name=None):
    def binary_call(lhs, rhs):
        return _broadcast_binary_op(op, lhs, rhs)

    name = op.__name__ if name is None else name
    binary_call.__name__ = f"__{name}__"
    return binary_call


def _rev_binary_op(op, name=None):
    def binary_call(lhs, rhs):
        return _broadcast_binary_op(op, rhs, lhs)

    name = op.__name__ if name is None else name
    binary_call.__name__ = f"__r{name}__"
    return binary_call


def _fwd_rev_binary_op(op, name=None):
    return (_binary_op(op, name=name), _rev_binary_op(op, name=name))


@register_pytree_node_class
class Vector():
    """Value storage for arbitrary objects with added numerics."""
    def __init__(self, tree):
        """Instantiates a vector.

        Parameters
        ----------
        tree : object
            Arbitrary, flatten-able objects.
        """
        self._tree = tree

    def tree_flatten(self):
        return ((self._tree, ), None)

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @property
    def tree(self):
        """Retrieves a **view** of the vector's values."""
        return self._tree

    def __len__(self):
        return size(self)

    @property
    def size(self):
        return len(self)

    @property
    def shape(self):
        return (len(self), )

    def copy(self):
        return tree_map(_copy, self)

    def ravel(self):
        return self

    def __repr__(self):
        rep = pformat(self.tree).replace("\n", "\n\t").strip()
        s = f"{self.__class__.__name__}(\n\t{rep}\n)"
        s = s.replace("\n", "").replace("\t", "") if s.count("\n") <= 2 else s
        return s

    def __str__(self):
        return repr(self)

    __bool__ = _value_op(bool)

    def __hash__(self):
        return hash(tuple(tree_leaves(self)))

    # NOTE, this partly redundant code could be abstracted away using
    # `setattr`. However, static code analyzers will not be able to infer the
    # properties then.

    __add__, __radd__ = _fwd_rev_binary_op(operator.add)
    __sub__, __rsub__ = _fwd_rev_binary_op(operator.sub)
    __mul__, __rmul__ = _fwd_rev_binary_op(operator.mul)
    __truediv__, __rtruediv__ = _fwd_rev_binary_op(operator.truediv)
    __floordiv__, __rfloordiv__ = _fwd_rev_binary_op(operator.floordiv)
    __pow__, __rpow__ = _fwd_rev_binary_op(operator.pow)
    __mod__, __rmod__ = _fwd_rev_binary_op(operator.mod)
    __matmul__ = __rmatmul__ = matmul  # arguments of matmul commute

    def __divmod__(self, other):
        return self // other, self % other

    def __rdivmod__(self, other):
        return other // self, other % self

    __or__, __ror__ = _fwd_rev_binary_op(operator.or_, "or")
    __xor__, __rxor__ = _fwd_rev_binary_op(operator.xor)
    __and__, __rand__ = _fwd_rev_binary_op(operator.and_, "and")
    __lshift__, __rlshift__ = _fwd_rev_binary_op(operator.lshift)
    __rshift__, __rrshift__ = _fwd_rev_binary_op(operator.rshift)

    __lt__ = _binary_op(operator.lt)
    __le__ = _binary_op(operator.le)
    __eq__ = _binary_op(operator.eq)
    __ne__ = _binary_op(operator.ne)
    __ge__ = _binary_op(operator.ge)
    __gt__ = _binary_op(operator.gt)

    __neg__ = _unary_op(operator.neg)
    __pos__ = _unary_op(operator.pos)
    __abs__ = _unary_op(operator.abs)
    __invert__ = _unary_op(operator.invert)

    conj = conjugate = _unary_op(jnp.conj)
    real = property(_unary_op(jnp.real))
    imag = property(_unary_op(jnp.imag))
    dot = matmul

    def max(self):
        return max(self)

    def min(self):
        return min(self)

    def sum(self):
        return sum(self)

    __getitem__ = _value_op(operator.getitem)
    __contains__ = _value_op(operator.contains)
    __iter__ = _value_op(iter)
