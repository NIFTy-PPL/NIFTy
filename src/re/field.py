# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import operator
from pprint import pformat
from jax import numpy as jnp
from jax.tree_util import (
    register_pytree_node_class, tree_leaves, tree_map, tree_structure
)


def _value_op(op, name=None):
    def value_call(lhs, *args, **kwargs):
        return op(lhs.val, *args, **kwargs)

    name = op.__name__ if name is None else name
    value_call.__name__ = f"__{name}__"
    return value_call


def _unary_op(op, name=None):
    def unary_call(lhs):
        return tree_map(op, lhs)

    name = op.__name__ if name is None else name
    unary_call.__name__ = f"__{name}__"
    return unary_call


def _enforce_flags(lhs, rhs):
    flags = lhs.flags if isinstance(lhs, Field) else set()
    flags |= rhs.flags if isinstance(rhs, Field) else set()
    if "strict_domain_checking" in flags:
        ts_lhs = tree_structure(lhs)
        ts_rhs = tree_structure(rhs)

        if not hasattr(rhs, "domain"):
            te = f"RHS of type {type(rhs)} does not have a `domain` property"
            raise TypeError(te)
        if not hasattr(lhs, "domain"):
            te = f"LHS of type {type(lhs)} does not have a `domain` property"
            raise TypeError(te)
        if rhs.domain != lhs.domain or ts_rhs != ts_lhs:
            raise ValueError("domains and/or structures are incompatible")
    return flags


def _broadcast_binary_op(op, lhs, rhs):
    from itertools import repeat

    flags = _enforce_flags(lhs, rhs)

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

    out = tree_map(op, lhs, rhs)
    if flags != set():
        out._flags = flags
    return out


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


def matmul(lhs, rhs):
    """Returns the dot product of the two fields.

    Parameters
    ----------
    lhs : object
        Arbitrary, flatten-able objects.
    other : object
        Arbitrary, flatten-able objects.

    Returns
    -------
    out : float
        Dot product of fields.
    """
    from .forest_util import dot

    _enforce_flags(lhs, rhs)

    ts_lhs = tree_structure(lhs)
    ts_rhs = tree_structure(rhs)
    if ts_lhs.num_nodes != ts_rhs.num_nodes:
        ve = f"invalid operation for {ts_lhs!r} and {ts_rhs!r}"
        raise ValueError(ve)

    return dot(lhs, rhs)


dot = matmul


@register_pytree_node_class
class Field():
    """Value storage for arbitrary objects with added numerics."""
    supported_flags = {"strict_domain_checking"}

    def __init__(self, val, domain=None, flags=None):
        """Instantiates a field.

        Parameters
        ----------
        val : object
            Arbitrary, flatten-able objects.
        domain : dict or None, optional
            Domain of the field, e.g. with description of modes and volume.
        flags : set, str or None, optional
            Capabilities and constraints of the field.
        """
        self._val = val
        self._domain = {} if domain is None else dict(domain)

        flags = (flags, ) if isinstance(flags, str) else flags
        flags = set() if flags is None else set(flags)
        if not flags.issubset(Field.supported_flags):
            ve = (
                f"specified flags ({flags!r}) are not a subset of the"
                f" supported flags ({Field.supported_flags!r})"
            )
            raise ValueError(ve)
        self._flags = flags

    def tree_flatten(self):
        """Recipe for flattening fields.

        Returns
        -------
        flat_tree : tuple of two tuples
            Pair of an iterable with the children to be flattened recursively,
            and some opaque auxiliary data.
        """
        return ((self._val, ), (self._domain, self._flags))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recipe to construct fields from flattened Pytrees.

        Parameters
        ----------
        aux_data : tuple of a dict and a set
            Opaque auxiliary data describing a field.
        children: tuple
            Value of the field, i.e. unflattened children.

        Returns
        -------
        unflattened_tree : :class:`nifty8.field.Field`
            Re-constructed field.
        """
        return cls(*children, domain=aux_data[0], flags=aux_data[1])

    @property
    def val(self):
        """Retrieves a **view** of the field's values."""
        return self._val

    @property
    def domain(self):
        """Retrieves a **copy** of the field's domain."""
        return self._domain.copy()

    @property
    def flags(self):
        """Retrieves a **copy** of the field's flags."""
        return self._flags.copy()

    @property
    def size(self):
        from .forest_util import size

        return size(self)

    def __repr__(self):
        s = "Field("
        rep = pformat(self.val).replace("\n", "\n\t").strip()
        s += f"\n\t{rep}"
        if self._domain:
            rep = pformat(self._domain).replace("\n", "\n\t").strip()
            s += f",\n\tdomain={rep}"
        if self._flags:
            rep = pformat(self._flags).replace("\n", "\n\t").strip()
            s += f",\n\tflags={rep}"
        s += "\n)"
        s = s.replace("\n", "").replace("\t", "") if s.count("\n") <= 2 else s
        return s

    def __str__(self):
        return repr(self)

    def ravel(self):
        return tree_map(jnp.ravel, self)

    def __bool__(self):
        return bool(self.val)

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
    real = _unary_op(jnp.real)
    imag = _unary_op(jnp.imag)
    dot = matmul

    __getitem__ = _value_op(operator.getitem)
    __contains__ = _value_op(operator.contains)
    __len__ = _value_op(len)
    __iter__ = _value_op(iter)
