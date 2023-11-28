#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import operator

from jax.tree_util import register_pytree_node_class, tree_map


def _unary_op(op, name=None):
    def unary_call(lhs):
        return op(lhs._str)

    name = op.__name__ if name is None else name
    unary_call.__name__ = f"__{name}__"
    return unary_call


def _binary_op(op, name=None):
    def binary_call(lhs, rhs):
        lhs = lhs._str if isinstance(lhs, PyTreeString) else lhs
        rhs = rhs._str if isinstance(rhs, PyTreeString) else rhs
        out = op(lhs, rhs)
        return PyTreeString(out) if isinstance(out, str) else out

    name = op.__name__ if name is None else name
    binary_call.__name__ = f"__{name}__"
    return binary_call


def _rev_binary_op(op, name=None):
    def binary_call(lhs, rhs):
        lhs = lhs._str if isinstance(lhs, PyTreeString) else lhs
        rhs = rhs._str if isinstance(rhs, PyTreeString) else rhs
        out = op(rhs, lhs)
        return PyTreeString(out) if isinstance(out, str) else out

    name = op.__name__ if name is None else name
    binary_call.__name__ = f"__r{name}__"
    return binary_call


def _fwd_rev_binary_op(op, name=None):
    return (_binary_op(op, name=name), _rev_binary_op(op, name=name))


@register_pytree_node_class
class PyTreeString():
    def __init__(self, str):
        self._str = str

    def tree_flatten(self):
        return ((), (self._str, ))

    @classmethod
    def tree_unflatten(cls, aux, _):
        return cls(*aux)

    def __str__(self):
        return self._str

    def __repr__(self):
        return f"{self.__class__.__name__}({self._str!r})"

    __lt__ = _binary_op(operator.lt)
    __le__ = _binary_op(operator.le)
    __eq__ = _binary_op(operator.eq)
    __ne__ = _binary_op(operator.ne)
    __ge__ = _binary_op(operator.ge)
    __gt__ = _binary_op(operator.gt)

    __add__, __radd__ = _fwd_rev_binary_op(operator.add)
    __mul__, __rmul__ = _fwd_rev_binary_op(operator.mul)

    lower = _unary_op(str.lower)
    upper = _unary_op(str.upper)

    __hash__ = _unary_op(str.__hash__)

    startswith = _binary_op(str.startswith)


def hide_strings(a):
    return tree_map(lambda x: PyTreeString(x) if isinstance(x, str) else x, a)
