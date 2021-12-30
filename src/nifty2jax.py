#!/usr/bin/env python3

import operator
from typing import Any, Callable, Mapping, Tuple

from jax.tree_util import tree_map
import jifty1 as jft

from . import DomainTuple, Field, MultiDomain, MultiField, Operator


def translator(from_to: Mapping) -> Callable:
    """Convenience method to rename keys in a dictionary."""
    def translate(x):
        x = x.copy()
        for old, new in from_to.items():
            if old in x:
                x[new] = x.pop(old)
        return x

    return translate


def translate_call(apply: Callable, from_to: Mapping):
    translate = translator(from_to)

    def translated_call(x):
        if isinstance(x, jft.Field):
            dipl_x = jft.Field(translate(x.val))
        else:
            dipl_x = translate(x)
        return apply(dipl_x)

    return translated_call


def unite(x, y, op=operator.add):
    """Unites two Fields or MultiFields.

    If a key is contained in both MultiFields, then the fields at that key
    combined.
    """
    x = x.val if hasattr(x, "val") else x
    y = y.val if hasattr(y, "val") else y
    if not hasattr(x, "keys") and not hasattr(y, "keys"):
        return x + y

    out = {}
    for k in x.keys() | y.keys():
        if k in x and k in y:
            out[k] = op(x[k], y[k])
        elif k in x:
            out[k] = x[k]
        else:
            out[k] = y[k]
    return jft.Field(out)


def shapewithdtype_from_domain(domain, dtype):
    if isinstance(dtype, dict):
        dtp_fallback = float  # Fallback to `float` for unspecified keys
        k2dtp = dtype
    else:
        dtp_fallback = dtype
        k2dtp = {}

    if isinstance(domain, MultiDomain):
        parameter_tree = {}
        for k, dom in domain.items():
            parameter_tree[k] = jft.ShapeWithDtype(
                dom.shape, k2dtp.get(k, dtp_fallback)
            )
    elif isinstance(domain, DomainTuple):
        parameter_tree = jft.ShapeWithDtype(domain.shape, dtype)
    else:
        raise TypeError(f"incompatible domain {domain!r}")
    return parameter_tree


def convert(op: Operator, dtype=float) -> Tuple[Any, Any]:
    if not isinstance(op, Operator):
        raise TypeError(f"invalid input type {type(op)!r}")

    if isinstance(op, (Field, MultiField)):
        parameter_tree = tree_map(jft.ShapeWithDtype.from_leave, op.val)
    else:
        parameter_tree = shapewithdtype_from_domain(op.domain, dtype)
    parameter_tree = jft.Field(parameter_tree)

    if isinstance(op, (Field, MultiField)):
        assert op.jax_expr is None
        expr = jft.Field(op.val)
    else:
        expr = op.jax_expr
        if not callable(expr):
            # TODO: implement conversion via host_callback and custom_vjp
            raise NotImplementedError("Sorry, not yet done :(")

    return expr, parameter_tree
