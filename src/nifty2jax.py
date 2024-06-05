# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import operator
from functools import partial, reduce
from typing import Any, Callable, Tuple, Union, Dict
from warnings import warn

from . import re as jft
from .domain_tuple import DomainTuple
from .field import Field
from .multi_domain import MultiDomain
from .multi_field import MultiField
from .operators.operator import Operator
from .sugar import makeField


def spaces_to_axes(domain, spaces) -> Union[Tuple, int, None]:
    """Converts spaces in a domain to axes of the underlying NumPy array."""
    if spaces is None:
        return None

    domain = DomainTuple.make(domain)
    axes = tuple(domain.axes[sp_index] for sp_index in spaces)
    axes = reduce(operator.add, axes) if len(axes) > 0 else axes
    return axes


def shapewithdtype_from_domain(
    domain, dtype
) -> Union[jft.ShapeWithDtype, Dict[str, jft.ShapeWithDtype]]:
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


def wrap_nifty_call(op, target_dtype=float) -> Callable[[Any], jft.Vector]:
    from jax import pure_callback

    if callable(op.jax_expr):
        warn("wrapping operator that has a callable `.jax_expr`")

    def nifty_call(x):
        # Minimal parts that must run outside of JAX
        x = makeField(op.domain, x)
        return op(x).val

    # TODO: define custom JVP and VJP rules
    pt = shapewithdtype_from_domain(op.target, target_dtype)
    hcb_call = partial(pure_callback, nifty_call, pt)

    def wrapped_call(x) -> jft.Vector:
        x = x.tree if isinstance(x, jft.Vector) else x
        return jft.Vector(hcb_call(x))

    return wrapped_call


def convert(
    nifty_obj: Union[Operator, DomainTuple, MultiDomain],
    dtype=float
) -> Union[jft.Model, jft.Vector, jft.ShapeWithDtype, Dict[str,
                                                          jft.ShapeWithDtype]]:
    if not isinstance(nifty_obj, (Operator, DomainTuple, MultiDomain)):
        raise TypeError(f"invalid input type {type(nifty_obj)!r}")

    if isinstance(nifty_obj, (Field, MultiField)):
        return jft.Vector(nifty_obj.val)
    elif isinstance(nifty_obj, (DomainTuple, MultiDomain)):
        return shapewithdtype_from_domain(nifty_obj, dtype)
    else:
        expr = nifty_obj.jax_expr
        parameter_tree = shapewithdtype_from_domain(nifty_obj.domain, dtype)
        if not callable(expr):
            # TODO: implement conversion via host_callback and custom_vjp
            raise NotImplementedError("Sorry, not yet done :(")

    return jft.Model(expr, domain=parameter_tree)
