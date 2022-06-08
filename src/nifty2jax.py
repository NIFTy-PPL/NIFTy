# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import operator
from functools import partial, reduce
from typing import Any, Callable, Optional, Union
from warnings import warn

from jax.tree_util import register_pytree_node_class

from . import re as jft
from .domain_tuple import DomainTuple
from .field import Field
from .multi_domain import MultiDomain
from .multi_field import MultiField
from .operators.operator import Operator
from .sugar import makeField


def spaces_to_axes(domain, spaces):
    """Converts spaces in a domain to axes of the underlying NumPy array."""
    if spaces is None:
        return None

    domain = DomainTuple.make(domain)
    axes = tuple(domain.axes[sp_index] for sp_index in spaces)
    axes = reduce(operator.add, axes) if len(axes) > 0 else axes
    return axes


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


@register_pytree_node_class
class Model(jft.Field):
    """Modified field class with an additional call method taking itself as
    input.
    """
    def __init__(self, apply: Optional[Callable], val, domain=None, flags=None):
        """Instantiates a modified field with an accompanying callable.

        Parameters
        ----------
        apply : callable
            Method acting on `val`.
        val : object
            Arbitrary, flatten-able objects.
        domain : dict or None, optional
            Domain of the field, e.g. with description of modes and volume.
        flags : set, str or None, optional
            Capabilities and constraints of the field.
        """
        super().__init__(val, domain, flags)
        self._apply = apply

    def tree_flatten(self):
        return ((self._val, ), (self._apply, self._domain, self._flags))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            aux_data[0], *children, domain=aux_data[1], flags=aux_data[2]
        )

    def __call__(self, *args, **kwargs):
        if self._apply is None:
            nie = "no `apply` method specified; behaving like field"
            raise NotImplementedError(nie)
        return self._apply(*args, **kwargs)

    @property
    def field(self):
        return jft.Field(self.val, domain=self.domain, flags=self.flags)

    def __str__(self):
        s = f"Model(\n{self._apply},\n{self.val}"
        if self._domain:
            s += f",\ndomain={self._domain}"
        if self._flags:
            s += f",\nflags={self._flags}"
        s += ")"
        return s

    def __repr__(self):
        s = f"Model(\n{self._apply!r},\n{self.val!r}"
        if self._domain:
            s += f",\ndomain={self._domain!r}"
        if self._flags:
            s += f",\nflags={self._flags!r}"
        s += ")"
        return s


def wrap_nifty_call(op, target_dtype=float) -> Callable[[Any], jft.Field]:
    from jax.experimental.host_callback import call

    if callable(op.jax_expr):
        warn("wrapping operator that has a callable `.jax_expr`")

    def pack_unpack_call(x):
        x = makeField(op.domain, x)
        return op(x).val

    # TODO: define custom JVP and VJP rules
    pt = shapewithdtype_from_domain(op.target, target_dtype)
    hcb_call = partial(call, pack_unpack_call, result_shape=pt)

    def wrapped_call(x) -> jft.Field:
        return jft.Field(hcb_call(x))

    return wrapped_call


def convert(nifty_obj: Union[Operator,DomainTuple,MultiDomain], dtype=float) -> Model:
    if not isinstance(nifty_obj, (Operator, DomainTuple, MultiDomain)):
        raise TypeError(f"invalid input type {type(nifty_obj)!r}")

    if isinstance(nifty_obj, (Field, MultiField)):
        expr = None
        parameter_tree = jft.Field(nifty_obj.val)
    elif isinstance(nifty_obj, (DomainTuple, MultiDomain)):
        expr = None
        parameter_tree = shapewithdtype_from_domain(nifty_obj, dtype)
    else:
        expr = nifty_obj.jax_expr
        parameter_tree = shapewithdtype_from_domain(nifty_obj.domain, dtype)
        if not callable(expr):
            # TODO: implement conversion via host_callback and custom_vjp
            raise NotImplementedError("Sorry, not yet done :(")

    return Model(expr, parameter_tree)
