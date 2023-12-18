# Copyright(C) 2022 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import abc
import dataclasses
from functools import partial
from pprint import pformat
from typing import Any, Callable, Optional
from warnings import warn

from jax import eval_shape
from jax import numpy as jnp
from jax import random
from jax.tree_util import (
    register_pytree_node, tree_leaves, tree_map, tree_structure, tree_unflatten
)

from .misc import wrap
from .tree_math import PyTreeString, ShapeWithDtype, random_like


class Initializer():
    domain = ShapeWithDtype((2, ), jnp.uint32)

    def __new__(cls, call_or_struct):
        if isinstance(call_or_struct, Initializer):
            return call_or_struct
        obj = super().__new__(cls)
        obj._call_or_struct = call_or_struct
        obj._target = None  # Used only for caching
        return obj

    def __call__(self, key, *args, **kwargs):
        if not self.stupid:
            struct = tree_structure(self._call_or_struct)
            # Cast the subkeys to the structure of `primals`
            subkeys = tree_unflatten(
                struct, random.split(key, struct.num_leaves)
            )

            def draw(init, key):
                return init(key, *args, **kwargs)

            return tree_map(draw, self._call_or_struct, subkeys)

        return self._call_or_struct(key, *args, **kwargs)

    @property
    def target(self):
        if self._target is None:
            self._target = eval_shape(self, self.domain)
        return self._target

    @property
    def stupid(self):
        return callable(self._call_or_struct)

    def __or__(self, other):
        other = Initializer(other)
        if not self.stupid and not other.stupid:
            return Initializer(self._call_or_struct | other._call_or_struct)
        # TODO: we can actually do better here and combine the output of both
        # calls in a new call
        return NotImplemented

    def __getitem__(self, key):
        if not self.stupid:
            return Initializer(self._call_or_struct[key])
        raise NotImplementedError("'stupid' initializer not supported")

    def __len__(self):
        if not self.stupid:
            return len(self._call_or_struct)
        return len(self.target)

    def __repr__(self):
        s = "Initializer("
        rep = pformat(self._call_or_struct).replace("\n", "\n\t").strip()
        s += f"\n\t{rep}\n)"
        s = s.replace("\n", "").replace("\t", "") if s.count("\n") <= 2 else s
        return s

    def __str__(self):
        return repr(self)


class AbstractModelMeta(abc.ABCMeta):
    """Join a callable with a domain.

    From a domain and a callable, this class can automatically derive the target
    as well as instantiate a default initializer. Both can also be set
    explicitly.
    """
    def __new__(mcs, name, bases, dict_, /, **kwargs):
        cls = super().__new__(mcs, name, bases, dict_, **kwargs)

        def tree_flatten(self):
            field4static = "_static_fields"
            static = [(field4static, getattr(self, field4static))]
            dynamic = []
            for k, v in self.__dict__.items():
                if k == field4static:
                    continue
                if k in getattr(self, field4static):
                    static.append((k, v))
                else:
                    v = v if not isinstance(v, str) else PyTreeString(v)
                    dynamic.append((PyTreeString(k), v))
            return (tuple(dynamic), tuple(static))

        @partial(partial, cls=cls)
        def tree_unflatten(aux, children, *, cls):
            static, dynamic = aux, children
            obj = object.__new__(cls)
            for nm, m in dynamic + static:
                object.__setattr__(
                    obj, str(nm), m
                )  # unwrap any potential `PyTreeSring`s
            return obj

        # Register class and all classes deriving from it
        register_pytree_node(cls, tree_flatten, tree_unflatten)
        return cls


class _NoValue():
    pass


@dataclasses.dataclass(init=False, repr=False, eq=False)
class AbstractModel(metaclass=AbstractModelMeta):
    _domain: Any = _NoValue
    _target: Any = _NoValue
    _init: Any = _NoValue
    _static_fields: tuple[str, ...] = (
        "_domain",
        "_target",
        "_init",
        "_static_fields",
    )

    def __init__(
        self,
        domain=_NoValue,
        target=_NoValue,
        init=_NoValue,
        static_fields=_NoValue,
    ):
        self._domain = domain
        self._target = target
        self._init = Initializer(init) if init is not _NoValue else init
        fields = dataclasses.fields(self.__class__)
        if static_fields is _NoValue:
            static_fields = fields["_static_fields"].default
        self._static_fields = static_fields

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def domain(self):
        if self._domain is _NoValue and self._init is not _NoValue:
            return eval_shape(self.init, Initializer.domain)
        if self._domain is not _NoValue:
            return self._domain
        raise NotImplementedError()

    @property
    def target(self):
        if self._target is _NoValue:
            return eval_shape(self.__call__, self.domain)
        return self._target

    @property  # Needs to be a property to enable `model_a.init | model_B.init`
    def init(self) -> Initializer:
        if self._init is _NoValue:
            msg = (
                "drawing white parameters"
                ";\nto silence this warning, overload the `init` method"
            )
            warn(msg)
            return Initializer(
                tree_map(
                    lambda p: partial(random_like, primals=p), self.domain
                )
            )
        return self._init


class Model(AbstractModel):
    """Thin wrapper for a callable to jointly store it with the shape of its
    primals (`domain`) and optionally an initialization method.
    """
    def __init__(
        self,
        call: Optional[Callable] = None,
        *,
        domain=_NoValue,
        target=_NoValue,
        init=_NoValue,
        white_init=False,
        static_fields=_NoValue,
    ):
        """Wrap a callable and associate it with a `domain`.

        Parameters
        ----------
        call : callable
            Method acting on objects of type `domain`.
        domain : tree-like structure of ShapeWithDtype, optional
            PyTree of objects with a shape and dtype attribute. Inferred from
            init if not specified.
        target : tree-like structure of ShapeWithDtype, optional
            PyTree of objects with a shape and dtype attribute akin to the
            output of `call`. Inferrred from `call` and `domain` if not set.
        init : callable, optional
            Initialization method taking a PRNG key as first argument and
            creating an object of type `domain`. Inferred from `domain`
            assuming a white normal prior if not set.
        white_init : bool, optional
            If `True`, the domain is set to a white normal prior. Defaults to
            `False`.
        """
        self._call = call
        if init is _NoValue and domain is not _NoValue and white_init is True:
            init = tree_map(lambda p: partial(random_like, primals=p), domain)
        elif init is _NoValue and domain is _NoValue:
            raise ValueError("one of `init` or `domain` must be set")

        if domain is _NoValue and init is not _NoValue:
            domain = eval_shape(init, Initializer.domain)
        if target is _NoValue and domain is not _NoValue:
            target = eval_shape(call, domain)
        if static_fields is _NoValue:
            static_fields = AbstractModel._static_fields + ("_call", )
        super().__init__(
            domain=domain,
            init=init,
            target=target,
            static_fields=static_fields,
        )

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        if self._call:
            rep = pformat(self._call).replace("\n", "\n\t").strip()
        else:
            rep = f"<bound method {self.__class__.__name__}.__call__>"
        s += f"\n\t{rep}"
        if self._domain is not None:
            rep = pformat(self._domain).replace("\n", "\n\t").strip()
            s += f",\n\tdomain={rep}"
        if self._init is not None:
            rep = pformat(self._init).replace("\n", "\n\t").strip()
            s += f",\n\tinit={rep}"
        s += "\n)"
        s = s.replace("\n", "").replace("\t", "") if s.count("\n") <= 2 else s
        return s

    def __str__(self):
        return repr(self)


class WrappedCall(Model):
    def __init__(
        self,
        call: Callable,
        *,
        name=None,
        shape=(),
        dtype=None,
        white_init=False,
        target=_NoValue,
    ):
        """Transforms `call` such that instead of it acting on `input` it
        selects `name` from `input` using `input[name]`.

        Parameters
        ----------
        call : callable
            Callable to wrap.
        name : hashable, optional
            New name of the `input` on which `call` acts.
        shape : tuple or tree-like structure of ShapeWithDtype
            Shape of old `input` on which `call` acts. This can also be an
            arbitrary shape-dtype structure in which case `dtype` is ignored.
            Defaults to a scalar.
        dtype : dtype or tree-like structure of ShapeWithDtype
            If `shape` is a tuple, this is the dtype of the old `input` on
            which `call` acts. This is redundant if `shape` already encodes the
            `dtype`.

        See :class:`Model` for details on the remaining arguments.
        """
        leaves = tree_leaves(shape)
        isswd = all(hasattr(e, "shape") and hasattr(e, "dtype") for e in leaves)
        isswd &= len(leaves) > 0
        domain = ShapeWithDtype(shape, dtype) if not isswd else shape

        if name is not None:
            call = wrap(call, name=name)
            domain = {name: domain}
        super().__init__(
            call, domain=domain, target=target, white_init=white_init
        )
