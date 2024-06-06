# Copyright(C) 2022 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Gordian Edenhofer, Philipp Frank

import abc
from dataclasses import dataclass, field
from functools import partial
from pprint import pformat
from typing import Any, Callable, Iterable, Optional
from warnings import warn

from jax import eval_shape
from jax import numpy as jnp
from jax import random, vmap
from jax.tree_util import (
    register_pytree_node,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
)

from .misc import wrap
from .tree_math import PyTreeString, ShapeWithDtype, random_like, Vector


class Initializer:
    domain = ShapeWithDtype((2,), jnp.uint32)

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
            subkeys = tree_unflatten(struct, random.split(key, struct.num_leaves))

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


class ModelMeta(abc.ABCMeta):
    """Register all derived classes as PyTrees in JAX using metaprogramming.

    For any dataclasses.Field property with a metadata-entry named "static",
    we will either hide or expose the property to JAX depending on the value.
    """

    def __new__(mcs, name, bases, dict_, /, **kwargs):
        cls = super().__new__(mcs, name, bases, dict_, **kwargs)
        cls = dataclass(init=False, repr=False, eq=False)(cls)
        IS_STATIC_DEFAULT = True

        def tree_flatten(self):
            static = []
            dynamic = []
            for k, v in self.__dict__.items():
                # Inspired by how equinox registers properties as static in JAX
                fm = self.__dataclass_fields__.get(k)
                fm = fm.metadata if fm is not None else {}
                if fm.get("static", IS_STATIC_DEFAULT) is False:
                    dynamic.append((PyTreeString(k), v))
                else:
                    static.append((k, v))
            return (tuple(dynamic), tuple(static))

        @partial(partial, cls=cls)
        def tree_unflatten(aux, children, *, cls):
            static, dynamic = aux, children
            obj = object.__new__(cls)
            for nm, m in dynamic + static:
                setattr(obj, str(nm), m)  # unwrap any potential `PyTreeSring`s
            return obj

        # Register class and all classes deriving from it
        register_pytree_node(cls, tree_flatten, tree_unflatten)
        return cls


class NoValue:
    pass


class LazyModel(metaclass=ModelMeta):
    _domain: Any = field()
    _target: Any = field()
    _init: Any = field()

    def __init__(self, domain=NoValue, target=NoValue, init=NoValue):
        self._domain = domain
        self._target = target
        self._init = Initializer(init) if init is not NoValue else init

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def domain(self):
        if self._domain is NoValue and self._init is not NoValue:
            return eval_shape(self.init, Initializer.domain)
        return self._domain

    @property
    def target(self):
        if self._target is NoValue and self.domain is not NoValue:
            return eval_shape(self.__call__, self.domain)
        return self._target

    @property  # Needs to be a property to enable `model_a.init | model_B.init`
    def init(self) -> Initializer:
        if self._init is NoValue:
            msg = (
                "drawing white parameters"
                ";\nto silence this warning, overload the `init` method"
            )
            warn(msg)
            return Initializer(
                tree_map(lambda p: partial(random_like, primals=p), self.domain)
            )
        return self._init


class Model(LazyModel):
    """Join a callable with a domain, target, and an init method.

    From a domain and a callable, this class automatically derives the target as
    well as instantiate a default initializer if not set explicitly. More
    importantly though, it registers the class as PyTree in JAX using
    metaprogramming. By default all properties are hidden from JAX except those
    marked via `dataclasses.field(metadata=dict(static=False))` as non-static.
    """

    def __init__(
        self,
        call: Optional[Callable] = None,
        *,
        domain=NoValue,
        target=NoValue,
        init=NoValue,
        white_init=False,
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
        if init is NoValue and domain is not NoValue and white_init is True:
            init = tree_map(lambda p: partial(random_like, primals=p), domain)
        elif init is NoValue and domain is NoValue:
            raise ValueError("one of `init` or `domain` must be set")

        if domain is NoValue and init is not NoValue:
            domain = eval_shape(init, Initializer.domain)
        if target is NoValue and domain is not NoValue:
            target = eval_shape(self, domain)  # Honor overloaded `__call__`
        super().__init__(domain=domain, init=init, target=target)

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
        target=NoValue,
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
        super().__init__(call, domain=domain, target=target, white_init=white_init)


def _is_none_or_int(x):
    # Workaround as jax.tree_util.none_leaf_registry is not exposed
    return isinstance(x, int) or (x is None)


def _parse_axes(axes, domain, name=""):
    struct = tree_structure(domain)
    if isinstance(axes, int):
        axes = tree_unflatten(struct, (axes,) * struct.num_leaves)
    else:
        # Shortcut for dict only models
        if isinstance(axes, str):
            axes = (axes,)
        if isinstance(axes, Iterable) and all((isinstance(ii, str) for ii in axes)):
            dom = dict(domain)
            if dom != domain:
                msg = f"{name} must be dict-like if axes are strings"
                raise ValueError(msg)
            axes = {k: (0 if k in axes else None) for k in dom.keys()}

        ax_struct = tree_structure(axes, is_leaf=_is_none_or_int)
        if ax_struct != struct:
            msg = f"{name} structure {struct} does not match axis structure {ax_struct}"
            raise ValueError(msg)
    return axes


class VModel(LazyModel):
    model: LazyModel = field(metadata=dict(static=False))
    in_axes: Any
    out_axes: Any
    axis_size: int

    def __init__(self, model, axis_size, in_axes=0, out_axes=0):
        if not isinstance(model, LazyModel):
            raise ValueError(f"Model {model} of invalid type")
        if model.init.stupid:
            raise ValueError("can only vmap models with a non-'stupid' init")
        self.model = model

        if not isinstance(axis_size, int) and axis_size <= 0:
            raise ValueError(f"invalid axis size {axis_size}")

        self.in_axes = _parse_axes(in_axes, model.domain, "Model domain")
        self.out_axes = _parse_axes(out_axes, model.target, "Model target")

        def _init(key, func, axes):
            ks = random.split(key, axis_size)
            return vmap(func, out_axes=axes)(ks)

        def _parse_init(func, axes):
            if axes is NoValue:
                return func
            return partial(_init, func=func, axes=axes)

        parse_axes = tree_map(
            lambda x: NoValue if x is None else x, self.in_axes, is_leaf=_is_none_or_int
        )
        init = tree_map(_parse_init, self.model.init._call_or_struct, parse_axes)
        super().__init__(init=init)

    def __call__(self, x):
        axs = self.in_axes
        axs_tr = axs.tree if isinstance(axs, Vector) else axs
        x_tr = x.tree if isinstance(x, Vector) else x
        if isinstance(axs_tr, dict) and isinstance(x_tr, dict):
            axs_tr = axs_tr | {k: None for k in x_tr.keys() - axs_tr.keys()}
        axs = Vector(axs_tr) if isinstance(x, Vector) else axs_tr
        return vmap(self.model, (axs,), self.out_axes)(x)
