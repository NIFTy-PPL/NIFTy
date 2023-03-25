# Copyright(C) 2022 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from pprint import pformat
from typing import Callable, Optional
from warnings import warn

from jax import numpy as jnp
from jax import eval_shape, linear_transpose
from jax import random
from jax.tree_util import tree_map, tree_structure, tree_unflatten

from .tree_math import ShapeWithDtype, random_like


class AbstractModel():
    # TODO: This should really be a custom JAX type such that domain and init
    # are preserved under JAX transformations.

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def domain(self):
        raise NotImplementedError()

    @property
    def target(self):
        if getattr(self, "_target", None) is None:
            self._target = eval_shape(self.__call__, self.domain)
        return self._target

    @property
    def init(self):
        if getattr(self, "_init", None) is None:
            msg = (
                "drawing white parameters"
                ";\nto silence this warning, overload the `init` method"
            )
            warn(msg)
            self._init = Initializer(partial(random_like, primals=self.domain))
        return self._init

    def transpose(self):
        # TODO: Split into linear model
        if getattr(self, "_linear_transpose", None) is None:
            self._linear_transpose = linear_transpose(
                self.__call__, self.domain
            )
        # Yield a concrete model b/c __init__ signature is unspecified
        return Model(
            self._linear_transpose,
            domain=self.target,
            _target=self.domain,
            _linear_transpose=self.__call__
        )

    @property
    def T(self):
        return self.transpose()


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


class Model(AbstractModel):
    """Thin wrapper of a method to jointly store the shape of its primals
    (`domain`) and optionally an initialization method and an associated
    inverse method.
    """
    def __init__(
        self,
        apply: Callable,
        *,
        domain=None,
        init=None,
        apply_inverse: Optional[Callable] = None,
        _linear_transpose: Optional[Callable] = None,
        _target=None,
    ):
        """Wrap a callable and associate it with a `domain`.

        Parameters
        ----------
        apply : callable
            Method acting on objects of type `domain`.
        domain : object, optional
            PyTree of objects with a shape and dtype attribute. Inferred from
            init if not specified.
        init : callable, optional
            Initialization method taking a PRNG key as first argument and
            creating an object of type `domain`. Inferred from `domain`
            assuming a white normal prior if not set.
        apply_inverse : callable, optional
            If the apply method has an inverse, this can be stored in addition
            to the apply method itself.
        """
        self.apply = apply
        if init is None and domain is None:
            raise ValueError("one of `init` or `domain` must be set")
        if domain is None and init is not None:
            domain = eval_shape(init, ShapeWithDtype((2, ), jnp.uint32))
        self._domain = domain
        self._init = init

        self._apply_inverse = apply_inverse
        self._linear_transpose = _linear_transpose
        self._target = _target

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        if self._target is not None:
            return self._target
        return super().target

    @property
    def init(self) -> Initializer:
        if self._init is not None:
            return Initializer(self._init)
        if self.domain is not None:
            return super().init
        raise NotImplementedError("must specify `init` or set the `domain`")

    def transpose(self):
        if self._linear_transpose is not None:
            apply_T = self._linear_transpose
        else:
            apply_T = linear_transpose(self.__call__, self.domain)
        return self.__class__(
            apply_T,
            domain=self.target,
            _target=self.domain,
            _linear_transpose=self.__call__
        )

    def inv(self):
        if not callable(self._apply_inverse):
            raise NotImplementedError("must specify callable `apply_inverse`")
        return self.__class__(
            self._apply_inverse, domain=self.target, _target=self.domain
        )

    def __repr__(self):
        s = "Model("
        rep = pformat(self.apply).replace("\n", "\n\t").strip()
        s += f"\n\t{rep}"
        if self._domain:
            rep = pformat(self._domain).replace("\n", "\n\t").strip()
            s += f",\n\tdomain={rep}"
        if self._init:
            rep = pformat(self._init).replace("\n", "\n\t").strip()
            s += f",\n\tinit={rep}"
        if self._apply_inverse:
            rep = pformat(self._apply_inverse).replace("\n", "\n\t").strip()
            s += f",\n\tapply_inverse={rep}"
        s += "\n)"
        s = s.replace("\n", "").replace("\t", "") if s.count("\n") <= 2 else s
        return s

    def __str__(self):
        return repr(self)
