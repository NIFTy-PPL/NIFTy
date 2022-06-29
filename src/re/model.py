# Copyright(C) 2022 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from pprint import pformat
from typing import Callable, Optional
from warnings import warn

from jax import numpy as jnp
from jax import eval_shape, linear_transpose

from .forest_util import ShapeWithDtype
from .sugar import random_like


class AbstractModel():
    def __call__(self, *args, **kwargs):
        return NotImplementedError()

    @property
    def domain(self):
        raise NotImplementedError()

    @property
    def target(self):
        return eval_shape(self.__call__, self.domain)

    def init(self, key, *args, **kwargs):
        msg = (
            "drawing white parameters"
            ";\nto silence this warning, overload the `init` method"
        )
        warn(msg)
        return random_like(key, self.domain)

    def transpose(self):
        apply_T = linear_transpose(self.__call__, self.domain)
        # Yield a concrete model b/c __init__ signature is unspecified
        return Model(
            apply_T,
            domain=self.target,
            _target=self.domain,
            _linear_transpose=self.__call__
        )

    @property
    def T(self):
        return self.transpose()


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
        init: Optional[Callable] = None,
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
        self._apply = apply
        if init is None and domain is None:
            raise ValueError("one of `init` or `domain` must be set")
        if domain is None and init is not None:
            domain = eval_shape(init, ShapeWithDtype((2, ), jnp.uint32))
        self._domain = domain
        self._init = init

        self._apply_inverse = apply_inverse
        self._linear_transpose = _linear_transpose
        self._target = _target

    def __call__(self, *args, **kwargs):
        return self._apply(*args, **kwargs)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        if self._target is not None:
            return self._target
        return super().target

    def init(self, *args, **kwargs):
        if callable(self._init):
            return self._init(*args, **kwargs)
        if self.domain is not None:
            return super().init(*args, **kwargs)
        raise NotImplementedError("must specify callable `init` or set the `domain`")

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
        rep = pformat(self._apply).replace("\n", "\n\t").strip()
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
