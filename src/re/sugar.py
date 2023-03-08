# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from collections.abc import Iterable
from typing import Any, Callable, Dict, Hashable, Mapping, TypeVar, Union

import jax
from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_map, tree_reduce, tree_structure, tree_unflatten

from .field import Field

O = TypeVar('O')
I = TypeVar('I')


def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]


def split(mappable, keys):
    """Split a dictionary into one containing only the specified keys and one
    with all of the remaining ones.
    """
    sel, rest = {}, {}
    for k, v in mappable.items():
        if k in keys:
            sel[k] = v
        else:
            rest[k] = v
    return sel, rest


def isiterable(candidate):
    try:
        iter(candidate)
        return True
    except (TypeError, AttributeError):
        return False


def is1d(ls: Any) -> bool:
    """Indicates whether the input is one dimensional.

    An object is considered one dimensional if it is an iterable of
    non-iterable items.
    """
    if hasattr(ls, "ndim"):
        return ls.ndim == 1
    if not isiterable(ls):
        return False
    return all(not isiterable(e) for e in ls)


def doc_from(original):
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


def ducktape(call: Callable[[I], O],
             key: Hashable) -> Callable[[Mapping[Hashable, I]], O]:
    def named_call(p):
        return call(p[key])

    return named_call


def ducktape_left(call: Callable[[I], O],
                  key: Hashable) -> Callable[[I], Dict[Hashable, O]]:
    def named_call(p):
        return {key: call(p)}

    return named_call


def sum_of_squares(tree) -> Union[jnp.ndarray, jnp.inexact]:
    return tree_reduce(jnp.add, tree_map(lambda x: jnp.sum(x**2), tree), 0.)


def mean(forest):
    from functools import reduce

    norm = 1. / len(forest)
    if isinstance(forest[0], Field):
        m = norm * reduce(Field.__add__, forest)
        return m
    else:
        m = norm * reduce(Field.__add__, (Field(t) for t in forest))
        return m.val


def mean_and_std(forest, correct_bias=True):
    if isinstance(forest[0], Field):
        m = mean(forest)
        mean_of_sq = mean(tuple(t**2 for t in forest))
    else:
        m = Field(mean(forest))
        mean_of_sq = mean(tuple(Field(t)**2 for t in forest))

    n = len(forest)
    scl = jnp.sqrt(n / (n - 1)) if correct_bias else 1.
    std = scl * tree_map(jnp.sqrt, mean_of_sq - m**2)
    if isinstance(forest[0], Field):
        return m, std
    else:
        return m.val, std.val


def random_like(key: Iterable, primals, rng: Callable = random.normal):
    import numpy as np

    struct = tree_structure(primals)
    # Cast the subkeys to the structure of `primals`
    subkeys = tree_unflatten(struct, random.split(key, struct.num_leaves))

    def draw(key, x):
        shp = x.shape if hasattr(x, "shape") else jnp.shape(x)
        dtp = x.dtype if hasattr(x, "dtype") else np.common_type(x)
        return rng(key=key, shape=shp, dtype=dtp)

    return tree_map(draw, subkeys, primals)


def interpolate(xmin=-7., xmax=7., N=14000) -> Callable:
    """Replaces a local nonlinearity such as jnp.exp with a linear interpolation

    Interpolating functions speeds up code and increases numerical stability in
    some cases, but at a cost of precision and range.

    Parameters
    ----------
    xmin : float
        Minimal interpolation value. Default: -7.
    xmax : float
        Maximal interpolation value. Default: 7.
    N : int
        Number of points used for the interpolation. Default: 14000
    """
    def decorator(f):
        from functools import wraps

        x = jnp.linspace(xmin, xmax, N)
        y = f(x)

        @wraps(f)
        def wrapper(t):
            return jnp.interp(t, x, y)

        return wrapper

    return decorator
