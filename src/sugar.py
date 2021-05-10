from typing import Callable
from collections.abc import Iterable

from jax import random
from jax import numpy as np
from jax.tree_util import (
    tree_structure, tree_leaves, tree_unflatten, tree_map, tree_multimap,
    tree_reduce
)

from .field import Field


def isiterable(candidate):
    try:
        iter(candidate)
        return True
    except TypeError:
        return False


def is1d(ls, object_type=(int, np.unsignedinteger)):
    if isinstance(ls, np.ndarray):
        ndim = np.ndim(ls)
        dtp_match = any(np.issubdtype(ls.dtype, dtp) for dtp in object_type)
        return (ndim == 1) & dtp_match

    if not isiterable(ls):
        return False
    return all(isinstance(e, object_type) for e in ls)


def ducktape(call, key):
    def named_call(p):
        return call(p.get(key))

    return named_call


def just_add(a, b):
    from jax.tree_util import tree_leaves

    return tree_leaves(Field(a) + Field(b))


def sum_of_squares(tree):
    from jax.numpy import add, sum

    return tree_reduce(add, tree_map(lambda x: sum(x**2), tree), 0.)


def norm(tree, ord, ravel=False):
    from jax.numpy import ndim, abs
    from jax.numpy.linalg import norm

    if ravel:
        def el_norm(x):
            return abs(x) if ndim(x) == 0 else norm(x.ravel(), ord=ord)
    else:
        def el_norm(x):
            return abs(x) if ndim(x) == 0 else norm(x, ord=ord)

    return norm(tree_leaves(tree_map(el_norm, tree)), ord=ord)


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
        mean_of_sq = Field(mean(tuple(Field(t)**2 for t in forest)))

    n = len(forest)
    scl = np.sqrt(n / (n - 1)) if correct_bias else 1.
    std = scl * tree_map(np.sqrt, mean_of_sq - m**2)
    if isinstance(forest[0], Field):
        return m, std
    else:
        return m.val, std.val


def random_like_shapewdtype(
    tree_of_shapes: Iterable, key: Iterable, rng: Callable = random.normal
):
    struct = tree_structure(tree_of_shapes)
    # Cast the subkeys to the structure of `primals`
    subkeys = tree_unflatten(struct, random.split(key, struct.num_leaves))

    def draw(swd, key):
        return rng(key=key, shape=swd.shape, dtype=swd.dtype)

    return tree_multimap(draw, tree_of_shapes, subkeys)


def random_like(
    primals: Iterable, key: Iterable, rng: Callable = random.normal
):
    import numpy as onp
    from jax import numpy as np

    struct = tree_structure(primals)
    # Cast the subkeys to the structure of `primals`
    subkeys = tree_unflatten(struct, random.split(key, struct.num_leaves))

    def draw(x, key):
        shp = np.shape(x)
        dtp = onp.common_type(x)
        return rng(key=key, shape=shp, dtype=dtp)

    return tree_multimap(draw, primals, subkeys)


def interpolate(xmin=-7., xmax=7., N=14000):
    """Replaces a local nonlinearity such as np.exp with a linear interpolation

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

        x = np.linspace(xmin, xmax, N)
        y = f(x)

        @wraps(f)
        def wrapper(t):
            return np.interp(t, x, y)

        return wrapper

    return decorator
