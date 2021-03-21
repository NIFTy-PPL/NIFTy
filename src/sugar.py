from typing import Callable
from collections.abc import Iterable

from jax import random
from jax.tree_util import (
    tree_structure, tree_leaves, tree_unflatten, tree_map, tree_multimap,
    tree_reduce
)

from .field import Field


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


def norm(tree, ord):
    from jax.numpy import ndim, abs
    from jax.numpy.linalg import norm

    enorm = lambda x: abs(x) if ndim(x) == 0 else norm(x, ord=ord)
    return norm(tree_leaves(tree_map(enorm, tree)), ord=ord)


def random_with_tree_shape(
    tree_shape: Iterable, key: Iterable, rng: Callable = random.normal
):
    if isinstance(tree_shape, dict):
        rvs = {}
        subkeys = random.split(key, len(tree_shape))
        for (k, v), sk in zip(tree_shape.items(), subkeys):
            rvs[k] = random_with_tree_shape(v, sk, rng=rng)
        return rvs
    elif isinstance(tree_shape, (list, tuple)):
        return rng(shape=tuple(tree_shape), key=key)
    raise TypeError(f"invalid type of `tree_shape` {type(tree_shape)!r}")


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
