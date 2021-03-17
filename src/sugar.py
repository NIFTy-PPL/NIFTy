from typing import Callable
from collections.abc import Iterable
from jax.numpy import isscalar, array
from jax import random
from jax.tree_util import tree_structure, tree_flatten, tree_unflatten, tree_multimap


def makeField(obj):
    if isscalar(obj):
        return obj
    val, domain = tree_flatten(obj)
    from .field import Field
    return Field(domain, [array(v) for v in val])


def fromField(f):
    if isscalar(f):
        return f
    return f.to_tree()


def just_add(a, b):
    return fromField(makeField(a) + makeField(b))


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
