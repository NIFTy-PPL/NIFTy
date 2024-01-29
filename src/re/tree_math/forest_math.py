# Copyright(C) 2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import operator
from collections.abc import Iterable
from functools import partial
from typing import Callable, Tuple, TypeVar, Union

from jax import lax
from jax import numpy as jnp
from jax import random
from jax.tree_util import (tree_leaves, tree_map, tree_structure,
                           tree_transpose, tree_unflatten)

T = TypeVar("T")

CORE_ARITHMETIC_ATTRIBUTES = (
    "__neg__", "__pos__", "__abs__", "__add__", "__radd__", "__sub__",
    "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
    "__floordiv__", "__rfloordiv__", "__pow__", "__rpow__", "__mod__",
    "__rmod__", "__matmul__", "__rmatmul__"
)


def has_arithmetics(obj, additional_attributes=()):
    desired_attrs = CORE_ARITHMETIC_ATTRIBUTES + additional_attributes
    return all(hasattr(obj, attr) for attr in desired_attrs)


def assert_arithmetics(obj, *args, **kwargs):
    if not has_arithmetics(obj, *args, **kwargs):
        ae = (
            f"input of type {type(obj)} does not support"
            " core arithmetic operations"
            "\nmaybe you forgot to wrap your object in a `Vector`"
        )
        raise AssertionError(ae)


def random_like(key: Iterable, primals, rng: Callable = random.normal):
    import numpy as np

    struct = tree_structure(primals)
    # Cast the subkeys to the structure of `primals`
    subkeys = tree_unflatten(struct, random.split(key, struct.num_leaves))

    def draw(key, x):
        shp = x.shape if hasattr(x, "shape") else jnp.shape(x)
        dtp = x.dtype if hasattr(x, "dtype") else np.result_type(x)
        return rng(key=key, shape=shp, dtype=dtp)

    return tree_map(draw, subkeys, primals)


def unite(x, y, op=operator.add):
    """Unites two Vector-like objects.

    If a key is contained in both objects, then the fields at that key
    are combined.
    """
    from .vector import Vector

    if isinstance(x, Vector) or isinstance(y, Vector):
        x = x.tree if isinstance(x, Vector) else x
        y = y.tree if isinstance(y, Vector) else y
        return Vector(unite(x, y, op=op))
    if not hasattr(x, "keys") and not hasattr(y, "keys"):
        return op(x, y)
    if not hasattr(x, "keys") or not hasattr(y, "keys"):
        te = (
            "one of the inputs does not have a `keys` property;"
            f" got {type(x)} and {type(y)}"
        )
        raise TypeError(te)

    out = {}
    for k in x.keys() | y.keys():
        if k in x and k in y:
            out[k] = op(x[k], y[k])
        elif k in x:
            out[k] = x[k]
        else:
            out[k] = y[k]
    return out


def _shape(x):
    return x.shape if hasattr(x, "shape") else jnp.shape(x)


def tree_shape(tree: T) -> T:
    return tree_map(_shape, tree)


def stack(arrays, axis=0):
    return tree_map(lambda *el: jnp.stack(el, axis=axis), *arrays)


def unstack(stack, axis=0):
    element_count = tree_leaves(stack)[0].shape[0]
    split = partial(jnp.split, indices_or_sections=element_count, axis=axis)
    unstacked = tree_transpose(
        tree_structure(stack), tree_structure((0., ) * element_count),
        tree_map(split, stack)
    )
    return tree_map(partial(jnp.squeeze, axis=axis), unstacked)


def _lax_map(fun, in_axes=0, out_axes=0):
    if in_axes not in (0, (0, )) or out_axes not in (0, (0, )):
        raise ValueError("`lax.map` maps only along first axis")
    return partial(lax.map, fun)


def get_map(map) -> Callable:
    from jax import pmap, vmap

    from ..custom_map import smap, lmap

    if isinstance(map, str):
        if map in ('vmap', 'v'):
            m = vmap
        elif map in ('pmap', 'p'):
            m = pmap
        elif map in ('lmap', 'l'):
            m = lmap
        elif map in ('smap', 's'):
            m = smap
        else:
            raise ValueError(f"unknown `map` {map!r}")
    elif callable(map):
        m = map
    else:
        raise TypeError(f"invalid `map` {map!r}; expected string or callable")
    return m


def map_forest(
    f: Callable,
    in_axes: Union[int, Tuple] = 0,
    out_axes: Union[int, Tuple] = 0,
    tree_transpose_output: bool = True,
    map: Union[str, Callable] = "vmap",
    **kwargs
) -> Callable:
    if out_axes != 0:
        raise NotImplementedError("`out_axis` not yet supported")
    in_axes = in_axes if isinstance(in_axes, tuple) else (in_axes, )
    i = None
    for idx, el in enumerate(in_axes):
        if el is not None and i is None:
            i = idx
        elif el is not None and i is not None:
            nie = "mapping over more than one axis is not yet supported"
            raise NotImplementedError(nie)
    if i is None:
        raise ValueError("must map over at least one axis")
    if not isinstance(i, int):
        te = "mapping over a non integer axis is not yet supported"
        raise TypeError(te)

    map = get_map(map)
    map_f = map(f, in_axes=in_axes, out_axes=out_axes, **kwargs)

    def apply(*xs):
        if not isinstance(xs[i], (list, tuple)):
            te = f"expected mapped axes to be a tuple; got {type(xs[i])}"
            raise TypeError(te)
        x_T = stack(xs[i])

        out_T = map_f(*xs[:i], x_T, *xs[i + 1:])
        # Since `out_axes` is forced to be `0`, we don't need to worry about
        # transposing only part of the output
        if not tree_transpose_output:
            return out_T
        return unstack(out_T)

    return apply


def map_forest_mean(method, map="vmap", *args, **kwargs) -> Callable:
    method_map = map_forest(
        method, *args, tree_transpose_output=False, map=map, **kwargs
    )

    def meaned_apply(*xs, **xs_kw):
        return tree_map(partial(jnp.mean, axis=0), method_map(*xs, **xs_kw))

    return meaned_apply


def mean(forest):
    from functools import reduce
    from .vector import Vector

    norm = 1. / len(forest)
    if isinstance(forest[0], Vector):
        m = norm * reduce(Vector.__add__, forest)
        return m
    else:
        m = norm * reduce(Vector.__add__, (Vector(t) for t in forest))
        return m.tree


def mean_and_std(forest, correct_bias=True):
    from .vector import Vector

    if isinstance(forest[0], Vector):
        m = mean(forest)
        mean_of_sq = mean(tuple(t**2 for t in forest))
    else:
        m = Vector(mean(forest))
        mean_of_sq = mean(tuple(Vector(t)**2 for t in forest))

    n = len(forest)
    scl = jnp.sqrt(n / (n - 1)) if correct_bias else 1.
    std = scl * tree_map(jnp.sqrt, mean_of_sq - m**2)
    if isinstance(forest[0], Vector):
        return m, std
    else:
        return m.tree, std.tree
