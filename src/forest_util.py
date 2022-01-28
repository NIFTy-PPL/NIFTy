from functools import partial
import operator
from jax import lax
from jax import numpy as jnp
from jax.tree_util import (
    all_leaves, tree_leaves, tree_map, tree_structure, tree_reduce
)

from typing import Any, Callable, Optional, Tuple, Union

from .sugar import is1d


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


class ShapeWithDtype():
    """Minimal helper class storing the shape and dtype of an object.

    Notes
    -----
    This class may not be transparent to JAX as it shall not be flattened
    itself. If used in a tree-like structure. It should only be used as leave.
    """
    def __init__(self, shape: Union[tuple, list, int], dtype=None):
        """Instantiates a storage unit for shape and dtype.

        Parameters
        ----------
        shape : tuple or list of int
            One-dimensional sequence of integers denoting the length of the
            object along each of the object's axis.
        dtype : dtype
            Data-type of the to-be-described object.
        """
        if isinstance(shape, int):
            shape = (shape, )
        if not is1d(shape):
            ve = f"invalid shape; got {shape!r}"
            raise TypeError(ve)

        self._shape = shape
        self._dtype = jnp.float64 if dtype is None else dtype

    @classmethod
    def from_leave(cls, element):
        """Convenience method for creating an instance of `ShapeWithDtype` from
        an object.

        To map a whole tree-like structure to a its shape and dtype use JAX's
        `tree_map` method like so:

            tree_map(ShapeWithDtype.from_leave, tree)

        Parameters
        ----------
        element : tree-like structure
            Object from which to take the shape and data-type.

        Returns
        -------
        swd : instance of ShapeWithDtype
            Instance storing the shape and data-type of `element`.
        """
        if not all_leaves((element, )):
            ve = "tree is not flat and still contains leaves"
            raise ValueError(ve)
        return cls(jnp.shape(element), get_dtype(element))

    @property
    def shape(self):
        """Retrieves the shape."""
        return self._shape

    @property
    def dtype(self):
        """Retrieves the data-type."""
        return self._dtype

    def __repr__(self):
        nm = self.__class__.__name__
        return f"{nm}(shape={self.shape}, dtype={self.dtype})"


def get_dtype(v: Any):
    if hasattr(v, "dtype"):
        return v.dtype
    else:
        return type(v)


def common_type(*trees):
    from numpy import find_common_type

    common_dtp = find_common_type(
        tuple(
            find_common_type(tuple(get_dtype(v) for v in tree_leaves(tr)), ())
            for tr in trees
        ), ()
    )
    return common_dtp


def size(tree, axis: Optional[int] = None) -> Union[int, jnp.ndarray]:
    if axis is not None:
        raise TypeError("axis of an arbitrary tree is ill defined")
    sizes = tree_map(jnp.size, tree)
    return tree_reduce(jnp.add, sizes)


def _zeros_like(x, dtype, shape):
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        shp = x.shape if shape is None else shape
        dtp = x.dtype if dtype is None else dtype
        return jnp.zeros(shape=shp, dtype=dtp)
    return jnp.zeros_like(x, dtype=dtype, shape=shape)


def zeros_like(a, dtype=None, shape=None):
    return tree_map(partial(_zeros_like, dtype=dtype, shape=shape), a)


def norm(tree, ord, *, ravel: bool):
    from jax.numpy.linalg import norm

    if ravel:

        def el_norm(x):
            return jnp.abs(x) if jnp.ndim(x) == 0 else norm(x.ravel(), ord=ord)
    else:

        def el_norm(x):
            return jnp.abs(x) if jnp.ndim(x) == 0 else norm(x, ord=ord)

    return norm(tree_leaves(tree_map(el_norm, tree)), ord=ord)


def _ravel(x):
    return x.ravel() if hasattr(x, "ravel") else jnp.ravel(x)


def dot(a, b, *, precision=None):
    tree_of_dots = tree_map(
        lambda x, y: jnp.dot(_ravel(x), _ravel(y), precision=precision), a, b
    )
    return tree_reduce(operator.add, tree_of_dots, 0.)


def vdot(a, b, *, precision=None):
    tree_of_vdots = tree_map(
        lambda x, y: jnp.vdot(_ravel(x), _ravel(y), precision=precision), a, b
    )
    return tree_reduce(jnp.add, tree_of_vdots, 0.)


def select(pred, on_true, on_false):
    return tree_map(partial(lax.select, pred), on_true, on_false)


def where(condition, x, y):
    """Selects a pytree based on the condition which can be a pytree itself.

    Notes
    -----
    If `condition` is not a pytree, then a partially evaluated selection is
    simply mapped over `x` and `y` without actually broadcasting `condition`.
    """
    import numpy as np
    from itertools import repeat

    ts_c = tree_structure(condition)
    ts_x = tree_structure(x)
    ts_y = tree_structure(y)
    ts_max = (ts_c, ts_x, ts_y)[np.argmax(
        [ts_c.num_nodes, ts_x.num_nodes, ts_y.num_nodes]
    )]

    if ts_x.num_nodes < ts_max.num_nodes:
        if ts_x.num_nodes > 1:
            raise ValueError("can not broadcast LHS")
        x = ts_max.unflatten(repeat(x, ts_max.num_leaves))
    if ts_y.num_nodes < ts_max.num_nodes:
        if ts_y.num_nodes > 1:
            raise ValueError("can not broadcast RHS")
        y = ts_max.unflatten(repeat(y, ts_max.num_leaves))

    if ts_c.num_nodes < ts_max.num_nodes:
        if ts_c.num_nodes > 1:
            raise ValueError("can not map condition")
        return tree_map(partial(jnp.where, condition), x, y)
    return tree_map(jnp.where, condition, x, y)


def vmap_forest(
    f: Callable,
    in_axes: Union[int, Tuple] = 0,
    out_axes: Union[int, Tuple] = 0,
    tree_transpose_output: bool = True,
    **kwargs
) -> Callable:
    from functools import partial
    from jax import vmap
    from jax import tree_util

    if out_axes != 0:
        raise TypeError("`out_axis` not yet supported")
    in_axes = in_axes if isinstance(in_axes, tuple) else (in_axes, )
    i = None
    for idx, el in enumerate(in_axes):
        if el is not None and i is None:
            i = idx
        elif el is not None and i is not None:
            ve = "mapping over more than one axis is not yet supported"
            raise ValueError(ve)
    if i is None:
        raise ValueError("must map over at least one axis")
    if not isinstance(i, int):
        te = "mapping over a non integer axis is not yet supported"
        raise TypeError(te)

    f_map = vmap(f, in_axes=in_axes, out_axes=out_axes, **kwargs)

    def apply(*xs):
        if not isinstance(xs[i], (list, tuple)):
            te = f"expected mapped axes to be a tuple; got {type(xs[i])}"
            raise TypeError(te)
        element_count = len(xs[i])
        x_T = tree_map(lambda *el: jnp.stack(el), *xs[i])

        out_T = f_map(*xs[:i], x_T, *xs[i + 1:])
        # Since `out_axes` is forced to be `0`, we don't need to worry about
        # transposing only part of the output
        if not tree_transpose_output:
            return out_T

        split = partial(jnp.split, indices_or_sections=element_count)
        out = tree_util.tree_transpose(
            tree_util.tree_structure(out_T),
            tree_util.tree_structure((0., ) * element_count),
            tree_map(split, out_T)
        )
        return tree_map(partial(jnp.squeeze, axis=0), out)

    return apply


def vmap_forest_mean(method, *args, **kwargs) -> Callable:
    method_map = vmap_forest(
        method, *args, tree_transpose_output=False, **kwargs
    )

    def meaned_apply(*xs, **xs_kw):
        return tree_map(partial(jnp.mean, axis=0), method_map(*xs, **xs_kw))

    return meaned_apply
