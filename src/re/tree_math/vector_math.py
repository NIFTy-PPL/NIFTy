# Copyright(C) 2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import operator
from functools import partial, reduce
from typing import Any, List, Optional, Tuple, Union

from jax import numpy as jnp
from jax.tree_util import (
    all_leaves,
    tree_leaves,
    tree_map,
    tree_reduce,
    tree_structure,
)


class ShapeWithDtype():
    """Minimal helper class storing the shape and dtype of an object.

    Notes
    -----
    This class may not be transparent to JAX as it shall not be flattened
    itself. If used in a tree-like structure. It should only be used as leave.
    """
    def __init__(
        self, shape: Union[Tuple[()], Tuple[int], List[int], int], dtype=None
    ):
        """Instantiates a storage unit for shape and dtype.

        Parameters
        ----------
        shape : tuple or list of int
            One-dimensional sequence of integers denoting the length of the
            object along each of the object's axis.
        dtype : dtype
            Data-type of the to-be-described object.
        """
        from ..misc import is_iterable_of_non_iterables

        if isinstance(shape, int):
            shape = (shape, )
        if isinstance(shape, list):
            shape = tuple(shape)
        if not is_iterable_of_non_iterables(shape):
            ve = f"invalid shape; got {shape!r}"
            raise TypeError(ve)

        self._shape = shape
        self._dtype = jnp.float64 if dtype is None else dtype
        self._size = None

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
        return cls(jnp.shape(element), _get_dtype(element))

    @property
    def shape(self) -> Tuple[int]:
        """Retrieves the shape."""
        return self._shape

    @property
    def dtype(self):
        """Retrieves the data-type."""
        return self._dtype

    @property
    def size(self) -> int:
        """Total number of elements."""
        if self._size is None:
            self._size = reduce(operator.mul, self.shape, 1)
        return self._size

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        if self.ndim > 0:
            return self.shape[0]
        else:  # mimic numpy
            raise TypeError("len() of unsized object")

    def __eq__(self, other) -> bool:
        if not isinstance(other, ShapeWithDtype):
            return False
        else:
            return (self.shape, self.dtype) == (other.shape, other.dtype)

    def __repr__(self):
        nm = self.__class__.__name__
        return f"{nm}(shape={self.shape}, dtype={self.dtype})"

    # TODO: overlaod basic arithmetics (see `np.broadcast_shapes((1, 2), (3,
    # 1), (3, 2))`)


def _get_dtype(v: Any):
    if hasattr(v, "dtype"):
        return v.dtype
    else:
        return type(v)


def result_type(*trees):
    from numpy import result_type as result_type_np

    common_dtp = result_type_np(
        *(
            result_type_np(*(_get_dtype(v) for v in tree_leaves(tr)))
            for tr in trees
        )
    )
    return common_dtp


def _size(x):
    return x.size if hasattr(x, "size") else jnp.size(x)


def size(a, axis: Optional[int] = None) -> int:
    if axis is not None:
        raise TypeError("axis of an arbitrary tree is ill defined")
    return tree_reduce(operator.add, tree_map(_size, a))


def shape(a):
    return (size(a), )


def _like(x, dtype, shape, like, new):
    # Catch structures that plain numpy might not handle such as SWD structs
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        shp = x.shape if shape is None else shape
        dtp = x.dtype if dtype is None else dtype
        return new(shape=shp, dtype=dtp)
    return like(x, dtype=dtype, shape=shape)


zeros_like = partial(
    tree_map,
    partial(_like, dtype=None, shape=None, like=jnp.zeros_like, new=jnp.zeros)
)

ones_like = partial(
    tree_map,
    partial(_like, dtype=None, shape=None, like=jnp.ones_like, new=jnp.ones)
)


def _ravel(x):
    return x.ravel() if hasattr(x, "ravel") else jnp.ravel(x)


def norm(tree, ord=2):
    """**Vector** norm.

    Notes
    -----
    This function assumes the input to be a vector, i.e. the default order `ord`
    is `2`.
    """
    from jax.numpy.linalg import norm

    def el_norm(x):
        if jnp.ndim(x) == 0:
            return jnp.abs(x)
        return norm(_ravel(x), ord=ord)

    return norm(jnp.array(tree_leaves(tree_map(el_norm, tree))), ord=ord)


def dot(a, b, *, precision=None):
    """Returns the dot product of the two vectors.

    Parameters
    ----------
    a : object
        Arbitrary, flatten-able objects.
    b : object
        Arbitrary, flatten-able objects.

    Returns
    -------
    out : float
        Dot product of vectors.
    """
    tree_of_dots = tree_map(
        lambda x, y: jnp.dot(_ravel(x), _ravel(y), precision=precision), a, b
    )
    return tree_reduce(operator.add, tree_of_dots, 0.)


matmul = dot


def vdot(a, b, *, precision=None):
    tree_of_vdots = tree_map(partial(jnp.vdot, precision=precision), a, b)
    return tree_reduce(jnp.add, tree_of_vdots, 0.)


def _conj(x):
    return x.conj() if hasattr(x, "conj") else jnp.conj(x)


def conjugate(a):
    """Returns the complex conjugate, component- and element-wise.

    Parameters
    ----------
    a : object
        Arbitrary, flatten-able objects.

    Returns
    -------
    out : object
        The complex conjugate of `a`, with same shape and dtype as `a`.
    """
    return tree_map(_conj, a)


conj = conjugate


def where(condition, x, y):
    """Selects a pytree based on the condition which can be a pytree itself.

    Notes
    -----
    If `condition` is not a pytree, then a partially evaluated selection is
    simply mapped over `x` and `y` without actually broadcasting `condition`.
    """
    from itertools import repeat

    import numpy as np

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


def _unary_reduction(jax_f, name=None):
    name = name if name is not None else jax_f.__name__

    def _safe_uni(a):
        return jax_f(jnp.atleast_1d(a))

    def _red(a, b):
        return jax_f(jnp.array([a, b]))

    def unary_reduction(a):
        return tree_reduce(_red, tree_map(_safe_uni, a))

    unary_reduction.__name__ = name
    return unary_reduction


sum = _unary_reduction(jnp.sum)
min = _unary_reduction(jnp.min)
max = _unary_reduction(jnp.max)
any = _unary_reduction(jnp.any)
all = _unary_reduction(jnp.all)
