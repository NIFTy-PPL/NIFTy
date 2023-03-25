#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import sys
from functools import partial
from typing import Tuple

import jax
import numpy as np
from jax import numpy as jnp
from numpy.typing import NDArray

from ..logger import logger


def unique(
    ar, *, return_inverse=False, axis=-1, atol=1e-10, rtol=1e-5, _verbosity=0
):
    """Find unique sub-arrays in `ar` along `axis` within a given tolerance.

    The algorithm is efficient if the number of approximately unique values is
    small compared to the overall size of `ar` along `axis`.
    """
    if not isinstance(axis, int):
        raise TypeError(f"`axis` needs to be of type `int`; got {type(axis)!r}")

    uniqs = None
    inverse = None
    if return_inverse:
        inverse = np.full(ar.shape[axis], -1, dtype=int)
    # Ensure positive axis required for identify the axis of reductions
    axis = np.arange(ar.ndim)[axis]
    ra = tuple(set(range(ar.ndim)) - {
        axis,
    })
    to_sort = np.ones(ar.shape[axis], dtype=bool)
    while np.sum(to_sort) != 0:
        i = np.nonzero(to_sort)[0][0]
        u = np.take(ar, (i, ), axis=axis)
        if uniqs is None:
            uniqs = u
        else:
            uniqs = np.concatenate((uniqs, u), axis=axis)
        isclose = np.zeros(to_sort.shape, dtype=bool)
        # Set the `mode` to work around `ar` potentially being a JAX array and
        # not supporting NumPy's default `mode='raise'`
        a = np.take(ar, np.nonzero(to_sort)[0], axis=axis, mode=None)
        isclose[to_sort] = np.all(
            np.abs(u - a) <= (atol + rtol * np.abs(a)), axis=ra
        )
        to_sort &= ~isclose
        if return_inverse:
            assert inverse is not None
            inverse[isclose] = uniqs.shape[axis] - 1
        if _verbosity > 0:
            logger.info(f"to-sort: {np.sum(to_sort)}", file=sys.stderr)

    if return_inverse:
        assert np.all(inverse != -1)
        return uniqs, inverse
    return uniqs


def amend_unique(ar,
                 el,
                 *,
                 axis=-1,
                 atol=1e-10,
                 rtol=1e-5) -> Tuple[NDArray, int]:
    """Amend the element `el` if it is unique up to the specified tolerance
    otherwise do nothing.
    """
    if not isinstance(axis, int):
        raise TypeError(f"`axis` needs to be of type `int`; got {type(axis)!r}")

    # Ensure positive axis required for identify the axis of reductions
    axis = np.arange(ar.ndim)[axis]
    ra = tuple(set(range(ar.ndim)) - {
        axis,
    })

    el = np.expand_dims(el, axis=axis)
    isclose = np.all(np.abs(ar - el) <= (atol + rtol * np.abs(el)), axis=ra)
    assert isclose.size == ar.shape[axis]
    if np.any(isclose):
        return ar, np.nonzero(isclose)[0][0]
    else:
        return np.concatenate((ar, el), axis=axis), ar.shape[axis]


@partial(jax.jit, static_argnames=("axis", ))
def amend_unique_(ar, el, *, axis=-1, atol=1e-10, rtol=1e-5):
    if not isinstance(axis, int):
        raise TypeError(f"`axis` needs to be of type `int`; got {type(axis)!r}")
    PLC = -1 << 63 if jnp.array(0).dtype == jnp.int64 else -1 << 31

    # Ensure positive axis required for identify the axis of reductions
    axis = np.arange(ar.ndim)[axis]
    ra = tuple(set(range(ar.ndim)) - {
        axis,
    })

    el = jnp.expand_dims(el, axis=axis)
    isclose = jnp.all(jnp.abs(ar - el) <= (atol + rtol * jnp.abs(el)), axis=ra)

    # Find the first not-NaN location in the array at which to potentially
    # insert a new value
    n = jnp.nonzero(jnp.all(jnp.isnan(ar), axis=ra), size=1,
                    fill_value=PLC)[0][0]

    # Replace NaN with NaN if the new element is close to any existing element,
    # else insert it at the first not-NaN location
    any_isclose = jnp.any(isclose)
    e = jnp.where(any_isclose, jnp.full_like(el, jnp.nan), el)
    ar = ar.at[(slice(None), ) * axis + (n, )].set(jnp.squeeze(e, axis=axis))
    idx = jnp.nonzero(isclose, size=1, fill_value=PLC)[0][0]
    idx = jnp.where(any_isclose, idx, n)
    return ar, idx
