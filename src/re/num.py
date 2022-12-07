#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import sys
import numpy as np


def unique(
    ar, *, return_inverse=False, axis=-1, atol=1e-10, rtol=1e-5, _verbose=0
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
        a = np.take(ar, np.nonzero(to_sort)[0], axis=axis)
        isclose[to_sort] = np.all(
            np.abs(u - a) <= (atol + rtol * np.abs(a)), axis=ra
        )
        to_sort &= ~isclose
        if return_inverse:
            assert inverse is not None
            inverse[isclose] = uniqs.shape[axis] - 1
        if _verbose > 0:
            print(f"to-sort: {np.sum(to_sort)}", file=sys.stderr)

    if return_inverse:
        assert np.all(inverse != -1)
        return uniqs, inverse
    return uniqs
