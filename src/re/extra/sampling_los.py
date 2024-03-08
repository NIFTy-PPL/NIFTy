#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import dataclasses
from functools import partial

import jax
from jax import numpy as jnp

from ..tree_math import ShapeWithDtype
from ..model import Model


def _los(x, /, start, end, *, distances, shape, n_sampling_points, order=1):
    from jax.scipy.ndimage import map_coordinates

    l2i = ((shape - 1) / shape) / distances
    start_iloc = start * l2i
    end_iloc = end * l2i
    ddi = (end_iloc - start_iloc) / n_sampling_points
    adi = jnp.arange(0, n_sampling_points) + 0.5
    dist = jnp.linalg.norm(end - start)
    pp = start_iloc[:, jnp.newaxis] + ddi[:, jnp.newaxis] * adi[jnp.newaxis]
    return map_coordinates(x, pp, order=order, cval=jnp.nan).sum() * (
        dist / n_sampling_points
    )


class SamplingCartesianGridLOS(Model):
    start: jax.Array = dataclasses.field(metadata=dict(static=False))
    end: jax.Array = dataclasses.field(metadata=dict(static=False))
    distances: jax.Array = dataclasses.field(metadata=dict(static=False))

    def __init__(
        self,
        start,
        end,
        *,
        shape,
        distances,
        n_sampling_points=500,
        interpolation_order=1,
        dtype=None,
    ):
        """Sampling Line-Of-Sight (LOS) intergrator.

        Samples the LOS at a number of points and sum up the result to estimate
        the integral from a starting point to an end point in n-dimensional space.

        Parameters
        ----------
        start :
            Location of the start point(s) in Cartesian space of shape
            `(n_points, n_dim)` or `(n_dim,)`.
        end :
            Location of the end point(s) in Cartesian space of shape
            `(n_points, n_dim)` or `(n_dim,)`.
        shape :
            Shape of the input.
        distances :
            Tuple of distances for each axis of the shape of the input
        n_sampling_points : int, optional
            Number of sampling points per LOS for the integration.
        interpolation_order : int, optional
            Order of the interpolation for reading out the sampling points.
        dtype : data-type, optional
            Hint specifying the dtype for the construction of the domain.
        """
        # We assume that `start` and `end` are of shape (n_points, n_dimensions)
        self.start = jnp.array(start)
        self.end = jnp.array(end)
        self.distances = jnp.array(distances)
        self._los = partial(
            _los,
            n_sampling_points=n_sampling_points,
            order=interpolation_order,
            distances=self.distances,
            shape=jnp.array(shape),
        )
        super().__init__(
            domain=ShapeWithDtype(shape, dtype), target=ShapeWithDtype(end.shape, dtype)
        )

    def __call__(self, x):
        in_axes = (None, 0, 0)
        if self.start.ndim < self.end.ndim:
            in_axes = (None, None, 0)
        elif self.start.ndim > self.end.ndim:
            in_axes = (None, 0, None)
        return jax.vmap(self._los, in_axes=in_axes)(x, self.start, self.end)
