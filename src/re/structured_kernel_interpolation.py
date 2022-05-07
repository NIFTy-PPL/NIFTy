# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import jax
from jax import numpy as jnp
import numpy as np

from .correlated_field import get_fourier_mode_distributor, hartley


def interp_mat(grid_shape, grid_bounds, sampling_points, *, distances=None):
    from scipy.sparse import coo_matrix  # TODO: use only JAX w/o SciPy or NumPy
    from jax.experimental.sparse import BCOO

    if sampling_points.ndim != 2:
        ve = f"invalid dimension of sampling_points {sampling_points.ndim!r}"
        raise ValueError(ve)
    ndim, n_points = sampling_points.shape
    if grid_bounds is not None and len(grid_bounds) != ndim:
        ve = (
            f"grid_bounds of length {len(grid_bounds)} incompatible with"
            " sampling_points of shape {sampling_points.shape!r}"
        )
        raise ValueError(ve)
    elif grid_bounds is not None:
        offset = np.array(list(zip(*grid_bounds))[0])
    else:
        offset = np.zeros(ndim)
    if distances is not None and np.size(distances) != ndim:
        ve = (
            f"distances of size {np.size(distances)} incompatible with"
            " sampling_points of shape {sampling_points.shape!r}"
        )
        raise ValueError(ve)
    distances = np.asarray(distances) if distances is not None else None
    if (distances is not None and grid_bounds
        is not None) or (distances is None and grid_bounds is None):
        raise ValueError("exactly one of `distances` or `grid_shape` expected")
    elif grid_bounds is not None:
        distances = np.array(
            [(b[1] - b[0]) / sz for b, sz in zip(grid_bounds, grid_shape)]
        )
    if distances is None:
        raise AssertionError()

    mg = np.mgrid[(slice(0, 2), ) * ndim].reshape(ndim, -1)
    pos = (sampling_points - offset.reshape(-1, 1)) / distances.reshape(-1, 1)
    excess, pos = np.modf(pos)
    pos = pos.astype(np.int64)
    # max_index = np.array(grid_shape).reshape(-1, 1)
    weights = np.zeros((2**ndim, n_points))
    ii = np.zeros((2**ndim, n_points), dtype=np.int64)
    jj = np.zeros((2**ndim, n_points), dtype=np.int64)
    for i in range(2**ndim):
        weights[i, :] = np.prod(
            np.abs(1 - mg[:, i].reshape(-1, 1) - excess), axis=0
        )
        fromi = (pos + mg[:, i].reshape(-1, 1))  # % max_index
        ii[i, :] = np.arange(n_points)
        jj[i, :] = np.ravel_multi_index(fromi, grid_shape)

    mat = coo_matrix(
        (weights.ravel(), (ii.ravel(), jj.ravel())),
        shape=(n_points, np.prod(grid_shape))
    )
    # BCOO(
    #     (weights.ravel(), jnp.stack((ii.ravel(), jj.ravel()), axis=1)),
    #     shape=(n_points, np.prod(grid_shape))
    # )
    return BCOO.from_scipy_sparse(mat)


class SKIHarmonicCovariance():
    def __init__(
        self,
        grid_shape,
        grid_bounds,
        sampling_points,
        harmonic_kernel,
        padding=0.5,
        subslice=None,
        jitter=True
    ):
        if jitter is True:
            if sampling_points.dtype.type == np.float64:
                self.jitter = 1e-8
            elif sampling_points.dtype.type == np.float32:
                self.jitter = 1e-6
            else:
                raise NotImplementedError()
        elif jitter is False:
            self.jitter = None
        else:
            self.jitter = jitter

        self.grid_unpadded_shape = np.asarray(grid_shape)
        self.grid_unpadded_bounds = np.asarray(grid_bounds)
        self.grid_unpadded_distances = jnp.diff(
            self.grid_unpadded_bounds, axis=1
        ).ravel() / self.grid_unpadded_shape
        self.grid_unpadded_total_volume = jnp.prod(
            self.grid_unpadded_shape * self.grid_unpadded_distances
        )
        self.w = interp_mat(grid_shape, grid_bounds, sampling_points)

        if padding is not None and padding != 0.:
            pad = 1. + padding
            grid_shape = np.asarray(grid_shape)
            grid_shape_wpad = np.ceil(grid_shape * pad).astype(int)
            scl = grid_shape_wpad / grid_shape
            scl_end = jnp.diff(jnp.asarray(grid_bounds), axis=1).ravel() * scl
            grid_bounds_wpad = jnp.asarray(grid_bounds)
            grid_bounds_wpad = grid_bounds_wpad.at[:, 1].set(
                grid_bounds_wpad[:, 0].ravel() + scl_end
            )
            if subslice is None:
                subslice = tuple(map(int, grid_shape))
            grid_shape = grid_shape_wpad
            grid_bounds = grid_bounds_wpad
        self.grid_shape = np.asarray(grid_shape)
        self.grid_bounds = np.asarray(grid_bounds)

        self.harmonic_kernel = harmonic_kernel

        if subslice is not None:
            if isinstance(subslice, slice):
                subslice = (subslice, ) * len(self.grid_shape)
            elif isinstance(subslice, int):
                subslice = (slice(subslice), ) * len(self.grid_shape)
            elif isinstance(subslice, tuple):
                if all(isinstance(el, slice) for el in subslice):
                    pass
                elif all(isinstance(el, int) for el in subslice):
                    subslice = tuple(slice(el) for el in subslice)
                else:
                    raise TypeError("elements of `subslice` of invalid type")
            else:
                raise TypeError("`subslice` of invalid type")
        self.grid_subslice = subslice

        self.grid_distances = jnp.diff(self.grid_bounds,
                                       axis=1).ravel() / self.grid_shape
        self.grid_total_volume = jnp.prod(self.grid_shape * self.grid_distances)

        self.power_distributor, ks, _ = get_fourier_mode_distributor(
            self.grid_shape, self.grid_distances
        )
        self.power = self.harmonic_kernel(ks)
        # Assume that the kernel scales linear with the total volume
        self.power *= self.grid_total_volume / self.grid_unpadded_total_volume
        self.amplitude = jnp.sqrt(self.power)

        def ht(x):
            return 1. / self.grid_total_volume * hartley(x)

        def correlated_field(x):
            f = ht(self.amplitude[self.power_distributor] * x)
            if self.grid_subslice is None:
                return f
            return f[self.grid_subslice]

        self.correlated_field = correlated_field

        def sandwich(x):
            if self.grid_subslice is None:
                x_wpad = x
            else:
                x_wpad = jnp.zeros(tuple(self.grid_shape))
                x_wpad = x_wpad.at[self.grid_subslice].set(x)

            ht_T = jax.linear_transpose(
                ht, jax.ShapeDtypeStruct(tuple(self.grid_shape), x.dtype)
            )
            s = ht(self.power[self.power_distributor] * ht_T(x_wpad)[0])
            if self.grid_subslice is None:
                return s
            return s[self.grid_subslice]

        self.sandwich = sandwich

    def __call__(self, x):
        """Applies the Covariance matrix."""
        x_shp = x.shape
        jitter = 0. if self.jitter is None else self.jitter * x

        x = (self.w.T @ x.ravel()).reshape(tuple(self.grid_unpadded_shape))
        x = self.sandwich(x)
        x = (self.w @ x.ravel()).reshape(x_shp)
        return x + jitter

    def evaluate(self):
        probe = jnp.zeros(self.w.shape[0])
        indices = jnp.arange(self.w.shape[0]).reshape(1, -1)

        return jax.lax.map(
            lambda idx: self(probe.at[tuple(idx)].set(1.)).ravel(), indices.T
        ).T  # vmap over `indices` w/ `in_axes=1, out_axes=-1`

    def evaluate_(self, kernel):
        from scipy.spatial import distance_matrix

        if self.jitter is None:
            jitter = 0.
        else:
            jitter = self.jitter * jnp.eye(self.w.shape[0])

        p = [
            np.linspace(*b, num=sz, endpoint=True) for b, sz in
            zip(self.grid_unpadded_bounds, self.grid_unpadded_shape)
        ]
        p = np.stack(np.meshgrid(*p, indexing="ij"),
                     axis=-1).reshape(-1, len(self.grid_unpadded_shape))
        kernel_inducing = kernel(distance_matrix(p, p))

        return self.w @ kernel_inducing @ self.w.T + jitter
