# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from typing import Callable, Optional, Tuple, Union

import jax
from jax import numpy as jnp
import numpy as np

from .correlated_field import get_fourier_mode_distributor, hartley

NDArray = Union[jnp.ndarray, np.ndarray]


def matmul_toeplitz(c, x):
    """Efficiently compute the product of an implicit Toeplitz matrix with a
    vector.

    Parameters
    ----------
    c : ndarray
        One-dimensional column of the Toeplitz matrix. It is assumed that the
        row `r` is given by `r = conjugate(c)`.
    x : (M,) or (M, K) ndarray
        Vector or matrix which to multiply from the right to the Toeplitz
        matrix.
    Returns
    -------
    Toeplitz @ x : (M,) or (M, K) ndarray
        Result of the matrix multiplication.
    """
    from jax.numpy.fft import fft, ifft, rfft, irfft

    c = c.ravel()
    r = c.conjugate()

    toepl_shp = (len(c), len(c))
    x_shp = x.shape
    if x.shape[0] != r.shape[0] or x.ndim > 2:
        raise ValueError('invalid matrix product dimensions')
    x = x.reshape(-1, 1) if x.ndim == 1 else x.reshape(x.shape[0], -1)
    m = x.shape[1]

    dtp = np.result_type(*(i.dtype for i in (r, c, x)))
    r, c, x = (jnp.asarray(i, dtype=dtp) for i in (r, c, x))
    embedded_col = jnp.concatenate((c, r[-1:0:-1]))

    # For real inputs use rfft
    cmplx = jnp.iscomplexobj(embedded_col) or jnp.iscomplexobj(x)
    ht, iht = (fft, ifft) if cmplx else (rfft, irfft)
    p = toepl_shp[0] + toepl_shp[1] - 1
    ht_mat = ht(embedded_col, axis=0).reshape(-1, 1)
    ht_x = ht(x, n=p, axis=0)

    toepl_times_x = iht(ht_mat * ht_x, axis=0, n=p)[:toepl_shp[0], :]

    out_shp = (toepl_shp[0], ) if len(x_shp) == 1 else (toepl_shp[0], m)
    return toepl_times_x.reshape(*out_shp)


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
            f" sampling_points of shape {sampling_points.shape!r}"
        )
        raise ValueError(ve)
    elif grid_bounds is not None:
        offset = np.array(list(zip(*grid_bounds))[0])
    else:
        offset = np.zeros(ndim)
    if distances is not None and np.size(distances) != ndim:
        ve = (
            f"distances of size {np.size(distances)} incompatible with"
            f" sampling_points of shape {sampling_points.shape!r}"
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
    return BCOO.from_scipy_sparse(mat).sort_indices()


class HarmonicSKI():
    def __init__(
        self,
        grid_shape: Tuple[int],
        grid_bounds: Tuple[Tuple[float, float]],
        sampling_points: NDArray,
        harmonic_kernel: Optional[Callable] = None,
        padding: float = 0.5,
        subslice=None,
        jitter: Union[bool, float, None] = True
    ):
        """Instantiate a KISS-GP model of the covariance using a harmonic
        representation of the kernel.

        Parameters
        ----------
        grid_shape :
            Number of pixels along each axes of the inducing points within
            `grid_bounds`.
        grid_bounds :
            Tuple of boundaries of length of the number of dimensions. The
            boundaries should denote the leftmost and rightmost edge of the
            modeling space.
        sampling_points :
            Locations of the modeled points within the grid.
        harmonic_kernel :
            Harmonically transformed kernel.
        padding :
            Padding factor which to apply along each axis.
        subslice :
            Slice of the inducing points which to use to model
            `sampling_points`. By default, the subslice is determined by the
            padding.
        jitter :
            Strength of the diagonal jitter which to add to the covariance.
        """
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
            p = jnp.diff(jnp.asarray(grid_bounds), axis=1).ravel() * (1. - scl)
            grid_bounds_wpad = jnp.asarray(grid_bounds)
            grid_bounds_wpad = grid_bounds_wpad.at[:, 0].set(
                grid_bounds_wpad[:, 0].ravel() - p
            )
            grid_bounds_wpad = grid_bounds_wpad.at[:, 1].set(
                grid_bounds_wpad[:, 1].ravel() + p
            )
            if subslice is None:
                subslice = tuple(map(int, grid_shape))
            grid_shape = grid_shape_wpad
            grid_bounds = grid_bounds_wpad
        self.grid_shape = np.asarray(grid_shape)
        self.grid_bounds = np.asarray(grid_bounds)
        self.grid_distances = jnp.diff(self.grid_bounds,
                                       axis=1).ravel() / self.grid_shape
        self.grid_total_volume = jnp.prod(self.grid_shape * self.grid_distances)

        self.power_distributor, self.unique_mode_lengths, _ = get_fourier_mode_distributor(
            self.grid_shape, self.grid_distances
        )

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

        self._harmonic_kernel = harmonic_kernel

    @property
    def harmonic_kernel(self) -> Callable:
        """Yields the harmonic kernel specified during initialization or throw
        a `TypeError`.
        """
        if self._harmonic_kernel is None:
            te = (
                "either specify a fixed harmonic kernel during initialization"
                f" of the {self.__class__.__name__} class or provide one here"
            )
            raise TypeError(te)
        return self._harmonic_kernel

    def power(self, harmonic_kernel=None) -> NDArray:
        if harmonic_kernel is None:
            harmonic_kernel = self.harmonic_kernel
        power = harmonic_kernel(self.unique_mode_lengths)
        power *= self.grid_total_volume / self.grid_unpadded_total_volume
        return power

    def amplitude(self, harmonic_kernel=None):
        power = self.power(harmonic_kernel)
        # Assume that the kernel scales linear with the total volume
        return jnp.sqrt(power)

    def harmonic_transform(self, x) -> NDArray:
        return 1. / self.grid_total_volume * hartley(x)

    def correlated_field(self, x, harmonic_kernel=None) -> NDArray:
        amp = self.amplitude(harmonic_kernel)
        f = self.harmonic_transform(amp[self.power_distributor] * x)
        if self.grid_subslice is None:
            return f
        return f[self.grid_subslice]

    def sandwich(self, x, harmonic_kernel=None) -> NDArray:
        if self.grid_subslice is None:
            x_wpad = x
        else:
            x_wpad = jnp.zeros(tuple(self.grid_shape))
            x_wpad = x_wpad.at[self.grid_subslice].set(x)

        swd = jax.ShapeDtypeStruct(tuple(self.grid_shape), x.dtype)
        ht = self.harmonic_transform
        ht_T = jax.linear_transpose(self.harmonic_transform, swd)

        power = self.power(harmonic_kernel=harmonic_kernel)
        s = ht(power[self.power_distributor] * ht_T(x_wpad)[0])
        if self.grid_subslice is None:
            return s
        return s[self.grid_subslice]

    def __call__(self, x, harmonic_kernel=None) -> NDArray:
        """Applies the Covariance matrix."""
        x_shp = x.shape
        jitter = 0. if self.jitter is None else self.jitter * x

        x = (self.w.T @ x.ravel()).reshape(tuple(self.grid_unpadded_shape))
        x = self.sandwich(x, harmonic_kernel=harmonic_kernel)
        x = (self.w @ x.ravel()).reshape(x_shp)
        return x + jitter

    def evaluate(self, harmonic_kernel=None):
        """Instantiate the full covariance matrix."""
        probe = jnp.zeros(self.w.shape[0])
        indices = jnp.arange(self.w.shape[0]).reshape(1, -1)

        return jax.lax.map(
            lambda idx: self(
                probe.at[tuple(idx)].set(1.), harmonic_kernel=harmonic_kernel
            ).ravel(), indices.T
        ).T  # vmap over `indices` w/ `in_axes=1, out_axes=-1`

    def evaluate_(self, kernel) -> NDArray:
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


class ToeplitzSKI():
    def __init__(
        self,
        grid_shape: Tuple[int],
        grid_bounds: Tuple[Tuple[float, float]],
        sampling_points: NDArray,
        kernel: Optional[Callable] = None,
        jitter: Union[bool, float, None] = True
    ):
        """Instantiate a KISS-GP model of the covariance using a Toeplitz matrix
        representation of the kernel.

        Parameters
        ----------
        grid_shape :
            Number of pixels along each axes of the inducing points within
            `grid_bounds`.
        grid_bounds :
            Tuple of boundaries of length of the number of dimensions. The
            boundaries should denote the leftmost and rightmost edge of the
            modeling space.
        sampling_points :
            Locations of the modeled points within the grid.
        kernel :
            Position space kernel.
        jitter :
            Strength of the diagonal jitter which to add to the covariance.
        """
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

        self.grid_shape = np.asarray(grid_shape)
        self.grid_bounds = np.asarray(grid_bounds)
        self.grid_distances = jnp.diff(self.grid_bounds,
                                       axis=1).ravel() / self.grid_shape
        self.ndim = len(grid_shape)

        d = jnp.mgrid[tuple(slice(s) for s in grid_shape)]
        d *= self.grid_distances.reshape((-1, ) + (1, ) * self.ndim)
        self.grid_distances_to_zero = jnp.linalg.norm(d, axis=0)

        self.w = interp_mat(grid_shape, grid_bounds, sampling_points)

        self._kernel = kernel

    @property
    def kernel(self) -> Callable:
        """Yields the harmonic kernel specified during initialization or throw
        a `TypeError`.
        """
        if self._kernel is None:
            te = (
                "either specify a fixed kernel during initialization of the"
                f" {self.__class__.__name__} class or provide one here"
            )
            raise TypeError(te)
        return self._kernel

    def __call__(self, x, kernel=None) -> NDArray:
        """Applies the Covariance matrix."""
        kernel = self.kernel if kernel is None else kernel
        x_shp = x.shape
        jitter = 0. if self.jitter is None else self.jitter * x

        x = (self.w.T @ x.ravel()).reshape(tuple(self.grid_shape))
        cov_row = kernel(self.grid_distances_to_zero)
        x = matmul_toeplitz(cov_row, x)
        x = (self.w @ x.ravel()).reshape(x_shp)
        return x + jitter

    def evaluate(self, kernel=None):
        """Instantiate the full covariance matrix."""
        probe = jnp.zeros(self.w.shape[0])
        indices = jnp.arange(self.w.shape[0]).reshape(1, -1)

        return jax.lax.map(
            lambda idx: self(probe.at[tuple(idx)].set(1.), kernel=kernel).ravel(
            ), indices.T
        ).T  # vmap over `indices` w/ `in_axes=1, out_axes=-1`

    def evaluate_(self, kernel=None) -> NDArray:
        from scipy.spatial import distance_matrix

        kernel = self.kernel if kernel is None else kernel

        if self.jitter is None:
            jitter = 0.
        else:
            jitter = self.jitter * jnp.eye(self.w.shape[0])

        p = [
            np.linspace(*b, num=sz, endpoint=True)
            for b, sz in zip(self.grid_bounds, self.grid_shape)
        ]
        p = np.stack(np.meshgrid(*p, indexing="ij"),
                     axis=-1).reshape(-1, len(self.grid_shape))
        kernel_inducing = kernel(distance_matrix(p, p))

        return self.w @ kernel_inducing @ self.w.T + jitter
