#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from collections import namedtuple
from functools import partial
from math import ceil
from typing import Callable, Iterable, Literal, Optional, Tuple, Union
from warnings import warn

import jax
import numpy as np
from jax import numpy as jnp
from scipy.spatial import distance_matrix

from ..tree_math import ShapeWithDtype, zeros_like
from ..logger import logger
from ..model import LazyModel

NDARRAY = Union[jnp.ndarray, np.ndarray]

RefinementMatrices = namedtuple(
    "RefinementMatrices",
    ("filter", "propagator_sqrt", "cov_sqrt0", "index_map")
)


def refinement_matrices(cov, n_fsz: int, coerce_fine_kernel: bool):
    cov_ff = cov[-n_fsz:, -n_fsz:]
    cov_fc = cov[-n_fsz:, :-n_fsz]
    cov_cc = cov[:-n_fsz, :-n_fsz]
    del cov
    cov_cc_inv = jnp.linalg.inv(cov_cc)
    del cov_cc

    olf = cov_fc @ cov_cc_inv
    # Also see Schur-Complement
    fine_kernel = cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
    del cov_cc_inv, cov_fc, cov_ff
    if coerce_fine_kernel:
        # TODO: Try to work with NaN to avoid the expensive eigendecomposition;
        # work with nan_to_num?
        # Implicitly assume a white power spectrum beyond the numerics limit.
        # Use the diagonal as estimate for the magnitude of the variance.
        fine_kernel_fallback = jnp.diag(jnp.abs(jnp.diag(fine_kernel)))
        # Never produce NaNs (https://github.com/google/jax/issues/1052)
        # This is expensive but necessary (worse but cheaper:
        # `jnp.all(jnp.diag(fine_kernel) > 0.)`)
        is_pos_def = jnp.all(jnp.linalg.eigvalsh(fine_kernel) > 0)
        fine_kernel = jnp.where(is_pos_def, fine_kernel, fine_kernel_fallback)
        # NOTE, subsequently use the Cholesky decomposition, even though
        # already having computed the eigenvalues, as to get consistent results
        # across platforms
    # Matrices are symmetrized by JAX, i.e. gradients are projected to the
    # subspace of symmetric matrices (see
    # https://github.com/google/jax/issues/10815)
    fine_kernel_sqrt = jnp.linalg.cholesky(fine_kernel)

    return olf, fine_kernel_sqrt


def get_cov_from_loc(
    kernel=None, cov_from_loc=None
) -> Callable[[NDARRAY, NDARRAY], NDARRAY]:
    if cov_from_loc is None and callable(kernel):
        # TODO: extend to non-stationary kernels

        def cov_from_loc_sngl(x, y):
            return kernel(jnp.linalg.norm(x - y))

        cov_from_loc = jax.vmap(
            jax.vmap(cov_from_loc_sngl, in_axes=(None, 0)), in_axes=(0, None)
        )
    else:
        if not callable(cov_from_loc):
            ve = "exactly one of `cov_from_loc` or `kernel` must be set and callable"
            raise ValueError(ve)
    # TODO: benchmark whether using `triu_indices(n, k=1)` and
    # `diag_indices(n)` is advantageous
    return cov_from_loc


def get_refinement_shapewithdtype(
    shape0: Union[int, tuple],
    depth: int,
    dtype=None,
    *,
    _coarse_size: int = 3,
    _fine_size: int = 2,
    _fine_strategy: Literal["jump", "extend"] = "jump",
    skip0: bool = False,
):
    if depth < 0:
        raise ValueError(f"invalid `depth`; got {depth!r}")
    csz = int(_coarse_size)  # coarse size
    fsz = int(_fine_size)  # fine size

    swd = partial(ShapeWithDtype, dtype=dtype)

    shape0 = (shape0, ) if isinstance(shape0, int) else shape0
    ndim = len(shape0)
    exc_shp = [swd(shape0)] if not skip0 else [None]
    if depth > 0:
        if _fine_strategy == "jump":
            exc_lvl = tuple(el - (csz - 1) for el in shape0) + (fsz**ndim, )
        elif _fine_strategy == "extend":
            exc_lvl = tuple(
                ceil((el - (csz - 1)) / (fsz // 2)) for el in shape0
            ) + (fsz**ndim, )
        else:
            raise ValueError(f"invalid `_fine_strategy`; got {_fine_strategy}")
        exc_shp += [swd(exc_lvl)]
    for lvl in range(1, depth):
        if _fine_strategy == "jump":
            exc_lvl = tuple(
                fsz * el - (csz - 1) for el in exc_shp[-1].shape[:-1]
            ) + (fsz**ndim, )
        elif _fine_strategy == "extend":
            exc_lvl = tuple(
                ceil((fsz * el - (csz - 1)) / (fsz // 2))
                for el in exc_shp[-1].shape[:-1]
            ) + (fsz**ndim, )
        else:
            raise AssertionError()
        if any(el <= 0 for el in exc_lvl):
            ve = (
                f"`shape0` ({shape0}) with `depth` ({depth}) yield an"
                f" invalid shape ({exc_lvl}) at level {lvl}"
            )
            raise ValueError(ve)
        exc_shp += [swd(exc_lvl)]

    return exc_shp


def coarse2fine_shape(
    shape0: Union[int, Iterable[int]],
    depth: int,
    *,
    _coarse_size: int = 3,
    _fine_size: int = 2,
    _fine_strategy: Literal["jump", "extend"] = "jump",
):
    """Translates a coarse shape to its corresponding fine shape."""
    shape0 = (shape0, ) if isinstance(shape0, int) else shape0
    csz = int(_coarse_size)  # coarse size
    fsz = int(_fine_size)  # fine size
    if _fine_size % 2 != 0:
        raise ValueError("only even numbers allowed for `_fine_size`")

    shape = []
    for shp in shape0:
        sz_at = shp
        for lvl in range(depth):
            if _fine_strategy == "jump":
                sz_at = fsz * (sz_at - (csz - 1))
            elif _fine_strategy == "extend":
                sz_at = fsz * ceil((sz_at - (csz - 1)) / (fsz // 2))
            else:
                ve = f"invalid `_fine_strategy`; got {_fine_strategy}"
                raise ValueError(ve)
            if sz_at <= 0:
                ve = (
                    f"`shape0` ({shape0}) with `depth` ({depth}) yield an"
                    f" invalid shape ({sz_at}) at level {lvl}"
                )
                raise ValueError(ve)
        shape.append(int(sz_at))
    return tuple(shape)


def fine2coarse_shape(
    shape: Union[int, Iterable[int]],
    depth: int,
    *,
    _coarse_size: int = 3,
    _fine_size: int = 2,
    _fine_strategy: Literal["jump", "extend"] = "jump",
    ceil_sizes: bool = False,
):
    """Translates a fine shape to its corresponding coarse shape."""
    shape = (shape, ) if isinstance(shape, int) else shape
    csz = int(_coarse_size)  # coarse size
    fsz = int(_fine_size)  # fine size
    if _fine_size % 2 != 0:
        raise ValueError("only even numbers allowed for `_fine_size`")

    shape0 = []
    for shp in shape:
        sz_at = shp
        for lvl in range(depth, 0, -1):
            if _fine_strategy == "jump":
                # solve for n: `fsz * (n - (csz - 1))`
                sz_at = sz_at / fsz + (csz - 1)
            elif _fine_strategy == "extend":
                # solve for n: `fsz * ceil((n - (csz - 1)) / (fsz // 2))`
                # NOTE, not unique because of `ceil`; use lower limit
                sz_at_max = (sz_at / fsz) * (fsz // 2) + (csz - 1)
                sz_at_min = ceil(sz_at_max - (fsz // 2 - 1))
                for sz_at_cand in range(sz_at_min, ceil(sz_at_max) + 1):
                    try:
                        shp_cand = coarse2fine_shape(
                            (sz_at_cand, ),
                            depth=depth - lvl + 1,
                            _coarse_size=csz,
                            _fine_size=fsz,
                            _fine_strategy=_fine_strategy
                        )[0]
                    except ValueError as e:
                        if "invalid shape" not in "".join(e.args):
                            ve = "unexpected behavior of `coarse2fine_shape`"
                            raise ValueError(ve) from e
                        shp_cand = -1
                    if shp_cand >= shp:
                        sz_at = sz_at_cand
                        break
                else:
                    ve = f"interval search within [{sz_at_min}, {ceil(sz_at_max)}] failed"
                    raise ValueError(ve)
            else:
                ve = f"invalid `_fine_strategy`; got {_fine_strategy}"
                raise ValueError(ve)

            sz_at = ceil(sz_at) if ceil_sizes else sz_at
            if sz_at != int(sz_at):
                raise ValueError(f"invalid shape at level {lvl}")
        shape0.append(int(sz_at))
    return tuple(shape0)


def coarse2fine_distances(
    distances0: Union[float, Iterable[float]],
    depth: int,
    *,
    _fine_size: int = 2,
    _fine_strategy: Literal["jump", "extend"] = "jump",
):
    """Translates coarse distances to its corresponding fine distances."""
    fsz = int(_fine_size)  # fine size
    if _fine_strategy == "jump":
        fpx_in_cpx = fsz**depth
    elif _fine_strategy == "extend":
        fpx_in_cpx = 2**depth
    else:
        ve = f"invalid `_fine_strategy`; got {_fine_strategy}"
        raise ValueError(ve)

    return jnp.atleast_1d(distances0) / fpx_in_cpx


def fine2coarse_distances(
    distances: Union[float, Iterable[float]],
    depth: int,
    *,
    _fine_size: int = 2,
    _fine_strategy: Literal["jump", "extend"] = "jump",
):
    """Translates fine distances to its corresponding coarse distances."""
    fsz = int(_fine_size)  # fine size
    if _fine_strategy == "jump":
        fpx_in_cpx = fsz**depth
    elif _fine_strategy == "extend":
        fpx_in_cpx = 2**depth
    else:
        ve = f"invalid `_fine_strategy`; got {_fine_strategy}"
        raise ValueError(ve)

    return jnp.atleast_1d(distances) * fpx_in_cpx


def _clipping_posdef_logdet(mat, msg_prefix=""):
    sign, logdet = jnp.linalg.slogdet(mat)
    if sign <= 0:
        ve = "not positive definite; clipping eigenvalues"
        warn(msg_prefix + ve)
        eps = jnp.finfo(mat.dtype.type).eps
        evs = jnp.linalg.eigvalsh(mat)
        logdet = jnp.sum(jnp.log(jnp.clip(evs, a_min=eps * evs.max())))
    return logdet


def gauss_kl(cov_desired, cov_approx, *, m_desired=None, m_approx=None):
    cov_t_dl = _clipping_posdef_logdet(cov_desired, msg_prefix="`cov_desired` ")
    cov_a_dl = _clipping_posdef_logdet(cov_approx, msg_prefix="`cov_approx` ")
    cov_a_inv = jnp.linalg.inv(cov_approx)

    kl = -cov_desired.shape[0]  # number of dimensions
    kl += cov_a_dl - cov_t_dl + jnp.trace(cov_a_inv @ cov_desired)
    if m_approx is not None and m_desired is not None:
        m_diff = m_approx - m_desired
        kl += m_diff @ cov_a_inv @ m_diff
    elif not (m_approx is None and m_approx is None):
        ve = "either both or neither of `m_approx` and `m_desired` must be `None`"
        raise ValueError(ve)
    return 0.5 * kl


def refinement_covariance(chart_or_model, kernel=None, jit=True):
    """Computes the implied covariance as modeled by the refinement scheme."""
    from .chart import CoordinateChart, HEALPixChart
    from .charted_field import RefinementField

    if isinstance(chart_or_model, CoordinateChart):
        cf = RefinementField(chart_or_model, kernel=kernel)
        shape = chart_or_model.shape
    elif isinstance(chart_or_model, HEALPixChart):
        cf = RefinementField(chart_or_model, kernel=kernel)
        shape = chart_or_model.shape
    elif isinstance(chart_or_model, LazyModel):
        cf = chart_or_model
        shape = chart_or_model.target.shape
    else:
        te = f"expected a model or a chart; got {type(chart_or_model)!r}"
        raise TypeError(te)
    ndim = len(shape)

    try:
        cf_T = jax.linear_transpose(cf, cf.domain)
        cov_implicit = lambda x: cf(*cf_T(x))
        cov_implicit = jax.jit(cov_implicit) if jit else cov_implicit
        _ = cov_implicit(jnp.zeros(shape))  # Test transpose
    except (NotImplementedError, AssertionError):
        # Workaround JAX not yet implementing the transpose of the scanned
        # refinement
        _, cf_T = jax.vjp(cf, zeros_like(cf.domain))
        cov_implicit = lambda x: cf(*cf_T(x))
        cov_implicit = jax.jit(cov_implicit) if jit else cov_implicit

    probe = jnp.zeros(shape)
    indices = np.indices(shape).reshape(ndim, -1)
    cov_empirical = jax.lax.map(
        lambda idx: cov_implicit(probe.at[tuple(idx)].set(1.)).ravel(),
        indices.T
    ).T  # vmap over `indices` w/ `in_axes=1, out_axes=-1`

    return cov_empirical


def true_covariance(chart, kernel, depth=None):
    """Computes the true covariance at the final grid."""
    depth = chart.depth if depth is None else depth

    c0_slc = tuple(slice(sz) for sz in chart.shape_at(depth))
    pos = jnp.stack(chart.ind2cart(jnp.mgrid[c0_slc], depth),
                    axis=-1).reshape(-1, chart.ndim)
    dist_mat = distance_matrix(pos, pos)
    return kernel(dist_mat)


def refinement_approximation_error(
    chart,
    kernel: Callable,
    cutout: Optional[Union[slice, int, Tuple[slice], Tuple[int]]] = None,
):
    """Computes the Kullback-Leibler (KL) divergence of the true covariance versus the
    approximative one for a given kernel and shape of the fine grid.

    If the desired shape can not be matched, the next larger one is used and
    the field is subsequently cropped to the desired shape.
    """

    suggested_min_shape = 2 * 4**chart.depth
    if any(s <= suggested_min_shape for s in chart.shape):
        msg = (
            f"shape {chart.shape} potentially too small"
            f" (desired {(suggested_min_shape, ) * chart.ndim} (=`2*4^depth`))"
        )
        warn(msg)

    cov_empirical = refinement_covariance(chart, kernel)
    cov_truth = true_covariance(chart, kernel)

    if cutout is None and all(s > suggested_min_shape for s in chart.shape):
        cutout = (suggested_min_shape, ) * chart.ndim
        logger.info(f"cropping field (w/ shape {chart.shape}) to {cutout}")
    if cutout is not None:
        if isinstance(cutout, slice):
            cutout = (cutout, ) * chart.ndim
        elif isinstance(cutout, int):
            cutout = (slice(cutout), ) * chart.ndim
        elif isinstance(cutout, tuple):
            if all(isinstance(el, slice) for el in cutout):
                pass
            elif all(isinstance(el, int) for el in cutout):
                cutout = tuple(slice(el) for el in cutout)
            else:
                raise TypeError("elements of `cutout` of invalid type")
        else:
            raise TypeError("`cutout` of invalid type")

        cov_empirical = cov_empirical.reshape(chart.shape * 2)[cutout * 2]
        cov_truth = cov_truth.reshape(chart.shape * 2)[cutout * 2]
        sz = np.prod(cov_empirical.shape[:chart.ndim])
        if np.prod(cov_truth.shape[:chart.ndim]) != sz or not sz.dtype == int:
            raise AssertionError()
        cov_empirical = cov_empirical.reshape(sz, sz)
        cov_truth = cov_truth.reshape(sz, sz)

    aux = {
        "cov_empirical": cov_empirical,
        "cov_truth": cov_truth,
    }
    return gauss_kl(cov_truth, cov_empirical), aux


REFINEMENT_STRATEGIES = [
    {
        "_coarse_size": 3,
        "_fine_size": 2
    },
    {
        "_coarse_size": 3,
        "_fine_size": 4
    },
    {
        "_coarse_size": 5,
        "_fine_size": 2
    },
    {
        "_coarse_size": 5,
        "_fine_size": 4
    },
    {
        "_coarse_size": 5,
        "_fine_size": 6
    },
]


def get_optimal_refinement_chart(
    kernel,
    *,
    shape0,
    refinement_strategies=REFINEMENT_STRATEGIES,
    **common_kwargs
):
    """Compute the Kullback-Leibler divergence for a given `kernel`, initial
    `shape0`, and parameters of a coordinate chart (`common_kwargs`); returning
    the set within `refinement_strategies` that has the lowest Kullback-Leibler
    divergence.
    """
    from .chart import CoordinateChart

    charts = []
    min_shape = (1 << 31, ) * len(shape0)  # Absurdly large placeholder value
    for kwargs in refinement_strategies:
        cc = CoordinateChart(shape0=shape0, **(common_kwargs | kwargs))
        charts.append(cc)
        min_shape = tuple(min(ms, s) for ms, s in zip(min_shape, cc.shape))

    errors = []
    for cc in charts:
        err, _ = refinement_approximation_error(cc, kernel, cutout=min_shape)
        errors.append(err)

    chosen = np.argmin(errors)
    return charts[chosen], tuple(zip(errors, charts))
