#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from math import ceil
import sys
from typing import Iterable, Literal, Union
from warnings import warn

import jax
from jax import numpy as jnp
import numpy as np
from scipy.spatial import distance_matrix

from .refine import get_fixed_power_correlated_field


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


def gauss_kl(cov_desired, cov_approx, *, m_desired=None, m_approx=None):
    cov_t_ds, cov_t_dl = jnp.linalg.slogdet(cov_desired)
    cov_a_ds, cov_a_dl = jnp.linalg.slogdet(cov_approx)
    if (cov_t_ds * cov_a_ds) <= 0.:
        raise ValueError("fraction of determinants must be positive")

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


def refinement_approximation_error(
    *,
    shape,
    distances,
    depth,
    kernel,
    cutout=None,
    _coarse_size=3,
    _fine_size=2,
    _fine_strategy="jump"
):
    """Computes the Kullback-Leibler (KL) divergence of the true covariance versus the
    approximative one for a given kernel and shape of the fine grid.

    If the desired shape can not be matched, the next larger one is used and
    the field is subsequently cropped to the desired shape.
    """
    if _fine_strategy == "jump":
        fpx_in_cpx = _fine_size**depth
    elif _fine_strategy == "extend":
        fpx_in_cpx = 2**depth
    else:
        ve = f"invalid `_fine_strategy`; got {_fine_strategy}"
        raise ValueError(ve)
    distances0 = tuple(d * fpx_in_cpx for d in distances)

    if any(s <= 2 * fpx_in_cpx for s in shape):
        msg = f"shape potentially too small (coarse pixel has {fpx_in_cpx} fine pixels)"
        warn(msg)

    shape0 = fine2coarse_shape(
        shape,
        depth,
        ceil_sizes=True,
        _coarse_size=_coarse_size,
        _fine_size=_fine_size,
        _fine_strategy=_fine_strategy
    )
    cf, dom = get_fixed_power_correlated_field(
        shape0=shape0,
        distances0=distances0,
        depth=depth,
        kernel=kernel,
        _coarse_size=_coarse_size,
        _fine_size=_fine_size,
        _fine_strategy=_fine_strategy
    )
    tgt = jax.eval_shape(cf, dom)
    cf_T = jax.linear_transpose(cf, dom)
    cov_implicit = jax.jit(lambda x: cf(*cf_T(x)))

    c0 = [
        d * jnp.arange(sz, dtype=float) for d, sz in zip(distances, tgt.shape)
    ]
    pos = jnp.stack(jnp.meshgrid(*c0, indexing="ij"), axis=0)

    probe = jnp.zeros(pos.shape[1:])
    indices = np.indices(pos.shape[1:]).reshape(pos.ndim - 1, -1)
    cov_empirical = jax.lax.map(
        lambda idx: cov_implicit(probe.at[tuple(idx)].set(1.)).ravel(),
        indices.T
    ).T  # vmap over `indices` w/ `in_axes=1, out_axes=-1`

    p = jnp.moveaxis(pos, 0, -1).reshape(-1, pos.shape[0])
    dist_mat = distance_matrix(p, p)
    del p
    cov_truth = kernel(dist_mat)

    if cutout is None and tgt.shape != shape:
        print(
            f"cropping enlarged field (w/ shape {tgt.shape}) to {shape}",
            file=sys.stderr
        )
        cutout = tuple(slice(s) for s in shape)
    if cutout is not None:
        sz = np.prod(shape)
        cov_empirical = cov_empirical.reshape(tgt.shape * 2
                                             )[cutout * 2].reshape(sz, sz)
        cov_truth = cov_truth.reshape(tgt.shape * 2)[cutout * 2].reshape(sz, sz)

    aux = {
        "cov_empirical": cov_empirical,
        "cov_truth": cov_truth,
        "shape0": shape0,
        "distances0": distances0,
        "correlated_field": cf,
        "domain": dom
    }
    return gauss_kl(cov_truth, cov_empirical), aux
