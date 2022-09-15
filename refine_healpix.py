#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from typing import Callable, Optional, Tuple
import warnings

from healpy import pixelfunc
import healpy as hp
from jax import vmap
import jax
from jax import numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np

from nifty8.re.refine import _get_cov_from_loc


# %%
def get_1st_hp_nbrs_idx(nside, pix, nest: bool = False):
    n_nbr = 8

    n_pix = 1 if np.ndim(pix) == 0 else len(pix)
    pix_nbr = np.empty((n_pix, n_nbr + 1), dtype=int)
    pix_nbr[:, 0] = pix
    nbr = pixelfunc.get_all_neighbours(nside, pix, nest=nest)
    nbr = nbr.reshape(n_nbr, n_pix)
    pix_nbr[:, 1:] = nbr.T

    # Account for unknown neighbors, encoded by -1
    # TODO: only do this for affected pixels
    with warnings.catch_warnings():
        wmsg = "invalid value encountered in _get_neigbors"
        warnings.filterwarnings("ignore", message=wmsg)
        nbr2 = pixelfunc.get_all_neighbours(nside, nbr, nest=nest)
    nbr2 = nbr2.reshape(n_nbr, n_nbr, n_pix)
    nbr2.T[nbr.T == -1] = -1

    n_2nbr = np.prod(nbr2.shape[:-1])
    setdiffer1d = partial(jnp.setdiff1d, size=n_nbr + 2, fill_value=-1)
    pix_2nbr = jax.vmap(setdiffer1d, (1, 0),
                        0)(nbr2.reshape(n_2nbr, n_pix), pix_nbr)
    # If there is a -1 in there, it will be at the first location
    pix_2nbr = pix_2nbr[:, 1:]

    # Select a "random" 2nd neighbor to fill in for the missing 1st order
    # neighbor
    pix_nbr = np.sort(pix_nbr, axis=1)
    pix_nbr = np.where(pix_nbr != -1, pix_nbr, pix_2nbr)
    return np.squeeze(pix_nbr, axis=0) if np.ndim(pix) == 0 else pix_nbr


def get_1st_hp_nbrs(nside, pix, nest: bool = False):
    return np.stack(
        pixelfunc.pix2vec(
            nside, get_1st_hp_nbrs_idx(nside, pix, nest=nest), nest=nest
        ),
        axis=-1
    )


# %%
nside = 256
pix = 0


def test_uniqueness(nside, nest):
    pix = np.arange(12 * nside**2)
    nbr = get_1st_hp_nbrs_idx(nside, pix, nest)
    n_non_uniq = np.sum(np.diff(np.sort(nbr, axis=1), axis=1) == 0, axis=1)
    np.testing.assert_equal(n_non_uniq, 0)


for nside in (1, 2):
    for n in (True, False):
        test_uniqueness(nside, n)

get_1st_hp_nbrs(nside, pix)


# %%
def _refinement_matrices(
    # level: int,
    gc_and_gf,
    # pixel_index: Optional[Iterable[int]] = None,
    kernel: Optional[Callable] = None,
    *,
    # nest: bool = False,
    coerce_fine_kernel: bool = True,
    _cov_from_loc: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    cov_from_loc = _get_cov_from_loc(kernel, _cov_from_loc)
    n_fsz = 4  # `n_csz = 9`

    gc, gf = gc_and_gf
    coord = jnp.concatenate((gc, gf), axis=0)
    cov = cov_from_loc(coord, coord)
    cov_ff = cov[-n_fsz:, -n_fsz:]
    cov_fc = cov[-n_fsz:, :-n_fsz]
    cov_cc = cov[:-n_fsz, :-n_fsz]
    cov_cc_inv = jnp.linalg.inv(cov_cc)

    olf = cov_fc @ cov_cc_inv
    # Also see Schur-Complement
    fine_kernel = cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
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


# %%
def matern_kernel(distance, scale, cutoff, dof):
    """Evaluates the Matern covariance kernel parametrized by its `scale`,
    length scale (a.k.a. `cutoff`) and degree-of-freedom parameter `dof` at
    `distance`.
    """
    if dof == 0.5:
        cov = scale**2 * jnp.exp(-distance / cutoff)
    elif dof == 1.5:
        reg_dist = jnp.sqrt(3) * distance / cutoff
        cov = scale**2 * (1 + reg_dist) * jnp.exp(-reg_dist)
    elif dof == 2.5:
        reg_dist = jnp.sqrt(5) * distance / cutoff
        cov = scale**2 * (1 + reg_dist + reg_dist**2 / 3) * jnp.exp(-reg_dist)
    else:
        from jax.scipy.special import gammaln
        from scipy.special import kv
        from warnings import warn

        warn("falling back to generic Matern covariance function")
        reg_dist = jnp.sqrt(2 * dof) * distance / cutoff
        cov = scale**2 * 2**(1 - dof) / jnp.exp(
            gammaln(dof)
        ) * (reg_dist)**dof * kv(dof, reg_dist)

    # NOTE, this is not safe for differentiating because `cov` still may
    # contain NaNs
    return jnp.where(distance < 1e-8 * cutoff, scale**2, cov)


nest = True
kernel = partial(matern_kernel, scale=1., cutoff=1., dof=1.5)
pix0s = np.stack(pixelfunc.pix2vec(1, np.arange(12), nest=nest), axis=-1)
cov_from_loc = _get_cov_from_loc(kernel, None)
fks_sqrt = jnp.linalg.cholesky(cov_from_loc(pix0s, pix0s))

key = random.PRNGKey(43)
r0 = random.normal(key, (12, ))
coarse_values = fks_sqrt @ r0


# %%
def refine_slice(
    coarse_values,
    excitations,
    kernel,
    precision=None,
):
    ndim = np.ndim(coarse_values)
    assert ndim == 1

    nside = (coarse_values.size / 12)**0.5
    if not nside.is_integer():
        raise ValueError("invalid nside of `coarse_values`")
    nside = int(nside)

    pix_idx = np.arange(coarse_values.size)
    pix_nbr_idx = get_1st_hp_nbrs_idx(nside, pix_idx, nest=nest)

    gc = np.stack(pixelfunc.pix2vec(nside, pix_nbr_idx, nest=nest), axis=-1)
    # gc = get_1st_hp_nbrs(2**nside, pix_idx, nest=nest)
    i = pixelfunc.ring2nest(nside, pix_idx) if nest is False else pix_idx
    gf = np.stack(
        pixelfunc.pix2vec(
            2 * nside, 4 * i[:, None] + jnp.arange(0, 4)[None, :], nest=True
        ),
        axis=-1
    )

    def refine(coarse_full, exc, idx, gc, gf):
        olf, fks = _refinement_matrices((gc, gf), kernel=kernel)

        refined = jnp.tensordot(
            olf, coarse_full[idx], axes=ndim, precision=precision
        )
        refined += jnp.matmul(fks, exc)
        return refined

    refined = vmap(refine,
                   in_axes=(None, 0, 0, 0, 0
                           ))(coarse_values, excitations, pix_nbr_idx, gc, gf)
    return refined.ravel()


jax.config.update("jax_debug_nans", True)
excitations = random.normal(key, (coarse_values.size, 4))
refined0 = refine_slice(coarse_values, excitations, kernel)
refined0 = refined.ravel()

# %%
refined = coarse_values
key = random.PRNGKey(42)
depth = 8
for i in range(depth):
    _, key = random.split(key)
    exc = random.normal(key, (refined.size, 4))
    refined = refine_slice(refined, exc, kernel)

# %%
hp.mollview(coarse_values, nest=nest, fig=0)
hp.mollview(refined.ravel(), nest=nest, fig=1)
plt.show()
