# Copyright(C) 2023 Philipp Frank
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import numpy as np
import jax.numpy as jnp
import healpy as hp
from functools import partial
from jax import vmap, jit
from .index_utils import my_setdiff_indices, id_to_axisids


def full_hp_radial_sorting(arrays, chart):
    """Returns the finest level sorted to match a full healpix+radial
    pixelization.

    Parameters:
    -----------
    arrays: List of Arrays
        A Field represented on `chart`.
    chart: MSChart
        Chart of the space. Must contain exactly two axes where one of them is
        a `HPAxis`. The chart is assumed to be fully resolved (with potentially
        open boundaries in the radial axis). This is checked for rudimentary by
        testing if the indices are a multiple of the number of pixels of the
        healpix sphere.

    Returns:
    --------
    Array:
        two-dimensional array of the pixels sorted according to the healpix
        nested scheme and radial bins.
    """
    axs = chart.axes(-1)
    if len(axs) != 2:
        msg = "Chart has to consist of one healpix and one extra axis!"
        raise ValueError(msg)
    from .axes import HPAxis
    hpaxis_id = int(isinstance(axs[1], HPAxis))
    raxis_id = (hpaxis_id+1)%2
    hpax = chart.axes(-1)[hpaxis_id]
    ids = chart.indices[-1]

    npixhp = 12 * hpax.nside**2
    npixr = ids.size // npixhp
    if npixhp*npixr != ids.size:
        raise ValueError("Inconsistent number of pixels in space!")

    ids = id_to_axisids(ids, chart.maxlevel, chart._axes)
    hpids = ids[hpaxis_id]
    rids = ids[raxis_id]
    array = arrays[-1]

    rsort = np.argsort(rids)
    hpids = hpids[rsort].reshape((npixr, npixhp))
    array = array[rsort].reshape((npixr, npixhp))

    array = array[np.arange(npixr, dtype=int)[:,np.newaxis],
                  np.argsort(hpids, axis=1)]
    if raxis_id == 1:
        array = array.T
    return array


def sorted_concat(arr1, inds1, arr2, inds2):
    """Concatenate two arrays and sort their values according to the supplied
    indices.

    Parameters:
    -----------
    arr1: jax.DeviceArray
        First array to be concatenated.
    inds1: numpy.ndarray
        Indices corresponding to the first array. Is assumed to be unique.
    arr2: jax.DeviceArray
        Second array to be concatenated
    inds2: numpy.ndarray
        Indices corresponding to the second array. Is assumed to be unique.

    Notes:
    ------
    If `inds2` containts indices that are in `inds1` an error is raised.
    """
    if my_setdiff_indices(inds2, inds1).size != inds2.size:
        raise ValueError
    res = jnp.concatenate((arr1, arr2), axis = 0)
    inds = np.concatenate((inds1,inds2))
    sort = np.argsort(inds)
    return res[sort], inds[sort]

@partial(jit, static_argnames = 'axes')
def axes_matmul(array, kernels, select, axes, ker_selects):
    """Performs tensor multiplication for a product tensor like structure, i.E.
    each axis has its own tensor that gets applied.
    """
    ndim = len(kernels)
    def my_mul(select, ker_selects):
        res = array[select]
        for kk, ss, aa in zip(kernels, ker_selects, axes):
            res = jnp.tensordot(res, kk[ss], axes = aa)
        return res
    return vmap(my_mul, (0, (0,)*ndim), 0)(select, ker_selects)

def get_all_kneighbours(nside, pix, level, nest=True):
    if level>1:
        raise NotImplementedError
    totpix = hp.nside2npix(nside)
    assert np.all(pix >= 0)
    assert np.all(pix < totpix)
    if level == 0:
        all_pix = pix[np.newaxis, ...]
        return all_pix, np.zeros(all_pix.shape, dtype=bool)
    shp = pix.shape
    pix = pix.flatten()
    new = hp.get_all_neighbours(nside, pix, nest=nest)
    new = np.concatenate((pix[np.newaxis, ...], new), axis=0)
    is_bad = new == -1

    good = is_bad.sum(axis=0) == 0
    new = new.T
    is_bad = is_bad.T

    tmp = new[~good]
    for id in range(tmp.shape[0]):
        tmp[id][tmp[id]==-1] = tmp[id,0]
    new[~good] = tmp
    return new.T.reshape((9,) + shp), is_bad.T.reshape((9,) + shp)




def old_get_all_kneighbours(nside, pix, level, nest):
    """Extends/Modifies healpy's `get_all_neighbours` function to the
    (k-)nearest-neighbours of a pixel on a healpixsphere. If a neighbour does
    not exist (see `healpy.get_all_neighbours` docu for the reasons), the id of
    the input pixel is returned at the position of the missing neighbour. In
    addition to the neighbours, a boolean array is returned that marks all
    missing `bad` pixels.

    Notes:
    ------
    Note that in `healpy.get_all_neighbours` the missing neighbours always get
    an id of `-1` rather than the input id as done here. We use th input id here
    to ensure that the selected array of the neighbours only contains valid ids
    that are close to the original one.
    """
    totpix = hp.nside2npix(nside)
    assert np.all(pix >= 0)
    assert np.all(pix < totpix)
    if level == 0:
        all_pix = pix[np.newaxis, ...]
        return all_pix, np.zeros(all_pix.shape, dtype=bool)
    neighbors = pix
    for _ in range(level):
        window = neighbors != -1
        new_neighbors = np.full((8,neighbors.size), -1)
        new_neighbors[:,window] = hp.get_all_neighbours(nside,
                                                        neighbors[window],
                                                        nest=nest)
        neighbors = new_neighbors.flatten()

    neighbors = neighbors.reshape((8**level,)+ pix.shape)
    neighbors.sort(axis = 0)
    window = neighbors == -1
    good = (window.sum(axis = 0) == 0)

    n_ngbrs = (1 + 2*level)**2
    all_pix = np.full((n_ngbrs, pix.size), -1, dtype=pix.dtype)
    good_pix = pix[good]
    bad_pix = pix[~good]
    if good_pix.size != 0:
        good_neighbors = neighbors[:, good]
        shp = good_neighbors.shape
        rem = (good_neighbors != good_pix)
        sz = rem.sum(axis=0)
        assert np.all(sz == sz[0])
        shp = (shp[1], sz[0])
        good_neighbors = (good_neighbors.T[rem.T]).reshape(shp).T
        unq = (good_neighbors[1:] - good_neighbors[:-1]) > 0
        unq = np.concatenate(
            (np.ones(unq.shape[1], dtype=bool)[np.newaxis, ...], unq), axis = 0)
        good_neighbors = good_neighbors.T[unq.T].reshape(
            (shp[0], n_ngbrs - 1)).T
        good_neighbors = np.concatenate((good_pix[np.newaxis, ...],
                                         good_neighbors))

        all_pix[:, good] = good_neighbors

    is_bad = np.zeros(all_pix.shape, dtype=bool)
    if bad_pix.size > 0:
        bad_neighbors = neighbors[:, ~good]
        bad_neighbors = list(np.concatenate((np.array([pp,]),
                            np.unique(nn[(nn!=-1)*(nn!=pp)]))) for nn,pp in
                            zip(bad_neighbors.T, bad_pix))
        ngood = tuple(bb.size for bb in bad_neighbors)
        bad_neighbors = [np.concatenate(
                        (nn, nn[0]*np.ones(n_ngbrs-sz, dtype=nn.dtype)))
                        for nn, sz in zip(bad_neighbors, ngood)]
        bad_neighbors = np.stack(bad_neighbors, axis=-1)
        bad_ones = [np.concatenate((np.zeros(ng, dtype=bool),
                                    np.ones(n_ngbrs-ng, dtype=bool)))
                    for ng in ngood]
        bad_ones = np.stack(bad_ones, axis=-1)
        all_pix[:, ~good] = bad_neighbors
        is_bad[:, ~good] = bad_ones
    sort = np.argsort(all_pix, axis=0)
    y = np.arange(sort.shape[1], dtype=sort.dtype)[np.newaxis, ...]
    all_pix, is_bad = all_pix[sort, y], is_bad[sort, y]

    vpix = hp.pix2ang(nside, pix, nest=True)
    vpix = np.stack(vpix, axis=0)[:, np.newaxis, ...]
    shp = all_pix.shape
    vnbr = hp.pix2ang(nside, all_pix.flatten(), nest=True)
    vnbr = np.stack(vnbr, axis=0).reshape((2,) + shp)
    dv = vnbr - vpix
    dv = dv[1] + 1.j*dv[0]
    s = np.argsort(dv, axis=0)
    y = np.arange(s.shape[1], dtype=s.dtype)[np.newaxis, ...]
    return all_pix[s, y], is_bad[s, y]
    return all_pix, is_bad

# Pure jax implementation for Bessel function from
# Copy taken from
# https://github.com/benjaminpope/sibylla/blob/main/notebooks/bessel_test.ipynb
RP1 = jnp.array([
-8.99971225705559398224E8, 4.52228297998194034323E11,
-7.27494245221818276015E13, 3.68295732863852883286E15,])
RQ1 = jnp.array([
 1.0, 6.20836478118054335476E2, 2.56987256757748830383E5,
 8.35146791431949253037E7, 2.21511595479792499675E10, 4.74914122079991414898E12,
 7.84369607876235854894E14, 8.95222336184627338078E16,
 5.32278620332680085395E18,])

PP1 = jnp.array([
 7.62125616208173112003E-4, 7.31397056940917570436E-2, 1.12719608129684925192E0,
 5.11207951146807644818E0, 8.42404590141772420927E0, 5.21451598682361504063E0,
 1.00000000000000000254E0,])
PQ1 = jnp.array([
 5.71323128072548699714E-4, 6.88455908754495404082E-2, 1.10514232634061696926E0,
 5.07386386128601488557E0, 8.39985554327604159757E0, 5.20982848682361821619E0,
 9.99999999999999997461E-1,])

QP1 = jnp.array([
 5.10862594750176621635E-2, 4.98213872951233449420E0, 7.58238284132545283818E1,
 3.66779609360150777800E2, 7.10856304998926107277E2, 5.97489612400613639965E2,
 2.11688757100572135698E2, 2.52070205858023719784E1,])
QQ1  = jnp.array([
 1.0, 7.42373277035675149943E1, 1.05644886038262816351E3,
 4.98641058337653607651E3, 9.56231892404756170795E3, 7.99704160447350683650E3,
 2.82619278517639096600E3, 3.36093607810698293419E2,])

Z1 = 1.46819706421238932572E1
Z2 = 4.92184563216946036703E1
THPIO4 = 2.35619449019234492885 # 3*pi/4
SQ2OPI = .79788456080286535588 # sqrt(2/pi)

def j1_small(x):
    z = x * x
    w = jnp.polyval(RP1, z) / jnp.polyval(RQ1, z)
    w = w * x * (z - Z1) * (z - Z2)
    return w

def j1_large_c(x):
    w = 5.0 / x
    z = w * w
    p = jnp.polyval(PP1, z) / jnp.polyval(PQ1, z)
    q = jnp.polyval(QP1, z) / jnp.polyval(QQ1, z)
    xn = x - THPIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)

def j1(x):
    """
    Bessel function of order one - using the implementation from CEPHES,
    translated to Jax.
    """
    return (jnp.sign(x)*jnp.where(jnp.abs(x) < 5.0,
                                  j1_small(jnp.abs(x)),
                                  j1_large_c(jnp.abs(x))))