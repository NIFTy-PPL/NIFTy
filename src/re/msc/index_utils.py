# Copyright(C) 2023 Philipp Frank
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import reduce
import numpy as np


def my_setdiff_indices(query, indices):
    """
    Parameters:
    -----------
    query : numpy.ndarray of int
        Array of coordinate indices that should be checked if they are in
        `indices`.
    indices : numpy.ndarray of int
        Array of coordinate indices that `query` is checked against
    Returns:
    --------
        numpy.ndarray
        Indices in `query` that are not in `indices`

    Note:
    -----
        Assumes `indices` to be unique!
    """
    query = np.unique(query.flatten())
    return query[~np.isin(query, indices, assume_unique=True)]

def get_table(indices):
    """
    Parameters:
    -----------
    indices : np.ndarray of int
        Indices of positions of elements for an array of the same shape
    Returns:
    --------
        dict
        Lookup table of array index according to position id
    """
    return {idx:i for i, idx in enumerate(indices)}

def table_to_array(table, dtype):
    """
    Parameters:
    -----------
    table : dict
        Lookup table of array index according to position id
    Returns:
    --------
        np.ndarray of int
        Indices of positions of elements for an array of the same shape
    """
    res = {vv:kk for kk,vv in table.items()}
    return np.array([res[i] for i in range(len(table.keys()))], dtype=dtype)

def get_selection(table, selection_indices):
    """
    Parameters:
    -----------
    table : dict
        Lookup table of array index according to position id
    selection_indices : np.ndarray of int
        Indices of positions that should be selected from an array
    Returns:
    --------
        np.ndarray of int
        Array indices of selected position ids
    """
    shp = selection_indices.shape
    return (np.array([table[ss] for ss in selection_indices.flatten()],
                     dtype=selection_indices.dtype).reshape(shp))

def update_table(table, new_ids, check = True):
    """Add new indices and append their corresponding indices of the array.
    """
    if check:
        for nn in new_ids:
            if nn in table.keys():
                raise ValueError
    table.update({idx : len(table.keys()) + i for i, idx in enumerate(new_ids)})

def batch_table_to_table(table, level, axes):
    if level == 0:
        raise ValueError
    ids = np.array(list(table.keys()))
    fids = get_fine_indices(ids, level-1, axes)
    sz = reduce(lambda a,b: a*b, fids.shape[1:])
    sel = np.array(list(table.values()))
    sel = np.add.outer(sz*sel, np.arange(sz, dtype=sel.dtype))
    return {kk:ii for kk,ii in zip(fids.flatten(), sel.flatten())}

def _get_axes_tuple(axes, maxlevel):
    axs = [axes,]
    for _ in range(maxlevel):
        axs.append(tuple((aa.fine_axis for aa in axs[-1])))
    return tuple(axs)

def _ax_to_pix(iax, shp0, bases):
    #TODO vectorize properly
    i = np.zeros(iax.shape[1:], dtype=iax.dtype)
    for n in range(bases.shape[1]+1):
        sh = shp0 if n == 0 else tuple(bases[:,n-1])
        j = 0
        for ax in range(len(sh)):
            j *= sh[ax]
            j += ((iax[ax]//reduce(lambda a,b:a*b, bases[ax,n:], 1))%sh[ax])
        i *= reduce(lambda a,b: a*b, sh)
        i += j
    return i

def _pix_to_ax(pix, shp0, bases):
    #TODO vectorize properly
    iax = np.zeros((bases.shape[0],) + pix.shape, dtype=pix.dtype)
    i = np.copy(pix)
    for n in range(bases.shape[1]+1)[::-1]:
        sh = shp0 if n == 0 else tuple(bases[:,n-1])
        fct = reduce(lambda a,b: a*b, sh)
        j = i % fct
        for ax in range(len(sh))[::-1]:
            iax[ax] += reduce(lambda a,b:a*b, bases[ax,n:], 1) * (j%sh[ax])
            j //= sh[ax]
        i //= fct
    return iax

def _get_base_shapes(lvl, axes):
    shp0 = tuple(ax.size for ax in axes)
    shpm = []
    bases = [[] for _ in range(len(axes))]
    bases = np.zeros((len(axes), lvl), dtype=int)
    for i, ax in enumerate(axes):
        for ll in range(lvl):
            bases[i, ll] = ax.base
            ax = ax.fine_axis
        shpm.append(ax.size)
    return bases, shp0, shpm

def axisids_to_id(index, lvl, axes):
    """Translates the axisids of a pixel to the corresponding pixelid"""
    bases, shp0, shpm = _get_base_shapes(lvl, axes)
    for ii, sh in zip(index, shpm):
        if np.any(ii > sh) or np.any(ii < 0):
            raise ValueError
    index = _ax_to_pix(index, shp0, bases)
    if np.any(index > reduce(lambda a,b: a*b, shpm)) or np.any(index < 0):
        raise ValueError
    return index

def id_to_axisids(index, lvl, axes):
    """Translates the pixelid of a pixel to the corresponding axisids"""
    bases, shp0, shpm = _get_base_shapes(lvl, axes)
    if np.any(index > reduce(lambda a,b: a*b, shpm)) or np.any(index < 0):
        raise ValueError
    index = _pix_to_ax(index, shp0, bases)
    for ii, sh in zip(index, shpm):
        if np.any(ii > sh) or np.any(ii < 0):
            raise ValueError
    return index

def my_axes_outer(slices):
    """Helper to create an axis meshgrid given slices along all axes"""
    res = np.zeros((len(slices),) +
                    tuple(ss.shape[-1] for ss in slices) +
                    (slices[0].shape[0],),
                    dtype=slices[0].dtype)
    for i, sl in enumerate(slices):
        res = np.moveaxis(res, 1 + i, -1)
        res[i] += sl
        res = np.moveaxis(res, -1, 1 + i)
    return np.moveaxis(res, -1, 1)

def get_fine_indices(index, lvl, axes):
    """Get all pixelids on the next fine level that `index` contains.

    Parameters:
    -----------
    index: numpy.ndarray of int
        Index on coarse level.
    lvl: int
        Refinment level of `index`
    axes: RegularAxes
        Axes which correspond to the chart.
    """
    myax = _get_axes_tuple(axes, lvl)[-1]
    index = id_to_axisids(index, lvl, axes)
    index = tuple(ax.get_fine_indices(ii) for ax, ii in zip(myax, index))
    index = my_axes_outer(index)
    return axisids_to_id(index, lvl+1, axes)

def get_coarse_index(index, lvl, axes):
    """Coarse pixelid of the pixel that `index` belongs to.

    Parameters:
    -----------
    index: int or numpy.ndarray of int
        Index for which the coarse index is requested.
    lvl: int
        Refinment level of `index`
    axes: RegularAxes
        Axes which correspond to the chart.
    """
    index = id_to_axisids(index, lvl, axes)
    cax = _get_axes_tuple(axes, lvl-1)[-1]
    index = np.stack(tuple(cax.get_binid_of(ii) for cax,ii in
                           zip(cax, index)), axis=0)
    return axisids_to_id(index, lvl-1, axes)

def get_kernel_window(index, lvl, axes):
    """The index window covered by the kernel for a given output `index`.

    Parameters:
    -----------
    index: numpy.ndarray of int
        Index around which the kernel window is requested.
    lvl: int
        Refinment level of `index`
    axes: RegularAxes
        Axes which correspond to the chart.
    """
    if len(index.shape) != 1:
        raise ValueError
    index = id_to_axisids(index, lvl, axes)
    myax = _get_axes_tuple(axes, lvl)[-1]
    slices = tuple(aa.get_kernel_window_ids(ii) for aa,ii in zip(myax, index))
    return axisids_to_id(my_axes_outer(slices), lvl, axes)

def get_inter_window(index, lvl, axes):
    """The index window covered by the kernel and interpolation for a given
    output `index`.

    Parameters:
    -----------
    index: numpy.ndarray of int
        Index around which the kernel window is requested.
    lvl: int
        Refinment level of `index`
    axes: RegularAxes
        Axes which correspond to the chart.
    """
    if len(index.shape) != 1:
        raise ValueError
    index = id_to_axisids(index, lvl, axes)
    myax = _get_axes_tuple(axes, lvl)[-1]
    slices = tuple(aa.get_inter_window_ids(ii) for aa,ii in zip(myax, index))
    return axisids_to_id(my_axes_outer(slices), lvl, axes)

def get_batch_kernel_window(index, lvl, axes):
    """The index window covered by the kernel for a given output `index`.

    Parameters:
    -----------
    index: numpy.ndarray of int
        Index around which the kernel window is requested.
    lvl: int
        Refinment level of `index`
    axes: RegularAxes
        Axes which correspond to the chart.
    """
    if len(index.shape) != 1:
        raise ValueError
    index = id_to_axisids(index, lvl, axes)
    myax = _get_axes_tuple(axes, lvl)[-1]
    slices = tuple(aa.get_batch_kernel_window(ii) for aa, ii in
                   zip(myax, index))
    return axisids_to_id(my_axes_outer(slices), lvl+1, axes)