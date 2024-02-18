# Copyright(C) 2023 Philipp Frank
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import numpy as np
from functools import reduce
from .index_utils import (id_to_axisids, axisids_to_id, my_setdiff_indices,
                          get_coarse_index)
from .axes import RegularAxis, HPAxis
from .chart import MSChart


def regular_grid_1D(base = 2, kernel_sizes = 3,
                    interpolation_method_in = 'linear',
                    interpolation_method_out = 'linear',
                    min_size = None, size0 = None,
                    binsize = None, binsize0 = None,
                    dtype = np.int32, boundary_condition = 'periodic',
                    nrefine = None, charted_trafos = None):
    if (min_size is None) and (size0 is None):
        raise ValueError("Either `min_size` or `size0` have to be set!")
    if (binsize is None) and (binsize0 is None):
        raise ValueError("Either `binsize` or `binsize0` have to be set!")
    if size0 is not None:
        if binsize0 is None:
            msg = "If `size0` is set, `binsize0` also has to be set!"
            raise ValueError(msg)
    if isinstance(base, tuple):
        if nrefine is not None:
            if len(base) != nrefine + 1:
                raise ValueError
        else:
            nrefine = len(base) - 1
    else:
        if nrefine is None:
            raise ValueError
        base = (base, ) * (nrefine + 1)

    if size0 is None:
        nref = []
        sz = min_size
        for bb in base[:-1][::-1]:
            sz = max(1, int(np.ceil(sz / bb)))
            nref.append(sz)
        nref = nref[::-1]
        size0 = nref[0]
    elif min_size is not None:
        print("`size0` is set, ignoring `min_size`...")
        min_size = None
    if binsize0 is None:
        fct = reduce(lambda a,b: a*b, base[:-1])
        binsize0 = binsize * fct
    elif binsize is not None:
        print("`binsize0` is set, ignoring `binsize`...")
        binsize = None

    if isinstance(kernel_sizes, tuple):
        if len(kernel_sizes) == nrefine:
            kernel_sizes = (size0 + (1-size0%2), ) + kernel_sizes
        elif len(kernel_sizes) != nrefine + 1:
            raise ValueError
    else:
        kernel_sizes = (size0 + (1-size0%2), ) + (kernel_sizes, ) * nrefine

    if isinstance(interpolation_method_in, tuple):
        if len(interpolation_method_in) == nrefine:
            interpolation_method_in = interpolation_method_in + ('nearest', )
        elif len(interpolation_method_in) != nrefine + 1:
            raise ValueError
    else:
        interpolation_method_in = (interpolation_method_in,) * nrefine
        interpolation_method_in = interpolation_method_in + ('nearest',)

    if isinstance(interpolation_method_out, tuple):
        if len(interpolation_method_out) != nrefine + 1:
            raise ValueError
    else:
        interpolation_method_out = (interpolation_method_out,) * (nrefine + 1)

    inds = np.arange(size0, dtype=dtype)
    if boundary_condition == 'periodic':
        mysize0 = size0
    elif boundary_condition == 'open':
        pd = 0
        for i in range(nrefine):
            extra = max(1, (kernel_sizes[nrefine-i]-1) // 2)
            pd = int(np.ceil((pd+extra) / base[nrefine-i-1]))
        pd += max(1, (kernel_sizes[0] - 1) // 2)
        mysize0 = size0 + 2*pd
        inds += pd
    else:
        raise ValueError(f"Unknown boundary condition: {boundary_condition}")

    inds = [inds, ]
    ax = RegularAxis(base[0], mysize0, binsize0, kernel_sizes[0], None,
                     interpolation_method_in[0], interpolation_method_out[0])
    chart = MSChart((np.arange(mysize0, dtype=dtype),), (ax,),
                    charted_trafos=charted_trafos)
    for i in range(1, nrefine + 1):
        ax = chart.axes(i-1)[0].copy().refine_axis(base[i], kernel_sizes[i],
                                                   interpolation_method_in[i],
                                                   interpolation_method_out[i])
        chart = chart.refine(inds, (ax,))
        if i != nrefine:
            refinds = chart.indices[-1]
            if min_size is not None:
                start = refinds[refinds.size//2] - nref[i]//2
                refinds = np.arange(nref[i], dtype=refinds.dtype) + start
            inds = [np.array([], dtype=dtype)] * i + [refinds,]
    return chart

def outer_product_chart(charts, charted_trafos = None):
    charts = tuple(charts)
    maxlevel = charts[0].maxlevel
    for cc in charts:
        if cc.maxlevel != maxlevel:
            raise ValueError("Inconsistent number of levels!")
        if cc.ndim != 1:
            raise NotImplementedError

    newinds = []
    for lvl in range(maxlevel + 1):
        ids = tuple(id_to_axisids(cc.main_indices[lvl], lvl, cc._axes) for cc in
                    charts)
        sizes = tuple(ii.shape[1] for ii in ids)
        nids = []
        for i, ids in enumerate(ids):
            sz = sizes[:i] + sizes[(i+1):]
            ids = np.multiply.outer(ids, np.ones(sz, dtype=ids.dtype)
                                    ).astype(ids.dtype)
            nids.append(np.moveaxis(ids, 1, i+1))
        ids = np.concatenate(nids, axis=0)
        ids = ids.reshape((ids.shape[0], -1))
        bax = reduce(lambda a,b: a+b, (cc._axes for cc in charts))
        ids = axisids_to_id(ids, lvl, bax)
        newinds.append(np.unique(ids))
    for lvl in range(maxlevel):
        #ax = tuple(cc.axes(lvl)[0] for cc in charts)
        #fax = tuple(aa.fine_axis for aa in ax)
        bax = reduce(lambda a,b: a+b, (cc._axes for cc in charts))
        newinds[lvl] = my_setdiff_indices(newinds[lvl],
                                          get_coarse_index(newinds[lvl+1],
                                                           lvl+1, bax))
    axes = reduce(lambda a,b: a+b, tuple(cc.axes(0) for cc in charts))
    return MSChart(tuple(newinds), axes, charted_trafos=charted_trafos)

def _to_tuple(x, ndim):
    if not isinstance(x, tuple):
        return (x,) * ndim
    return x

def regular_grid(base = 2, kernel_sizes = 3,
                 interpolation_method_in = 'linear',
                 interpolation_method_out = 'linear',
                 min_size = None, size0 = None,
                 binsize = None, binsize0 = None,
                 dtype = np.int32, boundary_condition = 'periodic',
                 nrefine = None, ndim = None, charted_trafos = None):
    if isinstance(base, int):
        base = (base, ) * ndim
    else:
        ndim = len(base)

    kernel_sizes = _to_tuple(kernel_sizes, ndim)
    interpolation_method_in = _to_tuple(interpolation_method_in, ndim)
    interpolation_method_out = _to_tuple(interpolation_method_out, ndim)
    min_size = _to_tuple(min_size, ndim)
    size0 = _to_tuple(size0, ndim)
    binsize = _to_tuple(binsize, ndim)
    binsize0 = _to_tuple(binsize0, ndim)
    boundary_condition = _to_tuple(boundary_condition, ndim)

    axes = []
    for i in range(ndim):
        ax = regular_grid_1D(base[i], kernel_sizes[i],
                             interpolation_method_in[i],
                             interpolation_method_out[i],
                             min_size[i], size0[i], binsize[i], binsize0[i],
                             dtype, boundary_condition[i], nrefine)
        axes.append(ax)
    return outer_product_chart(tuple(axes), charted_trafos)

def full_hp_chart(nside0, knn = 1, interpolation_method_out = 'linear',
                  nrefine = None, dtype = np.int32,
                  charted_trafos = None):
    if isinstance(knn, tuple):
        if nrefine is not None:
            if len(knn) == nrefine:
                knn = (-1,) + knn
            elif len(knn) != nrefine:
                raise ValueError
        else:
            nrefine = len(knn) - 1
    else:
        if nrefine is None:
            raise ValueError
        knn = (-1,) + (knn,) * nrefine


    if isinstance(interpolation_method_out, tuple):
        if len(interpolation_method_out) != nrefine + 1:
            raise ValueError
    else:
        interpolation_method_out = (interpolation_method_out,) * (nrefine + 1)

    ax = HPAxis(nside0, knn[0], None, 'nearest', interpolation_method_out[0])
    fax = ax
    for i in range(1, nrefine+1):
        fax = fax.refine_axis(knn[i], 'nearest', interpolation_method_out[i])
    inds = (np.array([], dtype=dtype), ) * nrefine
    inds = inds + (np.arange(fax.size, dtype=dtype),)
    return MSChart(inds, (ax,), charted_trafos=charted_trafos)

def random_refine(chart, rfrac = 0.3):
    if rfrac > 1.:
        raise ValueError
    ii = np.random.choice(chart.indices[-1], int(rfrac*chart.indices[-1].size),
                          replace=False).astype(chart.indices[-1].dtype)
    ii.sort()
    ii = (np.array([], dtype=ii.dtype), ) * chart.maxlevel + (ii,)
    return chart.refine(ii)