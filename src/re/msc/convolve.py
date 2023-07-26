# Copyright(C) 2023 Philipp Frank
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import numpy as np
import jax.numpy as jnp
from .index_utils import get_selection, get_table
from .utils import sorted_concat


def prepare_input(arrays, chart, volume_scaling = 1., indices = None):
    """Shapes and scales the input for MS convolutions

    Parameters:
    -----------
    arrays: list of jax.DeviceArray
        Input arrays to the convolution.
    chart: MSChart
        Chart on which convolution is performed.
    volume_scaling: float (default = 0.5)
        For convolution, the integrand is scaled with the volume of the bin of
        each input. Depending on the use case, one might want to scale the input
        not with the volume linearly, but rather with a fraction of it (See e.g.
        MSGp).
    indices: tuple of numpy.ndarray (optional)
        Corresponding indices of the bins the input is defined on. If `None` the
        input is assumed to be defined on all `indices` on `chart`.

    Returns:
    --------
    list of jax.DeviceArray
        Scaled and reshaped version of the input to match the required input
        of `MSConvolve`
    """
    if len(arrays) != chart.maxlevel + 1:
        raise ValueError
    res = []
    for lvl, aa in enumerate(arrays):
        inds = indices[lvl] if indices is not None else chart.indices[lvl]
        if aa.size != inds.size:
            raise ValueError
        rr = aa * chart.volume(inds, lvl)**volume_scaling
        for ax in chart.axes(lvl):
            if ax.is_linear:
                rr = jnp.stack((rr, jnp.zeros_like(rr)), axis=-1)
            else:
                rr = rr[..., jnp.newaxis]
        res.append(rr)
    return res

def charted_convolve(arrays, kernel, chart):
    if len(arrays) != chart.maxlevel + 1:
        raise ValueError
    result = [None,] * (chart.maxlevel + 1)
    resultids = [None,] * (chart.maxlevel + 1)
    for lvl in range(chart.maxlevel + 1)[::-1]:
        if lvl != chart.maxlevel:
            # Coarse grain
            res, resid = chart.coarse_grain(arrays[lvl+1], lvl+1)
            arrays[lvl], ids = sorted_concat(arrays[lvl], chart.indices[lvl],
                                             res, resid)
            assert np.all(ids == chart.main_indices[lvl])
        res = arrays[lvl]
        resid = chart.main_indices[lvl]
        if lvl != 0:
            # Get missing values for kernel windows from coarse layer.
            refined = chart.refine_input(arrays[lvl-1], lvl-1)
            if refined is not None:
                res, resid = sorted_concat(res, resid, refined[0], refined[1])
        result[lvl] = res
        resultids[lvl] = resid

    myres = None
    oldker = None
    for lvl in range(chart.maxlevel + 1):

        ker = kernel.evaluate_kernel_function(lvl, chart.main_indices[lvl])
        myres = chart.interpolate_convolve(result[lvl], resultids[lvl], ker, 
                                           lvl, myres, oldker)
        select = get_selection(get_table(chart.main_indices[lvl]), 
                               chart.indices[lvl])
        result[lvl] = myres[select].squeeze()
        oldker = ker
    return result

def charted_convolve_precompute(arrays, kernels, kerneltables, chart):
    if len(arrays) != chart.maxlevel + 1:
        raise ValueError
    if len(kernels) != chart.maxlevel + 1:
        raise ValueError
    result = []
    for lvl in range(chart.maxlevel + 1)[::-1]:
        if lvl != chart.maxlevel:
            # Coarse grain
            res, resid = chart.coarse_grain(arrays[lvl+1], lvl+1)
            if arrays[lvl].size > 0:
                res, resid = sorted_concat(arrays[lvl], chart.indices[lvl], res, 
                                           resid)
            assert np.all(resid == chart.main_indices[lvl])
            arrays[lvl] = res
        else:
            res, resid = arrays[lvl], chart.main_indices[lvl]
        if lvl != 0:
            # Get missing values for kernel windows from coarse layer.
            refined = chart.refine_input(arrays[lvl-1], lvl-1)
            if refined is not None:
                res, resid = sorted_concat(res, resid, refined[0], refined[1])
        # Convolve with kernel
        result.append(chart.convolve(res, resid, kernels[lvl], 
                                     kerneltables[lvl], lvl))
    result.reverse()
    # Add up and interpolate results
    for lvl in range(chart.maxlevel + 1):
        if lvl != chart.maxlevel:
            result[lvl+1] += chart.interpolate(result[lvl], lvl)
        select = get_selection(get_table(chart.main_indices[lvl]), 
                               chart.indices[lvl])
        result[lvl] = result[lvl][select].squeeze()
    return result