# Copyright(C) 2023 Philipp Frank
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
import jax.numpy as jnp
import numpy as np
from jax import random
from .chart import MSChart
from .index_utils import get_selection, get_table, my_setdiff_indices
from .convolve import prepare_input
from .utils import sorted_concat

def _get_binvals(values, actives, selection, shp):
    res = jnp.full(shp, jnp.nan).flatten()
    for lvl in range(len(actives)):
        if actives[lvl].size > 0:
            res = res.at[actives[lvl]].set(values[lvl][selection[lvl]])
    return res.reshape(shp)

# TODO also implement bilinear interpolation
def get_binvals(locs, chart, on_chart = False, want_bins = False):
    if not isinstance(chart, MSChart):
        raise ValueError
    shp = locs[0].shape[1:]
    locs = tuple(ll.reshape((ll.shape[0],-1)) for ll in locs)
    bins = np.full(shp, np.nan).flatten()
    binlevel = np.full(shp, np.nan, dtype=int).flatten()
    actives = []
    selection = []
    for lvl in range(chart.maxlevel+1):
        bin = chart.binid_from_coord(locs, lvl, on_chart = on_chart)
        active = np.array(tuple(np.isin(bb, chart.indices[lvl]) for bb in bin))
        actives.append(active)
        if active.size > 0:
            bin = bin[active]
            selection.append(get_selection(get_table(chart.indices[lvl]), bin))
            bins[active] = bin
            binlevel[active] = lvl
        else:
            selection.append(None)
    assert np.all(bins != np.nan)
    actives = tuple(actives)
    selection = tuple(selection)
    f = partial(_get_binvals, actives=actives, selection=selection, shp=shp)
    if want_bins:
        return f, bins.reshape(shp), binlevel.reshape(shp)
    return f

def refine_integrand(arrays, indices, chart, want_chart = True, key = None, 
           fine_axes = None, volume_scaling = 0.5):
    if not isinstance(chart, MSChart):
        raise ValueError
    if indices[-1].size != 0:
        arrays = arrays + [jnp.array([]),]

    new_chart, chart, pairs = chart.refine(indices, fine_axes=fine_axes,
                                           _want_ref_pairs=True)

    if key is not None:
        finds = tuple(pp[0] for pp in pairs)
        frnds = list(random.normal(kk, ff.shape) for kk, ff in 
                zip(random.split(key, len(pairs)), finds))
        frnds = prepare_input(frnds, chart, volume_scaling=volume_scaling,
                            indices=finds)

    result = prepare_input(arrays, chart, volume_scaling=volume_scaling)
    allinds = list(ii for ii in chart.indices)
    for lvl, pp in reversed(list(enumerate(pairs))):
        fine, ind = pp
        if fine.size > 0:
            table = get_table(chart.indices[lvl])
            newvals = chart._refine_input(result[lvl], fine, table, lvl)
            if key is not None:
                frnd = frnds[lvl]
                rnd, rndind = chart._coarse_grain(frnd, fine, lvl+1)
                rndtable = get_table(rndind)
                frnd -= chart._refine_input(rnd, fine, rndtable, lvl)
                newvals += frnd
            sel = get_selection(get_table(allinds[lvl]), ind)
            result[lvl] = jnp.delete(result[lvl], sel, axis=0)
            allinds[lvl] = np.delete(allinds[lvl], sel)
            result[lvl+1], allinds[lvl+1] = sorted_concat(
                result[lvl+1], allinds[lvl+1], newvals, fine)
    result = [rr[(slice(None),) + (0,)*new_chart.ndim].ravel() for rr in result]
    vol = (new_chart.volume(lvl, new_chart.indices[lvl])**volume_scaling for
           lvl in range(new_chart.maxlevel + 1))
    result = list(rr / vv for rr, vv in zip(result, vol))
    if want_chart:
        return result, chart
    return result

def refine_output(indices, values, chart, want_chart = True, fine_axes = None):
    if not isinstance(chart, MSChart):
        raise ValueError
    newchart = chart.refine(indices, fine_axes)
    new_vals = list(jnp.copy(vv) for vv in values)
    new_ids = list(np.copy(ii) for ii in chart.indices)
    if newchart.maxlevel > chart.maxlevel:
        new_vals.append(jnp.array([],dtype=new_vals[-1].dtype))
        new_ids.append(np.array([], dtype=new_ids[-1].dtype))
    for lvl in range(newchart.maxlevel + 1):
        if lvl != 0:
            if (chart.maxlevel == newchart.maxlevel) or (lvl != newchart.maxlevel):
                vv = my_setdiff_indices(newchart.indices[lvl], chart.indices[lvl])
            else:
                vv = newchart.indices[lvl]
            ll = newchart.get_coordinates(vv, lvl)
            newvals, _, binlvl = get_binvals(ll, values, chart, 
                                                want_bins=True)
            assert np.all(binlvl == lvl - 1)
            new_vals[lvl], new_ids[lvl] = sorted_concat(
                new_vals[lvl], new_ids[lvl], newvals, vv)

        if (chart.maxlevel == newchart.maxlevel) or (lvl != newchart.maxlevel):
            vv = my_setdiff_indices(chart.indices[lvl], newchart.indices[lvl])
            vv = get_selection(get_table(new_ids[lvl]), vv)
            new_ids[lvl] = np.delete(new_ids[lvl], vv)
            new_vals[lvl] = jnp.delete(new_vals[lvl], vv)
    for nn, cc in zip(new_ids, newchart.indices):
        assert np.all(nn == cc)
    if want_chart:
        return new_vals, newchart
    return new_vals

def fill_grid(values, chart, is_integrand = False, want_chart = True,
              key = None, volume_scaling = 0.5):
    for lvl in range(chart.maxlevel):
        if chart.indices[lvl].size > 0:
            if is_integrand:
                values, chart = refine_integrand(values, chart.indices[lvl], 
                                                chart, want_chart = True, 
                                                key = key, 
                                                volume_scaling = volume_scaling)
            else:
                values, chart = refine_output(chart.indices[lvl], values, chart, 
                                            want_chart = True)
    if want_chart:
        return values, chart
    return values