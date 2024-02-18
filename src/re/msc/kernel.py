# Copyright(C) 2023 Philipp Frank
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial, reduce
import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax.lax import scan, dynamic_slice_in_dim
from ..num.unique import amend_unique_
from .chart import MSChart
from .axes import HPAxis
from .index_utils import (axisids_to_id, batch_table_to_table, get_table,
                          id_to_axisids, my_setdiff_indices, get_fine_indices)


def _get_stationary_kernel_indices(indices, level, baseaxes, stataxes):
    if level == 0:
        raise ValueError
    ids = id_to_axisids(indices, level-1, baseaxes)
    for i in stataxes:
        ids[i] = 0
    ids = axisids_to_id(ids, level-1, baseaxes)
    _, usel, sel = np.unique(ids, return_index=True, return_inverse=True)
    uids = indices[usel]
    table = {indices[i]:sel[i] for i in range(indices.size)}
    return uids, table


class MSKernel:
    def __init__(self, func, chart, stationary_axes = False,
                 scan_kernel = False, test_kernel = None, atol = None,
                 rtol = None, buffer_size = 10000, nbatch = 10):
        """Representation of a convolution kernel to be applied via `MSConvolve`

        Parameters:
        -----------
        func: python function
            Kernel function that yields the value of the kernel as a function of
            location of the center of the kernel and distance to the center.
            I.e. in case of a 2D space labeled by x and y, `func` takes 4
            arguments (x,y,dx,dy) where the first two args are the location of
            the center of the kernel in space, and the second two are the
            evaluation locations relative to the center.
        chart: MSChart
            Chart on which the convolution is performed.
        stationary_axes: bool or tuple of bool (default False)
            Whether the kernel along the axes of `chart` are assumed to be
            stationary. If `True` along an axis, the kernel is broadcasted
            along that axis instead of evaluated everywhere. Note that this
            is relative to the axes on the chart, i.E. if `chart` has nontrivial
            `charted_trafos` it refers to the internal axes of the chart and not
            the kernel coordinates.
        scan_kernel: bool (default False)
            Option to scan the kernel (or `test_kernel`) for kernels that are
            simmilar (within `atol` and `rtol`) upon instantiation. If `True`,
            only kernels that are significantly different get evaluated during
            runtime and broadcasted to all other kernels that are simmilar
            enough.
        test_kernel: function (optional)
            Option to provide different kernel to scan over then `func` for
            `scan_kernel` could be used e.g. to scan over the distance matrix
            in cases where the kernel is unkonwn but assumed to be stationary.
        atol: float (optional)
            Absolute tolerance for `scan_kernel`.
        rtol: float (optional)
            Relative tolerance for `scan_kernel`.
        buffer_size: ind (default 10000)
            Size of the buffer for `scan_kernel`.
        """
        #TODO check kernel sizes for consistency!
        if not isinstance(chart, MSChart):
            raise ValueError
        self._chart = chart
        self._func = func
        self._kernel_indices = [self._chart._main_indices[0],]
        for lvl in range(1, chart.maxlevel+1):
            dd = my_setdiff_indices(chart.main_indices[lvl-1],
                                    chart.indices[lvl-1])
            self._kernel_indices.append(dd)# get_fine_indices(dd, lvl-1,
                                           # chart._axes).flatten())
        self._kernel_indices = tuple(self._kernel_indices)
        self._kernel_tables = tuple(get_table(kk) for kk in
                                    self._kernel_indices)
        if isinstance(stationary_axes, bool):
            stationary_axes = (stationary_axes, ) * self._chart.ndim
        elif len(stationary_axes) != self._chart.ndim:
            raise ValueError
        self._stat_axes = tuple(np.arange(self._chart.ndim, dtype=int)
                                [list(stationary_axes)])
        kers, tables = [self._kernel_indices[0], ], [self._kernel_tables[0], ]
        for lvl in range(1, self._chart.maxlevel + 1):
            kk, tt = _get_stationary_kernel_indices(
                self._kernel_indices[lvl], lvl, self._chart._axes,
                self._stat_axes)
            print("Stationary:", lvl, kk.shape[0],
                    self._kernel_indices[lvl].shape[0])
            kers.append(kk)
            tables.append(tt)
        self._kernel_indices = tuple(kers)
        self._kernel_tables = tuple(tables)
        if scan_kernel:
            if atol is None:
                atol = 0.
            if rtol is None:
                rtol = 0.
            test_kernel = func if test_kernel is None else test_kernel
            self._kernel_indices,self._kernel_tables = (
                self._get_kernelids_by_tol(self._kernel_indices,
                                           self._kernel_tables,
                                           test_kernel, atol, rtol,
                                           buffer_size, nbatch))

    @property
    def chart(self):
        """MSChart on which the kernel is defined"""
        return self._chart

    @property
    def kernelfunction(self):
        """Kernel as a function of location and distance."""
        return self._func

    @property
    def indices(self):
        """Indices of bins on `chart` the kernel is evaluated.

        Notes:
        ------
        In case of some `stationary_axes` or `scan_kernel` being True, the
        kernel actually only gets evaluated at the locations corresponding to
        a subset of all indices existing on `chart`, i.E. `chart.main_indices`.
        """
        return self._kernel_indices

    @property
    def tables(self):
        """Lookup tables to translate `self.indices` to `chart.main_indices`. Is
        used in order to broadcast the kernel to the required indices during
        convolution.
        """
        return self._kernel_tables

    def batchsize(self, level):
        if level == 0:
            raise ValueError
        return reduce(lambda a,b:a*b,
                      (ax.base for ax in self.chart.axes(level-1)))

    def kernel_shape(self, level):
        return tuple(ax.kernel_size for ax in self.chart.axes(level))

    def update_kernelfunction(self, func):
        self._func = func

    def _get_kernelids_by_tol(self, kernelids, kerneltables, testfunc, atol,
                              rtol, buffer_size, nbatch):
        kernel_ids, kernel_tables = [kernelids[0], ], [kerneltables[0],]
        prev_kernel = self._get_kernel(0, kernelids[0], None, None, testfunc)[0]
        prev_table = kerneltables[0]
        for lvl in range(1, self._chart.maxlevel + 1):
            allinds = kernelids[lvl]
            bs = int(np.ceil(allinds.shape[0]/nbatch))
            if bs == 1:
                bs = allinds.shape[0]
                mybatch = 1
            else:
                mybatch = nbatch
            # Successively amend the duplicate-free distance/covariance matrices
            u, inv = None, None
            sc_amend_uq = partial(amend_unique_, axis=0, atol=atol, rtol=rtol)
            getker = lambda id: self._get_kernel(lvl, id, prev_kernel,
                                                 prev_table, testfunc)
            for i in range(mybatch):
                myinds = allinds[i*bs:(i+1)*bs]
                if myinds.shape[0] == 0:
                    break
                dkers = getker(myinds)[1]
                if u is None:
                    u = jnp.full((buffer_size, ) + dkers.shape[1:], jnp.nan,
                                 dtype=dkers.dtype)

                u, idx = scan(sc_amend_uq, u, dkers)
                if inv is None:
                    inv = idx
                else:
                    inv = jnp.concatenate((inv, idx))
            # Cut away the placeholder for preserving static shapes
            n = np.unique(inv).size
            if n >= u.shape[0] or not np.all(np.isnan(u[n:])):
                raise ValueError("`mat_buffer_size` too small")
            inv = np.array(inv)
            assert inv.size == allinds.size
            usort = np.unique(inv, return_index=True)[1]
            uinds = allinds[usort]
            print("Scanned:", lvl, uinds.size, allinds.size)
            table = {allinds[i]:inv[i] for i in range(allinds.size)}
            table = {kk:table[kernelids[lvl][ii]] for kk,ii
                     in kerneltables[lvl].items()}
            kernel_ids.append(uinds)
            kernel_tables.append(table)
            prev_kernel = getker(uinds)[0]
            prev_table = table
        return tuple(kernel_ids), tuple(kernel_tables)

    def evaluate_kernel_function(self, level, indices, func = None):
        """Evaluate a kernel function on the locations defined by `indices` on
        `level`.

        Parameters:
        -----------
        level: int
            Refinment level on which the evaluation is performed.
        indices: numpy.ndarray
            Indices of the bins at which the kernel should be evaluated.
        func: function (optional)
            Alternative `self.kernelfunction` what should be evaluated instead.
            Must provide compatible input/output (see `func` in the constructor
            for further information).
        """
        func = self._func if func is None else func
        axes = self._chart.axes(level)
        ndim = self._chart.ndim
        indices = id_to_axisids(indices, level, self._chart._axes)
        locdists = (ax.get_coords_and_distances(ii) for ax, ii in
                    zip(axes,indices))
        locdists = reduce(lambda a,b:a+b, locdists)
        def evaluate(locdists):
            locs = []
            dists = []
            for i, (ll,dd) in enumerate(zip(locdists[::2], locdists[1::2])):
                dims = np.arange(ndim)
                partdims = np.delete(dims, i)
                ll = jnp.expand_dims(ll, tuple(1 + partdims) +
                                     tuple(1 + ndim + dims))
                dd = jnp.expand_dims(dd, tuple(1 + dims) +
                                     tuple(1 + ndim + partdims))
                locs.append(ll)
                dists.append(dd)

            if self._chart._trafos is not None:
                locp = self._chart._chart_to_kernel(*(ll+dd for ll,dd in
                                               zip(locs, dists)))
                locs = self._chart._chart_to_kernel(*locs)
                dists = tuple(lp-l for lp,l in zip(locp, locs))
            kershp = tuple(ks + (ax.in_size - 1) for ks, ax
                           in zip(self.kernel_shape(level),
                                  self.chart.axes(level)))
            ker = func(*(locs+dists))
            ker = ker.reshape(kershp)
            for i, ax in enumerate(axes):
                if ax.in_size == 2:
                    if isinstance(ax, HPAxis):
                        # TODO
                        raise NotImplementedError
                    mat = jnp.array([[0.5,0.5],[-1./ax.binsize, 1./ax.binsize]])
                    def eval(kernel, ind):
                        kk = dynamic_slice_in_dim(kernel, ind, 2, axis=i)
                        return jnp.tensordot(kk, mat, axes=((i,),(1,)))
                    ker = vmap(eval, (None, 0), i)(ker,
                                                   np.arange(ax.kernel_size))
                else:
                    ker =  ker[..., jnp.newaxis]
            return ker
        return vmap(evaluate, ((1,)*(2*ndim),), 0)(locdists)

    def interpolate_kernel(self, input, level, refine_inds, coarse_tbl):
        """Interpolate kernel values to the next level.

        Parameters:
        -----------
        input: jax.DeviceArray
            Input values on `level`.
        level: int
            On which coarse graining level the input is defined.
        fine_inds: numpy.ndarray
            The requested indices on the next level the kernel should be
            interpolated to.
        coarse_tbl: dict
            Lookup table on `level` that maps input kernels to all indices that
            exist on `level`.
        Returns:
        --------
        jax.DeviceArray
            Interpolated kernels at `fine_inds`.
        """
        axs = self._chart.axes(level)
        ndim = self._chart.ndim
        # Get the kernels and weigths required for interpolation along the
        # output axes
        select, weights, wselect = self.chart._batch_interpolation_selection(
            level, refine_inds, coarse_tbl
        )
        weights = tuple(jnp.array(ww) for ww in weights)

        trafos = tuple(ax.batch_window_to_window() for ax in axs)
        axids = id_to_axisids(refine_inds, level, self.chart._axes)
        fine_select = tuple(ax.get_batch_fine_kernel_selection(ii) for ax,ii in
                            zip(axs, axids))

        def batch_int(input, select, cmat, trafo, weight):
            res = jnp.tensordot(trafo, input, axes=((1,),(0,)))
            res = jnp.tensordot(res, cmat, axes=((2*ndim,),(0,)))
            res = jnp.moveaxis(res, -2, 2*ndim)
            res = jnp.moveaxis(res, -1, ndim+1)
            res = res.reshape(res.shape[:ndim] +
                              (res.shape[ndim]*res.shape[ndim+1],) +
                              res.shape[(ndim+2):])
            res = res[(select[0],) + (slice(None),)*(ndim-1) + (select[1],)]
            res = jnp.moveaxis(res, 1, ndim)
            res = jnp.tensordot(trafo.T, res, axes=((1,),(0,)))
            res = jnp.tensordot(weight, res, axes=((0,),(0,)))
            return res
        vint = vmap(batch_int, in_axes=(None, (0,0), None, 0, 0), out_axes=-1)

        def my_mul(select, ker_select, fine_select):
            res = input[select]
            for ww, ax, sel, tt in zip(weights, axs, fine_select, trafos):
                cmat = ax.coarse_grain()
                ww = ww[ker_select] if ww.shape[0] != 1 else ww[0]
                res = vint(res, sel, cmat, tt, ww)
            return jnp.moveaxis(res.reshape(res.shape[:2*ndim]+(-1,)), -1, 0)

        my_mul = vmap(my_mul, (0, 0, ((0,0),)*ndim), 0)
        return my_mul(select, wselect, fine_select)

    def to_batch_kernel(self, kernel, level):
        kshp = self.kernel_shape(level)
        ksize = reduce(lambda a,b:a*b, kshp)
        lsize = reduce(lambda a,b:a*b, (ax.in_size for ax in
                                        self.chart.axes(level)))
        if level == 0:
            return kernel.reshape((kernel.shape[0],1,ksize,lsize))
        bases = tuple(ax.base for ax in self.chart.axes(level-1))
        fct = reduce(lambda a,b:a*b, bases)
        kernel = kernel.reshape((kernel.shape[0],)+bases+kshp+(lsize,))
        axids = id_to_axisids(self._kernel_indices[level], level-1,
                              self.chart._axes)
        trafos = tuple(ax.kernel_to_batchkernel(id) for ax, id in
                       zip(self.chart.axes(level-1), axids))
        mapax = tuple(None if tt.shape[0]==1 else 0 for tt in trafos)
        trafos = tuple(tt[0] if mm is None else tt for tt,mm in
                       zip(trafos, mapax))
        axs = tuple(1+2*i for i in range(self.chart.ndim))
        axs += tuple(2+2*i for i in range(self.chart.ndim))
        axs += (0,)
        def trafo(inp, trafos):
            for i,tt in enumerate(trafos):
                inp = jnp.tensordot(inp, tt, axes=((0,self.chart.ndim-i),(2,3)))
            return inp.transpose(axs)
        kernel = vmap(trafo, in_axes=(0, mapax), out_axes=0)(kernel, trafos)
        ksize = reduce(lambda a,b: a*b,
                       kernel.shape[1+self.chart.ndim:1+2*self.chart.ndim])
        return kernel.reshape((kernel.shape[0],fct,ksize,lsize))

    def integral_kernels(self, func = None):
        """Get the list of kernels in the form required for `MSConvolve` by
        evaluating `self.func` (or func if func is not None).
        """
        func = self._func if func is None else func
        prev_kernel, prev_table = None, None
        kernels = []
        for lvl in range(self._chart.maxlevel + 1):
            inds = self._kernel_indices[lvl]
            prev_kernel, dker = self._get_kernel(lvl, inds, prev_kernel,
                                                 prev_table, func=func)
            prev_table = self._kernel_tables[lvl]
            kernels.append(self.to_batch_kernel(dker, lvl))
        return kernels

    def _good_kernel_window(self, level, indices):
        indices = id_to_axisids(indices, level, self._chart._axes)
        flagging = tuple(ax.flagged_kernel_window(ii) for ax,ii in
                         zip(self._chart.axes(level), indices))
        res = np.zeros(indices.shape[1:]+tuple(ff.shape[1] for ff in flagging),
                       dtype=bool)
        for i,ff in enumerate(flagging):
            res = np.moveaxis(res, 0, -1)
            res = np.moveaxis(res, i, -1)
            res += ff
            res = np.moveaxis(res, -1, i)
            res = np.moveaxis(res, -1, 0)
        return (~res)

    def _get_kernel(self, level, indices, prev_kernel = None,
                    prev_table = None, func = None):
        if level == 0 and prev_kernel is not None:
            raise ValueError
        if level > 0:
            evalinds = get_fine_indices(indices, level-1, self.chart._axes)
            shp = evalinds.shape
            shp = (evalinds.shape[0],
                   reduce(lambda a,b:a*b, evalinds.shape[1:]))
            evalinds = evalinds.flatten()
        else:
            evalinds = indices
            shp = evalinds.shape
        ker = self.evaluate_kernel_function(level, evalinds, func=func)
        ker = ker.reshape(shp + ker.shape[1:])
        if prev_kernel is not None:
            if prev_table is None:
                raise ValueError
            if level > 1:
                prev_table = batch_table_to_table(prev_table, level-1,
                                                  self.chart._axes)
                sh = (prev_kernel.shape[0]*prev_kernel.shape[1],)
                sh += prev_kernel.shape[2:]
                prev_kernel = prev_kernel.reshape(sh)
            dker = ker - self.interpolate_kernel(prev_kernel, level - 1,
                                                 indices, prev_table)
        else:
            dker = ker
        window = ~self._good_kernel_window(level, evalinds)
        window = window.reshape(shp + window.shape[1:])
        dker = dker.at[window].set(0.)
        return ker, dker