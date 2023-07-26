# Copyright(C) 2023 Philipp Frank
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial, reduce
import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax.lax import scan, dynamic_slice_in_dim
from .utils import amend_unique_
from .chart import MSChart
from .index_utils import axisids_to_id, get_table, id_to_axisids

def refine_kernel(axis, kernel, fine_selection, ndim):
    """Refines the kernel along the integrand axis to the `fine_index` on
    the next level. To do so, the coarse graining operation that has been
    applied to the input is applied to the integrand axis (adjoint) and the
    input indices of the fine window are selected.
    """
    res = kernel
    cmat = axis.coarse_grain()
    res = jnp.tensordot(res, cmat, axes=((3*ndim,),(0,)))
    res = jnp.moveaxis(res, -2, 3*ndim)
    res = jnp.moveaxis(res, -1, ndim+1)
    res = res.reshape(res.shape[:ndim] +
                        (res.shape[ndim]*res.shape[ndim+1],) +
                        res.shape[(ndim+2):])
    res = res[(fine_selection[0],) + (slice(None),)*(ndim-1) + 
                (fine_selection[1],)]
    return jnp.moveaxis(res, 1, ndim)

def _to_kernel_weights(axis, kernel, kaxis):
    """Transforms from values of the kernel function to the weights of each bin
    that get applied to the input during convolution along one input axis.

    Parameters:
    -----------
    axis: RegularAxis
        Axis along which th weights get computed
    kernel: jax.numpy.ndarray
        Kernel values after evaluating the kernel function
    kaxis: int
        Axis of the array `kernel` that corresponds to the subspace defined by
        `axis`.

    Returns:
    --------
    jax.numpy.ndarray
        The kernel with the values along `kaxis` transformed to the weights
    """
    if axis.is_linear:
        mat = jnp.array([[0.5,0.5],[-1./axis.binsize, 1./axis.binsize]])
        def eval(ind):
            kk = dynamic_slice_in_dim(kernel, ind, 2, axis=kaxis)
            return jnp.tensordot(kk, mat, axes=((kaxis,),(1,)))
        return vmap(eval, 0, kaxis)(np.arange(axis.kernel_size))
    return kernel[..., jnp.newaxis]

def _get_stationary_kernel_indices(indices, level, baseaxes, stataxes, 
                                   caxes = None):
    ids = id_to_axisids(indices, level, baseaxes)
    if caxes is None:
        for ss in stataxes:
            ids[ss] = 0
    else:
        base = (cc.base for cc in caxes)
        shape = (cc.fine_axis.size for cc in caxes)
        fct = list(bb if i in stataxes else sh for i, (sh,bb) in
                    enumerate(zip(shape, base)))
        fct = np.array(fct, dtype=ids.dtype)[:, np.newaxis]
        ids = ids%fct
    ids = axisids_to_id(ids, level, baseaxes)
    uids, sel = np.unique(ids, return_inverse=True)
    table = {indices[i]:sel[i] for i in range(indices.size)}
    #table.update({uids[i]:i for i in range(uids.size)})
    return uids, table

class MSKernel:
    def __init__(self, func, chart, stationary_axes = False, scan_kernel = False, 
                 test_kernel = None, atol = None, rtol = None, 
                 buffer_size = 10000):
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
        if not isinstance(chart, MSChart):
            raise ValueError
        self._chart = chart
        self._func = func
        self._kernel_indices = self._chart._main_indices
        self._kernel_tables = tuple(get_table(kk) for kk in 
                                    self._kernel_indices)
        if isinstance(stationary_axes, bool):
            stationary_axes = (stationary_axes, ) * self._chart.ndim
        elif len(stationary_axes) != self._chart.ndim:
            raise ValueError
        self._stat_axes = tuple(np.arange(self._chart.ndim, dtype=int)
                                [list(stationary_axes)])
        kers, tables = [], []
        for lvl in range(self._chart.maxlevel + 1):
            #FIXME this may break for non-fully resolved grids!
            kk, tt = _get_stationary_kernel_indices(
                self._kernel_indices[lvl], lvl, self._chart._axes,
                self._stat_axes, self._chart.axes(lvl-1) if lvl != 0 else None)
            print("Stationary:", lvl, kk.shape[0], 
                    self._kernel_indices[lvl].shape[0])
            kers.append(kk)
            #tt = {ii : tt[self._kernel_indices[lvl]
            #        [self._kernel_tables[lvl][ii]]] for ii in 
            #         self._chart._main_indices[lvl]}
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
                                           buffer_size))

    @property
    def chart(self):
        """MSChart on which the kernel is defined"""
        return self._chart

    @property
    def kernelfunction(self):
        """Kernel as a function of location and distance."""
        return self._func

    def update_kernelfunction(self, func):
        self._func = func

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

#    def _kernelids_from_batch(self, kernelids, kerneltables, testfunc, atol, 
#                              rtol, batchsize = 10000):
#        kernel_ids, kernel_tables = [], []
#        prev_kernel, prev_table = None, None
#        for lvl in range(self._chart.maxlevel + 1):
#            allinds = kernelids[lvl]
#            ids = np.array([], dtype=allinds.dtype)
#            tbl = {}
#            ker, dker = None, None
#            i = 0
#            while i*batchsize < allinds.size:
#                print(f"Batch: {i} ...")
#                ii = allinds[(i*batchsize):(i+1)*batchsize]
#
#                kk, dk = self._get_kernel(lvl, ii, prev_kernel, prev_table,
#                                          testfunc)
#                tm1 = np.round(dk, atol)
#                tm2 = np.round(dk/np.max(np.abs(dk)), rtol)

    def _get_kernelids_by_tol(self, kernelids, kerneltables, testfunc, atol, 
                              rtol, buffer_size):
        kernel_ids, kernel_tables = [], []
        prev_kernel, prev_table = None, None
        for lvl in range(self._chart.maxlevel + 1):
            allinds = kernelids[lvl]
            # TODO scan over batches instead of all kernels
            kers, dkers = self._get_kernel(lvl, allinds, prev_kernel,
                                           prev_table, testfunc)

            # TODO Author Gordian Edenhofer
            # Successively amend the duplicate-free distance/covariance matrices
            u = jnp.full((buffer_size, ) + dkers.shape[1:], jnp.nan, 
                         dtype=dkers.dtype)

            sc_amend_uq = partial(amend_unique_, axis=0, atol=atol, rtol=rtol)
            u, inv = scan(sc_amend_uq, u, dkers)
            # Cut away the placeholder for preserving static shapes
            n = np.unique(inv).size
            if n >= u.shape[0] or not np.all(np.isnan(u[n:])):
                raise ValueError("`mat_buffer_size` too small")
            #u = u[:n]
            inv = np.array(inv)
            assert inv.size == allinds.size
            assert inv.size == allinds.size
            usort = np.unique(inv, return_index=True)[1]
            uinds = allinds[usort]
            print("Scanned:", lvl, uinds.size, allinds.size)
            kernel_ids.append(uinds)
            prev_kernel = kers[usort]
            table = {allinds[i]:inv[i] for i in range(allinds.size)}
            table = {ii:table[kernelids[lvl][kerneltables[lvl][ii]]] for ii
                     in self._chart._main_indices[lvl]}
            kernel_tables.append(table)
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
            ker = func(*(locs+dists)).squeeze()
            ker = jnp.expand_dims(ker, axis=np.arange(ndim)+ndim)
            for i, ax in enumerate(axes):
                ker = _to_kernel_weights(ax, ker, i)
            return ker
        return vmap(evaluate, ((1,)*(2*ndim),), 0)(locdists)

    def interpolate_kernel(self, input, level, fine_inds, coarse_tbl):
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

        select, weights, ker_select = self._chart._interpolation_selection(
                                                level, fine_inds, coarse_tbl)
        fine_id = id_to_axisids(fine_inds, level+1, self._chart._axes)
        fine_select = tuple(ax.get_fine_kernel_selection(idx) for ax, idx in 
                            zip(axs, fine_id))

        def my_mul(select, ker_selects, fine_select):
            res = input[select]
            for i, (ww, ss, ax, sel) in enumerate(zip(weights, ker_selects, axs, 
                                                      fine_select)):

                res = refine_kernel(ax, res, sel, ndim)
                res = jnp.tensordot(ww[ss], res, ((0,2), (0, 2*ndim-i)))
                res = jnp.moveaxis(res, 0, 2*ndim-i-1)
            return res
        my_mul = vmap(my_mul, (0, (0,)*ndim, ((0,0),)*ndim), 0)
        return my_mul(select, ker_select, fine_select)

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
            kernels.append(dker)
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
        ker = self.evaluate_kernel_function(level, indices, func=func)
        window = ~self._good_kernel_window(level, indices)
        if prev_kernel is not None:
            if prev_table is None:
                raise ValueError
            dker = ker - self.interpolate_kernel(prev_kernel, level - 1,
                                                 indices, prev_table)
            dker = dker.at[window].set(0.)
        else:
            dker = ker.at[window].set(0.)
        return ker, dker