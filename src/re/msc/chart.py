# Copyright(C) 2023 Philipp Frank
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import numpy as np
import jax.numpy as jnp
from functools import reduce
from jax import vmap
from .index_utils import (get_selection, get_table, my_setdiff_indices,
                          get_kernel_window, get_coarse_index, my_axes_outer,
                          get_fine_indices, id_to_axisids, axisids_to_id,
                          get_inter_window, _get_axes_tuple, 
                          get_batch_kernel_window)
from .utils import axes_matmul


def _check_indices(indices, baseaxes):
    axes = _get_axes_tuple(baseaxes, len(indices) - 1)
    maxlevel = len(indices) - 1
    inds = indices[-1]
    for i, (ind, cax) in enumerate(zip(indices[:-1][::-1], axes[:-1][::-1])):
        nref = reduce(lambda a,b: a*b, (aa.base for aa in cax))
        coarse = get_coarse_index(inds, maxlevel - i, baseaxes)
        coarse, counts = np.unique(coarse, return_counts=True)
        if np.sum((counts%nref != 0)):
            msg = f"Indices not fully refined on level: {maxlevel-i-1}"
            raise ValueError(msg)
        missing = get_inter_window(inds, maxlevel - i, baseaxes)
        missing = my_setdiff_indices(missing, inds)
        missing = get_coarse_index(missing, maxlevel - i, baseaxes)
        missing = np.unique(missing)
        if my_setdiff_indices(missing, ind).size != 0:
            msg = f"Indices not in coarse on level: {maxlevel-i}"
            raise ValueError(msg)
        if my_setdiff_indices(coarse, ind).size != coarse.size:
            msg = f"Indices already set in coarse on level: {maxlevel-i}"
            raise ValueError(msg)
        inds = np.concatenate((ind, coarse))
    if inds.size != reduce(lambda a,b: a*b, (aa.size for aa in axes[0])):
        raise ValueError("Base grid not fully filled!")


class MSChart:
    def __init__(self, indices, axes, charted_trafos = None):
        """MultiScale chart for handling convolutions on sparsely resolved
        multigrids. The underlying abstract multigrid is represented as 
        levels i >=0, where 0 corresponds to the most coarse grid and higher
        levels indicate more refined grids.

        Parameters:
        -----------
        indices: tuple of numpy.ndarray
            Indices on each level of the multigrid that exist in this chart.
            All levels may have existing indices, however they are subject to
            several constraints (i - iii)
            i  : If an index exists on level i, all fine voxels that would 
                 correspond to refining the voxel of this index must not exist on
                 levels i+j (j>0). 
            ii : Furthermore, combining all voxels of all levels must correspond
                 to a full coverage of the entire space (but only a single 
                 coverage due to constraint i).
            iii: If a voxel exists on level i, but some of the voxels in the
                 neighbourhood do not exist on i, the coarse voxels (that would
                 contain the neighbours) must exist on level i-1. The relevant
                 neighbourhood is defined via the size kernel windows given by
                 the `axes`.
        axes: tuple of RegularAxis
            Axes of the chart on level 0. The axes on higher refinment levels
            are provided by axes[i].fine_axis for each axis i. All entries of
            axes must be of same depth, i.E. if axes[i].fine_axis.fine_axis is
            not none for i = 0, it must also not be None for all other entries
            of axes, and so on.
        charted_trafos: iterable of functions of len(3) (optional)
            Optional additional transformations of the charts coordinates to
            another coordinate system where the kernel is defined. Must contain
            three functions:
            chart_to_kernel: The transformation of the coordinates on the chart
                             to the coordinate system of the kernel.
            kernel_to_chart: Inverse of `chart_to_kernel`
            metric_sqrt_det: Determinant of the metric square root (the Jacobian
                             of the transformation in case of flat space). This
                             is used to weight the integrand input according to
                             the change of volume arising from a transformation.
        """
        self._axes = tuple(axes)
        self._ndim = len(self._axes)
        maxlevel = None
        for aa in axes:
            ll = 0
            while aa.fine_axis is not None:
                aa = aa.fine_axis
                ll += 1
            if maxlevel is not None:
                if ll != maxlevel:
                    raise ValueError("All axes must have same depth!")
            else:
                maxlevel = ll

        self._indices = tuple(indices)
        if len(indices) != maxlevel + 1:
            raise ValueError("Inconsistent number of layers in indices!")
        self._maxlevel = maxlevel
        _check_indices(self._indices, self._axes)
        self._main_indices = self._get_main_indices()
        # Validate ordering between levels
        for lvl in range(self.maxlevel):
            diff = my_setdiff_indices(self._main_indices[lvl], 
                                      self._indices[lvl])
            fine = get_fine_indices(diff, lvl, self._axes)
            assert np.all(fine.flatten() == self._main_indices[lvl+1])
        self._trafos = charted_trafos
        if self._trafos is not None:
            self._chart_to_kernel = self._trafos[0]
            self._kernel_to_chart = self._trafos[1]
            self._metric_sqrt_det = self._trafos[2]
        else:
            self._chart_to_kernel = None
            self._kernel_to_chart = None
            self._metric_sqrt_det = None

    @property
    def maxlevel(self):
        """Maximum refinment level of the chart."""
        return self._maxlevel

    @property
    def ndim(self):
        """Number of axes in the chart"""
        return self._ndim

    @property
    def nspacedims(self):
        """Number of spatial dimensions represented by the chart"""
        return reduce(lambda a,b: a+b, (ax.axdim for ax in self._axes))

    @property
    def indices(self):
        """Pixel indices of existing values on each level."""
        return self._indices

    @property
    def main_indices(self):
        """All eisting values on a level, including the ones one gets from
        coarse graining."""
        return self._main_indices

    def shape(self, level):
        """Theoretical shape of the chart on `level`.

        Notes:
        ------
            As the chart is sparse in general, the shape of arrays defined on
            the chart is not equal to `shape`. See `n_pixels` for this.
        """
        level = level%(self.maxlevel+1)
        return tuple(aa.size for aa in self.axes(level))

    def n_pixels(self, level):
        """Number of pixels that exist on the chart on `level`."""
        level = level%(self.maxlevel+1)
        return self._indices[level].size

    def axes(self, level):
        """Axes of the chart on each `level`."""
        level = level%(self.maxlevel+1)
        return _get_axes_tuple(self._axes, level)[-1]

    def add_trafos(self, charted_trafos):
        """Option to add coordinate transformations to the chart if the are not
        set yet.

        Parameters:
        -----------
        charted_trafos: iterable of functions of len(3)
            See constructors `charted_trafos` for the definition.
        """
        if self._trafos is not None:
            raise ValueError("Coordinate trafos already set for this chart!")
        self._trafos = charted_trafos
        if self._trafos is not None:
            self._chart_to_kernel = self._trafos[0]
            self._kernel_to_chart = self._trafos[1]
            self._metric_sqrt_det = self._trafos[2]

    def _get_main_indices(self):
        """All indices on each level including the ones that are obtained when
        coarse graining the finer levels."""
        inds = np.copy(self.indices[-1])
        main_inds = [inds,]
        for lvl in range(self._maxlevel)[::-1]:
            inds = get_coarse_index(inds, lvl+1, self._axes)
            inds = np.unique(inds)
            inds = np.append(self.indices[lvl], inds)
            inds.sort()
            main_inds.append(inds)
        return tuple(main_inds[::-1])

    def coordinates(self, index, level, on_chart = False):
        """Coordinates corresponding to the bincenter of the pixel indexed by
        `index` along each axis.

        Parameters:
        -----------
        index: numpy.ndarray of int
            Index of the requested pixel coordinate.
        level: int
            Refinment level of the requested pixel.
        on_chart: bool (default False)
            Whether the coordinate should be in the coordinate frame where the
            chart is defined or where kernels are defined.
        Returns:
        --------
        tuple of numpy.ndarray of float
            The coordinates of the bincenter along each axis of the chart.
        """
        locs = id_to_axisids(index, level, self._axes)
        locs = tuple(ax.coordinate(ll) for ax,ll in zip(self.axes(level), locs))
        if (self._trafos is None) or on_chart:
            return locs
        return self._chart_to_kernel(*locs)

    def binid_from_coord(self, coordinate, level, on_chart = False):
        """Pseudoinverse operation of `coordinate`. Yields the index of the 
        bin `coordinates` falls into.

        Parameters:
        -----------
        coordinate: tuple of numpy.ndarray of float
            The coordinates along each axis of the chart.
        level: int
            Refinment level of the requested pixel.
        on_chart: bool (default False)
            Whether the coordinates are in the coordinate frame where the
            chart is defined or where kernels are defined.
        Returns:
        --------
        numpy.ndarray of int
            Index of the bin coordinates falls into.
        """
        #TODO same keyword
        if (not on_chart) and (self._trafos is not None):
            coordinate = self._kernel_to_chart(*coordinate)
            if not isinstance(coordinate, tuple):
                coordinate = (coordinate, )
            coordinate = tuple(np.array(cc) for cc in coordinate)
        ids = tuple(ax.binid_from_coord(ll) for ax,ll in
                    zip(self.axes(level), coordinate))
        ids = np.stack(ids, axis = 0)
        return axisids_to_id(ids, level, self._axes)

    def volume(self, index, level):
        """Volume element of the bin of `index`.

        Parameters:
        -----------
        index: numpy.ndarray of int
            Index of bin the volume is requested for.
        level: int
            Refinment level of the bin.
        Returns:
        --------
        numpy.ndarray of float
            The volume of the bin of `index`.
        """
        #TODO jacdet
        level = level%(self.maxlevel + 1)
        vol = reduce(lambda a,b: a*b, (ax.binsize for ax in self.axes(level)))
        if self._trafos is None:
            return vol*np.ones_like(index)
        locs = self.coordinates(index, level, on_chart=True)
        return self._metric_sqrt_det(*locs)*vol

    def _coarse_grain(self, input, indices, level):
        caxs = self.axes(level-1)
        coarse = np.unique(get_coarse_index(indices, level, self._axes))
        window = id_to_axisids(coarse, level - 1, self._axes)
        window = tuple(cax.get_fine_indices(cc) for cax,cc in zip(caxs, window))
        window = axisids_to_id(my_axes_outer(window), level, self._axes)
        window = get_selection(get_table(indices), window)
        kernels = tuple(cax.coarse_grain()[jnp.newaxis, ...] for cax in caxs)
        ker_select = tuple(np.zeros(window.shape[0], dtype=window.dtype)
                           for _ in range(len(caxs)))
        axes = tuple(((0, self.ndim-i), (2,1)) for i in range(self.ndim))
        return axes_matmul(input, kernels, window, axes, ker_select), coarse

    def coarse_grain(self, input, level):
        """Coarse grain fine values to the previous level.

        Parameters:
        -----------
        input: jax.DeviceArray
            Input values on `level`.
        level: int
            On which refinment level the input is defined.
        """
        if level == 0:
            raise ValueError
        indices = self.main_indices[level]
        return self._coarse_grain(input, indices, level)

    def _refine_input(self, input, fine_missing, coarse_table, level):
        coarse, kernels, ker_select = (), (), ()
        for ax, idx in zip(self.axes(level), 
                           id_to_axisids(fine_missing, level+1, self._axes)):
            cc, ker, ker_sel = ax.refine_mat(idx)
            kernels += (ker,)
            ker_select += (ker_sel,)
            coarse += (cc,)
        coarse = axisids_to_id(np.stack(coarse, axis = 0), level, self._axes)
        coarse = get_selection(coarse_table, coarse)
        axes = tuple(((0,), (1,)) for _ in range(self.ndim))
        return axes_matmul(input, kernels, coarse, axes, ker_select)

    def refine_input(self, input, level):
        """Get missing values of the input on the next level to perform
        convolutions.

        Parameters:
        -----------
        input: jax.DeviceArray
            Input values on `level`.
        level: int
            On which refinment level the input is defined.
        """
        if level == self.maxlevel:
            raise ValueError
        fine_indices = self.main_indices[level + 1]
        table = get_table(self.indices[level])
        missing = get_kernel_window(fine_indices, level + 1, self._axes)
        missing = my_setdiff_indices(missing, fine_indices)
        if missing.size > 0:
            return self._refine_input(input, missing, table, level), missing
        return None

    def _batch_interpolation_selection(self, level, refine_indices, 
                                       coarse_table):
        if level == self.maxlevel:
            raise ValueError
        axs = self.axes(level)
        kids = id_to_axisids(refine_indices, level, self._axes)
        for i, aa in enumerate(axs):
            if aa.regular:
                kids[i] = 0
        kids = axisids_to_id(kids, level, self._axes)
        kids, isel = np.unique(kids, return_inverse=True)

        coarse, kernels = [], []
        for i, (ax, idx, kidx) in enumerate(zip(axs, 
                            id_to_axisids(refine_indices, level, self._axes), 
                            id_to_axisids(kids, level, self._axes))):
            cc, kk = ax.batch_interpolate(idx, kidx)
            kernels.append(kk)
            coarse.append(cc)
        coarse = my_axes_outer(coarse)
        coarse = axisids_to_id(coarse, level, self._axes)
        coarse = get_selection(coarse_table, coarse)
        ker_select = isel if kids.size > 1 else None
        return coarse, kernels, ker_select

    def batch_interconvolve(self, oldres, input, inputids, kernel, kerneltable, 
                            level):
        input = input.reshape((input.shape[0],-1))
        if level == 0:
            if oldres is not None:
                raise ValueError
            inds = self.main_indices[0]
            select = get_kernel_window(inds, 0, self._axes)
            interselect, interkernels, interker_select = None, None, None
        else:
            inds = my_setdiff_indices(self.main_indices[level-1], 
                                      self.indices[level-1])
            select = get_batch_kernel_window(inds, level-1, self._axes)
            coarse_tbl = get_table(self.main_indices[level-1])
            (interselect, interkernels, 
            interker_select) = self._batch_interpolation_selection(level-1, 
                                                                   inds, 
                                                                   coarse_tbl)
            dims = np.arange(2*self.ndim, dtype=int) + 1
            interkernels = (np.expand_dims(kk, 
                            tuple(np.delete(dims, [i, self.ndim+i])))
                            for i,kk in enumerate(interkernels))
            interkernels = reduce(lambda a,b: a*b, interkernels)
            shp = (interkernels.shape[0],
                   reduce(lambda a,b:a*b, interkernels.shape[1:self.ndim+1]),
                   reduce(lambda a,b:a*b, interkernels.shape[self.ndim+1:]))
            interselect = interselect.reshape((interselect.shape[0], shp[2]))
            shp = shp[1:] if interker_select is None else shp
            interkernels = jnp.array(interkernels.reshape(shp))

        select = get_selection(get_table(inputids), select)
        select = select.reshape((select.shape[0],-1))
        if kernel.shape[0] == 1:
            kernel = kernel[0]
            ker_select = None
        else:
            ker_select = get_selection(kerneltable, inds)

        def interconv(ores, rker, rsel, rksel, inp, ker, sel, ksel):
            ker = ker[ksel] if ksel is not None else ker
            res = jnp.tensordot(ker, inp[sel], axes=((1,2),(0,1)))
            if ores is not None:
                rker = rker[rksel] if rksel is not None else rker
                res += jnp.matmul(rker, ores[rsel])
            return res

        axes = (None, None, 0 if oldres is not None else None, 
                0 if interker_select is not None else None, 
                None, None, 0, 0 if ker_select is not None else None)
        return vmap(interconv, axes, 0)(oldres, interkernels, interselect,
                                        interker_select, input, kernel, select,
                                        ker_select).flatten()

    def copy(self):
        """Returns a copy of this instance of `MSChart`"""
        inds = tuple(ii.copy() for ii in self.indices)
        axes = tuple(ax.copy() for ax in self._axes)
        return MSChart(inds, axes, self._trafos)

    def refine(self, refine_indices, fine_axes = None, _want_ref_pairs = False):
        """Refine the chart at given `refine_indices`.
        
        Parameters:
        -----------
        refine_indices: iterable of numpy.ndarray
            Indices that exist on this chart that should get refined.
        fine_axes: tuple of RegularAxis (optional)
            If the currently finest level is refined, a new finest level is
            created. The axes for this new finest level may be provided here.
            If None, `RegularAxis.refine_axis` is used to create the new finest
            axes.
        _want_ref_pairs: bool (default False)
            Internal flag to also return the matching pairs of 
            (old_index, new_indices) for each refined index.
        Notes:
        ------
            Note that in order to obtain a consistent new `MSChart` additional
            indices may have to be refined to satisfy the conditions (i-iii)
            that indices must obey for a valid chart (see constructor of 
            `MSChart` for a definition). These additional indices get identified
            and automatically also refined by this function.
        """
        refine_indices = list(np.unique(rr) for rr in refine_indices)
        if len(refine_indices) != len(self.indices):
            raise ValueError
        chart = self.copy()
        if refine_indices[-1].size != 0:
            refine_indices.append(np.array([], dtype=refine_indices[-1].dtype))
            inds = chart._indices 
            inds += (np.array([], dtype=refine_indices[-1].dtype), )
            for i, aa in enumerate(chart.axes(chart.maxlevel)):
                if fine_axes is not None:
                    aa._fine_axis = fine_axes[i]
                else:
                    aa.refine_axis()
            chart = MSChart(inds, chart._axes, self._trafos)

        if _want_ref_pairs:
            pairs = [(np.array([]),)*2,]
        new_inds = [
            np.array([], dtype=chart.indices[0].dtype),] * chart.maxlevel
        new_inds.append(np.copy(chart.indices[-1]))
        for lvl in range(chart._maxlevel)[::-1]:
            if (my_setdiff_indices(refine_indices[lvl], 
                                   chart.indices[lvl]).size != 0):
                raise ValueError("Indices to be refined not in chart!")
            fine = get_fine_indices(refine_indices[lvl], lvl, chart._axes)
            fine = fine.flatten()
            fine.sort()
            if _want_ref_pairs:
                pairs.append((fine, refine_indices[lvl]))
            assert np.all(fine == np.unique(fine))
            missing = get_inter_window(fine, lvl+1, chart._axes)
            missing = get_coarse_index(missing, lvl+1, chart._axes)
            missing = my_setdiff_indices(missing, chart.main_indices[lvl])
            if missing.size > 0:
                if lvl == 0:
                    raise ValueError
                missing = get_coarse_index(missing, lvl, chart._axes)
                missing = my_setdiff_indices(missing, refine_indices[lvl-1])
                refine_indices[lvl-1] = np.append(refine_indices[lvl-1], 
                                                  missing)
                refine_indices[lvl-1].sort()
            # Check that none of the indices are in fine level
            sz = my_setdiff_indices(fine, chart.indices[lvl+1])
            if sz.size != fine.size:
                raise ValueError("Indices to be refined already in chart!")
            # Delete refined indices and insert new ones on fine level
            new_inds[lvl] = np.append(new_inds[lvl], 
                                      my_setdiff_indices(chart.indices[lvl], 
                                                         refine_indices[lvl]))
            new_inds[lvl].sort()
            new_inds[lvl+1] = np.append(new_inds[lvl+1], fine)
            new_inds[lvl+1].sort()
        new_inds = tuple(new_inds)
        new_chart = MSChart(new_inds, chart._axes, charted_trafos=chart._trafos)
        if _want_ref_pairs:
            return new_chart, chart, tuple(pairs[::-1])
        return new_chart