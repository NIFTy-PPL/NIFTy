# -*- coding: utf8 -*-

from __future__ import division
from builtins import range
import numpy as np

from d2o import STRATEGIES

from ..endomorphic_operator import EndomorphicOperator


class DirectSmoothingOperator(EndomorphicOperator):
    def __init__(self, domain, sigma, log_distances=False,
                 default_spaces=None):
        super(DirectSmoothingOperator, self).__init__(default_spaces)

        self._domain = self._parse_domain(domain)
        if len(self._domain) != 1:
            raise ValueError("DirectSmoothingOperator only accepts exactly one"
                             " space as input domain.")

        self._sigma = sigma
        self._log_distances = log_distances
        self._effective_smoothing_width = 3.01

    def _times(self, x, spaces):
        if self.sigma == 0:
            return x.copy()

        # the domain of the smoothing operator contains exactly one space.
        # Hence, if spaces is None, but we passed LinearOperator's
        # _check_input_compatibility, we know that x is also solely defined
        # on that space
        if spaces is None:
            spaces = (0,)

        return self._smooth(x, spaces)

    # ---Mandatory properties and methods---
    @property
    def domain(self):
        return self._domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False

    # ---Added properties and methods---

    @property
    def sigma(self):
        return self._sigma

    @property
    def log_distances(self):
        return self._log_distances

    def _precompute(self, x, sigma, dxmax=None):
        """ Does precomputations for Gaussian smoothing on a 1D irregular grid.

        Parameters
        ----------
        x: 1D floating point array or list containing the individual grid
            positions. Points must be given in ascending order.

        sigma: The sigma of the Gaussian with which the function living on x
            should be smoothed, in the same units as x.
        dxmax: (optional) The maximum distance up to which smoothing is
            performed, in the same units as x. Default is 3.01*sigma.

        Returns
        -------
        ibegin: integer array of the same size as x
            ibegin[i] is the minimum grid index to consider when computing the
            smoothed value at grid index i
        nval: integer array of the same size as x
            nval[i] is the number of indices to consider when computing the
            smoothed value at grid index i.
        wgt: list with the same number of entries as x
            wgt[i] is an array with nval[i] entries containing the
            normalized smoothing weights.
        """

        if dxmax is None:
            dxmax = self._effective_smoothing_width*sigma

        x = np.asarray(x)

        ibegin = np.searchsorted(x, x-dxmax)
        nval = np.searchsorted(x, x+dxmax) - ibegin

        wgt = []
        expfac = 1. / (2. * sigma*sigma)
        for i in range(x.size):
            if nval[i] > 0:
                t = x[ibegin[i]:ibegin[i]+nval[i]]-x[i]
                t = np.exp(-t*t*expfac)
                t *= 1./np.sum(t)
                wgt.append(t)
            else:
                wgt.append(np.array([]))

        return ibegin, nval, wgt

    def _apply_kernel_along_array(self, power, startindex, endindex,
                                  distances, smooth_length, smoothing_width,
                                  ibegin, nval, wgt):

        if smooth_length == 0.0:
            return power[startindex:endindex]

        p_smooth = np.zeros(endindex-startindex, dtype=power.dtype)
        for i in range(startindex, endindex):
            imin = max(startindex, ibegin[i])
            imax = min(endindex, ibegin[i]+nval[i])
            p_smooth[imin:imax] += (power[i] *
                                    wgt[i][imin-ibegin[i]:imax-imin+ibegin[i]])

        return p_smooth

    def _apply_along_axis(self, axis, arr, startindex, endindex, distances,
                          smooth_length, smoothing_width):

        nd = arr.ndim
        if axis < 0:
            axis += nd
        if (axis >= nd):
            raise ValueError(
                "axis must be less than arr.ndim; axis=%d, rank=%d."
                % (axis, nd))
        ibegin, nval, wgt = self._precompute(
                distances, smooth_length, smooth_length*smoothing_width)

        ind = np.zeros(nd-1, dtype=np.int)
        i = np.zeros(nd, dtype=object)
        shape = arr.shape
        indlist = np.asarray(list(range(nd)))
        indlist = np.delete(indlist, axis)
        i[axis] = slice(None, None)
        outshape = np.asarray(shape).take(indlist)

        i.put(indlist, ind)

        Ntot = np.product(outshape)
        holdshape = outshape
        slicedArr = arr[tuple(i.tolist())]
        res = self._apply_kernel_along_array(
                    slicedArr, startindex, endindex, distances,
                    smooth_length, smoothing_width, ibegin, nval, wgt)

        outshape = np.asarray(arr.shape)
        outshape[axis] = endindex - startindex
        outarr = np.zeros(outshape, dtype=arr.dtype)
        outarr[tuple(i.tolist())] = res
        k = 1
        while k < Ntot:
            # increment the index
            ind[nd-1] += 1
            n = -1
            while (ind[n] >= holdshape[n]) and (n > (1-nd)):
                ind[n-1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            slicedArr = arr[tuple(i.tolist())]
            res = self._apply_kernel_along_array(
                    slicedArr, startindex, endindex, distances,
                    smooth_length, smoothing_width, ibegin, nval, wgt)
            outarr[tuple(i.tolist())] = res
            k += 1

        return outarr

    def _smooth(self, x, spaces):
        # infer affected axes
        # we rely on the knowledge, that `spaces` is a tuple with length 1.
        affected_axes = x.domain_axes[spaces[0]]

        if len(affected_axes) > 1:
            raise ValueError("By this implementation only one-dimensional "
                             "spaces can be smoothed directly.")

        affected_axis = affected_axes[0]

        distance_array = x.domain[spaces[0]].get_distance_array(
            distribution_strategy='not')
        distance_array = distance_array.get_local_data(copy=False)

        #MR FIXME: this causes calls of log(0.) which should probably be avoided
        if self.log_distances:
            np.log(np.maximum(distance_array,1e-15), out=distance_array)

        # collect the local data + ghost cells
        local_data_Q = False

        if x.distribution_strategy == 'not':
            local_data_Q = True
        elif x.distribution_strategy in STRATEGIES['slicing']:
            # infer the local start/end based on the slicing information of
            # x's d2o. Only gets non-trivial for axis==0.
            if 0 != affected_axis:
                local_data_Q = True
            else:
                start_index = x.val.distributor.local_start
                start_distance = distance_array[start_index]
                augmented_start_distance = \
                    (start_distance -
                     self._effective_smoothing_width*self.sigma)
                augmented_start_index = \
                    np.searchsorted(distance_array, augmented_start_distance)
                true_start = start_index - augmented_start_index
                end_index = x.val.distributor.local_end
                end_distance = distance_array[end_index-1]
                augmented_end_distance = \
                    (end_distance + self._effective_smoothing_width*self.sigma)
                augmented_end_index = \
                    np.searchsorted(distance_array, augmented_end_distance)
                true_end = true_start + x.val.distributor.local_length
                augmented_slice = slice(augmented_start_index,
                                        augmented_end_index)

                augmented_data = x.val.get_data(augmented_slice,
                                                local_keys=True,
                                                copy=False)
                augmented_data = augmented_data.get_local_data(copy=False)

                augmented_distance_array = distance_array[augmented_slice]

        else:
            raise ValueError("Direct smoothing not implemented for given"
                             "distribution strategy.")

        if local_data_Q:
            # if the needed data resides on the nodes already, the necessary
            # are the same; no matter what the distribution strategy was.
            augmented_data = x.val.get_local_data(copy=False)
            augmented_distance_array = distance_array
            true_start = 0
            true_end = x.shape[affected_axis]

        # perform the convolution along the affected axes
        # currently only one axis is supported
        data_axis = affected_axes[0]

        local_result = self._apply_along_axis(
                              data_axis,
                              augmented_data,
                              startindex=true_start,
                              endindex=true_end,
                              distances=augmented_distance_array,
                              smooth_length=self.sigma,
                              smoothing_width=self._effective_smoothing_width)

        result = x.copy_empty()
        result.val.set_local_data(local_result, copy=False)
        return result
