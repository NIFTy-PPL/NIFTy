# -*- coding: utf8 -*-

from __future__ import division
from builtins import range
import numpy as np

from .smoothing_operator import SmoothingOperator


class DirectSmoothingOperator(SmoothingOperator):
    def __init__(self, domain, sigma, log_distances=False,
                 default_spaces=None):
        super(DirectSmoothingOperator, self).__init__(domain,
                                                      sigma,
                                                      log_distances,
                                                      default_spaces)
        self.effective_smoothing_width = 3.01

    def _add_attributes_to_copy(self, copy, **kwargs):
        copy.effective_smoothing_width = self.effective_smoothing_width

        copy = super(DirectSmoothingOperator, self)._add_attributes_to_copy(
                                                                copy, **kwargs)
        return copy

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
            dxmax = self.effective_smoothing_width*sigma

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

    def _smooth(self, x, spaces, inverse):
        # infer affected axes
        # we rely on the knowledge, that `spaces` is a tuple with length 1.
        affected_axes = x.domain_axes[spaces[0]]

        if len(affected_axes) > 1:
            raise ValueError("By this implementation only one-dimensional "
                             "spaces can be smoothed directly.")

        affected_axis = affected_axes[0]

        distance_array = x.domain[spaces[0]].get_distance_array()

        #MR FIXME: this causes calls of log(0.) which should probably be avoided
        if self.log_distances:
            np.log(np.maximum(distance_array,1e-15), out=distance_array)

        augmented_data = x.val
        augmented_distance_array = distance_array
        true_start = 0
        true_end = x.shape[affected_axis]

        # perform the convolution along the affected axes
        # currently only one axis is supported
        data_axis = affected_axes[0]

        if inverse:
            true_sigma = 1. / self.sigma
        else:
            true_sigma = self.sigma

        local_result = self._apply_along_axis(
                              data_axis,
                              augmented_data,
                              startindex=true_start,
                              endindex=true_end,
                              distances=augmented_distance_array,
                              smooth_length=true_sigma,
                              smoothing_width=self.effective_smoothing_width)

        return local_result
