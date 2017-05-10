# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

import nifty.nifty_utilities as utilities
from nifty.operators.endomorphic_operator import EndomorphicOperator
from nifty.operators.fft_operator import FFTOperator
#from nifty.operators.smoothing_operator import smooth_util as su
from d2o import STRATEGIES

def smoothingHelper (x,sigma,dxmax=None):
    """ Performs the precomputations for Gaussian smoothing on a 1D irregular grid.

    Parameters
    ----------
    x: 1D floating point array or list containing the individual grid positions.
        Points must be given in ascending order.

    sigma: The sigma of the Gaussian with which the function living on x should
        be smoothed, in the same units as x.
    dxmax: (optional) The maximum distance up to which smoothing is performed,
        in the same units as x. Default is 3.01*sigma.

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

    if dxmax==None:
        dxmax=3.01*sigma

    x=np.asarray(x)

    ibegin=np.searchsorted(x,x-dxmax)
    nval=np.searchsorted(x,x+dxmax)-ibegin

    wgt=[]
    expfac=1./(2.*sigma*sigma)
    for i in range(x.size):
        t=x[ibegin[i]:ibegin[i]+nval[i]]-x[i]
        t=np.exp(-t*t*expfac)
        t*=1./np.sum(t)
        wgt.append(t)

    return ibegin,nval,wgt

def apply_kernel_along_array(power, startindex, endindex, distances,
    smooth_length, smoothing_width,ibegin,nval,wgt):

    if smooth_length == 0.0:
        return power[startindex:endindex]

    p_smooth = np.empty(endindex-startindex, dtype=np.float64)

    for i in xrange(startindex, endindex):
        p_smooth[i-startindex]=np.sum(power[ibegin[i]:ibegin[i]+nval[i]]*wgt[i])

    return p_smooth

def getShape(a):
    return tuple(a.shape)

def apply_along_axis(axis, arr, startindex, endindex, distances,
    smooth_length, smoothing_width):

    nd = arr.ndim
    if axis < 0:
        axis += nd
    if (axis >= nd):
        raise ValueError("axis must be less than arr.ndim; axis=%d, rank=%d."
            % (axis, nd))
    ibegin,nval,wgt=smoothingHelper(distances,smooth_length,smooth_length*smoothing_width)

    ind = np.zeros(nd-1, dtype=np.int)
    i = np.zeros(nd, dtype=object)
    shape = getShape(arr)
    indlist = np.asarray(range(nd))
    indlist = np.delete(indlist,axis)
    i[axis] = slice(None, None)
    outshape = np.asarray(shape).take(indlist)

    i.put(indlist, ind)

    Ntot =np.product(outshape)
    holdshape = outshape
    slicedArr = arr[tuple(i.tolist())]

    res = apply_kernel_along_array(slicedArr,
                                   startindex,
                                   endindex,
                                   distances,
                                   smooth_length,
                                   smoothing_width, ibegin,nval,wgt)

    outshape = np.asarray(getShape(arr))
    outshape[axis] = endindex - startindex
    outarr = np.zeros(outshape, dtype=np.float64)
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
        res = apply_kernel_along_array(slicedArr,
                                       startindex,
                                       endindex,
                                       distances,
                                       smooth_length,
                                       smoothing_width,ibegin,nval,wgt)
        outarr[tuple(i.tolist())] = res
        k += 1

    return outarr

class SmoothingOperator(EndomorphicOperator):
    # ---Overwritten properties and methods---
    def __init__(self, domain, sigma, log_distances=False,
                 default_spaces=None):
        super(SmoothingOperator, self).__init__(default_spaces)

        self._domain = self._parse_domain(domain)

        if len(self.domain) != 1:
            raise ValueError(
                'ERROR: SmoothOperator accepts only exactly one '
                'space as input domain.'
            )

        self.sigma = sigma
        self.log_distances = log_distances

        self._direct_smoothing_width = 3.

    def _inverse_times(self, x, spaces):
        return self._smoothing_helper(x, spaces, inverse=True)

    def _times(self, x, spaces):
        return self._smoothing_helper(x, spaces, inverse=False)

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

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = np.float(sigma)

    @property
    def log_distances(self):
        return self._log_distances

    @log_distances.setter
    def log_distances(self, log_distances):
        self._log_distances = bool(log_distances)

    def _smoothing_helper(self, x, spaces, inverse):
        if self.sigma == 0:
            return x.copy()

        # the domain of the smoothing operator contains exactly one space.
        # Hence, if spaces is None, but we passed LinearOperator's
        # _check_input_compatibility, we know that x is also solely defined
        # on that space
        if spaces is None:
            spaces = (0,)
        else:
            spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))

        try:
            result = self._fft_smoothing(x, spaces, inverse)
        except ValueError:
            result = self._direct_smoothing(x, spaces, inverse)
        return result

    def _fft_smoothing(self, x, spaces, inverse):
        Transformator = FFTOperator(x.domain[spaces[0]])

        # transform to the (global-)default codomain and perform all remaining
        # steps therein
        transformed_x = Transformator(x, spaces=spaces)
        codomain = transformed_x.domain[spaces[0]]
        coaxes = transformed_x.domain_axes[spaces[0]]

        # create the kernel using the knowledge of codomain about itself
        axes_local_distribution_strategy = \
            transformed_x.val.get_axes_local_distribution_strategy(axes=coaxes)

        kernel = codomain.get_distance_array(
            distribution_strategy=axes_local_distribution_strategy)

        if self.log_distances:
            kernel.apply_scalar_function(np.log, inplace=True)

        kernel.apply_scalar_function(
            codomain.get_fft_smoothing_kernel_function(self.sigma),
            inplace=True)

        # now, apply the kernel to transformed_x
        # this is done node-locally utilizing numpys reshaping in order to
        # apply the kernel to the correct axes
        local_transformed_x = transformed_x.val.get_local_data(copy=False)
        local_kernel = kernel.get_local_data(copy=False)

        reshaper = [transformed_x.shape[i] if i in coaxes else 1
                    for i in xrange(len(transformed_x.shape))]
        local_kernel = np.reshape(local_kernel, reshaper)

        # apply the kernel
        if inverse:
            local_transformed_x /= local_kernel
        else:
            local_transformed_x *= local_kernel

        transformed_x.val.set_local_data(local_transformed_x, copy=False)

        smoothed_x = Transformator.inverse_times(transformed_x, spaces=spaces)

        result = x.copy_empty()
        result.set_val(smoothed_x, copy=False)

        return result

    def _direct_smoothing(self, x, spaces, inverse):
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

        if self.log_distances:
            np.log(distance_array, out=distance_array)

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
                    (start_distance - self._direct_smoothing_width*self.sigma)
                augmented_start_index = \
                    np.searchsorted(distance_array, augmented_start_distance)
                true_start = start_index - augmented_start_index
                end_index = x.val.distributor.local_end
                end_distance = distance_array[end_index-1]
                augmented_end_distance = \
                    (end_distance + self._direct_smoothing_width*self.sigma)
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
        local_result = self._direct_smoothing_single_axis(
                                                    augmented_data,
                                                    data_axis,
                                                    augmented_distance_array,
                                                    true_start,
                                                    true_end,
                                                    inverse)
        result = x.copy_empty()
        result.val.set_local_data(local_result, copy=False)
        return result

    def _direct_smoothing_single_axis(self, data, data_axis, distances,
                                      true_start, true_end, inverse):
        if inverse:
            true_sigma = 1. / self.sigma
        else:
            true_sigma = self.sigma

        #MR: do not split for complex arrays!
        if data.dtype is np.dtype('float32'):
            distances = distances.astype(np.float32, copy=False)
            smoothed_data = apply_along_axis(
                                  data_axis, data,
                                  startindex=true_start,
                                  endindex=true_end,
                                  distances=distances,
                                  smooth_length=true_sigma,
                                  smoothing_width=self._direct_smoothing_width)
        elif data.dtype is np.dtype('float64'):
            distances = distances.astype(np.float64, copy=False)
            smoothed_data = apply_along_axis(
                                  data_axis, data,
                                  startindex=true_start,
                                  endindex=true_end,
                                  distances=distances,
                                  smooth_length=true_sigma,
                                  smoothing_width=self._direct_smoothing_width)

        elif np.issubdtype(data.dtype, np.complexfloating):
            real = self._direct_smoothing_single_axis(data.real,
                                                      data_axis,
                                                      distances,
                                                      true_start,
                                                      true_end, inverse)
            imag = self._direct_smoothing_single_axis(data.imag,
                                                      data_axis,
                                                      distances,
                                                      true_start,
                                                      true_end, inverse)

            return real + 1j*imag

        else:
            raise TypeError("Dtype %s not supported" % str(data.dtype))

        return smoothed_data
