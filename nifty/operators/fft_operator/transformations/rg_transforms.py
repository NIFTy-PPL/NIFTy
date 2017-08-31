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
#
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from builtins import range
from builtins import object
import warnings

import numpy as np
from .... import nifty_utilities as utilities

from keepers import Loggable
from functools import reduce

import pyfftw


class Transform(Loggable, object):
    """
        A generic fft object without any implementation.
    """

    def __init__(self, domain, codomain):
        self.domain = domain
        self.codomain = codomain

        # initialize the dictionary which stores the values from
        # get_centering_mask
        self.centering_mask_dict = {}

    def get_centering_mask(self, to_center_input, dimensions_input,
                           offset_input=False):
        """
            Computes the mask, used to (de-)zerocenter domain and target
            fields.

            Parameters
            ----------
            to_center_input : tuple, list, numpy.ndarray
                A tuple of booleans which dimensions should be
                zero-centered.

            dimensions_input : tuple, list, numpy.ndarray
                A tuple containing the mask's desired shape.

            offset_input : int, boolean
                Specifies whether the zero-th dimension starts with an odd
                or and even index, i.e. if it is shifted.

            Returns
            -------
            result : np.ndarray
                A 1/-1-alternating mask.
        """
        # cast input
        to_center = np.array(to_center_input)
        dimensions = np.array(dimensions_input)

        # if none of the dimensions are zero centered, return a 1
        if np.all(to_center == 0):
            return 1

        if np.all(dimensions == np.array(1)) or \
                np.all(dimensions == np.array([1])):
            return dimensions
        # The dimensions of size 1 must be sorted out for computing the
        # centering_mask. The depth of the array will be restored in the
        # end.
        size_one_dimensions = []
        temp_dimensions = []
        temp_to_center = []
        for i in range(len(dimensions)):
            if dimensions[i] == 1:
                size_one_dimensions += [True]
            else:
                size_one_dimensions += [False]
                temp_dimensions += [dimensions[i]]
                temp_to_center += [to_center[i]]
        dimensions = np.array(temp_dimensions)
        to_center = np.array(temp_to_center)
        # cast the offset_input into the shape of to_center
        offset = np.zeros(to_center.shape, dtype=int)
        # if the first dimension has length 1 and has an offset, restore the
        # global minus by hand
        if not size_one_dimensions[0]:
            offset[0] = int(offset_input)
        # check for dimension match
        if to_center.size != dimensions.size:
            raise TypeError(
                'The length of the supplied lists does not match.')

        # build up the value memory
        # compute an identifier for the parameter set
        temp_id = tuple(
            (tuple(to_center), tuple(dimensions), tuple(offset)))
        if temp_id not in self.centering_mask_dict:
            # use np.tile in order to stack the core alternation scheme
            # until the desired format is constructed.
            core = np.fromfunction(
                lambda *args: (-1) **
                              (np.tensordot(to_center,
                                            args +
                                            offset.reshape(offset.shape +
                                                           (1,) *
                                                           (np.array(
                                                              args).ndim - 1)),
                                            1)),
                (2,) * to_center.size)
            # Cast the core to the smallest integers we can get
            core = core.astype(np.int8)

            centering_mask = np.tile(core, dimensions // 2)
            # for the dimensions of odd size corresponding slices must be
            # added
            for i in range(centering_mask.ndim):
                # check if the size of the certain dimension is odd or even
                if (dimensions % 2)[i] == 0:
                    continue
                # prepare the slice object
                temp_slice = (slice(None),) * i + (slice(-2, -1, 1),) + \
                             (slice(None),) * (centering_mask.ndim - 1 - i)
                # append the slice to the centering_mask
                centering_mask = np.append(centering_mask,
                                           centering_mask[temp_slice],
                                           axis=i)
            # Add depth to the centering_mask where the length of a
            # dimension was one
            temp_slice = ()
            for i in range(len(size_one_dimensions)):
                if size_one_dimensions[i]:
                    temp_slice += (None,)
                else:
                    temp_slice += (slice(None),)
            centering_mask = centering_mask[temp_slice]
            # if the first dimension has length 1 and has an offset, restore
            # the global minus by hand
            if size_one_dimensions[0] and offset_input:
                centering_mask *= -1

            self.centering_mask_dict[temp_id] = centering_mask
        return self.centering_mask_dict[temp_id]

    def _apply_mask(self, val, mask, axes):
        """
            Apply centering mask to an array.

            Parameters
            ----------
            val: distributed_data_object or numpy.ndarray
                The value-array on which the mask should be applied.

            mask: numpy.ndarray
                The mask to be applied.

            axes: tuple
                The axes which are to be transformed.

            Returns
            -------
            distributed_data_object or np.nd_array
                Mask input array by multiplying it with the mask.
        """
        # reshape mask if necessary
        if axes:
            mask = mask.reshape(
                [y if x in axes else 1
                 for x, y in enumerate(val.shape)]
            )
        return val * mask

    def transform(self, val, axes, **kwargs):
        """
            A generic ff-transform function.

            Parameters
            ----------
            field_val : distributed_data_object
                The value-array of the field which is supposed to
                be transformed.

            domain : nifty.rg.nifty_rg.rg_space
                The domain of the space which should be transformed.

            codomain : nifty.rg.nifty_rg.rg_space
                The taget into which the field should be transformed.
        """
        raise NotImplementedError


class SerialFFT(Transform):
    """
        The numpy fft pendant of a fft object.

    """
    def __init__(self, domain, codomain):
        super(SerialFFT, self).__init__(domain, codomain)

        pyfftw.interfaces.cache.enable()

    def transform(self, val, axes, **kwargs):
        """
            The scalar FFT transform function.

            Parameters
            ----------
            val : distributed_data_object or numpy.ndarray
                The value-array of the field which is supposed to
                be transformed.

            axes: tuple, None
                The axes which should be transformed.

            **kwargs : *optional*
                Further kwargs are passed to the create_mpi_plan routine.

            Returns
            -------
            result : np.ndarray or distributed_data_object
                Fourier-transformed pendant of the input field.
        """

        # Check if the axes provided are valid given the shape
        if axes is not None and \
                not all(axis in range(len(val.shape)) for axis in axes):
            raise ValueError("Provided axes does not match array shape")

        return_val = np.empty(val.shape, dtype=np.complex)

        local_val = val

        result_data = self._atomic_transform(local_val=local_val,
                                             axes=axes,
                                             local_offset_Q=False)
        return_val=result_data

        return return_val

    def _atomic_transform(self, local_val, axes, local_offset_Q):

        # some auxiliaries for the mask computation
        local_shape = local_val.shape
        shape = (local_shape if axes is None else
                 [y for x, y in enumerate(local_shape) if x in axes])

        # Apply codomain centering mask
        if reduce(lambda x, y: x + y, self.codomain.zerocenter):
            temp_val = np.copy(local_val)
            mask = self.get_centering_mask(self.codomain.zerocenter,
                                           shape,
                                           local_offset_Q)
            local_val = self._apply_mask(temp_val, mask, axes)

        # perform the transformation
        if self.codomain.harmonic:
            result_val = pyfftw.interfaces.numpy_fft.fftn(
                         local_val, axes=axes)
        else:
            result_val = pyfftw.interfaces.numpy_fft.ifftn(
                         local_val, axes=axes)

        # Apply domain centering mask
        if reduce(lambda x, y: x + y, self.domain.zerocenter):
            mask = self.get_centering_mask(self.domain.zerocenter,
                                           shape,
                                           local_offset_Q)
            result_val = self._apply_mask(result_val, mask, axes)

        # If both domain and codomain are zero-centered the result,
        # will get a global minus. Store the sign to correct it.
        sign = (-1) ** np.sum(np.array(self.domain.zerocenter) *
                              np.array(self.codomain.zerocenter) *
                              (np.array(self.domain.shape) // 2 % 2))
        if sign != 1:
            result_val *= sign

        return result_val
