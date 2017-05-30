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

import warnings

import numpy as np
from d2o import distributed_data_object, STRATEGIES
from nifty.config import dependency_injector as gdi
import nifty.nifty_utilities as utilities

from keepers import Loggable

pyfftw = gdi.get('pyfftw')
pyfftw_scalar = gdi.get('pyfftw_scalar')


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


class FFTW(Transform):
    """
        The pyfftw pendant of a fft object.
    """

    def __init__(self, domain, codomain):

        if 'pyfftw' not in gdi:
            raise ImportError("The module pyfftw is needed but not available.")

        super(FFTW, self).__init__(domain, codomain)

        # Enable caching for pyfftw.interfaces
        pyfftw.interfaces.cache.enable()

        # The plan_dict stores the FFTWTransformInfo objects which correspond
        # to a certain set of (field_val, domain, codomain) sets.
        self.info_dict = {}

    def _get_transform_info(self, domain, codomain, axes, local_shape,
                            local_offset_Q, is_local, transform_shape=None,
                            **kwargs):
        # generate a id-tuple which identifies the domain-codomain setting
        temp_id = (domain, codomain, transform_shape, is_local)

        # generate the plan_and_info object if not already there
        if temp_id not in self.info_dict:
            if is_local:
                self.info_dict[temp_id] = FFTWLocalTransformInfo(
                    domain, codomain, axes, local_shape,
                    local_offset_Q, self, **kwargs
                )
            else:
                self.info_dict[temp_id] = FFTWMPITransfromInfo(
                    domain, codomain, axes, local_shape,
                    local_offset_Q, self, transform_shape, **kwargs
                )

        return self.info_dict[temp_id]

    def _atomic_mpi_transform(self, val, info, axes):
        # Apply codomain centering mask
        if reduce(lambda x, y: x + y, self.codomain.zerocenter):
            temp_val = np.copy(val)
            val = self._apply_mask(temp_val, info.cmask_codomain, axes)

        p = info.plan
        # Load the value into the plan
        if p.has_input:
            p.input_array[None] = val
        # Execute the plan
        p()

        if p.has_output:
            result = p.output_array
        else:
            return None

        # Apply domain centering mask
        if reduce(lambda x, y: x + y, self.domain.zerocenter):
            result = self._apply_mask(result, info.cmask_domain, axes)

        # Correct the sign if needed
        result *= info.sign

        return result

    def _local_transform(self, val, axes, **kwargs):
        ####
        # val must be numpy array or d2o with slicing distributor
        ###

        try:
            local_val = val.get_local_data(copy=False)
        except(AttributeError):
            local_val = val

        current_info = self._get_transform_info(self.domain,
                                                self.codomain,
                                                axes,
                                                local_shape=local_val.shape,
                                                local_offset_Q=False,
                                                is_local=True,
                                                **kwargs)

        # Apply codomain centering mask
        if reduce(lambda x, y: x + y, self.codomain.zerocenter):
            temp_val = np.copy(local_val)
            local_val = self._apply_mask(temp_val,
                                         current_info.cmask_codomain, axes)

        local_result = current_info.fftw_interface(
            local_val,
            axes=axes,
            planner_effort='FFTW_ESTIMATE'
        )

        # Apply domain centering mask
        if reduce(lambda x, y: x + y, self.domain.zerocenter):
            local_result = self._apply_mask(local_result,
                                            current_info.cmask_domain, axes)

        # Correct the sign if needed
        if current_info.sign != 1:
            local_result *= current_info.sign

        try:
            # Create return object and insert results inplace
            return_val = val.copy_empty(global_shape=val.shape,
                                        dtype=np.complex)
            return_val.set_local_data(data=local_result, copy=False)
        except(AttributeError):
            return_val = local_result

        return return_val

    def _repack_to_fftw_and_transform(self, val, axes, **kwargs):
        temp_val = val.copy_empty(distribution_strategy='fftw')
        self.logger.info("Repacking d2o to fftw distribution strategy")
        temp_val.set_full_data(val, copy=False)

        # Recursive call to transform
        result = self.transform(temp_val, axes, **kwargs)

        return_val = result.copy_empty(
            distribution_strategy=val.distribution_strategy)
        return_val.set_full_data(data=result, copy=False)

        return return_val

    def _mpi_transform(self, val, axes, **kwargs):

        local_offset_list = np.cumsum(
            np.concatenate([[0, ], val.distributor.all_local_slices[:, 2]])
        )
        local_offset_Q = bool(local_offset_list[val.distributor.comm.rank] % 2)
        return_val = val.copy_empty(global_shape=val.shape,
                                    dtype=np.complex)

        # Extract local data
        local_val = val.get_local_data(copy=False)

        # Create temporary storage for slices
        temp_val = None

        # If axes tuple includes all axes, set it to None
        if axes is not None:
            if set(axes) == set(range(len(val.shape))):
                axes = None

        current_info = None
        for slice_list in utilities.get_slice_list(local_val.shape, axes):
            if slice_list == [slice(None, None)]:
                inp = local_val
            else:
                if temp_val is None:
                    temp_val = np.empty_like(
                        local_val,
                        dtype=np.complex
                    )
                inp = local_val[slice_list]

            # This is in order to make FFTW behave properly when slicing input
            # over MPI ranks when the input is 1-dimensional. The default
            # behaviour is to optimize to take advantage of byte-alignment,
            # which doesn't match the slicing strategy for multi-dimensional
            # data.
            original_shape = None
            if len(inp.shape) == 1:
                original_shape = inp.shape
                inp = inp.reshape(inp.shape[0], 1)
                axes = (0, )

            if current_info is None:
                transform_shape = list(inp.shape)
                transform_shape[0] = val.shape[0]

                current_info = self._get_transform_info(
                    self.domain,
                    self.codomain,
                    axes,
                    local_shape=val.local_shape,
                    local_offset_Q=local_offset_Q,
                    is_local=False,
                    transform_shape=tuple(transform_shape),
                    **kwargs
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self._atomic_mpi_transform(inp, current_info, axes)

            if result is None:
                temp_val = np.empty_like(local_val)
            elif slice_list == [slice(None, None)]:
                temp_val = result
            else:
                # Reverting to the original shape i.e. before the input was
                # augmented with 1 to make FFTW behave properly.
                if original_shape is not None:
                    result = result.reshape(original_shape)
                temp_val[slice_list] = result

        return_val.set_local_data(data=temp_val, copy=False)

        return return_val

    def transform(self, val, axes, **kwargs):
        """
            The pyfftw transform function.

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

        # If the input is a numpy array we transform it locally
        if not isinstance(val, distributed_data_object):
            # Cast to a np.ndarray
            temp_val = np.asarray(val)

            # local transform doesn't apply transforms inplace
            return_val = self._local_transform(temp_val, axes)
        else:
            if val.distribution_strategy in STRATEGIES['slicing']:
                if axes is None or 0 in axes:
                    if val.distribution_strategy != 'fftw':
                        return_val = \
                            self._repack_to_fftw_and_transform(
                                val, axes, **kwargs
                            )
                    else:
                        return_val = self._mpi_transform(
                            val, axes, **kwargs
                        )
                else:
                    return_val = self._local_transform(
                        val, axes, **kwargs
                    )
            else:
                return_val = self._repack_to_fftw_and_transform(
                    val, axes, **kwargs
                )

        return return_val


class FFTWTransformInfo(object):
    def __init__(self, domain, codomain, axes, local_shape,
                 local_offset_Q, fftw_context, **kwargs):
        if pyfftw is None:
            raise ImportError("The module pyfftw is needed but not available.")

        shape = (local_shape if axes is None else
                 [y for x, y in enumerate(local_shape) if x in axes])

        self._cmask_domain = fftw_context.get_centering_mask(domain.zerocenter,
                                                             shape,
                                                             local_offset_Q)

        self._cmask_codomain = fftw_context.get_centering_mask(
                                                         codomain.zerocenter,
                                                         shape,
                                                         local_offset_Q)

        # If both domain and codomain are zero-centered the result,
        # will get a global minus. Store the sign to correct it.
        self._sign = (-1) ** np.sum(np.array(domain.zerocenter) *
                                    np.array(codomain.zerocenter) *
                                    (np.array(domain.shape) // 2 % 2))

    @property
    def cmask_domain(self):
        return self._cmask_domain

    @property
    def cmask_codomain(self):
        return self._cmask_codomain

    @property
    def sign(self):
        return self._sign


class FFTWLocalTransformInfo(FFTWTransformInfo):
    def __init__(self, domain, codomain, axes, local_shape,
                 local_offset_Q, fftw_context, **kwargs):
        super(FFTWLocalTransformInfo, self).__init__(domain,
                                                     codomain,
                                                     axes,
                                                     local_shape,
                                                     local_offset_Q,
                                                     fftw_context,
                                                     **kwargs)
        if codomain.harmonic:
            self._fftw_interface = pyfftw.interfaces.numpy_fft.fftn
        else:
            self._fftw_interface = pyfftw.interfaces.numpy_fft.ifftn

    @property
    def fftw_interface(self):
        return self._fftw_interface


class FFTWMPITransfromInfo(FFTWTransformInfo):
    def __init__(self, domain, codomain, axes, local_shape,
                 local_offset_Q, fftw_context, transform_shape, **kwargs):
        super(FFTWMPITransfromInfo, self).__init__(domain,
                                                   codomain,
                                                   axes,
                                                   local_shape,
                                                   local_offset_Q,
                                                   fftw_context,
                                                   **kwargs)
        self._plan = pyfftw.create_mpi_plan(
            input_shape=transform_shape,
            input_dtype='complex128',
            output_dtype='complex128',
            direction='FFTW_FORWARD' if codomain.harmonic else 'FFTW_BACKWARD',
            flags=["FFTW_ESTIMATE"],
            **kwargs
        )

    @property
    def plan(self):
        return self._plan


class NUMPYFFT(Transform):
    """
        The numpy fft pendant of a fft object.

    """

    def transform(self, val, axes, **kwargs):
        """
            The pyfftw transform function.

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
        # Enable caching for pyfftw_scalar.interfaces
        if 'pyfftw_scalar' in gdi:
            pyfftw_scalar.interfaces.cache.enable()

        # Check if the axes provided are valid given the shape
        if axes is not None and \
                not all(axis in range(len(val.shape)) for axis in axes):
            raise ValueError("Provided axes does not match array shape")

        return_val = val.copy_empty(global_shape=val.shape,
                                    dtype=np.complex)

        if (axes is None) or (0 in axes) or \
           (val.distribution_strategy not in STRATEGIES['slicing']):

            if val.distribution_strategy == 'not':
                local_val = val.get_local_data(copy=False)
            else:
                local_val = val.get_full_data()

            result_data = self._atomic_transform(local_val=local_val,
                                                 axes=axes,
                                                 local_offset_Q=False)
            return_val.set_full_data(result_data, copy=False)

        else:
            local_offset_list = np.cumsum(
                    np.concatenate([[0, ],
                                    val.distributor.all_local_slices[:, 2]]))
            local_offset_Q = \
                bool(local_offset_list[val.distributor.comm.rank] % 2)

            local_val = val.get_local_data()
            result_data = self._atomic_transform(local_val=local_val,
                                                 axes=axes,
                                                 local_offset_Q=local_offset_Q)

            return_val.set_local_data(result_data, copy=False)

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
        if 'pyfftw_scalar' in gdi:
            if self.codomain.harmonic:
                result_val = pyfftw_scalar.interfaces.numpy_fft.fftn(local_val, axes=axes)
            else:
                result_val = pyfftw_scalar.interfaces.numpy_fft.ifftn(local_val, axes=axes)
        else:
            if self.codomain.harmonic:
                result_val = np.fft.fftn(local_val, axes=axes)
            else:
                result_val = np.fft.ifftn(local_val, axes=axes)

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
