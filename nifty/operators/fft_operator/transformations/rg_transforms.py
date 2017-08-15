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

fftw = gdi.get('fftw')


class Transform(Loggable, object):
    """
        A generic fft object without any implementation.
    """

    def __init__(self, domain, codomain):
        self.domain = domain
        self.codomain = codomain

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


class MPIFFT(Transform):
    """
        The MPI-parallel FFTW pendant of a fft object.
    """

    def __init__(self, domain, codomain):

        if not hasattr(fftw, 'FFTW_MPI'):
            raise ImportError(
                "The MPI FFTW module is needed but not available.")

        super(MPIFFT, self).__init__(domain, codomain)

        # Enable caching
        fftw.interfaces.cache.enable()

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
        p = info.plan
        # Load the value into the plan
        if p.has_input:
            try:
                p.input_array[None] = val
            except ValueError:
                raise ValueError("Failed to load data into input_array of "
                                 "FFTW MPI-plan. Maybe the 1D slicing differs"
                                 "from n-D slicing?")
        # Execute the plan
        p()

        if p.has_output:
            result = p.output_array.copy()
            if result.shape != val.shape:
                raise ValueError("Output shape is different than input shape. "
                                 "Maybe fftw tries to optimize the "
                                 "bit-alignment? Try a different array-size.")
        else:
            return None

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

        local_result = current_info.fftw_interface(
            local_val,
            axes=axes,
            planner_effort='FFTW_ESTIMATE'
        )

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
                temp_val[slice_list] = result

        return_val.set_local_data(data=temp_val, copy=False)

        return return_val

    def transform(self, val, axes, **kwargs):
        """
            The MPI-parallel FFTW transform function.

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
        if not hasattr(fftw, 'FFTW_MPI'):
            raise ImportError(
                "The MPI FFTW module is needed but not available.")

        shape = (local_shape if axes is None else
                 [y for x, y in enumerate(local_shape) if x in axes])


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
            self._fftw_interface = fftw.interfaces.numpy_fft.fftn
        else:
            self._fftw_interface = fftw.interfaces.numpy_fft.ifftn

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
        self._plan = fftw.create_mpi_plan(
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


class SerialFFT(Transform):
    """
        The numpy fft pendant of a fft object.

    """
    def __init__(self, domain, codomain, use_fftw):
        super(SerialFFT, self).__init__(domain, codomain)

        if use_fftw and (fftw is None):
            raise ImportError(
                "The serial FFTW module is needed but not available.")

        self._use_fftw = use_fftw
        # Enable caching
        if self._use_fftw:
            fftw.interfaces.cache.enable()

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
        # perform the transformation
        if self._use_fftw:
            if self.codomain.harmonic:
                result_val = fftw.interfaces.numpy_fft.fftn(
                             local_val, axes=axes)
            else:
                result_val = fftw.interfaces.numpy_fft.ifftn(
                             local_val, axes=axes)
        else:
            if self.codomain.harmonic:
                result_val = np.fft.fftn(local_val, axes=axes)
            else:
                result_val = np.fft.ifftn(local_val, axes=axes)

        return result_val
