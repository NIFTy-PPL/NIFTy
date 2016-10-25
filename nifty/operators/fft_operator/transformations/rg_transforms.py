import warnings

import numpy as np
from d2o import distributed_data_object, STRATEGIES
from nifty.config import dependency_injector as gdi
import nifty.nifty_utilities as utilities

from keepers import Loggable

pyfftw = gdi.get('pyfftw')


class Transform(object, Loggable):
    """
        A generic fft object without any implementation.
    """

    def __init__(self, domain, codomain):
        pass

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
        self.domain = domain
        self.codomain = codomain

        if 'pyfftw' not in gdi:
            raise ImportError("The module pyfftw is needed but not available.")

        # Enable caching for pyfftw.interfaces
        pyfftw.interfaces.cache.enable()

        # The plan_dict stores the FFTWTransformInfo objects which correspond
        # to a certain set of (field_val, domain, codomain) sets.
        self.info_dict = {}

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
            self.centering_mask_dict[temp_id] = centering_mask
        return self.centering_mask_dict[temp_id]

    def _get_transform_info(self, domain, codomain, axes, local_shape,
                            local_offset_Q, is_local, transform_shape=None,
                            **kwargs):
        # generate a id-tuple which identifies the domain-codomain setting
        temp_id = (domain.__hash__() ^
                   (101 * codomain.__hash__()) ^
                   (211 * transform_shape.__hash__()) ^
                   (131 * is_local.__hash__())
                   )

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

    def _atomic_mpi_transform(self, val, info, axes):
        # Apply codomain centering mask
        if reduce(lambda x, y: x + y, self.codomain.zerocenter):
            temp_val = np.copy(val)
            val = self._apply_mask(temp_val, info.cmask_codomain, axes)

        p = info.plan
        # Load the value into the plan
        if p.has_input:
            p.input_array[:] = val
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
                                        dtype=self.codomain.dtype)
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
                                    dtype=self.codomain.dtype)

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
                        dtype=self.codomain.dtype
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

        self.cmask_domain = fftw_context.get_centering_mask(domain.zerocenter,
                                                            shape,
                                                            local_offset_Q)

        self.cmask_codomain = fftw_context.get_centering_mask(
                                                        codomain.zerocenter,
                                                        shape,
                                                        local_offset_Q)

        # If both domain and codomain are zero-centered the result,
        # will get a global minus. Store the sign to correct it.
        self.sign = (-1) ** np.sum(np.array(domain.zerocenter) *
                                   np.array(codomain.zerocenter) *
                                   (np.array(domain.shape) // 2 % 2))

    @property
    def cmask_domain(self):
        return self._domain_centering_mask

    @cmask_domain.setter
    def cmask_domain(self, cmask):
        self._domain_centering_mask = cmask

    @property
    def cmask_codomain(self):
        return self._codomain_centering_mask

    @cmask_codomain.setter
    def cmask_codomain(self, cmask):
        self._codomain_centering_mask = cmask

    @property
    def sign(self):
        return self._sign

    @sign.setter
    def sign(self, sign):
        self._sign = sign


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


class GFFT(Transform):
    """
        The gfft pendant of a fft object.

        Parameters
        ----------
        fft_module_name : String
            Switch between the gfft module used: 'gfft' and 'gfft_dummy'

    """

    def __init__(self, domain, codomain, fft_module=None):
        if fft_module is None:
            fft_module = gdi['gfft_dummy']

        self.domain = domain
        self.codomain = codomain
        self.fft_machine = fft_module

    def transform(self, val, axes, **kwargs):
        """
            The gfft transform function.

            Parameters
            ----------
            val : numpy.ndarray or distributed_data_object
                The value-array of the field which is supposed to
                be transformed.

            axes : None or tuple
                The axes which should be transformed.

            **kwargs : *optional*
                Further kwargs are not processed.

            Returns
            -------
            result : np.ndarray or distributed_data_object
                Fourier-transformed pendant of the input field.
        """
        # Check if the axes provided are valid given the shape
        if axes is not None and \
                not all(axis in range(len(val.shape)) for axis in axes):
            raise ValueError("Provided axes does not match array shape")

        # GFFT doesn't accept d2o objects as input. Consolidate data from
        # all nodes into numpy.ndarray before proceeding.
        if isinstance(val, distributed_data_object):
            temp_inp = val.get_full_data()
        else:
            temp_inp = val

        # Array for storing the result
        return_val = None

        for slice_list in utilities.get_slice_list(temp_inp.shape, axes):

            # don't copy the whole data array
            if slice_list == [slice(None, None)]:
                inp = temp_inp
            else:
                # initialize the return_val object if needed
                if return_val is None:
                    return_val = np.empty_like(temp_inp)
                inp = temp_inp[slice_list]

            inp = self.fft_machine.gfft(
                inp,
                in_ax=[],
                out_ax=[],
                ftmachine='fft' if self.codomain.harmonic else 'ifft',
                in_zero_center=map(bool, self.domain.zerocenter),
                out_zero_center=map(bool, self.codomain.zerocenter),
                # enforce_hermitian_symmetry=bool(self.codomain.complexity),
                enforce_hermitian_symmetry=False,
                W=-1,
                alpha=-1,
                verbose=False
            )
            if slice_list == [slice(None, None)]:
                return_val = inp
            else:
                return_val[slice_list] = inp

        if isinstance(val, distributed_data_object):
            new_val = val.copy_empty(dtype=self.codomain.dtype)
            new_val.set_full_data(return_val, copy=False)
            return_val = new_val
        else:
            return_val = return_val.astype(self.codomain.dtype, copy=False)

        return return_val
