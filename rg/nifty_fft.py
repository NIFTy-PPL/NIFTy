# -*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI
from d2o import distributed_data_object, STRATEGIES
from nifty.config import dependency_injector as gdi
from nifty.config import about
import nifty.nifty_utilities as utilities

pyfftw = gdi.get('pyfftw')
gfft = gdi.get('gfft')
gfft_dummy = gdi.get('gfft_dummy')


# Try to import pyfftw. If this fails fall back to gfft.
# If this fails fall back to local gfft_rg

# try:
#    import pyfftw
#    fft_machine='pyfftw'
# except(ImportError):
#    try:
#        import gfft
#        fft_machine='gfft'
#        about.infos.cprint('INFO: Using gfft')
#    except(ImportError):
#        import gfft_rg as gfft
#        fft_machine='gfft_fallback'
#        about.infos.cprint('INFO: Using builtin "plain" gfft version 0.1.0')


def fft_factory(fft_module_name):
    """
        A factory for fast-fourier-transformation objects.

        Parameters
        ----------
        fft_module_name : String
            Select an FFT module

        Returns
        -----
        fft : Returns a fft_object depending on the available packages.
        Hierarchy: pyfftw -> gfft -> built in gfft.

    """
    if fft_module_name == 'pyfftw':
        return FFTW()
    elif fft_module_name == 'gfft' or 'gfft_dummy':
        return GFFT(fft_module_name)
    else:
        raise ValueError('Given fft_module_name not known: ' +
                         str(fft_module_name))


class FFT(object):

    """
        A generic fft object without any implementation.
    """

    def __init__(self):
        pass

    def transform(self, val, domain, codomain, axes, **kwargs):
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


class FFTW(FFT):

    """
        The pyfftw pendant of a fft object.
    """

    def __init__(self):
        if 'pyfftw' not in gdi:
            raise ImportError("The module pyfftw is needed but not available.")

        self.name = 'pyfftw'
        # The plan_dict stores the FFTWTransformInfo objects which correspond
        # to a certain set of (field_val, domain, codomain) sets.
        self.info_dict = {}

        # initialize the dictionary which stores the values from
        # get_centering_mask
        self.centering_mask_dict = {}

        # Enable caching for pyfftw.interfaces
        pyfftw.interfaces.cache.enable()

    def get_centering_mask(self, to_center_input, dimensions_input,
                           offset_input=0):
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
                                             (np.array(args).ndim - 1)),
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

    def _get_transform_info(self, domain, codomain, is_local=False, **kwargs):
        # generate a id-tuple which identifies the domain-codomain setting
        temp_id = domain.__hash__() ^ (101 * codomain.__hash__())

        # generate the plan_and_info object if not already there
        if temp_id not in self.info_dict:
            if is_local:
                self.info_dict[temp_id] = FFTWLocalTransformInfo(
                    domain, codomain,
                    self, **kwargs
                )
            else:
                self.info_dict[temp_id] = FFTWMPITransfromInfo(
                    domain, codomain,
                    self, **kwargs
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

    def _local_transform(self, val, info, axes, domain, codomain):
        # Apply codomain centering mask
        if reduce(lambda x, y: x+y, codomain.paradict['zerocenter']):
            temp_val = np.copy(val)
            val = self._apply_mask(temp_val, info.cmask_codomain, axes)

        result = info.fftw_interface(val, axes=axes,
                                     planner_effort='FFTW_ESTIMATE')

        # Apply domain centering mask
        if reduce(lambda x, y: x+y, domain.paradict['zerocenter']):
            result = self._apply_mask(result, info.cmask_domain, axes)

        # Correct the sign if needed
        result *= info.sign

        return result

    def _mpi_transform(self, val, info, axes, domain, codomain):
        # Apply codomain centering mask
        if reduce(lambda x, y: x+y, codomain.paradict['zerocenter']):
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
            raise RuntimeError('ERROR: PyFFTW-MPI transform failed.')

        # Apply domain centering mask
        if reduce(lambda x, y: x+y, domain.paradict['zerocenter']):
            result = self._apply_mask(result, info.cmask_domain, axes)

        # Correct the sign if needed
        result *= info.sign

        return result

    def transform(self, val, domain, codomain, axes=None, **kwargs):
        """
            The pyfftw transform function.

            Parameters
            ----------
            val : distributed_data_object or numpy.ndarray
                The value-array of the field which is supposed to
                be transformed.

            domain : nifty.rg.nifty_rg.rg_space
                The domain of the space which should be transformed.

            codomain : nifty.rg.nifty_rg.rg_space
                The target into which the field should be transformed.

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
            raise ValueError("ERROR: Provided axes does not match array shape")

        # If the input is a numpy array we transform it locally
        if not isinstance(val, distributed_data_object):
            # Cast to a np.ndarray
            temp_val = np.asarray(val)

            current_info = self._get_transform_info(domain, codomain,
                                                    is_local=True,
                                                    **kwargs)

            # local transform doesn't apply transforms inplace
            return_val = self._local_transform(temp_val, current_info, axes,
                                               domain, codomain)
        else:
            if val.comm is not MPI.COMM_WORLD:
                raise RuntimeError('ERROR: Input array uses an unsupported \
                                   comm object')

            if val.distribution_strategy in STRATEGIES['slicing']:
                if axes is None or set(axes) == set(range(len(val.shape))) \
                        or 0 in axes:
                    if val.distribution_strategy != 'fftw':
                        temp_val = val.copy_empty(distribution_strategy='fftw')
                        temp_val.set_full_data(val, copy=False)

                        # Recursive call to transform
                        result = self.transform(temp_val, domain, codomain,
                                                axes, **kwargs)

                        return_val = result.copy_empty(
                            distribution_strategy=val.distribution_strategy
                        )
                        return_val.set_full_data(data=result, copy=False)
                    else:
                        current_info = self._get_transform_info(domain,
                                                                codomain,
                                                                **kwargs)

                        return_val = val.copy_empty(global_shape=val.shape,
                                                    dtype=codomain.dtype)

                        # Extract local data
                        local_val = val.get_local_data(copy=False)

                        # Create temporary storage for slices
                        temp_val = None

                        # If axes tuple includes all axes, set it to None
                        if set(axes) == set(range(len(val.shape))):
                            axes = None

                        for slice_list in\
                                utilities.get_slice_list(local_val.shape,
                                                         axes):
                            if slice_list == [slice(None, None)]:
                                inp = local_val
                            else:
                                if temp_val is None:
                                    temp_val = np.empty_like(local_val)
                                inp = local_val[slice_list]

                            result = self._mpi_transform(inp,
                                                         current_info,
                                                         axes, domain,
                                                         codomain)

                            if slice_list == [slice(None, None)]:
                                temp_val = result
                            else:
                                temp_val[slice_list] = result

                        return_val.set_local_data(data=temp_val, copy=False)
                else:
                    current_info = self._get_transform_info(domain, codomain,
                                                            is_local=True,
                                                            **kwargs)

                    # Compute transform for the local data
                    result = self._local_transform(
                        val.get_local_data(copy=False),
                        current_info, axes,
                        domain, codomain
                    )

                    # Create return object and insert results inplace
                    return_val = val.copy_empty(global_shape=val.shape,
                                                dtype=codomain.dtype)
                    return_val.set_local_data(data=result, copy=False)

                # If domain is purely real, the result of the FFT is hermitian
                if domain.paradict['complexity'] == 0:
                    return_val.hermitian = True
            else:
                temp_val = val.copy_empty(distribution_strategy='fftw')
                about.warnings.cprint('WARNING: Repacking d2o to fftw \
                                       distribution strategy')
                temp_val.set_full_data(val, copy=False)

                # Recursive call to take advantage of the fact that the data
                # necessary is already present on the nodes.
                result = self.transform(temp_val, domain, codomain, axes,
                                        **kwargs)

                return_val = val.copy_empty(
                    distribution_strategy=val.distribution_strategy
                )
                return_val.set_full_data(result, copy=False)

            # If domain is purely real, the result of the FFT is hermitian
            if domain.paradict['complexity'] == 0:
                return_val.hermitian = True

        return return_val


class FFTWTransformInfo(object):

    def __init__(self, domain, codomain, fftw_context, **kwargs):
        if pyfftw is None:
            raise ImportError("The module pyfftw is needed but not available.")

        # Create domain centering mask. The mask is the same size
        # as the domain's shape.
        self.cmask_domain = fftw_context.get_centering_mask(
            domain.paradict['zerocenter'],
            domain.get_shape(),
            True
        )

        self.cmask_codomain = fftw_context.get_centering_mask(
            codomain.paradict['zerocenter'],
            codomain.get_shape(),
            True
        )

        # If both domain and codomain are zero-centered the result,
        # will get a global minus. Store the sign to correct it.
        self.sign = (-1) ** np.sum(
            np.array(domain.paradict['zerocenter']) *
            np.array(codomain.paradict['zerocenter']) *
            (np.array(domain.get_shape()) // 2 % 2)
        )

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
    def __init__(self, domain, codomain, fftw_context, **kwargs):
        super(FFTWLocalTransformInfo, self).__init__(domain, codomain,
                                                     fftw_context, **kwargs)
        if codomain.harmonic:
            self._fftw_interface = pyfftw.interfaces.numpy_fft.fftn
        else:
            self._fftw_interface = pyfftw.interfaces.numpy_fftn.ifftn

    @property
    def fftw_interface(self):
        return self._fftw_interface

    @fftw_interface.setter
    def fftw_interface(self, interface):
        about.warnings.cprint('WARNING: FFTWLocalTransformInfo fftw_interface \
                               cannot be modified')


class FFTWMPITransfromInfo(FFTWTransformInfo):
    def __init__(self, domain, codomain, fftw_context, **kwargs):
        super(FFTWMPITransfromInfo, self).__init__(domain, codomain,
                                                   fftw_context, **kwargs)
        self._plan = pyfftw.create_mpi_plan(
            input_shape=domain.get_shape(),
            input_dtype='complex128',
            output_dtype='complex128',
            direction='FFTW_FORWARD' if codomain.harmonic else 'FFTW_BACKWARD',
            flags=["FFTW_ESTIMATE"],
            **kwargs
        )

    @property
    def plan(self):
        return self._plan

    @plan.setter
    def plan(self, plan):
        about.warnings.cprint('WARNING: FFTWMPITransfromInfo plan \
                               cannot be modified')


class GFFT(FFT):

    """
        The gfft pendant of a fft object.

        Parameters
        ----------
        fft_module_name : String
            Switch between the gfft module used: 'gfft' and 'gfft_dummy'

    """

    def __init__(self, fft_module_name):
        self.name = fft_module_name
        self.fft_machine = gdi.get(fft_module_name)
        if self.fft_machine is None:
            raise ImportError(
                "The gfft(_dummy)-module is needed but not available.")

    def transform(self, val, domain, codomain, axes=None, **kwargs):
        """
            The gfft transform function.

            Parameters
            ----------
            val : numpy.ndarray or distributed_data_object
                The value-array of the field which is supposed to
                be transformed.

            domain : nifty.rg.nifty_rg.rg_space
                The domain of the space which should be transformed.

            codomain : nifty.rg.nifty_rg.rg_space
                The target into which the field should be transformed.

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
            raise ValueError("ERROR: Provided axes does not match array shape")

        # GFFT doesn't accept d2o objects as input. Consolidate data from
        # all nodes into numpy.ndarray before proceeding.
        if isinstance(val, distributed_data_object):
            temp_inp = val.get_full_data()
        else:
            temp_inp = val

        # Cast input datatype to codomain's dtype
        temp_inp = temp_inp.astype(np.complex128, copy=False)

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
                ftmachine='fft' if codomain.harmonic else 'ifft',
                in_zero_center=map(bool, domain.paradict['zerocenter']),
                out_zero_center=map(bool, codomain.paradict['zerocenter']),
                enforce_hermitian_symmetry=bool(
                    codomain.paradict['complexity']
                ),
                W=-1,
                alpha=-1,
                verbose=False
            )
            if slice_list == [slice(None, None)]:
                return_val = inp
            else:
                return_val[slice_list] = inp

        if isinstance(val, distributed_data_object):
            new_val = val.copy_empty(dtype=codomain.dtype)
            new_val.set_full_data(return_val, copy=False)
            # If the values living in domain are purely real, the result of
            # the fft is hermitian
            if domain.paradict['complexity'] == 0:
                new_val.hermitian = True
            return_val = new_val
        else:
            return_val = return_val.astype(codomain.dtype, copy=False)

        return return_val
