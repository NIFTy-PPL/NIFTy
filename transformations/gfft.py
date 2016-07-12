import numpy as np
from transform import Transform
from d2o import distributed_data_object
import nifty.nifty_utilities as utilities


class GFFT(Transform):

    """
        The gfft pendant of a fft object.

        Parameters
        ----------
        fft_module_name : String
            Switch between the gfft module used: 'gfft' and 'gfft_dummy'

    """

    def __init__(self, domain, codomain, fft_module):
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
                ftmachine='fft' if self.codomain.harmonic else 'ifft',
                in_zero_center=map(
                    bool, self.domain.paradict['zerocenter']
                ),
                out_zero_center=map(
                    bool, self.codomain.paradict['zerocenter']
                ),
                enforce_hermitian_symmetry=bool(
                    self.codomain.paradict['complexity']
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
            new_val = val.copy_empty(dtype=self.codomain.dtype)
            new_val.set_full_data(return_val, copy=False)
            # If the values living in domain are purely real, the result of
            # the fft is hermitian
            if self.domain.paradict['complexity'] == 0:
                new_val.hermitian = True
            return_val = new_val
        else:
            return_val = return_val.astype(self.codomain.dtype, copy=False)

        return return_val
