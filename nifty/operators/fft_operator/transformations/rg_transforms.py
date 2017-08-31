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
        # perform the transformation
        if self.codomain.harmonic:
            result_val = pyfftw.interfaces.numpy_fft.fftn(
                         local_val, axes=axes)
        else:
            result_val = pyfftw.interfaces.numpy_fft.ifftn(
                         local_val, axes=axes)

        return result_val
