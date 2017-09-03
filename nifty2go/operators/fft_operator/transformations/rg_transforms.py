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

from builtins import object, range
import pyfftw


class SerialFFT(object):
    """
        The pyfftw pendant of a fft object.
    """
    def __init__(self, domain, codomain):
        self.domain = domain
        self.codomain = codomain

        pyfftw.interfaces.cache.enable()

    def transform(self, val, axes):
        """
            The scalar FFT transform function.

            Parameters
            ----------
            val : numpy.ndarray
                The value-array of the field which is supposed to
                be transformed.

            axes: tuple, None
                The axes which should be transformed.

            Returns
            -------
            result : numpy.ndarray
                Fourier-transformed pendant of the input field.
        """

        # Check if the axes provided are valid given the shape
        if axes is not None and \
                not all(axis in range(len(val.shape)) for axis in axes):
            raise ValueError("Provided axes does not match array shape")

        if self.codomain.harmonic:
            return pyfftw.interfaces.numpy_fft.fftn(val, axes=axes)
        else:
            return pyfftw.interfaces.numpy_fft.ifftn(val, axes=axes)
