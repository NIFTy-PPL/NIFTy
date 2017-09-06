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

from __future__ import division
import numpy as np
from .transformation import Transformation


class RGRGTransformation(Transformation):

    # ---Overwritten properties and methods---

    def __init__(self, domain, codomain=None):
        import pyfftw
        super(RGRGTransformation, self).__init__(domain, codomain)
        pyfftw.interfaces.cache.enable()
        self._fwd = self.codomain.harmonic

    # ---Mandatory properties and methods---

    @property
    def unitary(self):
        return True

    def _transform_helper(self, val, axes):
        from pyfftw.interfaces.numpy_fft import fftn, ifftn

        # Check if the axes provided are valid given the shape
        if axes is not None and \
                not all(axis in range(len(val.shape)) for axis in axes):
            raise ValueError("Provided axes does not match array shape")

        return fftn(val, axes=axes) if self._fwd else ifftn(val, axes=axes)

    def transform(self, val, axes=None):
        """
        RG -> RG transform method.

        Parameters
        ----------
        val : np.ndarray or distributed_data_object
            The value array which is to be transformed

        axes : None or tuple
            The axes along which the transformation should take place

        """
        fct=1.
        if self.codomain.harmonic:
            # correct for forward fft.
            # naively one would set power to 0.5 here in order to
            # apply effectively a factor of 1/sqrt(N) to the field.
            # BUT: the pixel volumes of the domain and codomain are different.
            # Hence, in order to produce the same scalar product, power===1.
            fct *= self.domain.weight()

        # Perform the transformation
        if issubclass(val.dtype.type, np.complexfloating):
            Tval_real = self._transform_helper(val.real, axes)
            Tval_imag = self._transform_helper(val.imag, axes)
            if self.codomain.harmonic:
                Tval_real.real += Tval_real.imag
                Tval_real.imag = Tval_imag.real + Tval_imag.imag
            else:
                Tval_real.real -= Tval_real.imag
                Tval_real.imag = Tval_imag.real - Tval_imag.imag

            Tval = Tval_real
        else:
            Tval = self._transform_helper(val, axes)
            if self.codomain.harmonic:
                Tval.real += Tval.imag
            else:
                Tval.real -= Tval.imag
            Tval = Tval.real

        if not self.codomain.harmonic:
            # correct for inverse fft.
            # See discussion above.
            fct /= self.codomain.weight()

        Tval *= fct
        return Tval
