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
from .rg_transforms import SerialFFT
from .... import RGSpace

class RGRGTransformation(Transformation):

    # ---Overwritten properties and methods---

    def __init__(self, domain, codomain=None):
        super(RGRGTransformation, self).__init__(domain, codomain)
        self._transform = SerialFFT(self.domain, self.codomain)

    # ---Mandatory properties and methods---

    @property
    def unitary(self):
        return True

    @classmethod
    def get_codomain(cls, domain, zerocenter=None):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  either a shifted grid or a Fourier conjugate
            grid.

            Parameters
            ----------
            domain: RGSpace
                Space for which a codomain is to be generated
            zerocenter : {bool, numpy.ndarray}, *optional*
                Whether or not the grid is zerocentered for each axis or not
                (default: None).

            Returns
            -------
            codomain : nifty.rg_space
                A compatible codomain.
        """
        if not isinstance(domain, RGSpace):
            raise TypeError("domain needs to be a RGSpace")

        # parse the zerocenter input
        if zerocenter is None:
            zerocenter = domain.zerocenter
        # if the input is something scalar, cast it to a boolean
        else:
            temp = np.empty_like(domain.zerocenter)
            temp[:] = zerocenter
            zerocenter = temp

        # calculate the initialization parameters
        distances = 1. / (np.array(domain.shape) *
                          np.array(domain.distances))

        new_space = RGSpace(domain.shape,
                            zerocenter=zerocenter,
                            distances=distances,
                            harmonic=(not domain.harmonic))

        # better safe than sorry
        cls.check_codomain(domain, new_space)
        return new_space

    @classmethod
    def check_codomain(cls, domain, codomain):
        if not isinstance(domain, RGSpace):
            raise TypeError("domain is not a RGSpace")

        if not isinstance(codomain, RGSpace):
            raise TypeError("domain is not a RGSpace")

        if not np.all(np.array(domain.shape) ==
                      np.array(codomain.shape)):
            raise AttributeError("The shapes of domain and codomain must be "
                                 "identical.")

        if domain.harmonic == codomain.harmonic:
            raise AttributeError("domain.harmonic and codomain.harmonic must "
                                 "not be the same.")

        # Check if the distances match, i.e. dist' = 1 / (num * dist)
        if not np.all(
            np.absolute(np.array(domain.shape) *
                        np.array(domain.distances) *
                        np.array(codomain.distances) - 1) <
                1e-7):
            raise AttributeError("The grid-distances of domain and codomain "
                                 "do not match.")

        super(RGRGTransformation, cls).check_codomain(domain, codomain)

    def transform(self, val, axes=None, **kwargs):
        """
        RG -> RG transform method.

        Parameters
        ----------
        val : np.ndarray or distributed_data_object
            The value array which is to be transformed

        axes : None or tuple
            The axes along which the transformation should take place

        """
        if self._transform.codomain.harmonic:
            # correct for forward fft.
            # naively one would set power to 0.5 here in order to
            # apply effectively a factor of 1/sqrt(N) to the field.
            # BUT: the pixel volumes of the domain and codomain are different.
            # Hence, in order to produce the same scalar product, power===1.
            val = self._transform.domain.weight(val, power=1, axes=axes)

        # Perform the transformation
        if issubclass(val.dtype.type, np.complexfloating):
            Tval_real = self._transform.transform(val.real, axes,
                                                  **kwargs)
            Tval_imag = self._transform.transform(val.imag, axes,
                                                  **kwargs)
            if self.codomain.harmonic:
                Tval_real.real += Tval_real.imag
                Tval_real.imag = \
                    Tval_imag.real + Tval_imag.imag
            else:
                Tval_real.real -= Tval_real.imag
                Tval_real.imag = \
                    Tval_imag.real - Tval_imag.imag

            Tval = Tval_real
        else:
            Tval = self._transform.transform(val, axes, **kwargs)
            if self.codomain.harmonic:
                Tval.real += Tval.imag
            else:
                Tval.real -= Tval.imag
            Tval = Tval.real

        if not self._transform.codomain.harmonic:
            # correct for inverse fft.
            # See discussion above.
            Tval = self._transform.codomain.weight(Tval, power=-1, axes=axes)

        return Tval
