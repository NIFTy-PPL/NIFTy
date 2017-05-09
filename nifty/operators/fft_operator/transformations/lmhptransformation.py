# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

import numpy as np
from nifty.config import dependency_injector as gdi
from nifty import HPSpace, LMSpace
from slicing_transformation import SlicingTransformation
import lm_transformation_helper

pyHealpix = gdi.get('pyHealpix')


class LMHPTransformation(SlicingTransformation):

    # ---Overwritten properties and methods---

    def __init__(self, domain, codomain=None, module=None):
        if module is None:
            module = 'pyHealpix'

        if module != 'pyHealpix':
            raise ValueError("Unsupported SHT module.")

        if gdi.get('pyHealpix') is None:
            raise ImportError(
                "The module pyHealpix is needed but not available.")

        super(LMHPTransformation, self).__init__(domain, codomain, module)

    # ---Mandatory properties and methods---

    @property
    def unitary(self):
        return False

    @classmethod
    def get_codomain(cls, domain):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  a pixelization of the two-sphere.

            Parameters
            ----------
            domain : LMSpace
                Space for which a codomain is to be generated

            Returns
            -------
            codomain : HPSpace
                A compatible codomain.

            References
            ----------
            .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
                   High-Resolution Discretization and Fast Analysis of Data
                   Distributed on the Sphere", *ApJ* 622..759G.
        """
        if not isinstance(domain, LMSpace):
            raise TypeError("domain needs to be a LMSpace.")

        nside = max((domain.lmax + 1)//2, 1)
        result = HPSpace(nside=nside)
        return result

    @classmethod
    def check_codomain(cls, domain, codomain):
        if not isinstance(domain, LMSpace):
            raise TypeError("domain is not a LMSpace.")

        if not isinstance(codomain, HPSpace):
            raise TypeError("codomain must be a HPSpace.")

        nside = codomain.nside
        lmax = domain.lmax

        if lmax != 2*nside:
            cls.logger.warn("Unrecommended: lmax != 2*nside.")

        super(LMHPTransformation, cls).check_codomain(domain, codomain)

    def _transformation_of_slice(self, inp, **kwargs):
        if inp.dtype not in (np.float, np.complex):
            self.logger.warn("The input array has dtype: %s. The FFT will "
                             "be performed at double precision." %
                             str(inp.dtype))

        nside = self.codomain.nside
        lmax = self.domain.lmax
        mmax = lmax

        if issubclass(inp.dtype.type, np.complexfloating):
            [resultReal,
             resultImag] = [lm_transformation_helper.buildLm(x, lmax=lmax)
                            for x in (inp.real, inp.imag)]

            [resultReal, resultImag] = [pyHealpix.alm2map(x, lmax, mmax, nside)
                                        for x in [resultReal, resultImag]]

            result = self._combine_complex_result(resultReal, resultImag)

        else:
            result = lm_transformation_helper.buildLm(inp, lmax=lmax)
            result = pyHealpix.alm2map(result, lmax, mmax, nside)

        return result
