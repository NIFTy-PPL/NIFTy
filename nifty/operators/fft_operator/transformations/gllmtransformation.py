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

import numpy as np

from nifty.config import dependency_injector as gdi
from nifty import GLSpace, LMSpace
from slicing_transformation import SlicingTransformation
import lm_transformation_helper

pyHealpix = gdi.get('pyHealpix')


class GLLMTransformation(SlicingTransformation):

    # ---Overwritten properties and methods---

    def __init__(self, domain, codomain=None, module=None):
        if module is None:
            module = 'pyHealpix'

        if module != 'pyHealpix':
            raise ValueError("Unsupported SHT module.")

        if 'pyHealpix' not in gdi:
            raise ImportError(
                "The module pyHealpix is needed but not available.")

        super(GLLMTransformation, self).__init__(domain, codomain, module)

    # ---Mandatory properties and methods---

    @property
    def unitary(self):
        return False

    @classmethod
    def get_codomain(cls, domain):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  an instance of the :py:class:`lm_space` class.

            Parameters
            ----------
            domain: GLSpace
                Space for which a codomain is to be generated

            Returns
            -------
            codomain : LMSpace
                A compatible codomain.
        """

        if not isinstance(domain, GLSpace):
            raise TypeError("domain needs to be a GLSpace")

        nlat = domain.nlat
        lmax = nlat - 1

        result = LMSpace(lmax=lmax)
        return result

    @classmethod
    def check_codomain(cls, domain, codomain):
        if not isinstance(domain, GLSpace):
            raise TypeError("domain is not a GLSpace")

        if not isinstance(codomain, LMSpace):
            raise TypeError("codomain must be a LMSpace.")

        nlat = domain.nlat
        nlon = domain.nlon
        lmax = codomain.lmax
        mmax = codomain.mmax

        if lmax != mmax:
            cls.logger.warn("Unrecommended: codomain has lmax != mmax.")

        if lmax != nlat - 1:
            cls.logger.warn("Unrecommended: codomain has lmax != nlat - 1.")

        if nlon != 2*nlat - 1:
            cls.logger.warn("Unrecommended: domain has nlon != 2*nlat - 1.")

        super(GLLMTransformation, cls).check_codomain(domain, codomain)

    def _transformation_of_slice(self, inp, **kwargs):
        nlat = self.domain.nlat
        nlon = self.domain.nlon
        lmax = self.codomain.lmax
        mmax = self.codomain.mmax

        sjob = pyHealpix.sharpjob_d()
        sjob.set_Gauss_geometry(nlat, nlon)
        sjob.set_triangular_alm_info(lmax, mmax)
        if issubclass(inp.dtype.type, np.complexfloating):
            [resultReal, resultImag] = [sjob.map2alm(x)
                                        for x in (inp.real, inp.imag)]

            [resultReal,
             resultImag] = [lm_transformation_helper.buildIdx(x, lmax=lmax)
                            for x in [resultReal, resultImag]]

            result = self._combine_complex_result(resultReal, resultImag)
        else:
            result = sjob.map2alm(inp)
            result = lm_transformation_helper.buildIdx(result, lmax=lmax)

        return result
