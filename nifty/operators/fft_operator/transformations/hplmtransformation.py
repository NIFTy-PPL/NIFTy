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
from nifty import HPSpace, LMSpace
from slicing_transformation import SlicingTransformation

import lm_transformation_helper

pyHealpix = gdi.get('pyHealpix')


class HPLMTransformation(SlicingTransformation):

    # ---Overwritten properties and methods---

    def __init__(self, domain, codomain=None, module=None):
        if module is None:
            module = 'pyHealpix'

        if module != 'pyHealpix':
            raise ValueError("Unsupported SHT module.")

        if 'pyHealpix' not in gdi:
            raise ImportError(
                "The module pyHealpix is needed but not available")

        super(HPLMTransformation, self).__init__(domain, codomain, module)

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
            domain: HPSpace
                Space for which a codomain is to be generated

            Returns
            -------
            codomain : LMSpace
                A compatible codomain.
        """

        if not isinstance(domain, HPSpace):
            raise TypeError("domain needs to be a HPSpace")

        lmax = 2*domain.nside

        result = LMSpace(lmax=lmax)
        return result

    @classmethod
    def check_codomain(cls, domain, codomain):
        if not isinstance(domain, HPSpace):
            raise TypeError("domain is not a HPSpace")

        if not isinstance(codomain, LMSpace):
            raise TypeError("codomain must be a LMSpace.")

        lmax = codomain.lmax
        nside = domain.nside

        if lmax != 2*nside:
            cls.logger.warn("Unrecommended: lmax != 2*nside.")

        super(HPLMTransformation, cls).check_codomain(domain, codomain)

    def _transformation_of_slice(self, inp, **kwargs):
        lmax = self.codomain.lmax
        mmax = lmax

        if issubclass(inp.dtype.type, np.complexfloating):
            [resultReal,
             resultImag] = [pyHealpix.map2alm(x, lmax, mmax)
                            for x in (inp.real, inp.imag)]

            [resultReal,
             resultImag] = [lm_transformation_helper.buildIdx(x, lmax=lmax)
                            for x in [resultReal, resultImag]]

            result = self._combine_complex_result(resultReal, resultImag)

        else:
            result = pyHealpix.map2alm(inp, lmax, mmax)
            result = lm_transformation_helper.buildIdx(result, lmax=lmax)

        return result
