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

import lm_transformation_factory as ltf

pyHealpix = gdi.get('pyHealpix')


class HPLMTransformation(SlicingTransformation):

    # ---Overwritten properties and methods---

    def __init__(self, domain, codomain=None, module=None):
        if 'pyHealpix' not in gdi:
            raise ImportError(
                "The module pyHealpix is needed but not available")

        super(HPLMTransformation, self).__init__(domain, codomain,
                                                 module=module)

    # ---Mandatory properties and methods---

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
            raise TypeError(
                "domain needs to be a HPSpace")

        lmax = 2 * domain.nside

        result = LMSpace(lmax=lmax, dtype=np.dtype('float64'))
        cls.check_codomain(domain, result)
        return result

    @staticmethod
    def check_codomain(domain, codomain):
        if not isinstance(domain, HPSpace):
            raise TypeError(
                'ERROR: domain is not a HPSpace')

        if not isinstance(codomain, LMSpace):
            raise TypeError(
                'ERROR: codomain must be a LMSpace.')

        nside = domain.nside
        lmax = codomain.lmax

        return None

    def _transformation_of_slice(self, inp, **kwargs):
        lmax = self.codomain.lmax
        mmax = lmax

        sjob=pyHealpix.sharpjob_d()
        sjob.set_Healpix_geometry(nside)
        sjob.set_triangular_alm_info(lmax,mmax)
        if issubclass(inp.dtype.type, np.complexfloating):
            [resultReal, resultImag] = [sjob.map2alm(x)
                                        for x in (inp.real, inp.imag)]

            [resultReal, resultImag] = [ltf.buildIdx(x, lmax=lmax)
                                        for x in [resultReal, resultImag]]

            result = self._combine_complex_result(resultReal, resultImag)

        else:
            result = sjob.map2alm(inp)
            result = ltf.buildIdx(result, lmax=lmax)

        return result
