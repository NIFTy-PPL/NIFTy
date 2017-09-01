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

from .... import GLSpace, LMSpace
from .slicing_transformation import SlicingTransformation
from . import lm_transformation_helper

import pyHealpix


class GLLMTransformation(SlicingTransformation):

    def __init__(self, domain, codomain=None):
        super(GLLMTransformation, self).__init__(domain, codomain)

    @property
    def unitary(self):
        return False

    @classmethod
    def get_codomain(cls, domain):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  an instance of the :py:class:`LMSpace` class.

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

        return LMSpace(lmax=domain.nlat-1,mmax=(domain.nlon-1)//2)

    @classmethod
    def check_codomain(cls, domain, codomain):
        if not isinstance(domain, GLSpace):
            raise TypeError("domain is not a GLSpace")
        if not isinstance(codomain, LMSpace):
            raise TypeError("codomain must be a LMSpace.")
        super(GLLMTransformation, cls).check_codomain(domain, codomain)

    def _transformation_of_slice(self, inp):
        nlat = self.domain.nlat
        nlon = self.domain.nlon
        lmax = self.codomain.lmax
        mmax = self.codomain.mmax

        sjob = pyHealpix.sharpjob_d()
        sjob.set_Gauss_geometry(nlat, nlon)
        sjob.set_triangular_alm_info(lmax, mmax)
        if issubclass(inp.dtype.type, np.complexfloating):
            return self._combine_complex_result(
                lm_transformation_helper.buildIdx(sjob.map2alm(inp.real),
                                                  lmax=lmax),
                lm_transformation_helper.buildIdx(sjob.map2alm(inp.imag),
                                                  lmax=lmax))
        else:
            return lm_transformation_helper.buildIdx(sjob.map2alm(inp),
                                                     lmax=lmax)
