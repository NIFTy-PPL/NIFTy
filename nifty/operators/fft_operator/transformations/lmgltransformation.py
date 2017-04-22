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
from nifty import GLSpace, LMSpace

from slicing_transformation import SlicingTransformation
import lm_transformation_factory

pyHealpix = gdi.get('pyHealpix')


class LMGLTransformation(SlicingTransformation):

    # ---Overwritten properties and methods---

    def __init__(self, domain, codomain=None, module=None):
        if 'pyHealpix' not in gdi:
            raise ImportError(
                "The module pyHealpix is needed but not available.")

        super(LMGLTransformation, self).__init__(domain, codomain,
                                                 module=module)

    # ---Mandatory properties and methods---

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
            .. [#] M. Reinecke and D. Sverre Seljebotn, 2013,
                   "Libsharp - spherical
                   harmonic transforms revisited";
                   `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_
        """
        if not isinstance(domain, LMSpace):
            raise TypeError("domain needs to be a LMSpace")

        nlat = domain.lmax + 1
        nlon = domain.lmax*2 + 1

        result = GLSpace(nlat=nlat, nlon=nlon, dtype=domain.dtype)
        return result

    @classmethod
    def check_codomain(cls, domain, codomain):
        if not isinstance(domain, LMSpace):
            raise TypeError("domain is not a LMSpace")

        if not isinstance(codomain, GLSpace):
            raise TypeError("codomain must be a GLSpace.")

        nlat = codomain.nlat
        nlon = codomain.nlon
        lmax = domain.lmax
        mmax = domain.mmax

        if lmax != mmax:
            cls.Logger.warn("Unrecommended: codomain has lmax != mmax.")

        if nlat != lmax + 1:
            cls.Logger.warn("Unrecommended: codomain has nlat != lmax + 1.")

        if nlon != 2*lmax + 1:
            cls.Logger.warn("Unrecommended: domain has nlon != 2*lmax + 1.")

        super(LMGLTransformation, cls).check_codomain(domain, codomain)

    def _transformation_of_slice(self, inp, **kwargs):
        nlat = self.codomain.nlat
        nlon = self.codomain.nlon
        lmax = self.domain.lmax
        mmax = self.domain.mmax

        sjob = pyHealpix.sharpjob_d()
        sjob.set_Gauss_geometry(nlat, nlon)
        sjob.set_triangular_alm_info(lmax, mmax)
        if issubclass(inp.dtype.type, np.complexfloating):
            [resultReal,
             resultImag] = [lm_transformation_factory.buildLm(x, lmax=lmax)
                                        for x in (inp.real, inp.imag)]

            [resultReal, resultImag] = [sjob.alm2map(x)
                                        for x in [resultReal, resultImag]]

            result = self._combine_complex_result(resultReal, resultImag)

        else:
            result = lm_transformation_factory.buildLm(inp, lmax=lmax)
            result = sjob.alm2map(result)

        return result
