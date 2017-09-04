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
from .slicing_transformation import SlicingTransformation
from . import lm_transformation_helper


class LMGLTransformation(SlicingTransformation):
    def __init__(self, domain, codomain=None):
        super(LMGLTransformation, self).__init__(domain, codomain)

    @property
    def unitary(self):
        return False

    def _transformation_of_slice(self, inp):
        from pyHealpix import sharpjob_d

        lmax = self.domain.lmax
        mmax = self.domain.mmax

        sjob = sharpjob_d()
        sjob.set_Gauss_geometry(self.codomain.nlat, self.codomain.nlon)
        sjob.set_triangular_alm_info(lmax, mmax)
        if issubclass(inp.dtype.type, np.complexfloating):
            rr = lm_transformation_helper.buildLm(inp.real, lmax=lmax)
            ri = lm_transformation_helper.buildLm(inp.imag, lmax=lmax)
            return sjob.alm2map(rr) + 1j*sjob.alm2map(ri)

        else:
            result = lm_transformation_helper.buildLm(inp, lmax=lmax)
            return sjob.alm2map(result)
