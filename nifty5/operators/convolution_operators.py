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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domains.lm_space import LMSpace
from ..domains.hp_space import HPSpace
from ..domains.gl_space import GLSpace
from .endomorphic_operator import EndomorphicOperator
from .harmonic_operators import HarmonicTransformOperator
from ..domain_tuple import DomainTuple
from ..field import Field


def SphericalFuncConvolutionOperator(domain, func):
    """Convolves input with a radially symmetric kernel defined by `func`

    Parameters
    ----------
    domain: DomainTuple
            Domain of the operator. Must have exactly one entry, which is
            of type `HPSpace` or `GLSpace`.
    func:   function
            This function needs to take exactly one argument, which is
            colatitude in radians, and return the kernel amplitude at that
            colatitude.
    """
    if len(domain) != 1:
        raise ValueError("need exactly one domain")
    if not isinstance(domain[0], (HPSpace, GLSpace)):
        raise TypeError("need a spherical domain")
    kernel = domain[0].get_default_codomain().get_conv_kernel_from_func(func)
    return _SphericalConvolutionOperator(domain, kernel)


class _SphericalConvolutionOperator(EndomorphicOperator):
    """Convolves with kernel living on the appropriate LMSpace"""

    def __init__(self, domain, kernel):

        if len(domain) != 1:
            raise ValueError("need exactly one domain")
        if len(kernel.domain) != 1:
            raise ValueError("kernel needs exactly one domain")
        if not isinstance(domain[0], (HPSpace, GLSpace)):
            raise TypeError("need a spherical domain")
        self._domain = domain
        self.lm = domain[0].get_default_codomain()
        if self.lm != kernel.domain[0]:
            raise ValueError("Input domain and kernel are incompatible")
        self.kernel = kernel
        self.HT = HarmonicTransformOperator(self.lm, domain[0])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x_lm = self.HT.adjoint_times(x.weight(1))
        x_lm = x_lm * self.kernel * (4. * np.pi)
        return self.HT(x_lm)
