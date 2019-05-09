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

from ..domains.rg_space import RGSpace
from ..domains.lm_space import LMSpace
from ..domains.hp_space import HPSpace
from ..domains.gl_space import GLSpace
from .endomorphic_operator import EndomorphicOperator
from .harmonic_operators import HarmonicTransformOperator
from .diagonal_operator import DiagonalOperator
from .simple_linear_operators import WeightApplier
from ..domain_tuple import DomainTuple
from ..field import Field
from .. import utilities


def FuncConvolutionOperator(domain, func, space=None, without_mean=False):
    """Convolves input with a radially symmetric kernel defined by `func`

    Parameters
    ----------
    domain: DomainTuple
        Domain of the operator.
    func: function
        This function needs to take exactly one argument, which is
        colatitude in radians, and return the kernel amplitude at that
        colatitude.
    space: int, optional
        The index of the subdomain on which the operator should act
        If None, it is set to 0 if `domain` contains exactly one space.
        `domain[space]` must be of type `RGSpace`, `HPSpace`, or `GLSpace`.
    without_mean: bool, optional
        If `True`, subtracts the input mean before applying the convolution
        and adds it back afterwards.

    Notes
    -----
    The operator assumes periodic boundaries in the input domain. This means
    for a sufficiently broad function a point source close to the boundary will
    blur into the opposite side of the image. Zero padding can be applied to
    avoid this behaviour.
    """
    domain = DomainTuple.make(domain)
    space = utilities.infer_space(domain, space)
    if not isinstance(domain[space], (RGSpace, HPSpace, GLSpace)):
        raise TypeError("unsupported domain")
    codomain = domain[space].get_default_codomain()
    kernel = codomain.get_conv_kernel_from_func(func)
    return _ConvolutionOperator(domain, kernel, space, without_mean)


def _ConvolutionOperator(domain, kernel, space=None, without_mean=False):
    domain = DomainTuple.make(domain)
    space = utilities.infer_space(domain, space)
    if len(kernel.domain) != 1:
        raise ValueError("kernel needs exactly one domain")
    if not isinstance(domain[space], (HPSpace, GLSpace, RGSpace)):
        raise TypeError("need RGSpace, HPSpace, or GLSpace")
    lm = [d for d in domain]
    lm[space] = lm[space].get_default_codomain()
    lm = DomainTuple.make(lm)
    if lm[space] != kernel.domain[0]:
        raise ValueError("Input domain and kernel are incompatible")
    HT = HarmonicTransformOperator(lm, domain[space], space)
    diag = DiagonalOperator(kernel*domain[space].total_volume, lm, (space,))
    wgt = WeightApplier(domain, space, 1)
    op = HT(diag(HT.adjoint(wgt)))
    if without_mean:
        return _ApplicationWithoutMeanOperator(domain, op)
    else:
        return op


class _ApplicationWithoutMeanOperator(EndomorphicOperator):
    def __init__(self, domain, op):
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._domain = domain
        self._op = op
        if (op.domain != domain) or (op.domain != op.target):
            raise TypeError("domains incompatible")

    def apply(self, x, mode):
        self._check_input(x, mode)
        mean = x.mean()
        return mean + self._op.apply(x - mean, mode)

    def __repr__(self):
        from ..utilities import indent
        return "\n".join((
            "_ApplicationWithoutMeanOperator:",
            indent(self._op.__repr__())))
