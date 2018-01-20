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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from .. import Field, FieldArray, DomainTuple
from .linear_operator import LinearOperator
from .fft_smoothing_operator import FFTSmoothingOperator
from .diagonal_operator import DiagonalOperator
from .scaling_operator import ScalingOperator
import numpy as np


class GeometryRemover(LinearOperator):
    def __init__(self, domain):
        super(GeometryRemover, self).__init__()
        self._domain = DomainTuple.make(domain)
        target_list = [FieldArray(dom.shape) for dom in self._domain]
        self._target = DomainTuple.make(target_list)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return Field(self._target, val=x.val)
        return Field(self._domain, val=x.val).weight(power=-1)


def ResponseOperator(domain, sigma, exposure):
    domain = DomainTuple.make(domain)
    ncomp = len(exposure)
    if len(sigma) != ncomp or len(domain) != ncomp:
        raise ValueError("length mismatch between sigma, exposure "
                         "and domain")
    ncomp = len(sigma)
    if ncomp == 0:
        raise ValueError("Empty response operator not allowed")

    kernel = None
    expo = None
    for i in range(ncomp):
        if sigma[i] > 0:
            op = FFTSmoothingOperator(domain, sigma[i], space=i)
            kernel = op if kernel is None else op*kernel
        if np.isscalar(exposure[i]):
            if exposure[i] != 1.:
                op = ScalingOperator(exposure[i], domain)
                expo = op if expo is None else op*expo
        elif isinstance(exposure[i], Field):
            op = DiagonalOperator(exposure[i], domain=domain, spaces=i)
            expo = op if expo is None else op*expo

    res = GeometryRemover(domain)
    if expo is not None:
        res = res * expo
    if kernel is not None:
        res = res * kernel
    return res