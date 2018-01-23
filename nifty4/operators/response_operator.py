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

from ..field import Field
from ..spaces.field_array import FieldArray
from ..domain_tuple import DomainTuple
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
            return Field(self._target, val=x.weight(1).val)
        return Field(self._domain, val=x.val).weight(1)


def ResponseOperator(domain, sigma, sensitivity):
    # sensitivity has units 1/field/volume and gives a measure of how much
    # the instrument will excited when it is exposed to a certain field
    # volume amplitude
    domain = DomainTuple.make(domain)
    ncomp = len(sensitivity)
    if len(sigma) != ncomp or len(domain) != ncomp:
        raise ValueError("length mismatch between sigma, sensitivity "
                         "and domain")
    ncomp = len(sigma)
    if ncomp == 0:
        raise ValueError("Empty response operator not allowed")

    kernel = None
    sensi = None
    for i in range(ncomp):
        if sigma[i] > 0:
            op = FFTSmoothingOperator(domain, sigma[i], space=i)
            kernel = op if kernel is None else op*kernel
        if np.isscalar(sensitivity[i]):
            if sensitivity[i] != 1.:
                op = ScalingOperator(sensitivity[i], domain)
                sensi = op if sensi is None else op*sensi
        elif isinstance(sensitivity[i], Field):
            op = DiagonalOperator(sensitivity[i], domain=domain, spaces=i)
            sensi = op if sensi is None else op*sensi

    res = GeometryRemover(domain)
    if sensi is not None:
        res = res * sensi
    if kernel is not None:
        res = res * kernel
    return res
