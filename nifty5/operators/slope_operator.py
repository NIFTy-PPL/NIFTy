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

from ..domain_tuple import DomainTuple
from ..domains.log_rg_space import LogRGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from .linear_operator import LinearOperator


class SlopeOperator(LinearOperator):
    """Creates a slope on target.

    This operator creates a field on a LogRGSpace, which is created according
    to a slope of given entries, (mean, y-intercept). The slope mean is the
    power law of the field in normal-space.

    Parameters
    ----------
    target : LogRGSpace
        The target of the operator which needs to be one-dimensional.
    """

    def __init__(self, target):
        if not isinstance(target, LogRGSpace) or not len(target.shape) != 1:
            raise TypeError("Slope Operator only works for ndim == 1")
        self._domain = DomainTuple.make(UnstructuredDomain((2,)))
        self._target = DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        pos = self.target[0].get_k_array() - self.target[0].t_0[0]
        self._pos = pos[0, 1:]

    def apply(self, x, mode):
        self._check_input(x, mode)
        inp = x.to_global_data()
        if mode == self.TIMES:
            res = np.empty(self.target.shape, dtype=x.dtype)
            res[0] = 0
            res[1:] = inp[1] + inp[0]*self._pos
        else:
            res = np.array(
                [np.sum(self._pos*inp[1:]),
                 np.sum(inp[1:])], dtype=x.dtype)
        return Field.from_global_data(self._tgt(mode), res)
