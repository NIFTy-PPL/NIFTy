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
# Copyright(C) 2022 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

from ..domain_tuple import DomainTuple
from ..sugar import makeField
from ..utilities import myassert
from .linear_operator import LinearOperator


class TransposeOperator(LinearOperator):
    def __init__(self, domain, indices):
        self._domain = DomainTuple.make(domain)
        indices = tuple(indices)
        if len(indices) != len(self._domain):
            raise IndexError("Either too many or too few indices given.")
        self._target = DomainTuple.make(self._domain[ind] for ind in indices)
        if self._domain.size != self._target.size:
            raise ValueError("List of indices not complete")
        self._capability = self._all_ops
        self._np_indices = _niftyspace_to_np_indices(self._domain, indices)
        self._indices = indices

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode in (self.TIMES, self.ADJOINT_INVERSE_TIMES):
            x = np.transpose(x, self._np_indices)
        else:
            x = np.transpose(x, np.argsort(self._np_indices))
        return makeField(self._tgt(mode), x)

    def __repr__(self):
        return f'Transpose (indices={self._indices})'


def _niftyspace_to_np_indices(domain, indices):
    np_indices = []
    dimensions = np.cumsum((0,) + tuple(len(dd.shape) for dd in domain))
    for ind in indices:
        np_indices.extend(range(dimensions[ind], dimensions[ind+1]))
    res = tuple(np_indices)
    myassert(len(res) == len(domain.shape))
    return res

