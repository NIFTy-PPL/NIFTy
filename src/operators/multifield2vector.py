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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.unstructured_domain import UnstructuredDomain
from ..multi_domain import MultiDomain
from ..sugar import makeField
from .linear_operator import LinearOperator


class Multifield2Vector(LinearOperator):
    """Flatten a MultiField and return a Field with unstructured domain and the
    same number of degrees of freedom.

    Parameters
    ----------
    domain: MultiDomain
        Domain of the operator

    Notes
    -----
    This operator works only on MultiFields that have the same dtype for all entries.
    """

    def __init__(self, domain):
        if not isinstance(domain, MultiDomain):
            raise NotImplementedError("This operator only works on MultiDomains")
        self._dof = domain.size
        self._domain = domain
        self._target = DomainTuple.make(UnstructuredDomain(self._dof))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        ii = 0
        if mode == self.TIMES:
            res = np.empty(self.target.shape, _unique_dtype([vv.dtype for vv in x.values()]))
            for key in self.domain.keys():
                arr = x[key].flatten()
                res[ii:ii + arr.size] = arr
                ii += arr.size
        else:
            res = {}
            for key in self.domain.keys():
                n = self.domain[key].size
                shp = self.domain[key].shape
                res[key] = x[ii:ii + n].reshape(shp)
                ii += n
        return makeField(self._tgt(mode), res)


def _unique_dtype(lst):
    if all(ll == lst[0] for ll in lst):
        return lst[0]
    raise RuntimeError("Dtype is not unique")
