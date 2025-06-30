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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..field import Field
from .linear_operator import LinearOperator


class OuterProduct(LinearOperator):
    """Performs the point-wise outer product of two fields.

    Parameters
    ---------
    domain : DomainTuple, the domain of the input field
    field : :class:`nifty.cl.field.Field`
    ---------
    """
    def __init__(self, domain, field):
        self._domain = DomainTuple.make(domain)
        self._field = field
        self._target = DomainTuple.make(
            tuple(sub_d for sub_d in field.domain._dom + self._domain._dom))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def _device_preparation(self, x, mode):
        self._field = self._field.at(x.device_id)

    def apply(self, x, mode):
        self._check_input(x, mode)
        self._device_preparation(x, mode)
        if mode == self.TIMES:
            res = np.multiply.outer(self._field.val, x.val)
        else:
            axes = len(self._field.shape)
            res = np.tensordot(self._field.val, x.val, axes)
        return Field(self._tgt(mode), res)
