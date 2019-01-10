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

import itertools

import numpy as np

from .. import dobj, utilities
from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..multi_field import MultiField, MultiDomain
from ..field import Field
from .linear_operator import LinearOperator
import operator


class OuterProduct(LinearOperator):
    """Performs the pointwise outer product of two fields.

    Parameters
    ---------
    field: Field,
    domain: DomainTuple, the domain of the input field
    ---------
    """

    def __init__(self, field, domain):

        self._domain = domain
        self._field = field
        self._target = DomainTuple.make(
            tuple(sub_d for sub_d in field.domain._dom + domain._dom))

        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return Field.from_global_data(
                self._target, np.multiply.outer(
                    self._field.to_global_data(), x.to_global_data()))
        axes = len(self._field.shape)
        return Field.from_global_data(
            self._domain, np.tensordot(
                self._field.to_global_data(), x.to_global_data(),  axes))
