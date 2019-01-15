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
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from .linear_operator import LinearOperator


# MR FIXME: this needs a redesign to avoid most _global_data() calls
# Possible approach: keep everything defined on `domain` distributed and only
# collect the unstructured Fields.
class MaskOperator(LinearOperator):
    """Implementation of a mask response

    Takes a field, applies flags and returns the values of the field in a
    :class:`UnstructuredDomain`.

    Parameters
    ----------
    flags : Field
        Is converted to boolean. Where True, the input field is flagged.
    """
    def __init__(self, flags):
        if not isinstance(flags, Field):
            raise TypeError
        self._domain = DomainTuple.make(flags.domain)
        self._flags = np.logical_not(flags.to_global_data())
        self._target = DomainTuple.make(UnstructuredDomain(self._flags.sum()))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()
        if mode == self.TIMES:
            res = x[self._flags]
            return Field.from_global_data(self.target, res)
        res = np.empty(self.domain.shape, x.dtype)
        res[self._flags] = x
        res[~self._flags] = 0
        return Field.from_global_data(self.domain, res)
