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
from ..field import Field
from .linear_operator import LinearOperator


class DomainTupleFieldInserter(LinearOperator):
    """Writes the content of a :class:`Field` into one slice of a
    :class:`DomainTuple`.

    Parameters
    ----------
    target : Domain, tuple of Domain or DomainTuple
    space : int
       The index of the sub-domain which is inserted.
    index : tuple
        Slice in new sub-domain in which the input field shall be written into.
    """

    def __init__(self, target, space, index):
        if not space <= len(target) or space < 0:
            raise ValueError
        self._target = DomainTuple.make(target)
        dom = list(self.target)
        dom.pop(space)
        self._domain = DomainTuple.make(dom)
        self._capability = self.TIMES | self.ADJOINT_TIMES

        new_space = target[space]
        nshp = new_space.shape
        fst_dims = sum(len(dd.shape) for dd in self.target[:space])

        if len(index) != len(nshp):
            raise ValueError("shape mismatch between new_space and position")
        for s, p in zip(nshp, index):
            if p < 0 or p >= s:
                raise ValueError("bad position value")

        self._slc = (slice(None),)*fst_dims + index

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = np.zeros(self.target.shape, dtype=x.dtype)
            res[self._slc] = x.val
            return Field(self.target, res)
        else:
            return Field(self.domain, x.val[self._slc])
