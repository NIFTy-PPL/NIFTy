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

from __future__ import absolute_import, division, print_function

import numpy as np

from ..compat import *
from ..domain_tuple import DomainTuple
from ..field import Field
from .linear_operator import LinearOperator


class DomainTupleFieldInserter(LinearOperator):
    def __init__(self, domain, new_space, index, position):
        '''Writes the content of a field into one slice of a DomainTuple.

        Parameters
        ----------
        domain : Domain, tuple of Domain or DomainTuple
        new_space : Domain, tuple of Domain or DomainTuple
        index : Integer
            Index at which new_space shall be added to domain.
        position : tuple
            Slice in new_space in which the input field shall be written into.
        '''
        self._domain = DomainTuple.make(domain)
        tgt = list(self.domain)
        tgt.insert(index, new_space)
        self._target = DomainTuple.make(tgt)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        fst_dims = sum(len(dd.shape) for dd in self.domain[:index])
        nshp = new_space.shape
        if len(position) != len(nshp):
            raise ValueError("shape mismatch between new_space and position")
        for s, p in zip(nshp, position):
            if p < 0 or p >= s:
                raise ValueError("bad position value")
        self._slc = (slice(None),)*fst_dims + position

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = np.zeros(self.target.shape, dtype=x.dtype)
            res[self._slc] = x.to_global_data()
            return Field.from_global_data(self.target, res)
        else:
            return Field.from_global_data(self.domain,
                                          x.to_global_data()[self._slc])
