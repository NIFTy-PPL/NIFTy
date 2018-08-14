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

from .. import dobj
from ..compat import *
from ..domain_tuple import DomainTuple
from ..field import Field
from .linear_operator import LinearOperator
from .. import utilities


class DomainDistributor(LinearOperator):
    def __init__(self, target, spaces):
        self._target = DomainTuple.make(target)
        self._spaces = utilities.parse_spaces(spaces, len(self._target))
        self._domain = [tgt for i, tgt in enumerate(self._target)
                        if i in self._spaces]
        self._domain = DomainTuple.make(self._domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            ldat = x.local_data if 0 in self._spaces else x.to_global_data()
            shp = []
            for i, tgt in enumerate(self._target):
                tmp = tgt.shape if i > 0 else tgt.local_shape
                shp += tmp if i in self._spaces else(1,)*len(tgt.shape)
            ldat = np.broadcast_to(ldat.reshape(shp), self._target.local_shape)
            return Field.from_local_data(self._target, ldat)
        else:
            return x.sum([s for s in range(len(x.domain))
                          if s not in self._spaces])
