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

from .. import dobj
from ..compat import *
from ..domain_tuple import DomainTuple
from ..domains.log_rg_space import LogRGSpace
from ..field import Field
from .endomorphic_operator import EndomorphicOperator


class SymmetrizingOperator(EndomorphicOperator):
    def __init__(self, domain, space=0):
        self._domain = DomainTuple.make(domain)
        self._space = int(space)
        dom = self._domain[self._space]
        if not (isinstance(dom, LogRGSpace) and not dom.harmonic):
            raise TypeError

    @property
    def domain(self):
        return self._domain

    def apply(self, x, mode):
        self._check_input(x, mode)
        tmp = x.val.copy()
        ax = dobj.distaxis(tmp)
        globshape = tmp.shape
        for i in self._domain.axes[self._space]:
            lead = (slice(None),)*i
            if i == ax:
                tmp = dobj.redistribute(tmp, nodist=(ax,))
            tmp2 = dobj.local_data(tmp)
            tmp2[lead+(slice(1, None),)] -= tmp2[lead+(slice(None, 0, -1),)]
            if i == ax:
                tmp = dobj.redistribute(tmp, dist=ax)
        return Field(self.target, val=tmp)

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
